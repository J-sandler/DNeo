"""HF verifier and shared DNeo cohort graphs; no native worker dependency."""
import contextlib
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from sglang.srt.speculative.dneo.runtime.graph_target import GraphHFTarget
class RowKVCache:
    """Static KV storage with independent logical write positions per request."""

    def __init__(self, layers, batch, heads, capacity, head_dim, device, dtype):
        self.capacity = capacity
        self.buffers = tuple(tuple(torch.zeros(batch, heads, capacity, head_dim,
                                              device=device, dtype=dtype)
                                   for _ in range(2)) for _ in range(layers))
        self.positions = None
        self.row_offset = 0

    def update(self, key, value, layer_idx, cache_kwargs=None):
        batch, heads, width, dim = key.shape
        positions = self.positions[:batch, :width]
        indices = positions[:, None, :, None].expand(batch, heads, width, dim)
        result = tuple(buffer[self.row_offset:self.row_offset + batch]
                       for buffer in self.buffers[layer_idx])
        for destination, source in zip(result, (key, value)):
            destination.scatter_(2, indices, source)
        return result

    def get_seq_length(self, layer_idx=0):
        return 0

    def get_max_cache_shape(self, layer_idx=0):
        return self.capacity


class CohortHFTarget(GraphHFTarget):
    @torch.inference_mode()
    def __init__(self, target, max_batch=32, capacity=1024, feature_layers=(), feature_source="attention_input"):
        super().__init__(target, capacity, capture_features=False)
        self.features = [None] * len(self.backbone.layers)
        self.feature_layers = tuple(feature_layers)
        if feature_source not in {"attention_input", "block_output"}:
            raise ValueError("unknown verifier feature convention")
        self.feature_source = feature_source
        self.capture_features = bool(feature_layers)
        self.cache = RowKVCache(target.config.num_hidden_layers, max_batch,
                                target.config.num_key_value_heads, capacity,
                                target.config.head_dim, target.device, target.dtype)
        self.graphs = {}
        self.lengths = torch.zeros(max_batch, device=target.device, dtype=torch.long)
        self.key_indices = torch.arange(capacity, device=target.device)

    @contextlib.contextmanager
    def packed_layers(self):
        handles = []
        for index, layer in enumerate(self.backbone.layers):
            if index in self.feature_layers:
                def capture(module, inputs, output, index=index):
                    self.features[index] = output
                source = layer.input_layernorm if self.feature_source == "attention_input" else layer
                handles.append(source.register_forward_hook(capture))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _run(self, ids, positions, mask, last_logit_only=False, logit_positions=None):
        self.cache.positions = positions
        with self.packed_layers():
            output = self.backbone(input_ids=ids, past_key_values=self.cache,
                                   position_ids=positions, attention_mask={"full_attention": mask},
                                   use_cache=True, output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state
        if logit_positions is not None:
            rows = torch.arange(ids.shape[0], device=ids.device)
            hidden = hidden[rows, logit_positions][:, None]
        elif last_logit_only:
            hidden = hidden[:, -1:]
        features = (torch.stack([self.features[index] for index in self.feature_layers], dim=2)
                    if self.capture_features else None)
        return self.target.lm_head(hidden), features

    def _mask(self, positions):
        return torch.where(self.key_indices[None, None, None] <= positions[:, None, :, None],
                           0.0, torch.finfo(self.target.dtype).min).to(self.target.dtype)

    @torch.inference_mode()
    def prepare(self, batches=(1, 4, 16, 32), widths=(1, 16)):
        from sglang.srt.speculative.dneo.runtime.cuda_graph import capture
        pool = torch.cuda.graph_pool_handle()
        for batch in batches:
            for width in widths:
                ids = torch.zeros(batch, width, device=self.target.device, dtype=torch.long)
                positions = torch.arange(width, device=ids.device)[None].repeat(batch, 1)
                mask = self._mask(positions)
                graph, output = capture(lambda: self._run(ids, positions, mask), pool=pool)
                self.graphs[batch, width] = graph, ids, positions, mask, output
                print(f"hf_target_graph=batch{batch}_width{width}", flush=True)

    @torch.inference_mode()
    def forward_at(self, ids, positions, last_logit_only=False, logit_positions=None):
        key = tuple(ids.shape)
        if key in self.graphs and not last_logit_only and logit_positions is None:
            graph, static_ids, static_positions, mask, output = self.graphs[key]
            static_ids.copy_(ids)
            static_positions.copy_(positions)
            mask.copy_(self._mask(positions))
            graph.replay()
            return output
        return self._run(ids, positions, self._mask(positions), last_logit_only, logit_positions)

    @torch.inference_mode()
    def prefill_at(self, ids, positions, logit_positions, microbatch=64):
        if ids.shape[0] <= microbatch:
            return self.forward_at(ids, positions, logit_positions=logit_positions)
        logits, features = [], []
        try:
            for start in range(0, ids.shape[0], microbatch):
                stop = start + microbatch
                self.cache.row_offset = start
                value, hidden = self.forward_at(ids[start:stop], positions[start:stop],
                                                logit_positions=logit_positions[start:stop])
                logits.append(value)
                if hidden is not None:
                    features.append(hidden)
        finally:
            self.cache.row_offset = 0
        return torch.cat(logits), torch.cat(features) if features else None


class DNeoCohortDraft:
    def __init__(self, model, capacity, max_batch=32):
        from sglang.srt.speculative.dneo.runtime.engine import DNeoEngine
        from sglang.srt.speculative.dneo.runtime.feature_layout import feature_layer_ids
        worker = DNeoEngine()
        worker._target_worker = None
        worker.model_runner = SimpleNamespace(decode_cuda_graph_runner=SimpleNamespace(max_bs=max_batch))
        worker._head_profiles = {}
        worker.dneo_model = model
        worker.block_size = model.block_size
        worker.capacity = capacity
        worker._max_batch = max_batch
        worker._captured = [None] * len(feature_layer_ids(model.conditioner))
        worker._mask_token_id = model.mask_token_id
        worker.draft_temperature = 1.0
        worker._disable_graph = False
        worker._profile_path = None
        worker._reuse_memory_kv = True
        worker._verify_feature_buffer = torch.zeros(
            max_batch * model.block_size, len(worker._captured), model.embed_tokens.weight.shape[1],
            device="cuda", dtype=torch.bfloat16)
        worker._init_batched_state()
        self.worker = worker
        self.model = model
        self.feature_layers = feature_layer_ids(model.conditioner)

    def reset(self, batch):
        worker = self.worker
        worker._active_slots = worker._graph_slots[:batch]
        worker._memory_valid.zero_()
        worker._prefix.zero_()

    def commit(self, features, positions, commit=None):
        worker = self.worker
        batch, width = positions.shape
        if commit is None:
            contexts, (cos, sin) = worker.runtime.route(features, positions)
            if worker.runtime.grouped_prefix_commit is not None:
                worker.runtime.grouped_prefix_commit(contexts, cos, sin, positions, worker._graph_slots[:batch])
            else:
                for layer, context, buffers in zip(worker.runtime.draft.layers, contexts, worker.runtime.prefix_buffers):
                    projected = layer._project_prefix(context, cos, sin)
                    for destination, value in zip(buffers, projected):
                        index = positions[:, None, :, None].expand_as(value)
                        destination[:batch].scatter_(2, index, value)
            return
        worker._verify_feature_buffer[:batch * width].copy_(features.flatten(0, 1))
        worker._graph_positions[:batch].copy_(positions)
        worker._commit[:batch].copy_(commit)
        worker._batch_graphs[batch][3].replay()

    def propose(self, anchor, lengths):
        worker = self.worker
        batch = anchor.numel()
        worker._prefix[:batch].copy_(lengths)
        worker._graph_block_ids[:batch, 0].copy_(anchor)
        graph, output, _, _ = worker._batch_graphs[batch]
        worker._draft_output = output
        graph.replay()
        return output

    def accept(self, logits, temperature):
        worker = self.worker
        batch = logits.shape[0]
        worker._graph_logits[:batch].copy_(logits)
        graph, output = worker._batch_graphs[batch][2][int(temperature)]
        graph.replay()
        return output

