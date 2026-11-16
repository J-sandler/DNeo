"""Native Transformers Qwen3 CUDA graphs with explicit static-KV commit length."""
import torch
from transformers import StaticCache


class GraphHFTarget:
    def __init__(self, target, capacity=1024, capture_features=True, batch_size=1, feature_layer_ids=None):
        self.target = target
        self.backbone = target.model
        self.capacity = capacity
        self.capture_features = capture_features
        self.feature_capture = None
        if capture_features and feature_layer_ids is not None:
            from sglang.srt.speculative.dneo.runtime.feature_layout import NormalizedFeatureCapture
            self.feature_capture = NormalizedFeatureCapture(self.backbone, feature_layer_ids)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.cache = StaticCache(config=target.config, max_cache_len=capacity)
        self.length = 0
        self.graphs = {}
        self.force_eager = False
        self.pool = torch.cuda.graph_pool_handle()

    def close(self):
        if self.feature_capture is not None:
            self.feature_capture.close()

    def reset(self):
        self.length = 0
        # Stale KV slots are inaccessible through the explicit causal mask;
        # accepted prefix slots are overwritten during the next prefill.

    def crop(self, length):
        if not 0 <= length <= self.length:
            raise ValueError("invalid static target cache commit")
        self.length = length

    def _run(self, ids, positions, mask, last_logit_only=False):
        from sglang.srt.speculative.dneo.modeling_dneo import target_attention_input_features
        # Transformers 6 derives StaticCache write positions from this counter,
        # rather than the explicit cache_position argument used by 4.x.
        for layer in self.cache.layers:
            counter = getattr(layer, "cumulative_length", None)
            if isinstance(counter, torch.Tensor) and counter.device == positions.device:
                counter.copy_(positions[:1])
        output = self.backbone(
            input_ids=ids, past_key_values=self.cache, cache_position=positions,
            position_ids=positions[None], attention_mask={"full_attention": mask},
            use_cache=True, output_hidden_states=self.capture_features and self.feature_capture is None, return_dict=True,
        )
        features = (self.feature_capture.stack() if self.feature_capture is not None else
                    target_attention_input_features(self.backbone, output.hidden_states)
                    if self.capture_features else None)
        hidden = output.last_hidden_state[:, -1:] if last_logit_only else output.last_hidden_state
        return self.target.lm_head(hidden), features

    @torch.inference_mode()
    def prepare(self, widths=range(1, 17)):
        device, dtype = self.target.device, self.target.dtype
        for width in widths:
            ids = torch.zeros((self.batch_size, width), device=device, dtype=torch.long)
            positions = torch.arange(width, device=device)
            mask = torch.full((self.batch_size, 1, width, self.capacity), torch.finfo(dtype).min, device=device, dtype=dtype)
            mask.masked_fill_(torch.arange(self.capacity, device=device)[None, None, None] <= positions[None, None, :, None], 0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._run(ids, positions, mask)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = self._run(ids, positions, mask)
            self.graphs[width] = graph, ids, positions, mask, output
        self.reset()

    @torch.inference_mode()
    def forward(self, ids, last_logit_only=False):
        if ids.ndim != 2 or ids.shape[0] != self.batch_size:
            raise ValueError(
                f"target graph expected batch {self.batch_size}, got {tuple(ids.shape)}"
            )
        width = ids.shape[1]
        if self.length + width > self.capacity:
            raise ValueError("target static cache capacity exceeded")
        positions = torch.arange(self.length, self.length + width, device=ids.device)
        if width in self.graphs and not self.force_eager and not last_logit_only:
            graph, static_ids, static_positions, mask, output = self.graphs[width]
            static_ids.copy_(ids)
            static_positions.copy_(positions)
            mask.fill_(torch.finfo(mask.dtype).min)
            mask.masked_fill_(torch.arange(self.capacity, device=ids.device)[None, None, None] <= positions[None, None, :, None], 0)
            graph.replay()
        else:
            mask = torch.full((self.batch_size, 1, width, self.capacity), torch.finfo(self.target.dtype).min,
                              device=ids.device, dtype=self.target.dtype)
            mask.masked_fill_(torch.arange(self.capacity, device=ids.device)[None, None, None] <= positions[None, None, :, None], 0)
            output = self._run(ids, positions, mask, last_logit_only=last_logit_only)
        self.length += width
        return output
