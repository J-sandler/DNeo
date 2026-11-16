"""Shared draft/accept/commit engine extracted from qualified native DNeo."""
import os
import time
import json
from pathlib import Path
import torch
import triton
import triton.language as tl
from sglang.srt.speculative.dneo.runtime.dneo import DNeoRuntime
from sglang.srt.speculative.dneo.runtime.kernels import _leviathan_count, _leviathan_race, _leviathan_winner, full_vocab_probabilities
from sglang.srt.speculative.dneo.runtime.structural_kernels import write_kv_at_positions
from sglang.srt.speculative.dneo.runtime.serving_kernels import batched_leviathan, write_packed_kv
def _native_graph_batch_sizes(request_capacity, target_graph_max_batch, descending):
    """Keep request-slot capacity independent of the upstream graph ceiling."""
    limit = min(request_capacity, target_graph_max_batch)
    if limit < 1:
        raise ValueError('Native graph capture needs a positive batch limit')
    return range(limit, 0, -1) if descending else range(1, limit + 1)


def _set_inference_block_size(model, block_size):
    """Use a prefix of a checkpoint's trained draft positions at inference."""
    trained = int(model.block_size)
    requested = int(block_size)
    if not 2 <= requested <= trained:
        raise ValueError(
            f"dNEO inference block size must be in [2, {trained}], got {requested}"
        )
    if requested == trained:
        return
    if getattr(model.draft, "internal_mofa", False):
        raise ValueError("short-block dNEO inference does not support internal MoFA")
    if getattr(model.draft, "grouped_online_mofa", False):
        raise ValueError("short-block dNEO inference does not support grouped-online MoFA")
    model.block_size = requested
    model.draft.block_size = requested
    for layer in model.draft.layers:
        layer.block_size = requested
        router = getattr(layer, "grouped_online_router", None)
        if router is not None:
            raise ValueError("short-block dNEO inference found a grouped-online router")
        for conv in (getattr(layer, "attention_conv", None), getattr(layer, "mlp_conv", None)):
            if conv is not None and getattr(conv, "position_count", 0):
                raise ValueError("short-block dNEO inference requires position-shared convolutions")


@triton.jit
def _finish_accept(DRAFT, META, PREFIX, ACCEPT, COMMIT, BONUS, OUT, LENGTH,
                   B: tl.constexpr, TILE: tl.constexpr):
    i = tl.arange(0, TILE)
    count = tl.load(META)
    bonus = tl.load(META + 1)
    token = tl.load(DRAFT + i, i < B - 1, 0)
    tl.store(OUT + i, tl.where(i < count, token, tl.where(i == count, bonus, 0)), i < B)
    tl.store(ACCEPT, count)
    tl.store(COMMIT, count + 1)
    tl.store(BONUS, bonus)
    tl.store(LENGTH, tl.load(PREFIX) + count + 1)


class DNeoEngine:
    @torch.inference_mode()
    def init_cuda_graphs(self):
        if self._max_batch > 1:
            self._init_batched_state()
            return
        # The scheduler captures the target graph before calling this method.
        self._verify_features = tuple(self._captured)
        if any(x is None for x in self._verify_features):
            raise RuntimeError("target graph did not capture dNEO attention-input features")
        r = DNeoRuntime(
            None, self.dneo_model, "static-prefix", target_backend=self,
            context_capacity=self.capacity, draft_graph_capacities=(), optimize_head=True,
            draft_kernel=os.environ.get("DNEO_NATIVE_DRAFT_KERNEL", "auto"),
        )
        self.runtime = r
        b = self.block_size
        d = self.dneo_model.embed_tokens.weight.shape[1]
        n = r.feature_count
        if n != len(self._verify_features):
            raise ValueError("checkpoint router feature count differs from target capture")
        self._features = torch.zeros((1, b, n, d), device="cuda", dtype=torch.bfloat16)
        self._positions = torch.arange(b, device="cuda")[None]
        self._prefix = torch.zeros((1,), device="cuda", dtype=torch.int64)
        self._anchor = torch.zeros((1,), device="cuda", dtype=torch.int64)
        self._commit = torch.ones((1,), device="cuda", dtype=torch.int32)
        self._memory_valid = torch.zeros((1, 1, b - 1), device="cuda", dtype=torch.bool)
        self._memory_positions = torch.zeros((1, 1, b - 1), device="cuda", dtype=torch.int64)
        self._raw_memory = torch.zeros((1, b - 1, n, d), device="cuda", dtype=torch.bfloat16)
        self._routed_memory = tuple(torch.zeros((1, 1, b - 1, 1, d), device="cuda", dtype=torch.bfloat16) for _ in r.draft.layers)
        self._block_ids = torch.full((1, 1, b), self._mask_token_id, device="cuda", dtype=torch.int64)
        self._graph_events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)] if self._profile_path else None

        def draft():
            if self._graph_events:
                self._graph_events[0].record()
            self._block_ids[:, :, 0].copy_(self._anchor.view(1, 1))
            if self._cache_routed_memory:
                memory = self._routed_memory
            else:
                contexts, _ = r.route(self._raw_memory, self._memory_positions[:, 0], memory=True)
                memory = tuple(c[:, None, :, :1] for c in contexts)
            with r.operator_kernels():
                hidden = r.structural_draft(
                    self.dneo_model.embed_tokens(self._block_ids), r.prefix_buffers,
                    self._prefix.view(1, 1), memory, self._memory_valid, self._memory_positions,
                    reuse_memory_kv=self._reuse_memory_kv,
                )[:, 0, 1:]
                self._draft_hidden = hidden
                if self._graph_events:
                    self._graph_events[1].record()
                output = r.rollout_eager(hidden, self._anchor, b - 1, self.draft_temperature, 1.0, r.generator)
                if self._graph_events:
                    self._graph_events[2].record()
                return output

        def commit():
            with r.operator_kernels():
                contexts, (cos, sin) = r.route(self._features, self._positions)
                if self._cache_routed_memory and not self._reuse_memory_kv:
                    if getattr(self.dneo_model.conditioner, "rejection_router_logits", None) is None:
                        memory_contexts = tuple(c[:, 1:] for c in contexts)
                    else:
                        memory_contexts, _ = r.route(self._features[:, 1:], self._positions[:, 1:], memory=True)
                    for destination, context in zip(self._routed_memory, memory_contexts):
                        destination.copy_(context[:, None, :, :1])
                if r.grouped_prefix_commit is not None:
                    r.grouped_prefix_commit(
                        contexts, cos, sin, self._positions, fused_norm=True)
                else:
                    for layer, context, destination in zip(r.draft.layers, contexts, r.prefix_buffers):
                        if os.environ.get("DNEO_FUSED_PREFIX_COMMIT", "0") == "1":
                            from sglang.srt.speculative.dneo.runtime.prefix_commit import project_prefix_into_cache
                            project_prefix_into_cache(layer, context, cos, sin, self._positions, destination)
                            continue
                        kv = layer._project_prefix(context, cos, sin)
                        # Rejected slots stay outside the committed prefix mask.
                        write_kv_at_positions(kv, destination, self._positions)
            if not self._reuse_memory_kv:
                self._raw_memory.copy_(self._features[:, 1:])
            self._memory_positions.copy_(self._positions[:, None, 1:])
            self._memory_valid.copy_(torch.arange(1, b, device="cuda")[None, None] >= self._commit.view(1, 1, 1))

        self._draft_graph, self._draft_output = self._capture(draft, r.generator)
        self._commit_graph, _ = self._capture(commit)
        vocab = self.dneo_model.lm_head.weight.shape[0]
        self._target_logits = torch.zeros((b, vocab), device="cuda", dtype=torch.float32)
        self._accept_graphs = {}
        self._accept_slot = 0
        for temp in (0.0, 1.0):
            for slot in (0, 1):
                self._accept_graphs[temp, slot] = self._capture(lambda temp=temp: self._accept_gpu(temp), r.generator)
        self._native_graphs_ready = True
        self._audit_draft()
        print("DNeoWorker ready: native scheduler, GPU-resident draft/accept/commit, folded dense head", flush=True)


    def _init_batched_state(self):
        r = DNeoRuntime(
            None, self.dneo_model, "static-prefix", target_backend=self,
            context_capacity=self.capacity, batch_size=self._max_batch,
            draft_graph_capacities=(), optimize_head=True,
            draft_kernel=os.environ.get("DNEO_NATIVE_DRAFT_KERNEL", "auto"),
        )
        self.runtime = r
        b = self.block_size
        d = self.dneo_model.embed_tokens.weight.shape[1]
        n = r.feature_count
        if n != len(self._captured):
            raise ValueError("checkpoint router feature count differs from target capture")
        device = torch.device("cuda")
        self._prefix = torch.zeros(self._max_batch, device=device, dtype=torch.int64)
        self._commit = torch.ones(self._max_batch, device=device, dtype=torch.int32)
        self._memory_valid = torch.zeros(
            self._max_batch, 1, b - 1, device=device, dtype=torch.bool
        )
        self._memory_positions = torch.zeros(
            self._max_batch, 1, b - 1, device=device, dtype=torch.int64
        )
        self._raw_memory = torch.zeros(
            self._max_batch, b - 1, n, d, device=device, dtype=torch.bfloat16
        )
        self._routed_memory = tuple(
            torch.zeros(
                self._max_batch, 1, b - 1, 1, d,
                device=device, dtype=torch.bfloat16,
            )
            for _ in r.draft.layers
        )
        self._slot_prefix_lens = torch.zeros(
            self._max_batch, device=device, dtype=torch.int64
        )
        self._graph_slots = torch.arange(self._max_batch, device=device)
        self._graph_block_ids = torch.full(
            (self._max_batch, b), self._mask_token_id, device=device, dtype=torch.long)
        self._graph_positions = torch.arange(b, device=device)[None].repeat(self._max_batch, 1)
        self._graph_logits = torch.zeros(
            (self._max_batch, b, self.dneo_model.lm_head.weight.shape[0]),
            device=device, dtype=torch.float32)
        # A retained Q tensor for every captured batch grows quadratically.
        # Graphs run serially, so high-capacity captures can share one output.
        self._shared_draft_q = (
            torch.empty((self._max_batch, b - 1, self.dneo_model.lm_head.weight.shape[0]),
                        device=device, dtype=torch.float32)
            if self._max_batch > 32
            else None
        )
        self._batch_graphs = {}
        self._batch_profiles = {}
        self._graph_pool = torch.cuda.graph_pool_handle()
        if not self._disable_graph:
            target_graph = self.model_runner.decode_cuda_graph_runner
            batches = _native_graph_batch_sizes(
                self._max_batch, target_graph.max_bs, self._shared_draft_q is not None)
            benchmark_batches = os.environ.get('DNEO_BENCHMARK_GRAPH_BATCHES')
            if benchmark_batches:
                if not os.environ.get('SPEC_STEP_PROFILE'):
                    raise ValueError('Restricted graph capture is benchmark-only')
                selected = sorted(set(map(int, benchmark_batches.split(','))))
                if not selected or selected[0] < 1 or selected[-1] > self._max_batch:
                    raise ValueError('Invalid benchmark graph batches')
                batches = [bs for bs in batches if bs in selected]
            for bs in batches:
                self._active_slots = self._graph_slots[:bs]
                self._batch_events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)] if self._profile_path else None
                r.structural_draft.profile_events = [
                    [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
                    for _ in r.draft.layers] if self._profile_path else None
                self._batch_profiles[bs] = self._batch_events, r.structural_draft.profile_events
                r.head_profile_events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(5)] if self._profile_path else None
                if not hasattr(self, "_head_profiles"):
                    self._head_profiles = {}
                self._head_profiles[bs] = r.head_profile_events
                draft_graph, proposals = self._capture(
                    lambda bs=bs: self._draft_block_batched(
                        self._graph_block_ids[:bs], self._prefix[:bs]),
                    r.generator, self._graph_pool)
                draft_output = self._draft_output
                acceptance = []
                for temp in (0.0, 1.0):
                    self._target_temperature = temp
                    graph, output = self._capture(
                        lambda bs=bs: self._accept_batched(self._graph_logits[:bs]),
                        r.generator, self._graph_pool)
                    acceptance.append((graph, output))
                commit_graph, _ = self._capture(
                    lambda bs=bs: self._commit_decode_batched(
                        self._graph_positions[:bs], self._commit[:bs]),
                    pool=self._graph_pool)
                self._batch_graphs[bs] = (draft_graph, draft_output, acceptance, commit_graph)
                print(f"dneo_graph_capture_batch={bs}/{self._max_batch}", flush=True)
            self._target_temperature = 1.0
            self._memory_valid.zero_()
            self._prefix.zero_()
        self._native_graphs_ready = True
        print(
            f"DNeoWorker ready: indexed request caches, batched acceptance, "
            f"cuda_graphs={len(self._batch_graphs)}, max_batch={self._max_batch}",
            flush=True,
        )


    def _slots_for_batch(self, batch):
        rids = [req.rid for req in batch.reqs]
        for rid, req in list(self._requests.items()):
            if rid not in rids and (req.finished() or getattr(req, "is_retracted", False)):
                self._request_slots.pop(rid, None)
                self._requests.pop(rid, None)
        missing = [rid for rid in rids if rid not in self._request_slots]
        free = [slot for slot in range(self._max_batch) if slot not in self._request_slots.values()]
        if len(missing) > len(free):
            raise RuntimeError("dNEO request capacity exceeded; live cache slots cannot be reused")
        for rid in missing:
            self._request_slots[rid] = free.pop(0)
        self._requests.update({req.rid: req for req in batch.reqs})
        slots = [self._request_slots[rid] for rid in rids]
        self._active_slots_list = slots
        return torch.tensor(slots, device="cuda", dtype=torch.long)


    @staticmethod
    def _captured_feature_matrix(captured):
        layers = []
        for value in captured:
            if value is None:
                raise RuntimeError("target feature hook did not run")
            layers.append(value[0] if value.ndim == 3 and value.shape[0] == 1 else value)
        return torch.stack(layers, dim=1)


    def _commit_prefill_batched(self, positions):
        batch = self._current_batch
        if batch is None or batch.extend_lens is None:
            raise RuntimeError("missing batch metadata for batched dNEO prefill")
        features = self._captured_feature_matrix(self._captured)
        positions = positions.reshape(-1)
        token_slots = torch.tensor(
            [slot for slot, length in zip(self._active_slots_list, batch.extend_lens)
             for _ in range(length)], device="cuda", dtype=torch.long)
        if token_slots.numel() != features.shape[0]:
            raise RuntimeError("packed target features do not match prefill lengths")
        # Route and project the entire ragged prefill as one GEMM per layer.
        contexts, (cos, sin) = self.runtime.route(features[None], positions[None])
        for layer, context, destinations in zip(
                self.runtime.draft.layers, contexts, self.runtime.prefix_buffers):
            source = layer._project_prefix(context, cos, sin)
            write_packed_kv(source, destinations, positions, token_slots)
        lengths = torch.tensor(
            [p + n for p, n in zip(batch.prefix_lens, batch.extend_lens)],
            device="cuda", dtype=torch.long)
        self._slot_prefix_lens.index_copy_(0, self._active_slots, lengths)
        self._memory_valid.index_fill_(0, self._active_slots, False)
        self._memory_positions.index_fill_(0, self._active_slots, 0)
        self._raw_memory.index_fill_(0, self._active_slots, 0)
        for routed in self._routed_memory:
            routed.index_fill_(0, self._active_slots, 0)


    def _draft_block_batched(self, block_ids, prefix_lens):
        slots = self._active_slots
        bs = block_ids.shape[0]
        if slots is None or slots.numel() != bs:
            raise RuntimeError("active request slots do not match draft batch")
        r = self.runtime
        events = getattr(self, "_batch_events", None)
        if events:
            events[0].record()
        prefix_kv = r.prefix_buffers
        memory = (self._routed_memory if self._reuse_memory_kv else
                  tuple(value.index_select(0, slots) for value in self._routed_memory))
        valid = self._memory_valid.index_select(0, slots)
        memory_positions = self._memory_positions.index_select(0, slots)
        embeddings = self.dneo_model.embed_tokens(block_ids[:, None])
        with r.operator_kernels():
            hidden = r.structural_draft(
                embeddings, prefix_kv, prefix_lens[:, None], memory,
                valid, memory_positions, request_slots=slots,
                reuse_memory_kv=self._reuse_memory_kv,
            )[:, 0, 1:]
            if events:
                events[1].record()
            proposed, q = r.rollout_fused_sampling(
                hidden, block_ids[:, 0], self.block_size - 1,
                self.draft_temperature, r.generator,
                **({"probabilities_out": self._shared_draft_q[:bs]}
                   if self._shared_draft_q is not None else {}),
            )
            if events:
                events[2].record()
        if proposed.ndim == 1:
            proposed, q = proposed.unsqueeze(0), q.unsqueeze(0)
        self._draft_hidden = hidden
        self._draft_output = proposed, q
        self._prefix[:bs].copy_(prefix_lens)
        return proposed


    def _accept_batched(self, next_token_logits):
        proposed, q = self._draft_output
        bs, gamma, vocab = q.shape
        logits = next_token_logits.view(bs, self.block_size, vocab)
        if self._target_temperature == 0 and self._shared_draft_q is not None:
            p = torch.zeros_like(logits, dtype=torch.float32)
            p.scatter_(-1, logits.argmax(-1, keepdim=True), 1.)
        else:
            p = (torch.nn.functional.one_hot(logits.argmax(-1), vocab).float()
                 if self._target_temperature == 0
                 else full_vocab_probabilities(logits, self._target_temperature))
        output = batched_leviathan(proposed, q, p, self._prefix[:bs],
                                  generator=self.runtime.generator)
        self._commit[:bs].copy_(output[1])
        return output


    def _commit_decode_batched(self, positions, commit_lens):
        slots = self._active_slots
        bs = slots.numel()
        b = self.block_size
        features = self._verify_feature_buffer[:bs * b]
        features = features.view(bs, b, features.shape[1], features.shape[2])
        positions = positions.view(bs, b)
        r = self.runtime
        contexts, (cos, sin) = r.route(features, positions)
        memory_contexts = (
            tuple(context[:, 1:] for context in contexts)
            if getattr(self.dneo_model.conditioner, "rejection_router_logits", None) is None
            else r.route(features[:, 1:], positions[:, 1:], memory=True)[0]
        )
        if r.grouped_prefix_commit is not None:
            r.grouped_prefix_commit(contexts, cos, sin, positions, slots)
        else:
            for layer, context, destinations in zip(r.draft.layers, contexts, r.prefix_buffers):
                if os.environ.get("DNEO_FUSED_PREFIX_COMMIT", "0") == "1":
                    from sglang.srt.speculative.dneo.runtime.prefix_commit import project_prefix_into_cache
                    project_prefix_into_cache(layer, context, cos, sin, positions, destinations, slots)
                else:
                    source = layer._project_prefix(context, cos, sin)
                    write_kv_at_positions(source, destinations, positions, slots)
        arange = torch.arange(1, b, device="cuda")[None]
        self._memory_positions.index_copy_(0, slots, positions[:, None, 1:])
        self._memory_valid.index_copy_(0, slots, (arange >= commit_lens[:, None])[:, None])
        self._slot_prefix_lens.index_copy_(0, slots, self._prefix[:bs] + commit_lens)
        if not self._reuse_memory_kv:
            for destination, context in zip(self._routed_memory, memory_contexts):
                destination.index_copy_(0, slots, context[:, None, :, :1])


    def _audit_draft(self):
        """Compare padded device-state replay with the original compact memory path."""
        import json
        from sglang.srt.speculative.dneo.runtime.dneo import probabilities

        r = self.runtime
        b, n, d = self._features.shape[1:]
        from sglang.srt.speculative.dneo.runtime.structural_draft import StructuralDraft
        native_draft = r.structural_draft
        r.structural_draft = StructuralDraft(r.draft, native_draft.variant)
        r.structural_draft.attention_backend = native_draft.attention_backend
        r.structural_draft.tuning = native_draft.tuning
        r.structural_draft.attention_capacity = native_draft.attention_capacity
        audit_capacity = r.prefix_buffers[0][0].shape[2]
        r.draft_graphs.prepare(audit_capacity, b - 1)
        r.draft_graphs.prepare(audit_capacity, 0)
        r.structural_draft = native_draft
        # Keep this startup guard independent of graph-capture order and the
        # benchmark sampling seed. Otherwise harmless changes in max batch size
        # select a different synthetic fixture and can flip the tolerance gate.
        audit_generator = torch.Generator(device="cuda").manual_seed(20260918)
        prefix = torch.randn(
            (1, 32, n, d), device="cuda", dtype=torch.bfloat16,
            generator=audit_generator,
        )
        branch = torch.randn(
            self._features.shape, device="cuda", dtype=self._features.dtype,
            generator=audit_generator,
        )
        rows = []
        for accepted in (0, 7, b - 1):
            r.memory = branch[:, 1 + accepted:] if accepted < b - 1 else None
            r.memory_positions = (
                torch.arange(33 + accepted, 32 + b, device="cuda")[None]
                if accepted < b - 1 else None
            )
            r.prefix_length = 0
            r.extend_prefix(prefix)
            self._features.copy_(branch)
            self._positions.copy_(torch.arange(32, 32 + b, device="cuda")[None])
            self._commit.fill_(1 + accepted)
            self._commit_graph.replay()
            self._prefix.fill_(33 + accepted)
            self._anchor.fill_(17)
            self._draft_graph.replay()
            actual = self._draft_hidden.clone()
            # Invoke the existing optimized production graph with compact
            # memory, at the same cache capacity as the native graph.
            r.prefix_length = 33 + accepted
            key = (audit_capacity, b - 1 if r.memory is not None else 0)
            graph, ids, anchor_pos, _, _, valid, positions, output, raw = r.draft_graphs.graphs[key]
            ids[:, :, 0].copy_(self._anchor.view(1, 1))
            anchor_pos.fill_(r.prefix_length)
            if r.memory is not None:
                count = r.memory.shape[1]
                valid.zero_()
                valid[:, :, :count] = True
                positions.zero_()
                positions[:, :, :count].copy_(r.memory_positions[:, None])
                raw.zero_()
                raw[:, :count].copy_(r.memory)
            graph.replay()
            expected = output
            relative = float((actual.float() - expected.float()).norm() / expected.float().norm())
            p = probabilities(self.dneo_model.lm_head(expected))
            q = probabilities(self.dneo_model.lm_head(actual))
            tv = float((p - q).abs().sum(-1).max() / 2)
            if relative > 0.02 or tv > 0.015:
                raise RuntimeError(f"native padded-memory parity failed: accepted={accepted}, rel={relative}, tv={tv}")
            rows.append(dict(accepted=accepted, hidden_relative_error=relative, max_marginal_tv=tv))
        print("dneo_native_parity=" + json.dumps(rows), flush=True)
        self._memory_valid.zero_()


    @staticmethod
    def _capture(function, generator=None, pool=None):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                function()
        torch.cuda.current_stream().wait_stream(stream)
        if os.environ.get("DNEO_EXPERIMENTAL_REQUEST_CAPACITY"):
            torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        if generator is not None:
            graph.register_generator_state(generator)
        with torch.cuda.graph(graph, pool=pool):
            output = function()
        return graph, output


    def _accept_gpu(self, temperature):
        proposed, q = self._draft_output
        p = (torch.nn.functional.one_hot(self._target_logits.argmax(-1), q.shape[-1]).float()
             if temperature == 0 else full_vocab_probabilities(self._target_logits, temperature))
        gamma, vocab = q.shape
        meta = torch.empty(2, device="cuda", dtype=torch.int64)
        uniforms = torch.rand(gamma, device="cuda", generator=self.runtime.generator)
        noise = torch.empty(vocab, device="cuda").exponential_(generator=self.runtime.generator)
        tiles = triton.cdiv(vocab, 1024)
        scores = torch.empty(tiles, device="cuda")
        ids = torch.empty(tiles, device="cuda", dtype=torch.int32)
        _leviathan_count[(1,)](proposed, q, p, uniforms, meta, vocab, gamma,
                              triton.next_power_of_2(gamma), enable_fp_fusion=False)
        _leviathan_race[(tiles,)](p, q, meta, noise, scores, ids, vocab, gamma, 1024, enable_fp_fusion=False)
        _leviathan_winner[(1,)](scores, ids, meta, tiles, triton.next_power_of_2(tiles))
        accept = torch.empty(1, device="cuda", dtype=torch.int32)
        commit = torch.empty_like(accept)
        bonus = torch.empty(1, device="cuda", dtype=torch.int64)
        output = torch.empty((1, self.block_size), device="cuda", dtype=torch.int64)
        length = torch.empty_like(self._prefix)
        _finish_accept[(1,)](proposed, meta, self._prefix, accept, commit, bonus, output,
                             length, self.block_size, triton.next_power_of_2(self.block_size))
        return accept, commit, bonus, output, length, None


    @torch.inference_mode()
    def _dneo_draft_block(self, block_ids, prefix_lens, batch):
        if self._max_batch > 1:
            bs = block_ids.shape[0]
            if bs in self._batch_graphs:
                self._graph_slots[:bs].copy_(self._active_slots)
                self._active_slots = self._graph_slots[:bs]
                self._prefix[:bs].copy_(prefix_lens)
                self._graph_block_ids[:bs].copy_(block_ids)
                graph, self._draft_output, _, _ = self._batch_graphs[bs]
                self._timed("draft", graph.replay)
                return self._draft_output[0]
            return self._timed(
                "draft", self._draft_block_batched, block_ids, prefix_lens
            )
        return self._timed("draft", self._draft_block, block_ids, prefix_lens, batch)


    def _draft_block(self, block_ids, prefix_lens, batch):
        if block_ids.shape[0] != 1 or batch.has_grammar:
            raise ValueError("native dNEO currently supports one ungrammared request")
        self._prefix.copy_(prefix_lens)
        self._anchor.copy_(block_ids[:, 0])
        self._draft_graph.replay()
        return self._draft_output[0].view(1, -1)


    def _validate_phase1_sampling_support(self, batch):
        if len(batch.reqs) > self._max_batch or batch.has_grammar or batch.return_logprob:
            raise ValueError("native dNEO batch is too large or uses unsupported output features")
        temperatures = {float(req.sampling_params.temperature) for req in batch.reqs}
        self._target_temperature = (
            0.0 if batch.sampling_info.is_all_greedy else
            (temperatures.pop() if len(temperatures) == 1 else None)
        )
        if getattr(batch.sampling_info, "need_min_p_sampling", False):
            raise ValueError("native dNEO does not yet support min-p filtering")
        for request in batch.reqs:
            if len(request.origin_input_ids) + request.sampling_params.max_new_tokens + self.block_size > self.capacity:
                raise ValueError("request exceeds the configured dNEO context capacity")
        if self._max_batch == 1:
            req = batch.reqs[0]
            params = req.sampling_params
        else:
            return
        if self._request_id != req.rid:
            if len(req.origin_input_ids) + params.max_new_tokens + self.block_size > self.capacity:
                raise ValueError("native dNEO context capacity must include prompt, output budget, and one spare block")
            seed = getattr(params, "sampling_seed", None)
            if seed is not None:
                self.runtime.generator.manual_seed(seed)
            self._request_id = req.rid


    @torch.inference_mode()
    def _accept_block(self, *, candidates, next_token_logits, sampling_info, draft_input, prefix_lens, bs):
        result = self._accept_block_impl(candidates=candidates, next_token_logits=next_token_logits,
            sampling_info=sampling_info, draft_input=draft_input, prefix_lens=prefix_lens, bs=bs)
        if os.environ.get("DNEO_ACCEPT_AUDIT") == "1" and sampling_info.is_all_greedy:
            expected = next_token_logits.view(bs, self.block_size, -1).argmax(-1)
            mask = torch.arange(self.block_size, device=expected.device)[None] < result[1][:, None]
            if not torch.equal(expected[mask], result[3][mask]):
                raise AssertionError("dNEO emitted a non-greedy token for the live verifier logits")
        return result


    def _accept_block_impl(self, *, candidates, next_token_logits, sampling_info, draft_input, prefix_lens, bs):
        filtered = not sampling_info.is_all_greedy and (
            sampling_info.need_top_p_sampling or sampling_info.need_top_k_sampling)
        if filtered or self._target_temperature not in (0.0, 1.0):
            return self._timed("accept", self._accept_general,
                               next_token_logits, sampling_info, prefix_lens, bs)
        if self._max_batch > 1:
            if bs in self._batch_graphs:
                self._graph_logits[:bs].copy_(next_token_logits.view(bs, self.block_size, -1))
                graph, output = self._batch_graphs[bs][2][int(self._target_temperature)]
                self._timed("accept", graph.replay)
                # The scheduler retains lengths and emitted tokens across overlap
                # iterations; detach them from storage reused by graph replay.
                return tuple(value.clone() if value is not None else None for value in output)
            return self._timed("accept", self._accept_batched, next_token_logits)
        return self._timed("accept", self._accept_native, next_token_logits, sampling_info)


    def _accept_general(self, logits, sampling_info, prefix_lens, bs):
        from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs

        proposed, q = self._draft_output
        if q.ndim == 2:
            proposed, q = proposed[None], q[None]
        p = build_dflash_verify_target_probs(
            next_token_logits=logits, sampling_info=sampling_info,
            draft_token_num=self.block_size, bs=bs, use_sparse_topk=False)
        result = batched_leviathan(proposed, q, p.float(), prefix_lens.contiguous(),
                                  generator=self.runtime.generator)
        self._commit[:bs].copy_(result[1])
        return result


    def _accept_native(self, next_token_logits, sampling_info):
        if not sampling_info.is_all_greedy and (sampling_info.need_top_p_sampling or sampling_info.need_top_k_sampling or sampling_info.need_min_p_sampling):
            raise ValueError("native dNEO currently supports unfiltered sampling only")
        temperature = self._target_temperature
        if (temperature, 0) not in self._accept_graphs:
            raise ValueError("native dNEO target temperature must be 0 or 1")
        self._target_logits.copy_(next_token_logits)
        self._accept_slot = 1 - self._accept_slot
        graph, output = self._accept_graphs[temperature, self._accept_slot]
        graph.replay()
        self._commit.copy_(output[1])
        return output


    @torch.inference_mode()
    def _append_target_hidden_to_draft_kv_by_loc(self, *, target_hidden, cache_loc, positions,
                                                cache_loc_2d=None, commit_lens=None):
        if self._max_batch > 1:
            if commit_lens is not None and commit_lens.numel() in self._batch_graphs:
                bs = commit_lens.numel()
                self._graph_positions[:bs].copy_(positions.view(bs, self.block_size))
                self._commit[:bs].copy_(commit_lens)
                return self._timed("commit", self._batch_graphs[bs][3].replay)
            function = (
                self._commit_prefill_batched
                if commit_lens is None
                else self._commit_decode_batched
            )
            arguments = (positions,) if commit_lens is None else (positions, commit_lens)
            return self._timed("commit", function, *arguments)
        return self._timed("commit", self._commit_native, positions, commit_lens)


    def _commit_native(self, positions, commit_lens):
        if commit_lens is None:
            # A new request begins at position zero (radix caching is disabled).
            features = torch.stack(tuple(self._captured), dim=1)[None]
            self.runtime.prefix_length = 0
            self.runtime.extend_prefix(features)
            self._memory_valid.zero_()
            self._raw_memory.zero_()
            for context in self._routed_memory:
                context.zero_()
            self._memory_positions.zero_()
        else:
            features = self._active_verify_features or self._verify_features
            torch.stack(tuple(x[:self.block_size] for x in features), dim=1, out=self._features[0])
            self._positions.copy_(positions.view(1, -1))
            self._commit.copy_(commit_lens)
            self._commit_graph.replay()


    def _timed(self, name, function, *args, **kwargs):
        if self._timings is None:
            return function(*args, **kwargs)
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        before = time.perf_counter()
        start.record()
        output = function(*args, **kwargs)
        end.record()
        self._timings[name] = (start, end, (time.perf_counter() - before) * 1000)
        return output


    @torch.inference_mode()
    def forward_batch_generation(self, batch, *args, **kwargs):
        if self._max_batch > 1:
            self._current_batch = batch
            self._active_slots = self._slots_for_batch(batch)
        if not self._profile_path or self._profile_cycles >= 100:
            return super().forward_batch_generation(batch, *args, **kwargs)
        self._profile_cycles += 1
        if self._profile_cycles <= 20:
            return super().forward_batch_generation(batch, *args, **kwargs)
        self._timings = {}
        result = self._timed("total", super().forward_batch_generation, batch, *args, **kwargs)
        torch.cuda.synchronize()
        if "draft" in self._timings:
            row = {key: {"gpu_ms": a.elapsed_time(b), "cpu_ms": cpu} for key, (a, b, cpu) in self._timings.items()}
            row["batch_size"] = len(batch.reqs)
            if self._max_batch == 1:
                row["backbone_gpu_ms"] = self._graph_events[0].elapsed_time(self._graph_events[1])
                row["head_gpu_ms"] = self._graph_events[1].elapsed_time(self._graph_events[2])
            elif self._batch_profiles[len(batch.reqs)][0]:
                events, layers = self._batch_profiles[len(batch.reqs)]
                row["backbone_gpu_ms"] = events[0].elapsed_time(events[1])
                row["head_gpu_ms"] = events[1].elapsed_time(events[2])
                row["layers"] = [dict(input_qkv_ms=e[0].elapsed_time(e[1]),
                                      attention_ms=e[1].elapsed_time(e[2]),
                                      output_mlp_ms=e[2].elapsed_time(e[3])) for e in layers]
                e = self._head_profiles[len(batch.reqs)]
                row["head_phases_ms"] = dict(lm_projection=e[0].elapsed_time(e[1]),
                    preparation=e[1].elapsed_time(e[2]), rollout=e[2].elapsed_time(e[3]),
                    normalization=e[3].elapsed_time(e[4]))
            self._profile_rows.append(row)
        self._timings = None
        if self._profile_cycles == 100:
            Path(self._profile_path).write_text(json.dumps(self._profile_rows, indent=2))
        return result


