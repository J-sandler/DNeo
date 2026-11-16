"""Bucketed CUDA-graph replay of the *unchanged* trained rejection-memory graph."""
import types
import torch


def memory_buckets(block_size):
    gamma = block_size - 1
    return tuple(sorted({0, gamma, *[min(gamma, n) for n in (1, 2, 4, 8, 12, 16, 24, 32)]}))


class DraftGraphs:
    def __init__(self, runtime):
        self.runtime = runtime
        self.graphs = {}
        self.static_buffers = {}
        self.expanded_kv = {}
        self.pool = torch.cuda.graph_pool_handle()

    def update_prefix(self, new_kv, start):
        from dneo.runtime.structural_kernels import write_expanded_kv
        for (capacity, _), caches in self.expanded_kv.items():
            width = min(new_kv[0][0].shape[2], capacity - start)
            if width > 0:
                for source, destination in zip(new_kv, caches):
                    write_expanded_kv(tuple(x[:, :, :width] for x in source), destination, start)

    @torch.inference_mode()
    def prepare(self, capacity, memory_width):
        runtime, model = self.runtime, self.runtime.model
        draft = model.draft
        b, d = model.block_size, draft.hidden_size
        batch = runtime.batch_size
        device, dtype = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
        ids = torch.full((batch, 1, b), model.mask_token_id, device=device, dtype=torch.long)
        anchor = torch.ones((batch, 1), device=device, dtype=torch.long)
        valid = torch.ones_like(anchor, dtype=torch.bool)
        prefix_valid = torch.ones((batch, capacity), device=device, dtype=torch.long)
        prefix_kv = tuple((torch.zeros((batch, layer.num_key_value_heads, capacity, layer.head_dim), device=device, dtype=dtype),
                           torch.zeros((batch, layer.num_key_value_heads, capacity, layer.head_dim), device=device, dtype=dtype))
                          for layer in draft.layers)
        if runtime.prefix_buffers is not None:
            prefix_kv = tuple(tuple(x[:, :, :capacity] for x in kv) for kv in runtime.prefix_buffers)
        attention_kv = None
        structural = getattr(runtime, 'structural_draft', None)
        if structural is not None and structural.variant == 'fused-sdpa-cachedkv':
            from dneo.runtime.structural_kernels import write_expanded_kv
            attention_kv = tuple(tuple(torch.empty(
                (batch, layer.num_attention_heads, capacity + (1 if structural.kv_only else b) + memory_width, layer.head_dim),
                device=device, dtype=dtype) for _ in range(2)) for layer in draft.layers)
            for source, destination in zip(prefix_kv, attention_kv):
                write_expanded_kv(source, destination)
            self.expanded_kv[capacity, memory_width] = attention_kv
        contexts = tuple(torch.zeros((batch, capacity, 1, 1), device=device, dtype=dtype).expand(-1, -1, 1, d)
                         for _ in draft.layers)
        memory = tuple(torch.zeros((batch, 1, memory_width, 1, d), device=device, dtype=dtype) for _ in draft.layers)
        memory_valid = torch.zeros((batch, 1, memory_width), device=device, dtype=torch.bool)
        memory_positions = torch.zeros((batch, 1, memory_width), device=device, dtype=torch.long)
        raw_memory = None
        if runtime.variant in {"io-graph", "fused-draft", "static-prefix"} and memory_width:
            raw_memory = torch.zeros((batch, memory_width, runtime.feature_count, d), device=device, dtype=dtype)
        cos = torch.ones((batch, capacity, draft.layers[0].head_dim), device=device, dtype=dtype)
        sin = torch.zeros_like(cos)
        originals = []
        for layer, kv in zip(draft.layers, prefix_kv):
            originals.append(layer._project_prefix)
            if getattr(draft, "rejection_memory_kv_only", False):
                original = layer._project_prefix
                def project(context, c, s, _kv=kv, _original=original):
                    if context.shape[1] == capacity:
                        return _kv
                    memory_kv = _original(context[:, capacity:], c[:, capacity:], s[:, capacity:])
                    return tuple(torch.cat((prefix, suffix), dim=2)
                                 for prefix, suffix in zip(_kv, memory_kv))
                layer._project_prefix = project
            else:
                layer._project_prefix = types.MethodType(lambda self, *a, _kv=kv, **kw: _kv, layer)

        def call():
            routed_memory = memory
            if raw_memory is not None:
                routed, _ = runtime.route(raw_memory, memory_positions[:, 0], memory=True)
                routed_memory = tuple(c[:, None, :, :1] for c in routed)
            structural = getattr(runtime, "structural_draft", None)
            # Empty-memory passes have a stricter numerical fallback: the
            # trained-checkpoint sweep exposed amplification of BF16 drift.
            if structural is not None and not (structural.variant == 'structural-safe' and memory_width == 0):
                return runtime.structural_draft(
                    model.embed_tokens(ids), prefix_kv, anchor, routed_memory,
                    memory_valid, memory_positions, attention_kv=attention_kv)[:, 0, 1:]
            return draft(model.embed_tokens(ids), contexts, anchor, valid, prefix_valid,
                         context_cos=cos, context_sin=sin, rejection_contexts=routed_memory,
                         rejection_keep_mask=memory_valid, rejection_position_ids=memory_positions)[:, 0, 1:]
        try:
            with runtime.convolution_kernels(), runtime.last_layer_pruning(memory_width), runtime.operator_kernels():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        call()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    output = call()
        finally:
            for layer, method in zip(draft.layers, originals):
                layer._project_prefix = method
        self.graphs[(capacity, memory_width)] = (graph, ids, anchor, prefix_kv, memory, memory_valid, memory_positions, output, raw_memory)
        # CUDA graphs retain addresses, NOT ownership of external tensor inputs.
        # In particular these masks must outlive every replay and allocator reuse.
        self.static_buffers[(capacity, memory_width)] = (valid, prefix_valid, contexts, cos, sin)

    def warmup(self, capacities=(64, 128, 256, 512, 1024)):
        for capacity in capacities:
            widths = (memory_buckets(self.runtime.model.block_size) if self.runtime.optimize_backbone
                      else (0, self.runtime.model.block_size - 1))
            for memory_width in widths:
                self.prepare(capacity, memory_width)

    @torch.inference_mode()
    def __call__(self, anchor):
        runtime = self.runtime
        capacity = max(64, 2 ** (max(1, runtime.prefix_length) - 1).bit_length())
        memory_width = 0 if runtime.memory is None else runtime.model.block_size - 1
        if runtime.optimize_backbone and runtime.memory is not None:
            memory_width = next(n for n in memory_buckets(runtime.model.block_size) if n >= runtime.memory.shape[1])
        if (capacity, memory_width) not in self.graphs:
            raise ValueError(f"draft graph bucket {(capacity, memory_width)} was not warmed; increase capacities")
        graph, ids, anchor_pos, prefix_kv, memory, memory_valid, memory_positions, output, raw_memory = self.graphs[(capacity, memory_width)]
        ids[:, :, 0].copy_(anchor.reshape(-1, 1))
        anchor_pos.fill_(runtime.prefix_length)
        if runtime.prefix_buffers is None:
            for destination, source in zip(prefix_kv, runtime.prefix_kv):
                for dst, src in zip(destination, source):
                    dst[:, :, :runtime.prefix_length].copy_(src)
        if memory_width:
            count = runtime.memory.shape[1]
            memory_valid.zero_()
            memory_valid[:, :, :count] = True
            memory_positions.zero_()
            memory_positions[:, :, :count].copy_(runtime.memory_positions[:, None])
            if raw_memory is not None:
                raw_memory.zero_()
                raw_memory[:, :count].copy_(runtime.compact_features(runtime.memory))
            else:
                routed, _ = runtime.route(runtime.memory, runtime.memory_positions, memory=True)
                for dst, src in zip(memory, routed):
                    dst.zero_()
                    dst[:, :, :count].copy_(src[:, None, :, :1])
        graph.replay()
        return output
