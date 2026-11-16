"""Graph incremental MoFA routing and prefix K/V projection, not the history."""
import os
import torch


class PrefixGraphs:
    def __init__(self, runtime):
        self.runtime = runtime
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.commits_prefix = (runtime.prefix_buffers is not None
                               and not runtime.draft_graphs.expanded_kv)

    @torch.inference_mode()
    def prepare(self):
        runtime = self.runtime
        w = runtime.model.embed_tokens.weight
        n = runtime.feature_count
        batch = runtime.batch_size
        for width in range(1, runtime.model.block_size + 1):
            features = torch.zeros((batch, width, n, w.shape[1]), device=w.device, dtype=w.dtype)
            positions = torch.arange(width, device=w.device)[None].expand(batch, -1).clone()

            def call():
                with runtime.operator_kernels():
                    contexts, (cos, sin) = runtime.route(features, positions)
                    if self.commits_prefix and runtime.grouped_prefix_commit is not None:
                        runtime.grouped_prefix_commit(
                            contexts, cos, sin, positions, fused_norm=True)
                        return contexts, None
                    if self.commits_prefix and os.environ.get("DNEO_FUSED_PREFIX_COMMIT", "0") == "1":
                        from dneo.runtime.prefix_commit import project_prefix_into_cache
                        for layer, context, destination in zip(runtime.draft.layers, contexts, runtime.prefix_buffers):
                            project_prefix_into_cache(layer, context, cos, sin, positions, destination)
                        # extend_prefix reads the committed cache directly in this mode.
                        return contexts, None
                    kv = tuple(layer._project_prefix(c, cos, sin) for layer, c in zip(runtime.draft.layers, contexts))
                if self.commits_prefix:
                    from dneo.runtime.structural_kernels import write_kv_at_positions
                    for source, destination in zip(kv, runtime.prefix_buffers):
                        write_kv_at_positions(source, destination, positions)
                return contexts, kv

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = call()
            self.graphs[width] = graph, features, positions, output

    def __call__(self, features, positions):
        if features.shape[1] not in self.graphs:
            contexts, (cos, sin) = self.runtime.route(features, positions)
            kv = tuple(layer._project_prefix(c, cos, sin) for layer, c in zip(self.runtime.draft.layers, contexts))
            return contexts, kv
        graph, static_features, static_positions, output = self.graphs[features.shape[1]]
        static_features.copy_(features)
        static_positions.copy_(positions)
        graph.replay()
        return output
