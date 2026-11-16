"""Graph the conditional rollout, preserving the actual full-vocabulary q."""
import torch


class HeadGraphs:
    def __init__(self, runtime):
        self.runtime = runtime
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()

    @torch.inference_mode()
    def prepare(self, temperature, top_p, gammas=None):
        runtime = self.runtime
        model = runtime.model
        gammas = range(1, model.block_size) if gammas is None else gammas
        for gamma in gammas:
            key = gamma, temperature, top_p
            if key in self.graphs:
                continue
            hidden = torch.zeros((runtime.batch_size, gamma, model.embed_tokens.weight.shape[1]),
                                 device=model.embed_tokens.weight.device, dtype=model.embed_tokens.weight.dtype)
            anchor = torch.zeros(runtime.batch_size, device=hidden.device, dtype=torch.long)

            def call():
                return runtime.rollout_eager(hidden, anchor, gamma, temperature, top_p, runtime.generator)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(runtime.generator)
            with torch.cuda.graph(graph, pool=self.pool):
                output = call()
            self.graphs[key] = graph, hidden, anchor, output

    def __call__(self, hidden, anchor, gamma, temperature, top_p):
        graph, static_hidden, static_anchor, output = self.graphs[gamma, temperature, top_p]
        static_hidden.copy_(hidden[:, :gamma])
        static_anchor.copy_(anchor)
        graph.replay()
        return output
