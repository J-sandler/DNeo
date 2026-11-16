"""CUDA graph capture shared by inference and live-training collectors."""
import torch


@torch.inference_mode()
def capture(function, generator=None, pool=None):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    if generator is not None:
        graph.register_generator_state(generator)
    with torch.cuda.graph(graph, pool=pool):
        output = function()
    return graph, output
