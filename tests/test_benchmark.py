import pytest
import ast
from pathlib import Path
from types import SimpleNamespace
import sys

from dneo.benchmark_summary import compare
from dneo.runtime.stream_timing import streamed_generate


def test_ar_engine_does_not_receive_speculative_rejection_sampling():
    from dneo import benchmark
    tree = ast.parse(Path(benchmark.__file__).read_text())
    engine = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "Engine")
    expression = next(value.value for value in engine.keywords if value.arg == "speculative_use_rejection_sampling")
    for method in ("ar", "dneo"):
        enabled = eval(compile(ast.Expression(expression), "benchmark engine options", "eval"),
                       {"args": SimpleNamespace(method=method)})
        assert enabled == (method == "dneo")


def test_performance_harness_rejects_correctness_attention_mode(monkeypatch, tmp_path, capsys):
    from dneo import benchmark
    monkeypatch.setenv("DNEO_REFERENCE_MATCHED_ATTENTION", "1")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--backend", "sglang", "--prompt-bank",
                                      str(tmp_path / "bank.json"), "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit):
        benchmark.main()
    assert "must not be reported as optimized performance" in capsys.readouterr().err


def test_summary_rejects_mismatched_conditions():
    settings = dict(backend="sglang", temperature=1, prompt_bank_sha256="abc", matched_requests=256,
                    max_new_tokens=512, target="Qwen/Qwen3-4B", target_revision="pinned", engine_capacity=256,
                    cublas_workspace_config=":4096:8")
    with pytest.raises(ValueError, match="temperature"):
        compare(dict(settings, method="dneo", conditions=[]), dict(settings, method="ar", temperature=0, conditions=[]))
    with pytest.raises(ValueError, match="memory_fraction_static"):
        compare(dict(settings, method="dneo", memory_fraction_static=.6), dict(settings, method="ar", memory_fraction_static=.72))
    with pytest.raises(ValueError, match="cublas_workspace_config"):
        compare(dict(settings, method="dneo"), dict(settings, method="ar", cublas_workspace_config=None))


def test_memory_fraction_is_an_explicit_recorded_engine_parameter():
    from dneo import benchmark
    source = Path(benchmark.__file__).read_text()
    assert '"--mem-fraction-static", type=float, default=.60' in source
    tree = ast.parse(source)
    engine = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "Engine")
    expression = next(value.value for value in engine.keywords if value.arg == "mem_fraction_static")
    assert eval(compile(ast.Expression(expression), "memory setting", "eval"), {"args": SimpleNamespace(mem_fraction_static=.6)}) == .6
    assert "memory_fraction_static=args.mem_fraction_static" in source
    assert 'cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"]' in source


def test_stream_timing_uses_first_actual_token_not_delayed_delivery_metadata():
    class Engine:
        def generate(self, **kwargs):
            yield dict(index=0, meta_info=dict(completion_tokens=0, first_token_latency=1000))
            yield dict(index=0, meta_info=dict(completion_tokens=1, first_token_latency=1000))
            yield dict(index=1, meta_info=dict(completion_tokens=1, first_token_latency=1000))
            yield dict(index=0, meta_info=dict(completion_tokens=5))
            yield dict(index=1, meta_info=dict(completion_tokens=5))
    clock = iter([0., .1, .2, .4, .8, 1., 1.1])
    _, elapsed, decode, first = streamed_generate(Engine(), [[1], [2]], [{}, {}], lambda: next(clock))
    assert elapsed == 1.1
    assert decode == pytest.approx(.9)
    assert first == [.2, .4]
