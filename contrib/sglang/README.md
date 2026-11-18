# Pinned SGLang Integration

Base: `https://github.com/sgl-project/sglang`, commit
`7ccbf5fd04f7ee23095fc38e49e749d58dc18282`.

Use CPython 3.11 and a working C/C++ compiler. Source builds of this upstream
revision require Rust 1.92, as specified by its `rust/rust-toolchain.toml`.
Install Rustup using the official <https://rustup.rs/> instructions if it is not
already available. No private compiler, wheel index or model checkpoint is needed.

```bash
rustup toolchain install 1.92 --profile minimal --component clippy --component rustfmt
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout 7ccbf5fd04f7ee23095fc38e49e749d58dc18282
git apply /path/to/dneo/contrib/sglang/0001-dneo-integration.patch
git apply /path/to/dneo/contrib/sglang/0002-pinned-runtime-compatibility.patch
git apply /path/to/dneo/contrib/sglang/0003-text-only-installation.patch
cp -R /path/to/dneo/contrib/sglang/python/. python/
pip install -c /path/to/dneo/contrib/sglang/requirements-linux-py311.lock -e python
```

`0001` contains the public config dispatch and existing verified-step runtime
changes. `0002` preserves the pinned environment's compatibility changes so
reproduction does not rely on an unrecorded working tree. An upstream PR should
review/split these compatibility changes rather than include unrelated fixes.
The package's `source_manifest.json` records the optimized DNeo provenance.
The integration sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` when resolving a DNeo draft
config, before target workers initialize CUDA model computations. An explicitly
conflicting workspace value is rejected. This adds no per-token work and does not
enable batch-invariant or FP32 correctness-mode target kernels.
The `python/sglang/srt/speculative/dneo/` payload contains the native model,
loader, shared engine and kernels in the standard SGLang namespace. Native
serving does not import the standalone `dneo-inference` package. The HF wheel
and this payload are generated from one source and have separate integrity hashes.

These are local release candidates. Patch application and CPU checks are not
equivalent to GPU parity/performance qualification. Consult `QUALIFICATION.md`.
The contribution archive includes its payload, constraints, license and notices;
the native worker does not require the separate Transformers wheel.

The shared-router update is version 0.1.0rc6. It loads a single MoFA router
from the checkpoint and reuses its coefficients across the five draft layers;
old per-layer-router weights remain supported. The update's decoding checks
are recorded in the bundle's `SHARED_ROUTER_QUALIFICATION.json`. Earlier
throughput measurements used the previous checkpoint, not these adapted weights.

`0003` is a text-only installation profile: the pinned upstream metadata requires
a now-unavailable CUDA Tile RC and audio packages tied to another Torch version.
It selects stable `cuda-tile==1.6.0` and excludes mismatched audio dependencies.
It retains `torchvision==0.28.0` and `timm==1.0.16`, because upstream common
utilities import vision helpers even for text-only serving. Torchvision 0.28
declares compatibility with the pinned Torch 2.13.
It does not change inference operators. This candidate does not claim multimodal
SGLang support, or binary equivalence of the updated dependency environment; GPU
qualification of that fresh installation is still required.
