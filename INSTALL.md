# Runtime Installation

Linux x86-64, Python 3.11, Git, GCC/G++, and Rustup are required.
Allow eight CPU cores and 64GB host RAM for the source build. Inference results
were measured on one H100 80GB. This setup is for the GPU server, not a Mac.

Use `dneo-inference-source.tar.gz` from the matching
[HF model revision](https://huggingface.co/J-sandler/Qwen3-4B-DNeo/tree/main),
with the model's complete export in `hf_model/` beside it. The matching bundle
is version 0.1.0rc6; older releases do not support single-router weights.
Check the supplied `SHA256SUMS` before installing.
If the repositories are private, authenticate before downloading.

The following is the previously GPU-tested installation recipe, unchanged:

```bash
export DNEO_MODEL="$(pwd)/hf_model"
tar -xzf dneo-inference-source.tar.gz
cd dneo-inference
export DNEO_ROOT="$PWD"
python3.11 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH PYTHONHOME LD_PRELOAD

rustup toolchain install 1.92 --profile minimal --component clippy --component rustfmt
git clone --no-checkout https://github.com/sgl-project/sglang.git sglang
git -C sglang checkout 7ccbf5fd04f7ee23095fc38e49e749d58dc18282
git -C sglang apply "$DNEO_ROOT/contrib/sglang/0001-dneo-integration.patch"
git -C sglang apply "$DNEO_ROOT/contrib/sglang/0002-pinned-runtime-compatibility.patch"
git -C sglang apply "$DNEO_ROOT/contrib/sglang/0003-text-only-installation.patch"
cp -R contrib/sglang/python/. sglang/python/
export CARGO_BUILD_JOBS=2 MAX_JOBS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export CARGO_TARGET_DIR="$(mktemp -d /tmp/dneo-cargo.XXXXXX)"
python -m pip install --index-url https://pypi.org/simple \
  -r requirements-h100.lock -c requirements-linux-py311.lock . ./sglang/python
python -m pip check

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SGLANG_SPEC_DRAFT_TEMPERATURE_OVERRIDE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
dneo-verify-install --model "$DNEO_MODEL"
```

Run the final verification on the GPU server. Expect `source_integrity=PASS`,
`model_load=PASS`, and `gpu_available=true`.

For a new shell, activate `.venv` and restore the runtime exports at the
end of the block. The code selects the optimized DNeo kernels automatically;
no kernel-tuning environment variables need to be set by hand.

## Dependencies

The supplied locks pin PyTorch 2.13.0 / CUDA 13.0, Transformers 5.12.1,
Triton 3.7.1, FlashInfer 0.6.18, and sglang-kernel 0.4.7. Both lock files and
the patched SGLang build are required, including for Transformers inference.
Installing only `torch` and `transformers` does not install this fast runtime.
The qualification driver was NVIDIA 580.126.09.

The model config pins the Qwen3-4B target revision. Target weights and tokenizer
download separately through Hugging Face. No SpecForge checkout is required.
