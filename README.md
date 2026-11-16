# DNeo Fast Parallel Drafter

Version: `0.1.0rc5`, optimized v5 runtime. Model: adapted T=7, KV-only
rejection-memory, folded dense head for `Qwen/Qwen3-4B`. This is a drafter,
not a standalone language model.

## Requirements

- Linux x86-64, Python 3.11, one H100 80GB, eight CPU cores per GPU process.
- Qualification used NVIDIA driver `580.126.09` and PyTorch's CUDA `13.0` runtime.
- Build tools: Git, GCC/G++, Rustup with Rust `1.92`. Use a 64GB-RAM build host;
  no GPU is needed until inference verification.
- Core Python pins: PyTorch `2.13.0`, Transformers `5.12.1`, Triton `3.7.1`.
- Kernel pins include FlashInfer `0.6.18` and `sglang-kernel==0.4.7`.
  Install both supplied lock files as shown below; do not install only these core packages.

Both backends require the pinned, patched SGLang installation. Transformers uses
its FA3 kernels but does not run an SGLang server. No SpecForge checkout,
training checkpoint, private package index, or Slurm installation is required.
Initial installation/model downloads need internet access.

## Install

Start with `dneo-inference-source.tar.gz` and the complete `hf_model/`
directory from the qualified release. Check their hashes against `RELEASE_INDEX.json`.
Use a fresh shell without Conda or conflicting CUDA library overrides.

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

Verification should report `source_integrity=PASS`, `model_load=PASS`, and
`gpu_available=true`. In each new inference shell, activate `.venv` and restore
the runtime exports above.
The model config pins target revision `1cfa9a7208912126459214e8b04321603b3df60c`;
target weights download separately. Do not edit the exported config or weights.

## Run With Transformers

```bash
dneo-generate --draft-model "$DNEO_MODEL" \
  --prompt "Compute 37 times 19 and explain the calculation." \
  --max-new-tokens 512 --context-length 1024 --target-temperature 1 --seed 42
```

Returns JSON containing generated text, token count, TPF, request/decode seconds,
and request/decode tokens per second. The CLI applies Qwen's thinking chat template.

## Run With SGLang

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --speculative-algorithm DFLASH --speculative-draft-model-path "$DNEO_MODEL" \
  --speculative-num-draft-tokens 16 --speculative-use-rejection-sampling \
  --trust-remote-code --tp-size 1 --dtype bfloat16 --attention-backend fa3 \
  --disable-radix-cache --cuda-graph-backend-prefill disabled \
  --context-length 1024 --max-running-requests 256 \
  --cuda-graph-max-bs-decode 64 --mem-fraction-static 0.60 --port 30000
```

Expect `DNeoWorker ready:` in the log. `DFLASH` selects the upstream scheduling
family; the model config selects DNeo's native worker, not a DFlash drafter.
The first start loads weights, compiles kernels, and captures graphs; wait for readiness.
From another terminal:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Compute 37 times 19."}],
    "temperature": 1, "top_p": 1, "top_k": -1, "seed": 42, "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

## Expected Performance

Observed release means over eight runs (seeds 42, 1042, 2042, 3042, 4042, 5042,
6042, 7042) on one H100 80GB, BF16, target/draft
temperature 1, block 16 (15 drafted tokens), prompt cap 256, and 512 generated
tokens per request (ignoring EOS). First 256 matched prompts from GSM8K, MATH-500, HumanEval,
MBPP, GPQA, and MT-Bench. No competing GPU workload.

All entries below are **total tokens/sec**, not tokens/sec/user.

| Concurrency | SGLang Decode | SGLang Request | Transformers Decode | Transformers Request |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 701 | 670 | 390 | 379 |
| 4 | 2,237 | 2,156 | 1,048 | 1,027 |
| 16 | 6,500 | 6,309 | 1,936 | 1,905 |
| 32 | 9,031 | 8,776 | 2,178 | 2,137 |
| 64 | 10,463 | 10,158 | 2,205 | 2,155 |

Decode timing excludes prefill/first token. Request timing includes them; neither
includes model loading, compilation, or initial graph capture. These are tokenized-input
benchmarks; HTTP transport and tokenization are not measured. TPF was approximately
4.35-4.55 on this prompt bank. Prompts, output length, hardware, and concurrency
change these numbers. A short arbitrary prompt is not a throughput benchmark.
The historical ~1,500 tokens/sec figure used an invalid timer and is not a target.

## Measure Speedup Against AR

Provide at least 256 Qwen-tokenized prompts, each at most 256 tokens, in this format
(the token IDs below only illustrate the schema):

```json
{"prompts": [{"benchmark": "gsm8k", "token_ids": [123, 456]}]}
```

Stop any running server before benchmarking on the same GPU. Run:

```bash
dneo-benchmark --backend sglang --method dneo --draft-model "$DNEO_MODEL" \
  --prompt-bank prompts.json --output dneo.json --temperature 1 \
  --batches 1 4 16 32 64 --engine-capacity 256 --mem-fraction-static 0.60
dneo-benchmark --backend sglang --method ar \
  --prompt-bank prompts.json --output ar.json --temperature 1 \
  --batches 1 4 16 32 64 --engine-capacity 256 --mem-fraction-static 0.60
dneo-benchmark-summary --dneo dneo.json --ar ar.json --output comparison.json
```

Defaults are 512 output tokens and 32 warmup tokens. Use `--backend transformers`
for that backend; use new output filenames for each run. Summary JSON reports
request/decode speedup separately, throughput, TPF, and conditional position acceptance.
The commands above run seed 42. To reproduce the table's averaging, repeat with
`--seed` set to each listed seed on both benchmark commands, using separate output
files, and take the arithmetic mean of each condition's `decode_tps` and `request_tps`.
Wall-clock throughput will vary; the rounded table is a reference, not an exact output assertion.
Exact replication needs the identical prompt bank; it is not bundled. Its hash and
the qualification evidence are recorded in `PUBLIC_QUALIFICATION.json`.

## Supported Scope

Qualified: H100, TP=1, BF16, concurrency 1/4/16/32/64, block 16, target temperature
0 or 1, draft temperature 1, `top_p=1`, `top_k=-1`, and total context cap 1024.
Changing target temperature to 0 does not make the drafter greedy.
Multi-GPU TP, trees, radix prefix caching, arbitrary sampling filters/penalties,
larger contexts, and other GPUs are not qualified by these results. Shared seeds
do not guarantee identical text between different speculative samplers and AR.

Code: https://github.com/J-sandler/DNeo
Weights: https://huggingface.co/J-sandler/Qwen3-4B-DNeo
