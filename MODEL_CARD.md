---
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3-4B
tags: [dneo, speculative-decoding, draft-model]
---

# DNeo Fast Parallel Drafter

DNeo is a speculative drafter for **Qwen/Qwen3-4B with thinking enabled**.
It combines seven-layer MoFA feature routing, KV-only rejection memory,
prefix-only attention, and a folded dense prediction head.
The target verifies the proposed tokens using Leviathan rejection sampling.
This checkpoint is a drafter, not a standalone language model.

On one H100 80GB, our verified Transformers run averages **390 decoding tokens/s
and 3.37x decode speedup over matched autoregressive decoding at batch 1**.
Request speedup, including prefill and the first token, is **3.30x**.

## Installation

Use Linux x86-64, Python 3.11, one H100 80GB, and eight CPU cores per inference
process. Building requires Git, GCC/G++, Rustup, and a 64GB-RAM build host.
The tested runtime uses PyTorch 2.13.0, CUDA 13.0, Transformers 5.12.1, and
Triton 3.7.1; the qualification driver was NVIDIA 580.126.09.

Download the matching `dneo-inference-source.tar.gz` release and place this
model's complete export in `hf_model/` beside it. Verify the release
checksums before installation. The public Qwen target weights are loaded
separately at revision `1cfa9a7208912126459214e8b04321603b3df60c`.

<details>
<summary>Install the pinned runtime once</summary>

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

</details>

Both lock files and the pinned, patched SGLang build are required, including
for Transformers: it uses SGLang's FA3 kernels without running an SGLang server.
No SpecForge checkout or training checkpoint is needed. Verification should
report `source_integrity=PASS`, `model_load=PASS`, and `gpu_available=true`.

## Transformers Inference

After installation, run:

```bash
dneo-generate --draft-model "$DNEO_MODEL" \
  --prompt "Compute 37 times 19 and explain the calculation." \
  --max-new-tokens 512 --context-length 1024 --target-temperature 1 --seed 42
```

This is the optimized Transformers entry point. It loads the target and drafter,
applies Qwen's thinking chat template, and returns JSON with the generated text,
TPF, and request/decode tokens per second. The drafter temperature is 1.
In a new shell, reactivate `.venv`, set `DNEO_MODEL` to the downloaded export,
and restore the runtime exports above. The first start loads weights and
prepares kernels and graphs; wait for it to finish.

One recorded run of this exact example returned the following metrics:

```json
{"tokens": 512, "tpf": 5.01, "decode_tps": 417.6, "request_tps": 281.6}
```

These are one prompt's measurements, not the benchmark average below. Timings,
response length, and TPF depend on the prompt; thinking may fill the token cap.
Do not call ordinary causal `.generate()` on the draft alone.

## Measured H100 Performance

**Transformers only; all token rates are total tokens/s, not tokens/s/user.**
Speedups compare DNeo with AR on the same prompts, GPU, backend, and seed.

| Concurrency | TPF | Decode tokens/s | Request tokens/s | Decode speedup | Request speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.54 | 390 | 379 | 3.37x | 3.30x |
| 4 | 4.40 | 1,048 | 1,027 | 2.96x | 2.92x |
| 16 | 4.38 | 1,936 | 1,905 | 2.70x | 2.67x |
| 32 | 4.37 | 2,178 | 2,137 | 2.47x | 2.44x |
| 64 | 4.36 | 2,205 | 2,155 | 2.19x | 2.16x |

Settings: one H100 80GB, TP=1, BF16, target and drafter temperature 1,
`top_p=1`, `top_k=-1`, block size 16 (15 draft tokens), context cap 1024,
prompt cap 256, 512 generated tokens per request with EOS ignored, and
32 warmup tokens per cohort. Each run uses the first 256 matched prompts
from a six-benchmark bank: GSM8K, MATH-500, HumanEval, MBPP, GPQA, and MT-Bench.
Only full cohorts enter the table.

Entries are arithmetic means over seeds 42, 1042, 2042, 3042, 4042, 5042,
6042, and 7042. For each seed, benchmark TPF is total generated tokens divided
by total verification calls on those matched requests; speedup is the paired
DNeo/AR throughput ratio. Decode timing excludes prefill and the first token;
request timing includes them. Neither includes initial model loading, graph
capture, HTTP transport, or tokenization. These are reference measurements,
not a latency or throughput guarantee for arbitrary prompts or hardware.

## Reproduce The Benchmark

Provide the matched Qwen-tokenized input bank as `prompts.json`, with at least
256 entries of the form `{"benchmark": "gsm8k", "token_ids": [...]}` and at most
256 tokens per prompt. This bank is not bundled with the weights; its SHA256
is `c68ec0b96dab9de0aed3d5054b563044ae8e52cfd04bf783780ef7ac6c6da6e6`.
Different prompts will not reproduce the same TPF or throughput.

```bash
dneo-benchmark --backend transformers --method dneo --draft-model "$DNEO_MODEL" \
  --prompt-bank prompts.json --output dneo.json --temperature 1 \
  --batches 1 4 16 32 64 --engine-capacity 256 --mem-fraction-static 0.60
dneo-benchmark --backend transformers --method ar \
  --prompt-bank prompts.json --output ar.json --temperature 1 \
  --batches 1 4 16 32 64 --engine-capacity 256 --mem-fraction-static 0.60
dneo-benchmark-summary --dneo dneo.json --ar ar.json --output comparison.json
```

Defaults are 512 output tokens and 32 warmup tokens. The commands above use
seed 42. Repeat both benchmark commands with `--seed` set to each listed seed,
using separate output paths, then average the per-seed metrics as described above.
`comparison.json` also contains per-benchmark TPF and conditional position
acceptance. Those per-benchmark summaries include all input prompts, whereas
the throughput table uses only the first 256 matched requests and full cohorts.

## Verification

These installation, generation, benchmark, and summary commands were executed
from a fresh source installation of `dneo-inference==0.1.0rc5`, not just checked
for syntax. The table above is computed from the saved outputs, not estimates.
Transformers acceptance counts and histograms matched the qualified reference
exactly over 10,240 request comparisons: eight seeds, five batch sizes, and
256 requests per condition. Wall-clock timings are not bitwise reproducible.

These results qualify H100, TP=1, BF16, block size 16, and the settings above.
Other GPUs, larger contexts, multi-GPU TP, trees, and sampling filters are not
covered by these performance claims. Shared seeds do not imply identical text
between speculative decoding and AR.

## Repositories And Licenses

Code: [J-sandler/DNeo](https://github.com/J-sandler/DNeo).
Weights: [J-sandler/Qwen3-4B-DNeo](https://huggingface.co/J-sandler/Qwen3-4B-DNeo).

Draft weights are licensed under Apache-2.0; see `LICENSE`. The included
configuration/modeling/reference Python files retain their MIT license; see
`LICENSE-MIT` and `NOTICE`. The inference code repository retains MIT, with
upstream notices preserved. The separately downloaded Qwen target and runtime
dependencies retain their respective upstream licenses.
