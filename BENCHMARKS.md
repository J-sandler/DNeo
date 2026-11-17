# Benchmarks

Fresh-source measurements on one H100 80GB, BF16, TP=1, target/draft
temperature 1, and blocks of 16 (15 draft tokens). Rates are total tokens/s,
not per-user rates. Decode excludes prefill and the first token; request
timing includes them. Neither includes startup, HTTP transport, or tokenization.

| Concurrency | SGLang Decode | SGLang Request | Transformers Decode | Transformers Request |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 701 | 670 | 390 | 379 |
| 4 | 2,237 | 2,156 | 1,048 | 1,027 |
| 16 | 6,500 | 6,309 | 1,936 | 1,905 |
| 32 | 9,031 | 8,776 | 2,178 | 2,137 |
| 64 | 10,463 | 10,158 | 2,205 | 2,155 |

These are means over eight seeds: 42, 1042, 2042, 3042, 4042, 5042, 6042, 7042.
Each run times the first 256 matched requests from GSM8K, MATH-500, HumanEval,
MBPP, GPQA, and MT-Bench. Only full cohorts enter throughput measurements.
Prompts are capped at 256 tokens; outputs are fixed at 512 with EOS ignored;
total context cap is 1024. There was no competing workload on the measured GPU.

## Run A Comparison

Supply a Qwen-tokenized prompt bank as `prompts.json`:

```json
{"prompts": [{"benchmark": "gsm8k", "token_ids": [123, 456]}]}
```

The IDs above illustrate the format only. Use at least 256 real prompts of
at most 256 tokens. Stop other servers on the same GPU, then run:

```bash
dneo-benchmark --backend sglang --method dneo --draft-model "$DNEO_MODEL" \
  --prompt-bank prompts.json --output dneo.json --temperature 1 \
  --batches 1 4 16 32 64 --engine-capacity 256 --mem-fraction-static 0.60
dneo-benchmark --backend sglang --method ar \
  --prompt-bank prompts.json --output ar.json --temperature 1 \
  --batches 1 4 16 32 64 --engine-capacity 256 --mem-fraction-static 0.60
dneo-benchmark-summary --dneo dneo.json --ar ar.json --output comparison.json
```

Use `--backend transformers` for that backend. Defaults are seed 42, 512
output tokens, and 32 warmup tokens per cohort. Repeat both benchmark commands
with each listed `--seed`, keeping separate output files. Average per-seed
throughputs and paired same-seed speedup ratios, not ratios of global means.

Benchmark TPF pools generated tokens / verification calls over the matched
requests within each seed. The single-prompt Python/CLI decoder excludes the
prefill token from its TPF numerator. The summary's per-benchmark acceptance
statistics use all supplied records, not just the throughput-qualified subset.

The exact prompt bank is not bundled. Its SHA256 is
`c68ec0b96dab9de0aed3d5054b563044ae8e52cfd04bf783780ef7ac6c6da6e6`.
Different prompts change TPF and speed. The old ~1,500 tokens/s measurement
used an invalid timer and is not a reproduction target.

Other GPUs, contexts above 1024, TP>1, trees, radix prefix caching, and arbitrary
sampling filters are outside these performance claims. Shared seeds do not
guarantee the same text across AR and speculative samplers.
