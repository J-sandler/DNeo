# Reproduce Measurements

Provide a licensed, pretokenized JSON prompt bank:

```json
{"prompts": [{"benchmark": "gsm8k", "token_ids": [1, 2, 3]}]}
```

Use the **same bank and order**, target revision, hardware, temperatures and
budgets for DNeo and AR. The internal suite uses 50 prompts from each of the six
families in `settings.json`. The first 256 requests form the common full-cohort
timing comparison at every concurrency. All requests, including partial tails,
remain available for TPF/acceptance analysis; partial cohorts never enter the
frontier throughput numerator or denominator.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 dneo-benchmark --backend sglang --method dneo --draft-model /path/to/DNeo \
  --prompt-bank prompts.json --output dneo_sglang_t1.json --temperature 1 \
  --batches 1 4 16 32 64 --engine-capacity 256 --mem-fraction-static 0.60
CUBLAS_WORKSPACE_CONFIG=:4096:8 dneo-benchmark --backend sglang --method ar \
  --prompt-bank prompts.json --output ar_sglang_t1.json --temperature 1 \
  --batches 1 4 16 32 64 --engine-capacity 256 --mem-fraction-static 0.60
dneo-benchmark-summary --dneo dneo_sglang_t1.json --ar ar_sglang_t1.json --output comparison.json
```

Repeat with `--backend transformers` and temperature 0. Draft temperature remains
1, including against a greedy target. SGLang uses real streamed first-token/cohort
timestamps. HF uses synchronized fixed-cohort timing after prefill and first
token. Full-request speedup and decode-only speedup are reported separately;
their boundaries are deliberately not interchangeable.

The summary rejects mismatched target, temperature, prompt bank, output budget,
admission capacity and matched-request count. Raw accepted-run histograms are
preserved for reached-position conditional acceptance. Report TTFT and startup
separately; startup/capture is outside steady-state timing, not silently omitted
from an end-to-end request metric.

The memory fraction and cuBLAS workspace configuration are recorded and must match
the AR comparison. Both processes must start with `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
otherwise BF16 verifier arithmetic and stochastic trajectories may differ even with
identical weights and seeds. This is a library initialization setting, not an FP32
or slow reference-attention mode. Do not label
measurements as reproducing a release without its passing, hash-bound GPU and
paired-performance attestation under the protocol in `QUALIFICATION.md`.
