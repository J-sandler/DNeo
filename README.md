# DNeo Fast Parallel Drafter

[**GitHub**](https://github.com/J-sandler/DNeo) |
[**Quickstart**](https://github.com/J-sandler/DNeo/blob/main/QUICKSTART.md#install) |
[**Benchmarks**](https://github.com/J-sandler/DNeo/blob/main/BENCHMARKS.md)

**DNeo drafts 15 tokens from a single backbone pass for Qwen3-4B in thinking
mode.** Instead of discarding rejected predictions, it reuses their target-model
features as feedback for the next draft.

This checkpoint is the **drafter**, not a standalone language model. It must
be paired with [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), which
verifies proposals using speculative rejection sampling.



## Architecture

- **Parallel backbone.** Five transformer layers produce the draft block;
  masked positions do not attend to one another.
- **MoFA feature routing.** One learned mixture of normalized features from
  seven selected target layers is shared across all five drafter layers,
  KV heads, prefix, and rejection memory.
- **rejection memory.** Features from rejected tokens become attention
  memory for the next draft, without passing through the drafter's MLPs.
- **Two-tap convolutions.** DFlash-2-style convolutions transfer information
  between neighboring draft positions.
- **DNeo fused causal head.** A folded dense correction head conditions each
  prediction on the preceding token, without another backbone pass.


![Screenshot 2026-11-16 at 8.35.40 PM](https://cdn-uploads.huggingface.co/production/uploads/66ca057ecee5cf1456ba278a/EDjzANDO0hMEnXtMTHvkb.png)

*Historical figure from the previous per-layer-router checkpoint.*

## Performance

The current default is the **shared-router checkpoint**, adapted for 100k
prompts. In a matched online Leviathan test:

| Checkpoint | Tokens / forward |
| --- | ---: |
| Current shared-router DNeo | 4.21 |
| Previous per-layer-router DNeo | 4.30 |

192 prompts across six benchmarks, two seeds, 512 output tokens per request;
target and draft temperature 1, draft length 15. The shared checkpoint retains
97.87% of the previous checkpoint's TPF. Its serving speed has not been
remeasured in a full benchmark sweep.

**Historical serving results for the previous per-layer checkpoint:**

| Backend | Tokens / forward | Decode tokens/s | Decode speedup vs. AR |
| --- | ---: | ---: | ---: |
| SGLang | 4.51 | 701 | 2.96x |
| Transformers | 4.54 | 390 | 3.37x |

Batch 1 on one H100 80GB, BF16, block size 16, target and draft temperature 1.
Eight-seed means on 256 matched prompts from GSM8K, MATH-500, HumanEval, MBPP,
GPQA, and MT-Bench. Prompts are capped at 256 tokens; outputs are fixed at 512
tokens with EOS ignored; context cap is 1024. Decode timing excludes prefill,
startup, tokenization, and HTTP transport. Results depend on the workload.


![Screenshot 2026-11-16 at 8.46.44 PM](https://cdn-uploads.huggingface.co/production/uploads/66ca057ecee5cf1456ba278a/nCCgMMPfHjVY1cYbqpT65.png)

*Historical figure from the previous per-layer-router checkpoint.*

## Inference

This revision requires the **shared-router runtime (0.1.0rc6 or newer)**.
Download the matching [standalone inference source](https://huggingface.co/J-sandler/Qwen3-4B-DNeo/resolve/main/dneo-inference-source.tar.gz)
and follow its `INSTALL.md` for Transformers and native SGLang. The bundle
includes both implementations; older five-router loaders cannot load these
single-router weights. See [INFERENCE_UPDATE.md](./INFERENCE_UPDATE.md).

Follow the [SGLang guide](https://github.com/J-sandler/DNeo/blob/main/QUICKSTART.md#run-with-sglang) or
[Transformers guide](https://github.com/J-sandler/DNeo/blob/main/QUICKSTART.md#run-with-transformers).
Both use the pinned DNeo runtime documented in the code repository.

## License

Draft weights: [Apache-2.0](https://huggingface.co/J-sandler/Qwen3-4B-DNeo/blob/main/LICENSE).
Inference and included model code: [MIT](https://github.com/J-sandler/DNeo/blob/main/LICENSE).
See [LICENSE-MIT](https://huggingface.co/J-sandler/Qwen3-4B-DNeo/blob/main/LICENSE-MIT)
and [NOTICE](https://huggingface.co/J-sandler/Qwen3-4B-DNeo/blob/main/NOTICE) for attribution.
