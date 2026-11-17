# DNeo Quickstart

Run Qwen3-4B with the DNeo drafter using Transformers or SGLang.
This checkpoint uses thinking mode and blocks of 16 tokens: one anchor plus
15 draft tokens.

## Install

Complete the one-time [runtime installation](INSTALL.md), then use the same
activated environment below. The tested setup is Linux, Python 3.11, and an
H100 80GB. The optimized Transformers path also uses SGLang's FA3 kernels;
it does not run an SGLang server.

## Run With Transformers

Load the two models, prepare a chat prompt, and decode:

```python
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dneo import DNeoDraftModel
from dneo.generate import decode
from dneo.runtime.dneo_production import activate_production_runtime

activate_production_runtime()
torch.set_num_threads(8)

model_id = os.getenv("DNEO_MODEL", "J-sandler/Qwen3-4B-DNeo")
draft = DNeoDraftModel.from_pretrained(model_id, dtype=torch.bfloat16)
draft = draft.to("cuda").eval().requires_grad_(False)
target_id = draft.config.target_model_name_or_path
target = AutoModelForCausalLM.from_pretrained(
    target_id, revision=draft.config.target_revision,
    dtype=torch.bfloat16, attn_implementation="sdpa",
).to("cuda").eval().requires_grad_(False)
draft.bind_target(target)

tokenizer = AutoTokenizer.from_pretrained(target_id, revision=draft.config.target_revision)
messages = [{"role": "user", "content": "Compute 37 times 19 and explain the calculation."}]
input_ids = tokenizer.apply_chat_template(
    messages, enable_thinking=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=False,
).to(target.device)

result = decode(
    target, draft, input_ids, max_new_tokens=512,
    target_temperature=1, seed=42, context_length=1024,
    eos_token_id=target.generation_config.eos_token_id,
)
print(tokenizer.decode(result["token_ids"], skip_special_tokens=True))
print(f"\n{result['tpf']:.2f} tokens/forward; {result['decode_tps']:.0f} decode tokens/s")
```

`result` contains continuation token IDs, TPF, and request/decode throughput.
For a downloaded model, set `DNEO_MODEL` to its local directory. Hub model IDs
require access to that repository; private uploads require authentication.
The first call loads/compiles kernels and captures graphs.

For command-line use:

```bash
dneo-generate --draft-model J-sandler/Qwen3-4B-DNeo \
  --prompt "Compute 37 times 19 and explain the calculation." \
  --max-new-tokens 512 --context-length 1024 --target-temperature 1 --seed 42
```

## Run With SGLang

[Launch the server and send a chat request.](SERVING.md)
It uses the same checkpoint and an OpenAI-compatible endpoint.

## Expected Performance

| Backend | Tokens / forward | Decode tokens/s | Decode speedup vs. AR |
| --- | ---: | ---: | ---: |
| SGLang | 4.51 | 701 | 2.96x |
| Transformers | 4.54 | 390 | 3.37x |

Batch 1, one H100 80GB, BF16, temperature 1. These are matched benchmark
averages, not expected output for the single prompt above. See
[benchmark settings and concurrency results](BENCHMARKS.md).

The tested total context cap is 1024 tokens, including the prompt and output.
Target temperature may be 0 or 1; the drafter remains at temperature 1.
