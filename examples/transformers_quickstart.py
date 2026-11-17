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
