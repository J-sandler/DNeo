"""Public model-ID loading without trainer, saved arguments or private caches."""
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from .modeling_dneo import DNeoDraftModel


def resolve_model(path, revision=None):
    local = Path(path).expanduser()
    return local.resolve() if local.is_dir() else Path(snapshot_download(str(path), revision=revision))


def load_draft(path, device="cpu", revision=None):
    root = resolve_model(path, revision)
    model, info = DNeoDraftModel.from_pretrained(root, dtype=torch.bfloat16, output_loading_info=True)
    problems = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if problems:
        raise ValueError(f"DNeo checkpoint does not match the declared architecture: {problems}")
    model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    mask = model.conditioner.shared_mofa_candidate_mask.nonzero().flatten().tolist()
    if mask != model.config.target_layer_ids:
        raise ValueError("checkpoint MoFA mask disagrees with portable configuration")
    return model


def load_checkpoint(path, device="cuda"):
    model = load_draft(path, device)
    target = AutoModelForCausalLM.from_pretrained(model.config.target_model_name_or_path,
                revision=model.config.target_revision, dtype=torch.bfloat16,
                attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    model.bind_target(target)
    return target, model, SimpleNamespace(), {"architecture": model.config.to_dict()}, resolve_model(path)
