"""Portable qualified kernel selection, with no user-machine source paths."""
import hashlib
import json
import os
from pathlib import Path


def activate_production_runtime(hardware=None, checkpoint=None, verify_sources=True):
    profile = json.loads(Path(__file__).with_name("profile.json").read_text())
    if hardware is None:
        import torch
        name = torch.cuda.get_device_name().lower()
        hardware = "h100" if "h100" in name else "a10" if "a10" in name else None
    if hardware not in ("h100", "a10"):
        raise ValueError("this release currently qualifies H100 and A10 kernels only")
    values = {**profile["environment"]["common"], **profile["environment"][hardware]}
    # Serving capacity is a scheduler decision; kernel selection is independent.
    values.pop("DNEO_ENGINE_MAX_RUNNING_REQUESTS", None)
    conflicts = {key: os.environ[key] for key, value in values.items() if key in os.environ and os.environ[key] != value}
    if conflicts:
        raise ValueError(f"conflicting optimized DNeo settings: {conflicts}")
    os.environ.update(values)
    if verify_sources:
        manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_text())
        root = Path(__file__).resolve().parents[1]
        for relative, record in manifest.items():
            path = root / relative.removeprefix("src/dneo/")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["packaged_sha256"]:
                raise RuntimeError(f"modified DNeo runtime: {relative}")
    return profile
