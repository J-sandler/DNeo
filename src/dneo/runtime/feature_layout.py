"""Checkpoint-defined feature support; no inference-time top-k approximation."""
import os

import torch


def feature_layer_ids(conditioner, *, compact=True):
    total = int(conditioner.num_target_layers)
    mask = getattr(conditioner, "shared_mofa_candidate_mask", None)
    if mask is None:
        return tuple(range(total))
    if mask.dtype != torch.bool or mask.shape != (total,) or not bool(mask.any()):
        raise ValueError("invalid checkpoint MoFA candidate mask")
    if not compact or os.environ.get("DNEO_COMPACT_MOFA", "1") == "0":
        return tuple(range(total))
    return tuple(mask.nonzero().flatten().cpu().tolist())


def router_weights(conditioner, logits, layer_ids, dtype):
    mask = getattr(conditioner, "shared_mofa_candidate_mask", None)
    values = logits.detach().float()
    if mask is not None:
        values = values.masked_fill(~mask, torch.finfo(values.dtype).min)
    probabilities = values.softmax(-1)
    selected = torch.tensor(layer_ids, device=values.device)
    return probabilities[:, 0].index_select(-1, selected).to(dtype).contiguous()


class NormalizedFeatureCapture:
    """Reuse the existing HF input RMSNorm output instead of recomputing it."""
    def __init__(self, backbone, layer_ids):
        self.values = [None] * len(layer_ids)
        self.handles = []
        for slot, layer_id in enumerate(layer_ids):
            def capture(module, inputs, output, slot=slot):
                self.values[slot] = output
            self.handles.append(backbone.layers[layer_id].input_layernorm.register_forward_hook(capture))

    def stack(self):
        if any(value is None for value in self.values):
            raise RuntimeError("selected target features were not captured")
        return torch.stack(self.values, dim=2).detach()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()
