"""Packed QKV/gate-up and fused normalization for the existing MoFA blocks."""
import contextlib
import types
import os
import torch
import torch.nn.functional as F
from dneo.runtime.packed_target import kernel_mlp, kernel_norm


class DraftOperators:
    @torch.inference_mode()
    def __init__(self, draft):
        self.draft = draft
        self.tuning = os.environ.get('DNEO_BACKBONE_TUNING', 'none')
        self.qkv = []
        for layer in draft.layers:
            packs = (
                (layer, (layer.q_proj, layer.k_proj, layer.v_proj), "_dneo_qkv"),
                (layer.mlp, (layer.mlp.gate_proj, layer.mlp.up_proj), "_dneo_gate_up"),
            )
            for owner, projections, name in packs:
                if any(p.bias is not None for p in projections):
                    raise ValueError("packed draft operators require bias-free projections")
                if not hasattr(owner, name):
                    packed = torch.cat([p.weight for p in projections])
                    setattr(owner, name, packed)
                    offset = 0
                    for projection in projections:
                        projection.weight.set_(packed[offset:offset + projection.out_features])
                        offset += projection.out_features

    @contextlib.contextmanager
    def activate(self):
        methods = []
        for layer in self.draft.layers:
            state = {}
            sizes = (layer.q_proj.out_features, layer.k_proj.out_features, layer.v_proj.out_features)
            for index, projection in enumerate((layer.q_proj, layer.k_proj, layer.v_proj)):
                methods.append((projection, projection.forward))
                def projected(module, hidden, index=index, state=state, owner=layer, sizes=sizes):
                    if index == 0:
                        state["input"] = hidden
                        state["parts"] = F.linear(hidden, owner._dneo_qkv).split(sizes, dim=-1)
                    if state.get("input") is hidden:
                        return state["parts"][index]
                    return F.linear(hidden, module.weight, module.bias)
                projection.forward = types.MethodType(projected, projection)
            methods.append((layer.mlp, layer.mlp.forward))
            mlp_function = kernel_mlp
            if self.tuning in ('activation', 'all'):
                from dneo.runtime.optimized_backbone_ops import kernel_mlp as tuned_mlp
                mlp_function = tuned_mlp
            layer.mlp.forward = types.MethodType(mlp_function, layer.mlp)
        for module in self.draft.modules():
            if module.__class__.__name__ == "Qwen3RMSNorm":
                methods.append((module, module.forward))
                module.forward = types.MethodType(kernel_norm, module)
        try:
            yield
        finally:
            for module, method in methods:
                module.forward = method
