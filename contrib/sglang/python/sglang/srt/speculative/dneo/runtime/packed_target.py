"""Inference-only packing of native Transformers Qwen3 projection weights.

Qwen3 attention algebra follows transformers/models/qwen3/modeling_qwen3.py
(Apache-2.0, Qwen/Hugging Face). Packed GEMMs follow the SGLang Qwen3 layout.
No target weights or accumulation precision are quantized.
"""
import contextlib
import types
import torch
import torch.nn.functional as F
from sglang.srt.speculative.dneo.runtime.graph_target import GraphHFTarget


def attention(module, hidden_states, position_embeddings, attention_mask,
              past_key_values=None, cache_position=None, **kwargs):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    shape = hidden_states.shape[:-1]
    qsize, ksize = module.q_proj.out_features, module.k_proj.out_features
    q, k, v = F.linear(hidden_states, module._dneo_qkv).split((qsize, ksize, ksize), dim=-1)
    q = module.q_norm(q.reshape(*shape, -1, module.head_dim)).transpose(1, 2)
    k = module.k_norm(k.reshape(*shape, -1, module.head_dim)).transpose(1, 2)
    v = v.reshape(*shape, -1, module.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, module.layer_idx, dict(sin=sin, cos=cos, cache_position=cache_position))
    interface = eager_attention_forward if module.config._attn_implementation == "eager" else ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
    output, weights = interface(module, q, k, v, attention_mask, dropout=0., scaling=module.scaling,
                                sliding_window=module.sliding_window, **kwargs)
    return module.o_proj(output.reshape(*shape, -1).contiguous()), weights


def mlp(module, hidden):
    gate, up = F.linear(hidden, module._dneo_gate_up).chunk(2, dim=-1)
    return module.down_proj(module.act_fn(gate) * up)


def kernel_mlp(module, hidden):
    from sglang.srt.speculative.dneo.runtime.kernels import silu_mul
    return module.down_proj(silu_mul(F.linear(hidden, module._dneo_gate_up)))


def kernel_norm(module, hidden):
    from sglang.srt.speculative.dneo.runtime.kernels import rmsnorm
    return rmsnorm(hidden, module.weight, module.variance_epsilon)


class PackedHFTarget(GraphHFTarget):
    @torch.inference_mode()
    def __init__(self, target, capacity=1024, fused_kernels=False):
        super().__init__(target, capacity)
        if target.config.model_type != "qwen3":
            raise ValueError("packed HF verifier supports Qwen3 only")
        self.features = [None] * len(self.backbone.layers)
        self.fused_kernels = fused_kernels
        for layer in self.backbone.layers:
            for module, projections, name in (
                (layer.self_attn, (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj), "_dneo_qkv"),
                (layer.mlp, (layer.mlp.gate_proj, layer.mlp.up_proj), "_dneo_gate_up"),
            ):
                if any(p.bias is not None for p in projections):
                    raise ValueError("packed Qwen3 path requires bias-free projections")
                if not hasattr(module, name):
                    packed = torch.cat([p.weight for p in projections], dim=0)
                    setattr(module, name, packed)
                    offset = 0
                    # Retain the original modules for the reference checker,
                    # but let them view the same packed storage to save VRAM.
                    for projection in projections:
                        projection.weight.set_(packed[offset:offset + projection.out_features])
                        offset += projection.out_features

    @contextlib.contextmanager
    def packed_layers(self):
        methods, handles = [], []
        for i, layer in enumerate(self.backbone.layers):
            for module, forward in ((layer.self_attn, attention), (layer.mlp, kernel_mlp if self.fused_kernels else mlp)):
                methods.append((module, module.forward))
                module.forward = types.MethodType(forward, module)
            def capture(module, inputs, output, index=i):
                self.features[index] = output
            if self.capture_features:
                handles.append(layer.input_layernorm.register_forward_hook(capture))
        if self.fused_kernels:
            for module in self.backbone.modules():
                if module.__class__.__name__ == "Qwen3RMSNorm":
                    methods.append((module, module.forward))
                    module.forward = types.MethodType(kernel_norm, module)
        try:
            yield
        finally:
            for module, method in methods:
                module.forward = method
            for handle in handles:
                handle.remove()

    def _run(self, ids, positions, mask, last_logit_only=False):
        with self.packed_layers():
            output = self.backbone(input_ids=ids, past_key_values=self.cache, cache_position=positions,
                                   position_ids=positions[None], attention_mask={"full_attention": mask},
                                   use_cache=True, output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state[:, -1:] if last_logit_only else output.last_hidden_state
        return self.target.lm_head(hidden), (torch.stack(self.features, dim=2) if self.capture_features else None)
