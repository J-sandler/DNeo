"""The trained dNEO computation with dead final-layer work removed."""
import os
import torch
import torch.nn.functional as F

from sglang.srt.speculative.dneo.runtime.structural_kernels import conv_residual_norm, inject_norm, prepare_full, qkv_norm_rope, structural_attention, pack_attention_kv, write_expanded_kv


def default_draft_kernel(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 9:
        return "fused-sdpa-native"
    return "fused-sdpa-pack"


class StructuralDraft:
    def __init__(self, draft, variant="structural", splits=1):
        if draft.training:
            raise ValueError('structural kernels are inference-only; call eval() first')
        if not draft.prefix_only_mask_attention or draft.dual_attention_normalization:
            raise ValueError("structural kernels require dNEO prefix-only joint-softmax attention")
        if draft.is_mla or draft.prefix_drafter_prefill or draft.internal_mofa or draft.intermediate_sequential_markov:
            raise ValueError("unsupported structural draft architecture")
        if splits not in (1, 2, 4, 8):
            raise ValueError('attention splits must be 1, 2, 4 or 8')
        if any(layer.attention_conv is None or layer.mlp_conv is None or
               layer.mlp_intermediate_hadamard is not None for layer in draft.layers):
            raise ValueError('structural kernels require ordinary DF2 attention/MLP convolutions')
        self.draft, self.variant, self.splits = draft, variant, splits
        self.kv_only = bool(getattr(draft, "rejection_memory_kv_only", False))
        self.layer_residual = bool(getattr(draft, "rejection_memory_layer_residual", False))
        self.reference_arithmetic = bool(getattr(draft, "rejection_memory_fusion_mlp_hidden_size", 0))
        accumulation = os.environ.get("DNEO_KV_ONLY_ATTENTION_ACCUMULATION", "fp32")
        if accumulation not in ("bf16", "fp32"):
            raise ValueError("DNEO_KV_ONLY_ATTENTION_ACCUMULATION must be bf16 or fp32")
        self.accurate_attention = not self.kv_only or accumulation == "fp32"
        self.attention_backend = os.environ.get("DNEO_KV_ONLY_ATTENTION_BACKEND", "triton")
        if self.attention_backend not in ("triton", "triton-tf32", "fa3-split"):
            raise ValueError("unsupported K/V-only attention backend")
        self.attention_capacity = None
        if os.environ.get("DNEO_REFERENCE_MATCHED_ATTENTION", "0") == "1":
            self.attention_backend = "sdpa-checked"
        self.profile_events = None
        self.audit_layer_outputs = None
        self.rotary_tables = None
        self.tuning = os.environ.get('DNEO_BACKBONE_TUNING', 'none')
        if self.tuning not in ('none', 'activation', 'rope', 'merge', 'all'):
            raise ValueError('unsupported DNEO_BACKBONE_TUNING')

    def __call__(self, embeddings, prefix_kv, anchors, memory, valid, positions, attention_kv=None,
                 request_slots=None, reuse_memory_kv=False):
        from sglang.srt.speculative.dneo.modeling_dneo import apply_rope
        draft = self.draft
        batch, _, b, d = embeddings.shape
        width = valid.shape[-1]
        if reuse_memory_kv and not self.kv_only:
            raise ValueError("cached memory projections require K/V-only rejection memory")
        local = embeddings[:, 0]
        feedback_valid = valid
        feedback = memory
        if self.layer_residual and not self.kv_only:
            # Fusion supplies residuals, not additional attention keys or queries.
            valid = valid[..., :0]
            positions = positions[..., :0]
            memory = tuple(value[:, :, :0] for value in memory)
            width = 0
        memory_state = None
        cached_rope = (self.tuning in ('rope', 'all') and self.rotary_tables is not None
                       and self.kv_only and reuse_memory_kv and not self.reference_arithmetic)
        if cached_rope:
            cos = sin = None
        else:
            block_positions = anchors + torch.arange(b, device=local.device)[None]
            all_positions = torch.cat((block_positions, positions[:, 0]), 1)
            cos, sin = draft.rotary_emb(local, all_positions)
        use_fa3 = (self.kv_only and self.variant == 'fused-sdpa-native'
                   and self.attention_backend == 'fa3-split' and local.dtype != torch.float32)
        fa3_lengths = anchors.reshape(batch).to(torch.int32) if use_fa3 else None
        fa3_slots = request_slots.to(torch.int32) if use_fa3 and request_slots is not None else None
        cached_merge = self.tuning in ('merge', 'all') and use_fa3 and reuse_memory_kv
        conv_tile = 1024 if self.tuning == 'all' else 256
        fuse_next_norm = (self.tuning == 'all' and self.kv_only
                          and not self.layer_residual and not self.reference_arithmetic
                          and self.variant.startswith('fused-sdpa')
                          and all(norm.__class__.__name__ == 'Qwen3RMSNorm' for norm in
                                  [layer.input_layernorm for layer in draft.layers[1:]] + [draft.norm]))
        next_normalized = None
        full_mask = compact_mask = None
        if self.variant.startswith('fused-sdpa') and self.variant != 'fused-sdpa-native':
            from sglang.srt.speculative.dneo.modeling_dneo import rejection_memory_attention_mask
            cap = prefix_kv[0][0].shape[2]
            full_mask = rejection_memory_attention_mask(anchors, torch.ones_like(anchors, dtype=torch.bool),
                torch.ones(batch, cap, device=local.device, dtype=torch.long), valid, b, prefix_only=True)
            if self.kv_only or self.layer_residual or self.variant in ('fused-sdpa-pruned', 'fused-sdpa-pack-pruned'):
                selected = torch.cat((torch.arange(cap + 1, device=local.device),
                                      torch.arange(cap + b, cap + b + width, device=local.device)))
                compact_mask = full_mask[:, :, :b].index_select(-1, selected)
        for index, layer in enumerate(draft.layers):
            events = self.profile_events[index] if self.profile_events is not None else None
            if events:
                events[0].record()
            fused_layout = self.variant.startswith('fused-sdpa') and not self.reference_arithmetic
            if self.layer_residual:
                count = feedback_valid.shape[-1]
                updated = draft.apply_rejection_fusion(
                    local[:, 1:1 + count], feedback[index][:, 0, :, 0],
                    feedback_valid[:, 0], index)
                local = torch.cat((local[:, :1], updated, local[:, 1 + count:]), 1)
            if self.kv_only:
                hidden = local
                normalized = (next_normalized if next_normalized is not None
                              else layer.input_layernorm(hidden))
            elif width and fused_layout:
                hidden, normalized = inject_norm(local, memory_state, memory[index][:, 0, :, 0],
                    draft.rejection_source_embeddings[index].to(local.dtype), layer.input_layernorm)
            elif width:
                # Preserve the original two BF16 additions and their order.
                memory_state = (memory[index][:, 0, :, 0] + draft.rejection_source_embeddings[index].to(local.dtype)) if memory_state is None else (
                    memory_state + memory[index][:, 0, :, 0]
                ) + draft.rejection_source_embeddings[index].to(local.dtype)
                hidden = torch.cat((local, memory_state), 1)
            else:
                hidden = local
            if not self.kv_only and not (width and fused_layout):
                normalized = layer.input_layernorm(hidden)
            if fused_layout:
                normalized, conv_kernel = prepare_full(layer.attention_conv, normalized, b, tile=conv_tile)
            else:
                convolved, conv_kernel = layer.attention_conv.prepare(normalized[:, :b], (b,))
                normalized = torch.cat((convolved, normalized[:, b:]), 1) if width else convolved
            compact = self.kv_only or self.layer_residual or ((not fused_layout or self.variant in ('fused-sdpa-pruned', 'fused-sdpa-pack-pruned')) and (index == len(draft.layers) - 1 or width == 0))
            if self.reference_arithmetic and not self.kv_only:
                compact = False
                q, k, v = layer.q_proj(normalized), layer.k_proj(normalized), layer.v_proj(normalized)
            elif compact:
                schedule = os.environ.get('DNEO_ANCHOR_PROJECTION', 'auto')
                if schedule not in ('auto', 'split', 'packed'):
                    raise ValueError('unsupported anchor projection schedule')
                packed_anchor = reuse_memory_kv and (schedule == 'packed' or (schedule == 'auto' and batch <= 16))
                if packed_anchor:
                    projected = F.linear(normalized[:, :b], layer._dneo_qkv)
                    q = projected[..., :layer.q_proj.out_features]
                    kv = projected[:, :1, layer.q_proj.out_features:]
                else:
                    q = F.linear(normalized[:, :b], layer.q_proj.weight)
                # MoFA vectors use the prefix projection, without the draft
                # input norm, convolution, source tag, or evolving MLP state.
                if reuse_memory_kv:
                    # Avoid broadcasting the projection matrix across strided
                    # one-token requests instead of issuing one flattened GEMM.
                    kv_input = normalized[:, :1].contiguous()
                else:
                    memory_input = (memory[index][:, 0, :, 0].masked_fill(~valid[:, 0, :, None], 0)
                                    if self.kv_only else normalized[:, b:])
                    kv_input = torch.cat((normalized[:, :1], memory_input), 1)
                if not packed_anchor:
                    kv = F.linear(kv_input, layer._dneo_qkv[layer.q_proj.out_features:])
                k, v = kv.split((layer.k_proj.out_features, layer.v_proj.out_features), -1)
            else:
                packed = F.linear(normalized, layer._dneo_qkv)
                q, k, v = packed.split((layer.q_proj.out_features, layer.k_proj.out_features, layer.v_proj.out_features), -1)
            nq, nk = q.shape[1], k.shape[1]
            if cached_rope:
                from sglang.srt.speculative.dneo.runtime.optimized_backbone_ops import cached_anchor_qkv
                q, k, v = cached_anchor_qkv(q, k, v, layer.q_norm.weight, layer.k_norm.weight,
                                          self.rotary_tables, anchors,
                                          layer.q_norm.variance_epsilon, layer.k_norm.variance_epsilon)
            elif self.variant == "structural-attention" or self.reference_arithmetic:
                q = layer.q_norm(q.reshape(batch, nq, layer.num_attention_heads, layer.head_dim)).transpose(1, 2)
                k = layer.k_norm(k.reshape(batch, nk, layer.num_key_value_heads, layer.head_dim)).transpose(1, 2)
                v = v.reshape(batch, nk, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)
                kc, ks = (torch.cat((x[:, :1], x[:, b:]), 1) for x in (cos, sin)) if compact else (cos, sin)
                q, k = apply_rope(q, cos[:, :nq], sin[:, :nq]), apply_rope(k, kc, ks)
            else:
                q, k, v = qkv_norm_rope(q, k, v, layer.q_norm.weight, layer.k_norm.weight,
                                      cos, sin, b, compact,
                                      layer.q_norm.variance_epsilon, layer.k_norm.variance_epsilon)
            if reuse_memory_kv and not cached_merge:
                from sglang.srt.speculative.dneo.runtime.kv_only_attention import gather_local_kv
                k, v = gather_local_kv(k, v, prefix_kv[index], positions[:, 0], valid[:, 0], request_slots)
            if events:
                events[1].record()
            if self.variant == 'fused-sdpa-native' and self.attention_backend == 'sdpa-checked':
                from sglang.srt.speculative.dneo.runtime.reference_matched_attention import reference_matched_attention
                attended = reference_matched_attention(q, prefix_kv[index], k, v,
                    anchors.reshape(batch), valid[:, 0], request_slots, self.attention_capacity)
            elif self.variant == 'fused-sdpa-native':
                if cached_merge:
                    from sglang.srt.speculative.dneo.runtime.optimized_backbone_ops import cached_prefix_attention
                    attended = cached_prefix_attention(q, prefix_kv[index], k, v, fa3_lengths,
                                                       positions[:, 0], valid[:, 0], fa3_slots)
                elif use_fa3:
                    from sglang.srt.speculative.dneo.runtime.kv_only_attention import fa3_prefix_local_attention
                    attended = fa3_prefix_local_attention(q, prefix_kv[index], k, v, fa3_lengths,
                                                         valid[:, 0], fa3_slots)
                else:
                    attended = structural_attention(q, prefix_kv[index], k, v, anchors.reshape(batch),
                                                    valid[:, 0], b, compact, self.splits, self.accurate_attention,
                                                    request_slots=request_slots, dynamic=True,
                                                    pv_precision=('tf32' if self.kv_only and self.attention_backend == 'triton-tf32'
                                                                  and q.dtype != torch.float32 else 'tf32x3'))
            elif self.variant.startswith('fused-sdpa'):
                pk, pv = prefix_kv[index]
                mask = compact_mask if compact else full_mask[:, :, :nq]
                if self.variant == 'fused-sdpa-cachedkv':
                    if attention_kv is None:
                        raise ValueError('cached expanded KV requires managed inference caches')
                    key, value = attention_kv[index]
                    write_expanded_kv((k, v), (key, value), pk.shape[2])
                elif self.variant in ('fused-sdpa-pack', 'fused-sdpa-pack-pruned'):
                    key, value = pack_attention_kv((pk, pv), k, v, q.shape[1], request_slots)
                else:
                    key, value = torch.cat((pk, k), 2), torch.cat((pv, v), 2)
                attended = F.scaled_dot_product_attention(q, key, value, attn_mask=mask, enable_gqa=True) if self.variant == 'fused-sdpa-gqa' else layer._attention_once(q, key, value, mask)
            else:
                attended = structural_attention(q, prefix_kv[index], k, v, anchors.reshape(batch),
                                                valid[:, 0], b, compact, self.splits, self.variant == 'structural-precise',
                                                request_slots=request_slots)
            attended = attended.transpose(1, 2).reshape(batch, nq, -1)
            if events:
                events[2].record()
            if not self.reference_arithmetic and (
                    self.variant in ('structural-fused', 'structural-precise', 'structural-safe')
                    or self.variant.startswith('fused-sdpa')):
                residual, normalized = conv_residual_norm(
                    layer.o_proj(attended), conv_kernel, layer.attention_conv.base_kernel[1],
                    hidden[:, :nq], b, layer.attention_conv.group_size, layer.post_attention_layernorm)
                if fused_layout:
                    normalized, mlp_kernel = prepare_full(layer.mlp_conv, normalized, b, tile=conv_tile)
                else:
                    convolved, mlp_kernel = layer.mlp_conv.prepare(normalized[:, :b], (b,))
                    normalized = torch.cat((convolved, normalized[:, b:]), 1) if nq > b else convolved
                next_norm = (draft.layers[index + 1].input_layernorm
                             if index + 1 < len(draft.layers) else draft.norm) if fuse_next_norm else None
                hidden, next_normalized = conv_residual_norm(layer.mlp(normalized), mlp_kernel,
                    layer.mlp_conv.base_kernel[1], residual, b, layer.mlp_conv.group_size, next_norm)
                if not fuse_next_norm:
                    next_normalized = None
            else:
                hidden = layer._finish_attention_and_mlp(
                    hidden[:, :nq], attended, conv_kernel,
                    (b,) if compact else (b, width), not compact)
            local = hidden[:, :b]
            if self.audit_layer_outputs is not None:
                self.audit_layer_outputs.append(local.clone())
            if events:
                events[3].record()
            if not compact:
                memory_state = hidden[:, b:]
        return (next_normalized if fuse_next_norm else draft.norm(local))[:, None]
