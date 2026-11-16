"""Last-block dead-computation elimination for dNEO prefix-only attention."""
import torch
import torch.nn.functional as F


def split_attention(layer, query, key, value, attention_mask, prefix_length=None):
    """Remove masked draft K/V columns from local queries, not memory queries."""
    if prefix_length is None:
        raise ValueError("structural attention requires an explicit prefix length")
    b = layer.block_size
    stop = prefix_length + b
    compact_k = torch.cat((key[:, :, :prefix_length + 1], key[:, :, stop:]), 2)
    compact_v = torch.cat((value[:, :, :prefix_length + 1], value[:, :, stop:]), 2)
    compact_mask = torch.cat((attention_mask[:, :, :b, :prefix_length + 1],
                              attention_mask[:, :, :b, stop:]), -1)
    local = layer._attention_once(query[:, :, :b], compact_k, compact_v, compact_mask)
    memory = layer._attention_once(query[:, :, b:], key, value, attention_mask[:, :, b:])
    return torch.cat((local, memory), 2)


def forward(layer, hidden_states, prefix_context, cos, sin, attention_mask,
            context_cos=None, context_sin=None, **kwargs):
    from dneo.modeling_dneo import apply_rope
    b = layer.block_size
    original = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    local, kernel = layer.attention_conv.prepare(normalized[:, :b], (b,))
    # Only the anchor and rejection memory are keys for local queries. Memory
    # queries/MLPs in this LAST block cannot affect any returned local state.
    kv_input = torch.cat((local[:, :1], normalized[:, b:]), dim=1)
    # Bypass the ordinary packed-QKV wrapper: local K/V are precisely the work
    # being removed here. Only anchor and memory K/V are needed in this block.
    query = layer.q_norm(F.linear(local, layer.q_proj.weight, layer.q_proj.bias).view(
        1, b, layer.num_attention_heads, layer.head_dim)).transpose(1, 2)
    if hasattr(layer, "_dneo_qkv"):
        packed = F.linear(kv_input, layer._dneo_qkv[layer.q_proj.out_features:])
        key, value = packed.split((layer.k_proj.out_features, layer.v_proj.out_features), dim=-1)
    else:
        key, value = layer.k_proj(kv_input), layer.v_proj(kv_input)
    key = layer.k_norm(key.view(1, -1, layer.num_key_value_heads, layer.head_dim)).transpose(1, 2)
    value = value.view(1, -1, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)
    query = apply_rope(query, cos[:, :b], sin[:, :b])
    key = apply_rope(key, torch.cat((cos[:, :1], cos[:, b:]), 1), torch.cat((sin[:, :1], sin[:, b:]), 1))
    pk, pv = layer._project_prefix(prefix_context, context_cos, context_sin)
    prefix_len = pk.shape[2]
    mask = torch.cat((attention_mask[:, :, :b, :prefix_len + 1],
                      attention_mask[:, :, :b, prefix_len + b:]), dim=-1)
    attended = layer._attention(query, torch.cat((pk, key), 2), torch.cat((pv, value), 2), mask,
                                prefix_length=prefix_len).transpose(1, 2).reshape(1, b, -1)
    local = layer._finish_attention_and_mlp(original[:, :b], attended, kernel, (b,))
    return torch.cat((local, original[:, b:]), dim=1)
