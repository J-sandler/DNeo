"""Explicit SDPA correctness path; not the production latency profile."""
import torch
import torch.nn.functional as F

from .structural_kernels import pack_attention_kv


def reference_matched_attention(q, prefix_kv, k, v, lengths, valid,
                                request_slots=None, capacity=None):
    pk, pv = prefix_kv
    capacity = pk.shape[2] if capacity is None else int(capacity)
    if not 0 < capacity <= pk.shape[2]:
        raise ValueError("reference attention capacity exceeds allocated cache")
    if valid.shape != (q.shape[0], k.shape[2] - 1):
        raise ValueError("reference attention requires anchor plus masked memory")
    key, value = pack_attention_kv((pk[:, :, :capacity], pv[:, :, :capacity]),
                                   k, v, q.shape[1], request_slots)
    prefix = torch.arange(capacity, device=q.device)[None] < lengths.reshape(-1, 1)
    anchor = torch.ones((q.shape[0], 1), device=q.device, dtype=torch.bool)
    mask = torch.cat((prefix, anchor, valid), -1)[:, None, None]
    return F.scaled_dot_product_attention(q, key, value, attn_mask=mask)
