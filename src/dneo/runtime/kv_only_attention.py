"""Cached prefix FA3 plus a small local attention/LSE merge, without KV copies."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_local(AK, AV, PK, PV, POS, VALID, SLOTS, K, V,
                  A0: tl.constexpr, A1: tl.constexpr,
                  AV0: tl.constexpr, AV1: tl.constexpr,
                  P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                  PV0: tl.constexpr, PV1: tl.constexpr, PV2: tl.constexpr,
                  POS0: tl.constexpr, POS1: tl.constexpr, VALID0: tl.constexpr, VALID1: tl.constexpr,
                  H: tl.constexpr, D: tl.constexpr, NK: tl.constexpr,
                  CAP: tl.constexpr, INDEXED: tl.constexpr, TILE: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    x = tl.arange(0, TILE)
    row, dim = x // D, x % D
    slot = tl.load(SLOTS + batch) if INDEXED else batch
    pos = tl.load(POS + batch * POS0 + (row - 1) * POS1, (row > 0) & (row < NK), 0)
    valid = tl.load(VALID + batch * VALID0 + (row - 1) * VALID1, (row > 0) & (row < NK), 0)
    keep = (row > 0) & (row < NK) & valid & (pos >= 0) & (pos < CAP)
    key = tl.load(PK + slot * P0 + head * P1 + pos * P2 + dim, keep, 0)
    value = tl.load(PV + slot * PV0 + head * PV1 + pos * PV2 + dim, keep, 0)
    ak = tl.load(AK + batch * A0 + head * A1 + dim, row == 0, 0)
    av = tl.load(AV + batch * AV0 + head * AV1 + dim, row == 0, 0)
    dest = ((batch * H + head) * NK + row) * D + dim
    tl.store(K + dest, tl.where(row == 0, ak, key), row < NK)
    tl.store(V + dest, tl.where(row == 0, av, value), row < NK)


def gather_local_kv(anchor_k, anchor_v, prefix_kv, positions, valid, request_slots=None):
    """Reuse verifier-commit K/V at original memory positions; never read invalid slots."""
    batch, heads, _, d = anchor_k.shape
    nk = valid.shape[-1] + 1
    pk, pv = prefix_kv
    k, v = [torch.empty((batch, heads, nk, d), device=anchor_k.device, dtype=anchor_k.dtype) for _ in range(2)]
    _gather_local[(batch, heads)](
        anchor_k, anchor_v, pk, pv, positions, valid, request_slots, k, v,
        *anchor_k.stride()[:2], *anchor_v.stride()[:2], *pk.stride()[:3], *pv.stride()[:3],
        *positions.stride(), *valid.stride(),
        heads, d, nk, pk.shape[2], request_slots is not None, triton.next_power_of_2(nk*d),
        num_warps=4,
    )
    return k, v


@triton.jit
def _local_merge(Q, K, V, VALID, PREFIX, LSE, OUT,
                 Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                 K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                 V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                 P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                 L0: tl.constexpr, L1: tl.constexpr, L2: tl.constexpr,
                 VALID0: tl.constexpr, VALID1: tl.constexpr,
                 HQ: tl.constexpr, HK: tl.constexpr, NQ: tl.constexpr,
                 NK: tl.constexpr, D: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    bh = tl.program_id(0)
    batch, head = bh // HQ, bh % HQ
    kh = head // (HQ // HK)
    rows, keys, cols = tl.arange(0, M), tl.arange(0, N), tl.arange(0, D)
    q = tl.load(Q + batch * Q0 + head * Q1 + rows[:, None] * Q2 + cols[None, :], rows[:, None] < NQ, 0)
    valid = tl.load(VALID + batch * VALID0 + (keys - 1) * VALID1, (keys > 0) & (keys < NK), 0)
    valid = (keys < NK) & ((keys == 0) | valid)
    k = tl.load(K + batch * K0 + kh * K1 + keys[None, :] * K2 + cols[:, None], valid[None, :], 0)
    v = tl.load(V + batch * V0 + kh * V1 + keys[:, None] * V2 + cols[None, :], valid[:, None], 0)
    score = tl.dot(q, k, input_precision="ieee") * (D ** -0.5)
    score = tl.where(valid[None, :], score, -float("inf"))
    local_max = tl.max(score, 1)
    p = tl.exp(score - local_max[:, None])
    denom = tl.sum(p, 1)
    local_lse = local_max + tl.log(denom)
    prefix_lse = tl.load(LSE + batch * L0 + head * L1 + rows * L2, rows < NQ, -float("inf"))
    maximum = tl.maximum(local_lse, prefix_lse)
    a, b = tl.exp(local_lse - maximum), tl.exp(prefix_lse - maximum)
    prefix = tl.load(PREFIX + batch * P0 + rows[:, None] * P1 + head * P2 + cols[None, :], rows[:, None] < NQ, 0)
    local = tl.dot(p.to(q.dtype), v, input_precision="ieee") / denom[:, None]
    result = (local * a[:, None] + prefix * b[:, None]) / (a + b)[:, None]
    tl.store(OUT + ((batch * NQ + rows[:, None]) * HQ + head) * D + cols[None, :], result, rows[:, None] < NQ)


def fa3_prefix_local_attention(q, prefix_kv, k, v, lengths, valid, request_slots=None):
    """One joint softmax over prefix + anchor + valid memory; no cache mutation."""
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FA3 requires BF16/FP16 inputs")
    batch, heads, nq, d = q.shape
    if valid.shape != (batch, k.shape[2] - 1):
        raise ValueError("expected compact anchor + memory K/V")
    # Transposes are strided views: no O(context) copying or GQA expansion.
    result = flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=prefix_kv[0].transpose(1, 2),
        v_cache=prefix_kv[1].transpose(1, 2), cache_seqlens=lengths.to(torch.int32),
        cache_batch_idx=None if request_slots is None else request_slots.to(torch.int32),
        causal=False, return_softmax_lse=True, ver=3,
    )
    prefix, lse = result[:2]
    out = torch.empty((batch, nq, heads, d), device=q.device, dtype=q.dtype)
    _local_merge[(batch * heads,)](
        q, k, v, valid, prefix, lse, out,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        *prefix.stride()[:3], *lse.stride(), *valid.stride(), heads, k.shape[1], nq, k.shape[2], d,
        max(16, triton.next_power_of_2(nq)), max(16, triton.next_power_of_2(k.shape[2])),
        num_warps=4,
    )
    return out.transpose(1, 2)
