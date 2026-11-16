"""Exact-arithmetic Hopper draft operators, qualified independently of serving."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def kernel_mlp(module, hidden):
    from dneo.runtime.kernels import _silu_mul

    packed = F.linear(hidden, module._dneo_gate_up)
    width = packed.shape[-1] // 2
    rows = packed.numel() // (2 * width)
    tile = 1024 if rows >= 128 and width == 9728 else 256
    output = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=packed.dtype)
    _silu_mul[(rows, triton.cdiv(width, tile))](
        packed, output, width, 2 * width, tile,
        num_warps=4, enable_fp_fusion=False)
    return module.down_proj(output)


@triton.jit
def _query_norm_rope(Q, W, COS, SIN, ANCHORS, OUT,
                     Q0: tl.constexpr, Q1: tl.constexpr, A0: tl.constexpr,
                     HQ: tl.constexpr, NQ: tl.constexpr, D: tl.constexpr,
                     EPS: tl.constexpr, GROUP: tl.constexpr):
    token = tl.program_id(0)
    sample, row = token // NQ, token % NQ
    heads = tl.program_id(1) * GROUP + tl.arange(0, GROUP)
    cols = tl.arange(0, D)
    x = tl.load(Q + sample * Q0 + row * Q1 + heads[:, None] * D + cols[None],
                heads[:, None] < HQ, 0).to(tl.float32)
    rotated_cols = (cols + D // 2) % D
    xr = tl.load(Q + sample * Q0 + row * Q1 + heads[:, None] * D + rotated_cols[None],
                 heads[:, None] < HQ, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    normalized = (x * inv[:, None]).to(Q.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + cols)
    normalized = (normalized * weight[None]).to(Q.dtype.element_ty).to(tl.float32)
    rotated = (xr * inv[:, None]).to(Q.dtype.element_ty).to(tl.float32)
    rotated = (rotated * tl.load(W + rotated_cols)[None]).to(Q.dtype.element_ty).to(tl.float32)
    position = tl.load(ANCHORS + sample * A0) + row
    cos, sin = tl.load(COS + position * D + cols), tl.load(SIN + position * D + cols)
    result = (normalized * cos[None]).to(Q.dtype.element_ty).to(tl.float32) + (
        tl.where(cols[None] < D // 2, -rotated, rotated) * sin[None]
    ).to(Q.dtype.element_ty).to(tl.float32)
    tl.store(OUT + (token * HQ + heads[:, None]) * D + cols[None], result, heads[:, None] < HQ)


@triton.jit
def _anchor_norm_rope(K, V, W, COS, SIN, ANCHORS, OK, OV,
                      K0: tl.constexpr, V0: tl.constexpr, A0: tl.constexpr,
                      HK: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    sample, head = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, D)
    x = tl.load(K + sample * K0 + head * D + cols).to(tl.float32)
    rotated_cols = (cols + D // 2) % D
    xr = tl.load(K + sample * K0 + head * D + rotated_cols).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    normalized = (x * inv).to(K.dtype.element_ty).to(tl.float32)
    normalized = (normalized * tl.load(W + cols)).to(K.dtype.element_ty).to(tl.float32)
    rotated = (xr * inv).to(K.dtype.element_ty).to(tl.float32)
    rotated = (rotated * tl.load(W + rotated_cols)).to(K.dtype.element_ty).to(tl.float32)
    position = tl.load(ANCHORS + sample * A0)
    cos, sin = tl.load(COS + position * D + cols), tl.load(SIN + position * D + cols)
    key = (normalized * cos).to(K.dtype.element_ty).to(tl.float32) + (
        tl.where(cols < D // 2, -rotated, rotated) * sin
    ).to(K.dtype.element_ty).to(tl.float32)
    destination = (sample * HK + head) * D + cols
    tl.store(OK + destination, key)
    tl.store(OV + destination, tl.load(V + sample * V0 + head * D + cols))


def cached_anchor_qkv(q, k, v, q_weight, k_weight, tables, anchors,
                      eps_q=1e-6, eps_k=1e-6):
    batch, nq, _ = q.shape
    d = q_weight.numel()
    hq, hk = q.shape[-1] // d, k.shape[-1] // d
    cos, sin = tables
    if (k.shape[1] != 1 or v.shape != k.shape or d & (d - 1)
            or any(x.stride(-1) != 1 for x in (q, k, v))
            or not cos.is_contiguous() or not sin.is_contiguous()
            or cos.shape != sin.shape or cos.shape[-1] != d
            or anchors.numel() != batch):
        raise ValueError('cached QKV requires static RoPE tables and anchor-only K/V')
    oq = torch.empty((batch, nq, hq, d), device=q.device, dtype=q.dtype)
    ok = torch.empty((batch, hk, 1, d), device=k.device, dtype=k.dtype)
    ov = torch.empty_like(ok)
    _query_norm_rope[(batch * nq, triton.cdiv(hq, 4))](
        q, q_weight, cos, sin, anchors, oq, *q.stride()[:2], anchors.stride(0),
        hq, nq, d, eps_q, 4, num_warps=16, enable_fp_fusion=False)
    _anchor_norm_rope[(batch, hk)](
        k, v, k_weight, cos, sin, anchors, ok, ov,
        k.stride(0), v.stride(0), anchors.stride(0), hk, d, eps_k,
        num_warps=4, enable_fp_fusion=False)
    return oq.transpose(1, 2), ok, ov


@triton.jit
def _cached_local_merge(Q, AK, AV, PK, PV, POS, VALID, SLOTS, PREFIX, LSE, OUT,
                        Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                        A0: tl.constexpr, A1: tl.constexpr,
                        AV0: tl.constexpr, AV1: tl.constexpr,
                        K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                        V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                        POS0: tl.constexpr, POS1: tl.constexpr,
                        VALID0: tl.constexpr, VALID1: tl.constexpr,
                        P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                        L0: tl.constexpr, L1: tl.constexpr, L2: tl.constexpr,
                        HQ: tl.constexpr, HK: tl.constexpr, NQ: tl.constexpr,
                        NK: tl.constexpr, CAP: tl.constexpr, D: tl.constexpr,
                        M: tl.constexpr, N: tl.constexpr, INDEXED: tl.constexpr):
    bh = tl.program_id(0)
    batch, head = bh // HQ, bh % HQ
    kh = head // (HQ // HK)
    slot = tl.load(SLOTS + batch) if INDEXED else batch
    rows, keys, cols = tl.arange(0, M), tl.arange(0, N), tl.arange(0, D)
    q = tl.load(Q + batch * Q0 + head * Q1 + rows[:, None] * Q2 + cols[None], rows[:, None] < NQ, 0)
    memory = (keys > 0) & (keys < NK)
    valid = tl.load(VALID + batch * VALID0 + (keys - 1) * VALID1, memory, 0)
    pos = tl.load(POS + batch * POS0 + (keys - 1) * POS1, memory, 0)
    keep = memory & valid & (pos >= 0) & (pos < CAP)
    k = tl.load(PK + slot * K0 + kh * K1 + pos[None] * K2 + cols[:, None], keep[None], 0)
    v = tl.load(PV + slot * V0 + kh * V1 + pos[:, None] * V2 + cols[None], keep[:, None], 0)
    ak = tl.load(AK + batch * A0 + kh * A1 + tl.broadcast_to(cols[:, None], (D, N)), keys[None] == 0, 0)
    av = tl.load(AV + batch * AV0 + kh * AV1 + tl.broadcast_to(cols[None], (N, D)), keys[:, None] == 0, 0)
    k, v = tl.where(keys[None] == 0, ak, k), tl.where(keys[:, None] == 0, av, v)
    allowed = (keys < NK) & ((keys == 0) | valid)
    score = tl.dot(q, k, input_precision='ieee') * (D ** -0.5)
    score = tl.where(allowed[None], score, -float('inf'))
    maximum_local = tl.max(score, 1)
    p = tl.exp(score - maximum_local[:, None])
    denom = tl.sum(p, 1)
    local_lse = maximum_local + tl.log(denom)
    prefix_lse = tl.load(LSE + batch * L0 + head * L1 + rows * L2, rows < NQ, -float('inf'))
    maximum = tl.maximum(local_lse, prefix_lse)
    a, b = tl.exp(local_lse - maximum), tl.exp(prefix_lse - maximum)
    prefix = tl.load(PREFIX + batch * P0 + rows[:, None] * P1 + head * P2 + cols[None], rows[:, None] < NQ, 0)
    local = tl.dot(p.to(q.dtype), v, input_precision='ieee') / denom[:, None]
    result = (local * a[:, None] + prefix * b[:, None]) / (a + b)[:, None]
    tl.store(OUT + ((batch * NQ + rows[:, None]) * HQ + head) * D + cols[None], result, rows[:, None] < NQ)


def cached_prefix_attention(q, prefix_kv, anchor_k, anchor_v, lengths,
                            positions, valid, request_slots=None):
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

    batch, heads, nq, d = q.shape
    if anchor_k.shape[2] != 1 or positions.shape != valid.shape:
        raise ValueError('cached attention requires anchor-only K/V and matching memory metadata')
    pk, pv = prefix_kv
    result = flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=pk.transpose(1, 2), v_cache=pv.transpose(1, 2),
        cache_seqlens=lengths, cache_batch_idx=request_slots,
        causal=False, return_softmax_lse=True, ver=3)
    prefix, lse = result[:2]
    nk = valid.shape[-1] + 1
    out = torch.empty((batch, nq, heads, d), device=q.device, dtype=q.dtype)
    _cached_local_merge[(batch * heads,)](
        q, anchor_k, anchor_v, pk, pv, positions, valid, request_slots, prefix, lse, out,
        *q.stride()[:3], *anchor_k.stride()[:2], *anchor_v.stride()[:2],
        *pk.stride()[:3], *pv.stride()[:3], *positions.stride(), *valid.stride(),
        *prefix.stride()[:3], *lse.stride(), heads, anchor_k.shape[1], nq, nk, pk.shape[2], d,
        max(16, triton.next_power_of_2(nq)), max(16, triton.next_power_of_2(nk)), request_slots is not None,
        num_warps=4)
    return out.transpose(1, 2)

