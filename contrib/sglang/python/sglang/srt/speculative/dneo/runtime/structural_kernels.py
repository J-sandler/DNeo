"""Inference-only dNEO kernels with explicit prefix/local/memory connectivity."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_attention_kv(PK, PV, K, V, OK, OV, SLOTS,
                       P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                       PV0: tl.constexpr, PV1: tl.constexpr, PV2: tl.constexpr,
                       K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                       V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                       HQ: tl.constexpr, HK: tl.constexpr, CAP: tl.constexpr,
                       N: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr,
                       INDEXED: tl.constexpr = False):
    bh, tile = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HQ, (bh % HQ) // (HQ // HK)
    slot = tl.load(SLOTS + batch) if INDEXED else batch
    offset = tile * TILE + tl.arange(0, TILE)
    row, col = offset // D, offset % D
    p = row < CAP
    local = (row >= CAP) & (row < CAP + N)
    k = tl.load(PK + slot * P0 + head * P1 + row * P2 + col, p, 0)
    v = tl.load(PV + slot * PV0 + head * PV1 + row * PV2 + col, p, 0)
    lk = tl.load(K + batch * K0 + head * K1 + (row - CAP) * K2 + col, local, 0)
    lv = tl.load(V + batch * V0 + head * V1 + (row - CAP) * V2 + col, local, 0)
    tl.store(OK + bh * (CAP + N) * D + offset, tl.where(p, k, lk), row < CAP + N)
    tl.store(OV + bh * (CAP + N) * D + offset, tl.where(p, v, lv), row < CAP + N)


def pack_attention_kv(prefix_kv, k, v, query_heads, request_slots=None):
    """Bit-preserving concatenation + GQA expansion in one memory pass."""
    pk, pv = prefix_kv
    batch, heads, n, d = k.shape
    if query_heads % heads or any(x.stride(-1) != 1 for x in (pk, pv, k, v)):
        raise ValueError("invalid grouped attention layout")
    cap = pk.shape[2]
    out_k = torch.empty((batch, query_heads, cap + n, d), device=k.device, dtype=k.dtype)
    out_v = torch.empty_like(out_k)
    _pack_attention_kv[(batch * query_heads, triton.cdiv((cap + n) * d, 512))](
        pk, pv, k, v, out_k, out_v, request_slots,
        *pk.stride()[:3], *pv.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        query_heads, heads, cap, n, d, 512, request_slots is not None)
    return out_k, out_v


@triton.jit
def _write_expanded_kv(K, V, OK, OV,
                       K0, K1, K2, V0, V1, V2, O0, O1, O2,
                       HQ: tl.constexpr, HK: tl.constexpr, N,
                       D: tl.constexpr, START, TILE: tl.constexpr,
                       START_TENSOR: tl.constexpr = False, START_STRIDE: tl.constexpr = 0,
                       SLOTS=None, INDEXED: tl.constexpr = False):
    bh, tile = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HQ, bh % HQ
    slot = tl.load(SLOTS + batch) if INDEXED else batch
    kh = head // (HQ // HK)
    offset = tile * TILE + tl.arange(0, TILE)
    row, col = offset // D, offset % D
    k = tl.load(K + batch * K0 + kh * K1 + row * K2 + col, row < N, 0)
    v = tl.load(V + batch * V0 + kh * V1 + row * V2 + col, row < N, 0)
    start = tl.load(START + batch * START_STRIDE) if START_TENSOR else START
    dst = slot * O0 + head * O1 + (row + start) * O2 + col
    tl.store(OK + dst, k, row < N)
    tl.store(OV + dst, v, row < N)


def write_expanded_kv(kv, output, start=0):
    """Append only changed tokens to an expanded inference cache, without casts."""
    k, v = kv
    ok, ov = output
    batch, hk, n, d = k.shape
    hq = ok.shape[1]
    if hq % hk or start < 0 or start + n > ok.shape[2] or ok.stride() != ov.stride():
        raise ValueError("invalid expanded-cache append")
    _write_expanded_kv[(batch * hq, triton.cdiv(n * d, 512))](
        k, v, ok, ov, *k.stride()[:3], *v.stride()[:3], *ok.stride()[:3],
        hq, hk, n, d, start, 512)


def write_kv_at_positions(kv, output, positions, request_slots=None):
    """Graph-safe append at device-resident, per-request contiguous positions.

    Callers validate capacity before replay, and positions are increasing by one.
    """
    k, v = kv
    ok, ov = output
    batch, heads, n, d = k.shape
    if (k.shape != v.shape or ok.shape != ov.shape or ok.stride() != ov.stride()
            or ok.shape[1] != heads or (request_slots is None and ok.shape[0] != batch)
            or ok.shape[-1] != d
            or positions.shape != (batch, n) or positions.device != k.device):
        raise ValueError("incompatible graph-cache append shapes")
    _write_expanded_kv[(batch * heads, triton.cdiv(n * d, 512))](
        k, v, ok, ov, *k.stride()[:3], *v.stride()[:3], *ok.stride()[:3],
        heads, heads, n, d, positions, 512, True, positions.stride(0),
        request_slots, request_slots is not None)


@triton.jit
def _conv_residual_norm(X, DYNAMIC, BASE, RESIDUAL, WEIGHT, OUT, NORM,
                        X0: tl.constexpr, X1: tl.constexpr, R0: tl.constexpr, R1: tl.constexpr,
                        D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
                        WIDTH: tl.constexpr, GROUP: tl.constexpr, N: tl.constexpr, B: tl.constexpr,
                        NORMALIZE: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    sample, pos = tl.program_id(0) // N, tl.program_id(0) % N
    col = tl.arange(0, BLOCK)
    dtype = X.dtype.element_ty
    x = tl.load(X + sample * X0 + pos * X1 + col, col < WIDTH, 0).to(tl.float32)
    if pos < B:
        prev = tl.load(X + sample * X0 + (pos - 1) * X1 + col, (pos > 0) & (col < WIDTH), 0).to(tl.float32)
        a = tl.load(DYNAMIC + sample * D0 + pos * D1 + col // GROUP, col < WIDTH, 0).to(tl.float32)
        b = tl.load(DYNAMIC + sample * D0 + pos * D1 + D2 + col // GROUP, col < WIDTH, 0).to(tl.float32)
        a = (a + tl.load(BASE + col, col < WIDTH, 0)).to(dtype).to(tl.float32)
        b = (b + tl.load(BASE + WIDTH + col, col < WIDTH, 0)).to(dtype).to(tl.float32)
        x = ((x * a).to(dtype).to(tl.float32) + (prev * b).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    residual = tl.load(RESIDUAL + sample * R0 + pos * R1 + col, col < WIDTH, 0).to(tl.float32)
    x = (x + residual).to(dtype).to(tl.float32)
    tl.store(OUT + (sample * N + pos) * WIDTH + col, x, col < WIDTH)
    if NORMALIZE:
        inv = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
        weight = tl.load(WEIGHT + col, col < WIDTH, 0).to(tl.float32)
        y = (x * inv).to(dtype).to(tl.float32) * weight
        tl.store(NORM + (sample * N + pos) * WIDTH + col, y, col < WIDTH)


def conv_residual_norm(hidden, dynamic, base, residual, block_size, group_size, norm=None):
    """Post-convolution + residual (+ RMSNorm), never convolving memory rows."""
    batch, n, d = hidden.shape
    if hidden.shape != residual.shape or dynamic.shape[:2] != (batch, block_size):
        raise ValueError('inconsistent fused convolution layout')
    output = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    normalized = torch.empty_like(output) if norm is not None else output
    _conv_residual_norm[(batch * n,)](
        hidden, dynamic, base, residual, norm.weight if norm is not None else output, output, normalized,
        *hidden.stride()[:2], *residual.stride()[:2], *dynamic.stride()[:3], d, group_size, n, block_size,
        norm is not None, norm.variance_epsilon if norm is not None else 0., triton.next_power_of_2(d),
        num_warps=4, enable_fp_fusion=False)
    return output, normalized


@triton.jit
def _qkv_norm_rope(Q, K, V, QW, KW, COS, SIN, OQ, OK, OV,
                   QS0: tl.constexpr, QS1: tl.constexpr,
                   KS0: tl.constexpr, KS1: tl.constexpr,
                   VS0: tl.constexpr, VS1: tl.constexpr,
                   CS0: tl.constexpr, CS1: tl.constexpr,
                   HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                   NQ: tl.constexpr, NK: tl.constexpr, B: tl.constexpr,
                   COMPACT: tl.constexpr, EPSQ: tl.constexpr, EPSK: tl.constexpr):
    row = tl.program_id(0)
    packed_head = tl.program_id(1)
    sample, head = packed_head // (HQ + 2 * HK), packed_head % (HQ + 2 * HK)
    col = tl.arange(0, D)
    rotated = (col + D // 2) % D
    if head < HQ:
        x = tl.load(Q + sample * QS0 + row * QS1 + head * D + col, row < NQ, 0).to(tl.float32)
        xr = tl.load(Q + sample * QS0 + row * QS1 + head * D + rotated, row < NQ, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPSQ)
        x = ((x * inv).to(Q.dtype.element_ty).to(tl.float32) * tl.load(QW + col)).to(Q.dtype.element_ty).to(tl.float32)
        xr = ((xr * inv).to(Q.dtype.element_ty).to(tl.float32) * tl.load(QW + rotated)).to(Q.dtype.element_ty).to(tl.float32)
        c = tl.load(COS + sample * CS0 + row * CS1 + col, row < NQ, 0)
        s = tl.load(SIN + sample * CS0 + row * CS1 + col, row < NQ, 0)
        y = (x * c).to(Q.dtype.element_ty).to(tl.float32) + (tl.where(col < D // 2, -xr, xr) * s).to(Q.dtype.element_ty).to(tl.float32)
        tl.store(OQ + ((sample * HQ + head) * NQ + row) * D + col, y, row < NQ)
    elif head < HQ + HK:
        head -= HQ
        x = tl.load(K + sample * KS0 + row * KS1 + head * D + col, row < NK, 0).to(tl.float32)
        xr = tl.load(K + sample * KS0 + row * KS1 + head * D + rotated, row < NK, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPSK)
        x = ((x * inv).to(K.dtype.element_ty).to(tl.float32) * tl.load(KW + col)).to(K.dtype.element_ty).to(tl.float32)
        xr = ((xr * inv).to(K.dtype.element_ty).to(tl.float32) * tl.load(KW + rotated)).to(K.dtype.element_ty).to(tl.float32)
        pos = tl.where(row == 0, 0, row + B - 1) if COMPACT else row
        c = tl.load(COS + sample * CS0 + pos * CS1 + col, row < NK, 0)
        s = tl.load(SIN + sample * CS0 + pos * CS1 + col, row < NK, 0)
        y = (x * c).to(K.dtype.element_ty).to(tl.float32) + (tl.where(col < D // 2, -xr, xr) * s).to(K.dtype.element_ty).to(tl.float32)
        tl.store(OK + ((sample * HK + head) * NK + row) * D + col, y, row < NK)
    else:
        head -= HQ + HK
        x = tl.load(V + sample * VS0 + row * VS1 + head * D + col, row < NK, 0).to(tl.float32)
        tl.store(OV + ((sample * HK + head) * NK + row) * D + col, x, row < NK)


def qkv_norm_rope(q, k, v, q_weight, k_weight, cos, sin, block_size,
                  compact=False, eps_q=1e-6, eps_k=1e-6):
    """Retain the eager BF16 cast boundaries while avoiding intermediate tensors."""
    d = q_weight.numel()
    batch, nq = q.shape[:2]
    nk, hq, hk = k.shape[1], q.shape[-1] // d, k.shape[-1] // d
    if d & (d - 1) or q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("QKV requires contiguous head dimensions with power-of-two width")
    oq = torch.empty((batch, hq, nq, d), device=q.device, dtype=q.dtype)
    ok = torch.empty((batch, hk, nk, d), device=q.device, dtype=q.dtype)
    ov = torch.empty_like(ok)
    _qkv_norm_rope[(max(nq, nk), batch * (hq + 2 * hk))](
        q, k, v, q_weight, k_weight, cos, sin, oq, ok, ov,
        *q.stride()[:2], *k.stride()[:2], *v.stride()[:2], *cos.stride()[:2],
        hq, hk, d, nq, nk, block_size, compact, eps_q, eps_k,
        num_warps=4, enable_fp_fusion=False)
    return oq, ok, ov


@triton.jit
def _attention(Q, PK, PV, K, V, LENGTHS, VALID, OUT, LSE, SLOTS,
               Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
               P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
               PV0: tl.constexpr, PV1: tl.constexpr, PV2: tl.constexpr,
               K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
               V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
               B: tl.constexpr, MEMORY: tl.constexpr, CAP: tl.constexpr,
               HQ: tl.constexpr, HK: tl.constexpr, NQ: tl.constexpr,
               D: tl.constexpr, COMPACT: tl.constexpr, SPLITS: tl.constexpr, ACCURATE: tl.constexpr,
               CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               INDEXED: tl.constexpr = False, DYNAMIC: tl.constexpr = False,
               PV_PRECISION: tl.constexpr = "tf32x3"):
    tile, bh, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sample, head = bh // HQ, bh % HQ
    slot = tl.load(SLOTS + sample) if INDEXED else sample
    kh = head // (HQ // HK)
    local_tiles = tl.cdiv(B, BM)
    is_local = tile < local_tiles
    row = tl.where(is_local, tile * BM, B + (tile - local_tiles) * BM) + tl.arange(0, BM)
    qvalid = (row < NQ) & tl.where(is_local, row < B, True)
    if MEMORY > 0:
        mv = tl.load(VALID + sample * MEMORY + row - B, (row >= B) & (row < B + MEMORY), 0)
        qvalid = qvalid & ((row < B) | mv)
    dim = tl.arange(0, D)
    q = tl.load(Q + sample * Q0 + head * Q1 + row[:, None] * Q2 + dim[None, :], qvalid[:, None], 0)
    length = tl.load(LENGTHS + sample)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    end = tl.minimum((split + 1) * CHUNK, length + B + MEMORY) if DYNAMIC else (split + 1) * CHUNK
    prefix_end = length if DYNAMIC else CAP
    for start in range(split * CHUNK, end, BN):
        idx = start + tl.arange(0, BN)
        # Local queries read only anchor + memory. Memory queries still read
        # every local token. Neither path copies or expands the prefix cache.
        after = idx - prefix_end
        offset = tl.where(is_local & (after > 0), after + B - 1, after)
        local_limit = tl.where(is_local, 1 + MEMORY, B + MEMORY)
        valid_key = (idx < length) | ((idx >= prefix_end) & (after < local_limit))
        if MEMORY > 0:
            valid_memory = tl.load(VALID + sample * MEMORY + offset - B,
                                   (idx >= prefix_end) & (offset >= B) & (offset < B + MEMORY), 0)
            valid_key = valid_key & ((idx < prefix_end) | (offset < B) | valid_memory)
        actual = tl.where(offset == 0, 0, offset - B + 1) if COMPACT else offset
        pk = tl.load(PK + slot * P0 + kh * P1 + idx[None, :] * P2 + dim[:, None],
                     (idx[None, :] < length), 0)
        lk = tl.load(K + sample * K0 + kh * K1 + actual[None, :] * K2 + dim[:, None],
                     ((idx >= prefix_end) & valid_key)[None, :], 0)
        key = pk + lk
        score = tl.dot(q, key, input_precision="ieee") * (D ** -0.5)
        score = tl.where(qvalid[:, None] & valid_key[None, :], score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0., next_max)
        probability = tl.exp(score - safe_max[:, None])
        rescale = tl.exp(maximum - safe_max)
        pv = tl.load(PV + slot * PV0 + kh * PV1 + idx[:, None] * PV2 + dim[None, :],
                     idx[:, None] < length, 0)
        lv = tl.load(V + sample * V0 + kh * V1 + actual[:, None] * V2 + dim[None, :],
                     ((idx >= prefix_end) & valid_key)[:, None], 0)
        if ACCURATE:
            contribution = tl.dot(probability, (pv + lv).to(tl.float32), input_precision=PV_PRECISION)
        else:
            contribution = tl.dot(probability.to(q.dtype), pv + lv, input_precision="ieee")
        acc = acc * rescale[:, None] + contribution
        denom = denom * rescale + tl.sum(probability, 1)
        maximum = next_max
    result = acc / tl.where(denom > 0, denom, 1.)[:, None]
    out_base = ((bh * SPLITS + split) * NQ + row) * D
    tl.store(OUT + out_base[:, None] + dim[None, :], result, (row < NQ)[:, None] & tl.where(is_local, row < B, True)[:, None])
    if SPLITS > 1:
        tl.store(LSE + (bh * SPLITS + split) * NQ + row, maximum + tl.log(denom),
                 (row < NQ) & tl.where(is_local, row < B, True))


@triton.jit
def _merge_attention(PARTS, LSE, OUT, NQ: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    bh, row = tl.program_id(0), tl.program_id(1)
    splits, dim = tl.arange(0, SPLITS), tl.arange(0, D)
    lse = tl.load(LSE + (bh * SPLITS + splits) * NQ + row)
    maximum = tl.max(lse, 0)
    weight = tl.exp(lse - tl.where(maximum == -float("inf"), 0., maximum))
    weight = weight / tl.maximum(tl.sum(weight, 0), 1e-20)
    value = tl.load(PARTS + ((bh * SPLITS + splits[:, None]) * NQ + row) * D + dim[None, :])
    tl.store(OUT + (bh * NQ + row) * D + dim, tl.sum(value * weight[:, None], 0))


def structural_attention(q, prefix_kv, k, v, prefix_lengths, memory_valid, block_size,
                         compact=False, splits=1, accurate=False, request_slots=None,
                         dynamic=False, tile_size=64, num_stages=1, num_warps=4, pv_precision="tf32x3"):
    """Single joint softmax, native GQA, and bidirectional rejection exchange."""
    pk, pv = prefix_kv
    batch, heads, nq, d = q.shape
    memory = memory_valid.shape[-1]
    if splits not in (1, 2, 4, 8) or d not in (32, 64, 128, 256):
        raise ValueError("unsupported structural attention split count/head width")
    if any(x.stride(-1) != 1 for x in (q, pk, pv, k, v)):
        raise ValueError("attention head dimensions must be contiguous")
    if nq not in (block_size, block_size + memory) or k.shape[2] != ((1 if compact else block_size) + memory):
        raise ValueError("invalid local/rejection layout")
    cap = pk.shape[2]
    if tile_size not in (32, 64, 128, 256) or num_stages not in (1, 2, 3, 4) or num_warps not in (4, 8):
        raise ValueError("unsupported structural attention tile/pipeline configuration")
    if pv_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("unsupported probability/value dot precision")
    bm, bn = 16, tile_size
    tiles = triton.cdiv(block_size, bm) + (triton.cdiv(memory, bm) if nq > block_size else 0)
    chunk = triton.cdiv(cap + block_size + memory, splits * bn) * bn
    partial = torch.empty((batch, heads, splits, nq, d), device=q.device,
                          dtype=q.dtype if splits == 1 else torch.float32)
    lse = torch.empty((batch, heads, splits, nq), device=q.device, dtype=torch.float32) if splits > 1 else partial
    _attention[(tiles, batch * heads, splits)](
        q, pk, pv, k, v, prefix_lengths, memory_valid, partial, lse, request_slots,
        *q.stride()[:3], *pk.stride()[:3], *pv.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        block_size, memory, cap, heads, k.shape[1], nq, d, compact, splits, accurate, chunk, bm, bn,
        request_slots is not None, dynamic, pv_precision,
        num_warps=num_warps, num_stages=num_stages)
    if splits == 1:
        return partial[:, :, 0]
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _merge_attention[(batch * heads, nq)](partial, lse, output, nq, d, splits, num_warps=4)
    return output


@triton.jit
def _inject_norm(LOCAL, PREVIOUS, INJECT, TAG, WEIGHT, OUT, NORM,
                 L0: tl.constexpr, L1: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr,
                 I0: tl.constexpr, I1: tl.constexpr, B: tl.constexpr, M: tl.constexpr,
                 D: tl.constexpr, FIRST: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sample, pos = row // (B + M), row % (B + M)
    col = tl.arange(0, BLOCK)
    dtype = LOCAL.dtype.element_ty
    if pos < B:
        x = tl.load(LOCAL + sample * L0 + pos * L1 + col, col < D, 0).to(tl.float32)
    else:
        x = tl.load(INJECT + sample * I0 + (pos - B) * I1 + col, col < D, 0).to(tl.float32)
        if not FIRST:
            previous = tl.load(PREVIOUS + sample * P0 + (pos - B) * P1 + col, col < D, 0).to(tl.float32)
            x = (previous + x).to(dtype).to(tl.float32)
        x = (x + tl.load(TAG + col, col < D, 0).to(tl.float32)).to(dtype).to(tl.float32)
    tl.store(OUT + row * D + col, x, col < D)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = (x * inv).to(dtype).to(tl.float32) * tl.load(WEIGHT + col, col < D, 0).to(tl.float32)
    tl.store(NORM + row * D + col, y, col < D)


def inject_norm(local, previous, injected, tag, norm):
    batch, b, d = local.shape
    memory = injected.shape[1]
    output = torch.empty((batch, b + memory, d), device=local.device, dtype=local.dtype)
    normalized = torch.empty_like(output)
    prev = local if previous is None else previous
    _inject_norm[(batch * (b + memory),)](
        local, prev, injected, tag, norm.weight, output, normalized,
        *local.stride()[:2], *prev.stride()[:2], *injected.stride()[:2],
        b, memory, d, previous is None, norm.variance_epsilon, triton.next_power_of_2(d),
        num_warps=4, enable_fp_fusion=False)
    return output, normalized


@triton.jit
def _conv_passthrough(X, DELTA, BASE, OUT, X0: tl.constexpr, X1: tl.constexpr,
                      D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
                      N: tl.constexpr, B: tl.constexpr, D: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    sample, pos = tl.program_id(0) // N, tl.program_id(0) % N
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dtype = X.dtype.element_ty
    x = tl.load(X + sample * X0 + pos * X1 + col, col < D, 0).to(tl.float32)
    if pos < B:
        prev = tl.load(X + sample * X0 + (pos - 1) * X1 + col, (col < D) & (pos > 0), 0).to(tl.float32)
        a = tl.load(DELTA + sample * D0 + pos * D1 + col // G, col < D, 0).to(tl.float32)
        b = tl.load(DELTA + sample * D0 + pos * D1 + D2 + col // G, col < D, 0).to(tl.float32)
        a = (a + tl.load(BASE + col, col < D, 0).to(tl.float32)).to(dtype).to(tl.float32)
        b = (b + tl.load(BASE + D + col, col < D, 0).to(tl.float32)).to(dtype).to(tl.float32)
        x = (a * x).to(dtype).to(tl.float32) + (b * prev).to(dtype).to(tl.float32)
    tl.store(OUT + (sample * N + pos) * D + col, x, col < D)


def prepare_full(conv, hidden, block_size, tile=256):
    """Convolve the block and pass memory through without an extra concatenate."""
    batch, n, d = hidden.shape
    dynamic = conv.kernel_projection(hidden[:, :block_size]).view(batch, block_size, 2, 2, d // conv.group_size)
    pre = dynamic[..., 0, :, :]
    output = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    _conv_passthrough[(batch * n, triton.cdiv(d, tile))](
        hidden, pre, conv.base_kernel[0], output, *hidden.stride()[:2], *pre.stride()[:3],
        n, block_size, d, conv.group_size, tile, enable_fp_fusion=False)
    return output, dynamic[..., 1, :, :]
