"""Fast compact reduction with original-tree fallback at BF16 rounding ties."""
import torch
import triton
import triton.language as tl


@triton.jit
def _route_sparse(X, W, OUT, RISK, TOKENS: tl.constexpr, SEQUENCE: tl.constexpr,
                  LAYERS: tl.constexpr, DRAFT_LAYERS: tl.constexpr, DIM: tl.constexpr,
                  X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
                  N: tl.constexpr, M: tl.constexpr, TILE: tl.constexpr,
                  REUSE: tl.constexpr):
    token, tile = tl.program_id(0), tl.program_id(1)
    if X0 >= 2147483648 or TOKENS * X1 >= 2147483648:
        token = token.to(tl.int64)
    feature = tl.arange(0, N)
    col = tile * TILE + tl.arange(0, TILE)
    values = tl.load(X + (token // SEQUENCE) * X0 + (token % SEQUENCE) * X1
                     + feature[:, None] * X2 + col[None, :],
                     (feature[:, None] < LAYERS) & (col[None, :] < DIM), 0).to(tl.float32)
    if REUSE:
        draft = tl.arange(0, M)
        weights = tl.load(W + draft[:, None] * LAYERS + feature[None, :],
                         (draft[:, None] < DRAFT_LAYERS) & (feature[None, :] < LAYERS),
                         0).to(tl.float32)
        result = tl.sum(weights[:, :, None] * values[None, :, :], 1)
        absolute_sum = tl.sum(tl.abs(weights[:, :, None] * values[None, :, :]), 1)
        magnitude = tl.abs(result)
        bits = magnitude.to(tl.int32, bitcast=True)
        upper = ((bits & -65536) + 32768).to(tl.float32, bitcast=True)
        lower = ((bits & -65536) - 32768).to(tl.float32, bitcast=True)
        distance = tl.minimum(tl.abs(magnitude - upper), tl.abs(magnitude - lower))
        # Two FP32 addition trees, at most LAYERS-1 nonzero additions each.
        bound = absolute_sum * (2 * (LAYERS + 1) * 5.960464477539063e-8 * 1.001)
        risky = ((distance <= bound) | (magnitude < 7.52316384526264e-37)
                 | (magnitude != magnitude)) & (draft[:, None] < DRAFT_LAYERS) & (col[None, :] < DIM)
        any_risk = tl.sum(tl.sum(risky.to(tl.int32), 1), 0) > 0
        tl.store(RISK + token * tl.cdiv(DIM, TILE) + tile, any_risk)
        tl.store(OUT + (draft[:, None] * TOKENS + token) * DIM + col[None, :], result,
                 (draft[:, None] < DRAFT_LAYERS) & (col[None, :] < DIM))
    else:
        draft = tl.program_id(2)
        weights = tl.load(W + draft * LAYERS + feature, feature < LAYERS, 0).to(tl.float32)
        result = tl.sum(weights[:, None] * values, 0)
        tl.store(OUT + (draft * TOKENS + token) * DIM + col, result, col < DIM)


@triton.jit
def _route_fallback(X, W, OUT, RISK, TOKENS: tl.constexpr, LAYERS: tl.constexpr,
                    DIM: tl.constexpr, X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
                    SEQUENCE: tl.constexpr, DRAFT_LAYERS: tl.constexpr,
                    IDS: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr, REUSE: tl.constexpr):
    token = tl.program_id(0)
    tile = tl.program_id(1) if REUSE else tl.program_id(2)
    if tl.load(RISK + token * tl.cdiv(DIM, TILE) + tile):
        if X0 >= 2147483648 or TOKENS * X1 >= 2147483648:
            token = token.to(tl.int64)
        layer, col = tl.arange(0, N), tile * TILE + tl.arange(0, TILE)
        slot = tl.full((N,), -1, tl.int32)
        for i in tl.static_range(len(IDS)):
            slot = tl.where(layer == IDS[i], i, slot)
        valid = slot >= 0
        value = tl.load(X + (token // SEQUENCE) * X0 + (token % SEQUENCE) * X1
                        + slot[:, None] * X2 + col[None, :],
                        valid[:, None] & (col[None, :] < DIM), 0).to(tl.float32)
        if REUSE:
            for draft_layer in tl.static_range(DRAFT_LAYERS):
                probabilities = tl.load(W + draft_layer * LAYERS + slot, valid, 0).to(tl.float32)
                result = tl.sum(probabilities[:, None] * value, 0)
                tl.store(OUT + (draft_layer * TOKENS + token) * DIM + col, result, col < DIM)
        else:
            draft_layer = tl.program_id(1)
            probabilities = tl.load(W + draft_layer * LAYERS + slot, valid, 0).to(tl.float32)
            result = tl.sum(probabilities[:, None] * value, 0)
            tl.store(OUT + (draft_layer * TOKENS + token) * DIM + col, result, col < DIM)


def compact_mofa_sparse(features, weights, layer_ids, full_layer_count, *, reuse=True):
    batch, sequence, layers, dim = features.shape
    if (features.stride(-1) != 1 or weights.shape[1] != layers
            or not weights.is_contiguous() or layers > 16
            or features.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16):
        raise ValueError('unsupported sparse compact MoFA layout')
    count = weights.shape[0]
    out = torch.empty(count, batch, sequence, 1, dim,
                      device=features.device, dtype=features.dtype)
    risk = torch.empty(batch * sequence, triton.cdiv(dim, 128),
                       device=features.device, dtype=torch.bool)
    # Compute all five mixtures together, then repair only ambiguous tiles.
    _route_sparse[(batch * sequence, triton.cdiv(dim, 128))](features, weights, out, risk,
        batch * sequence, sequence, layers, count, dim, *features.stride()[:3],
        triton.next_power_of_2(layers), triton.next_power_of_2(count), 128, True,
        num_warps=4, enable_fp_fusion=False)
    grid = ((batch * sequence, triton.cdiv(dim, 128)) if reuse else
            (batch * sequence, count, triton.cdiv(dim, 128)))
    _route_fallback[grid](features, weights, out, risk, batch * sequence, layers, dim,
        *features.stride()[:3], sequence, count, tuple(layer_ids),
        triton.next_power_of_2(full_layer_count), 128, reuse,
        num_warps=4, enable_fp_fusion=False)
    return tuple(out.unbind(0))
