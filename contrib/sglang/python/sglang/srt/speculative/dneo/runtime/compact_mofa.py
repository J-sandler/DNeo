"""Sparse feature reads with the qualified dense kernel's reduction order."""
from functools import lru_cache
import torch
import triton
import triton.language as tl


@triton.jit
def _compact_route(X, W, OUT, MAP, TOKENS: tl.constexpr, LAYERS: tl.constexpr,
                   DIM: tl.constexpr, X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
                   SEQUENCE: tl.constexpr, DRAFT_LAYERS: tl.constexpr,
                   N: tl.constexpr, TILE: tl.constexpr, REUSE: tl.constexpr):
    token = tl.program_id(0)
    tile = tl.program_id(1) if REUSE else tl.program_id(2)
    if X0 >= 2147483648 or TOKENS * X1 >= 2147483648:
        token = token.to(tl.int64)
    layer, col = tl.arange(0, N), tile * TILE + tl.arange(0, TILE)
    slot = tl.load(MAP + layer)
    valid = slot >= 0
    value = tl.load(X + (token // SEQUENCE) * X0 + (token % SEQUENCE) * X1
                    + slot[:, None] * X2 + col[None, :],
                    valid[:, None] & (col[None, :] < DIM), 0).to(tl.float32)
    if REUSE:
        for draft_layer in tl.static_range(DRAFT_LAYERS):
            weights = tl.load(W + draft_layer * LAYERS + slot, valid, 0).to(tl.float32)
            result = tl.sum(weights[:, None] * value, 0)
            tl.store(OUT + (draft_layer * TOKENS + token) * DIM + col, result, col < DIM)
    else:
        draft_layer = tl.program_id(1)
        weights = tl.load(W + draft_layer * LAYERS + slot, valid, 0).to(tl.float32)
        result = tl.sum(weights[:, None] * value, 0)
        tl.store(OUT + (draft_layer * TOKENS + token) * DIM + col, result, col < DIM)


@lru_cache(maxsize=64)
def _layer_map(device, layer_ids, width):
    mapping = [-1] * width
    for slot, layer_id in enumerate(layer_ids):
        mapping[layer_id] = slot
    return torch.tensor(mapping, device=device, dtype=torch.int32)


def compact_mofa(features, weights, layer_ids, full_layer_count, *, reuse=True, output_dtype=None):
    batch, sequence, layers, dim = features.shape
    if features.stride(-1) != 1 or weights.shape[1] != layers or len(layer_ids) != layers:
        raise ValueError('invalid compact MoFA layout')
    out = torch.empty(weights.shape[0], batch, sequence, 1, dim,
                      device=features.device, dtype=output_dtype or features.dtype)
    reduction_width = triton.next_power_of_2(full_layer_count)
    mapping = _layer_map(str(features.device), tuple(layer_ids), reduction_width)
    grid = ((batch * sequence, triton.cdiv(dim, 128)) if reuse else
            (batch * sequence, weights.shape[0], triton.cdiv(dim, 128)))
    _compact_route[grid](features, weights, out, mapping, batch * sequence, layers, dim,
        *features.stride()[:3], sequence, weights.shape[0],
        reduction_width, 128, reuse, enable_fp_fusion=False, num_warps=4)
    return tuple(out.unbind(0))
