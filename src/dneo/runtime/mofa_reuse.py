"""Route all draft layers while reading each target feature tile only once."""
import torch
import triton
import triton.language as tl


@triton.jit
def _route(X, W, OUT, TOKENS: tl.constexpr, LAYERS: tl.constexpr,
           DIM: tl.constexpr, X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
           SEQUENCE: tl.constexpr, DRAFT_LAYERS: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    token, tile = tl.program_id(0), tl.program_id(1)
    if X0 >= 2147483648 or TOKENS * X1 >= 2147483648:
        token = token.to(tl.int64)
    layer, col = tl.arange(0, N), tile * TILE + tl.arange(0, TILE)
    value = tl.load(X + (token // SEQUENCE) * X0 + (token % SEQUENCE) * X1
                    + layer[:, None] * X2 + col[None, :],
                    (layer[:, None] < LAYERS) & (col[None, :] < DIM), 0).to(tl.float32)
    for draft_layer in tl.static_range(DRAFT_LAYERS):
        weights = tl.load(W + draft_layer * LAYERS + layer, layer < LAYERS, 0).to(tl.float32)
        result = tl.sum(weights[:, None] * value, 0)
        tl.store(OUT + (draft_layer * TOKENS + token) * DIM + col, result, col < DIM)


def shared_mofa_reuse(features, weights):
    batch, sequence, layers, dim = features.shape
    if features.stride(-1) != 1 or weights.ndim != 2 or weights.shape[1] != layers or not weights.is_contiguous():
        raise ValueError("invalid shared MoFA layout")
    out = torch.empty(weights.shape[0], batch, sequence, 1, dim, device=features.device, dtype=features.dtype)
    _route[(batch * sequence, triton.cdiv(dim,128))](
        features, weights, out, batch*sequence, layers, dim, *features.stride()[:3], sequence,
        weights.shape[0],triton.next_power_of_2(layers),128,enable_fp_fusion=False,num_warps=4,
    )
    return tuple(out.unbind(0))
