"""Compact MoFA kernels that prune zero leaves without reassociating additions."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=64)
def pruned_reduction_plan(layer_ids, width, order):
    if order == 'register8':
        active = {layer: slot for slot, layer in enumerate(layer_ids)}
        stages = []
        # Reference layout: eight rows per CTA, sequential register reduction.
        for start in range(8, width, 8):
            parents = sorted({index - start if start <= index < start + 8
                              else index for index in active})
            pairs, following = [], {}
            for parent in parents:
                left = active.get(parent)
                right = active.get(parent + start) if parent < 8 else None
                if left is None:
                    left, right = right, None
                pairs.append((left, -1 if right is None else right))
                following[parent] = len(pairs) - 1
            stages.append(tuple(pairs))
            active = following
        for stride in (1, 4, 2):
            parents = sorted({index & ~stride for index in active})
            pairs, following = [], {}
            for parent in parents:
                left, right = active.get(parent), active.get(parent | stride)
                if left is None:
                    left, right = right, None
                pairs.append((left, -1 if right is None else right))
                following[parent] = len(pairs) - 1
            stages.append(tuple(pairs))
            active = following
        return tuple(stages)
    if sorted(order) != [1 << i for i in range(width.bit_length() - 1)]:
        raise ValueError("Reduction order must contain each binary stride once")
    active = {layer: slot for slot, layer in enumerate(layer_ids)}
    stages = []
    for stride in order:
        parents = sorted({index & ~stride for index in active})
        pairs, following = [], {}
        for parent in parents:
            left, right = active.get(parent), active.get(parent | stride)
            if left is None:
                left, right = right, None
            pairs.append((left, -1 if right is None else right))
            following[parent] = len(pairs) - 1
        stages.append(tuple(pairs))
        active = following
    return tuple(stages)


@triton.jit
def _exact_route(X, W, OUT, POS, COS_TABLE, SIN_TABLE, COS_OUT, SIN_OUT,
                 P0: tl.constexpr, P1: tl.constexpr, ROPE_DIM: tl.constexpr,
                 WITH_ROPE: tl.constexpr, TOKENS: tl.constexpr, SEQUENCE: tl.constexpr,
                 LAYERS: tl.constexpr, DRAFT_LAYERS: tl.constexpr, DIM: tl.constexpr,
                 X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
                 LEFT: tl.constexpr, RIGHT: tl.constexpr, OFFSETS: tl.constexpr,
                 TILE: tl.constexpr, REUSE: tl.constexpr):
    token, tile = tl.program_id(0), tl.program_id(1)
    if X0 >= 2147483648 or TOKENS * X1 >= 2147483648:
        token = token.to(tl.int64)
    col = tile * TILE + tl.arange(0, TILE)
    if WITH_ROPE:
        if tile == 0:
            rd = tl.arange(0, ROPE_DIM)
            position = tl.load(POS + (token // SEQUENCE) * P0 + (token % SEQUENCE) * P1)
            cos = tl.load(COS_TABLE + position * ROPE_DIM + rd)
            sin = tl.load(SIN_TABLE + position * ROPE_DIM + rd)
            tl.store(COS_OUT + token * ROPE_DIM + rd, cos)
            tl.store(SIN_OUT + token * ROPE_DIM + rd, sin)
    base = X + (token // SEQUENCE) * X0 + (token % SEQUENCE) * X1 + col
    features = ()
    for slot in tl.static_range(LAYERS):
        features += (tl.load(base + slot * X2, col < DIM, 0).to(tl.float32),)
    for lane in tl.static_range(DRAFT_LAYERS if REUSE else 1):
        draft = lane if REUSE else tl.program_id(2)
        values = ()
        for slot in tl.static_range(LAYERS):
            weight = tl.load(W + draft * LAYERS + slot).to(tl.float32)
            values += (features[slot] * weight,)
        for stage in tl.static_range(len(OFFSETS) - 1):
            following = ()
            for pair in tl.static_range(OFFSETS[stage + 1] - OFFSETS[stage]):
                if RIGHT[OFFSETS[stage] + pair] < 0:
                    following += (values[LEFT[OFFSETS[stage] + pair]],)
                else:
                    following += (values[LEFT[OFFSETS[stage] + pair]]
                                  + values[RIGHT[OFFSETS[stage] + pair]],)
            values = following
        tl.store(OUT + (draft * TOKENS + token) * DIM + col, values[0], col < DIM)


def supports_exact_mofa(features, weights, layer_ids, full_layer_count):
    return (features.is_cuda and features.dtype == torch.bfloat16
            and features.shape[2:] == (7, 2560) and features.stride(-1) == 1
            and weights.shape == (5, 7) and weights.dtype == torch.bfloat16
            and weights.is_contiguous() and weights.device == features.device
            and full_layer_count == 36 and len(layer_ids) == 7
            and tuple(sorted(set(layer_ids))) == tuple(layer_ids)
            and min(layer_ids) >= 0 and max(layer_ids) < full_layer_count)


def compact_mofa_exact(features, weights, layer_ids, full_layer_count, *,
                       order='register8', tile=None, warps=4,
                       reuse=True, output_dtype=None, rotary=None):
    batch, sequence, layers, dim = features.shape
    if tile is None:
        tile = 128 if batch * sequence <= 16 else 512
    if (features.stride(-1) != 1 or weights.shape[1] != layers
            or not weights.is_contiguous() or len(layer_ids) != layers):
        raise ValueError("Unsupported compact MoFA layout")
    width = triton.next_power_of_2(full_layer_count)
    plan = pruned_reduction_plan(tuple(layer_ids), width, order)
    left = tuple(pair[0] for stage in plan for pair in stage)
    right = tuple(pair[1] for stage in plan for pair in stage)
    offsets = [0]
    for stage in plan:
        offsets.append(offsets[-1] + len(stage))
    out = torch.empty(weights.shape[0], batch, sequence, 1, dim,
                      device=features.device, dtype=output_dtype or features.dtype)
    grid = ((batch * sequence, triton.cdiv(dim, tile)) if reuse else
            (batch * sequence, triton.cdiv(dim, tile), weights.shape[0]))
    positions = cos_table = sin_table = cos = sin = None
    p0 = p1 = rope_dim = 0
    if rotary is not None:
        if not reuse:
            raise ValueError('Fused rotary routing requires load reuse')
        cos_table, sin_table, positions = rotary
        rope_dim = cos_table.shape[-1]
        if (cos_table.shape != sin_table.shape or not cos_table.is_contiguous()
                or not sin_table.is_contiguous() or positions.shape != (batch, sequence)
                or cos_table.device != features.device or sin_table.device != features.device
                or cos_table.dtype != features.dtype or sin_table.dtype != features.dtype):
            raise ValueError('Unsupported compact rotary layout')
        p0, p1 = positions.stride()
        cos = torch.empty(batch, sequence, rope_dim, device=features.device, dtype=features.dtype)
        sin = torch.empty_like(cos)
    _exact_route[grid](features, weights, out, positions, cos_table, sin_table, cos, sin,
        p0, p1, rope_dim, rotary is not None, batch * sequence, sequence,
        layers, weights.shape[0], dim, *features.stride()[:3],
        left, right, tuple(offsets), tile, reuse,
        num_warps=warps, enable_fp_fusion=False)
    contexts = tuple(out.unbind(0))
    return contexts if rotary is None else (contexts, (cos, sin))
