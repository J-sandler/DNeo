"""Graph-safe, batched exact rejection sampling for dNEO serving."""

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_mofa(X, W, OUT, TOKENS: tl.constexpr, LAYERS: tl.constexpr,
                  DIM: tl.constexpr, X0: tl.constexpr, X1: tl.constexpr,
                  X2: tl.constexpr, SEQUENCE: tl.constexpr,
                  N: tl.constexpr, TILE: tl.constexpr):
    token, draft_layer, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # Large packed prefills can exceed int32 element offsets, even with B=1.
    if X0 >= 2147483648 or TOKENS * X1 >= 2147483648:
        token = token.to(tl.int64)
    layer = tl.arange(0, N)
    col = tile * TILE + tl.arange(0, TILE)
    weights = tl.load(W + draft_layer * LAYERS + layer, layer < LAYERS, 0).to(tl.float32)
    value = tl.load(X + (token // SEQUENCE) * X0 + (token % SEQUENCE) * X1
                    + layer[:, None] * X2 + col[None, :],
                    (layer[:, None] < LAYERS) & (col[None, :] < DIM), 0).to(tl.float32)
    result = tl.sum(weights[:, None] * value, 0)
    tl.store(OUT + (draft_layer * TOKENS + token) * DIM + col, result, col < DIM)


def shared_mofa(features, weights):
    """Route each layer once, without materializing identical KV-head copies."""
    batch, sequence, layers, dim = features.shape
    if features.stride(-1) != 1 or weights.ndim != 2 or weights.shape[1] != layers:
        raise ValueError("invalid shared MoFA layout")
    out = torch.empty(weights.shape[0], batch, sequence, 1, dim,
                      device=features.device, dtype=features.dtype)
    _shared_mofa[(batch * sequence, weights.shape[0], triton.cdiv(dim, 128))](
        features, weights, out, batch * sequence, layers, dim,
        *features.stride()[:3], sequence, triton.next_power_of_2(layers), 128,
        enable_fp_fusion=False, num_warps=4)
    return tuple(out.unbind(0))


@triton.jit
def _write_packed_kv(K, V, OK, OV, POSITIONS, SLOTS,
                     K0: tl.constexpr, K1: tl.constexpr,
                     V0: tl.constexpr, V1: tl.constexpr,
                     O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
                     HEADS: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    dim = tl.arange(0, BLOCK)
    slot, pos = tl.load(SLOTS + token), tl.load(POSITIONS + token)
    key = tl.load(K + head * K0 + token * K1 + dim, dim < DIM, 0)
    value = tl.load(V + head * V0 + token * V1 + dim, dim < DIM, 0)
    dst = slot * O0 + head * O1 + pos * O2 + dim
    tl.store(OK + dst, key, dim < DIM)
    tl.store(OV + dst, value, dim < DIM)


def write_packed_kv(kv, output, positions, token_slots):
    """Scatter an entire ragged prefill after one batched feature projection."""
    k, v = kv
    ok, ov = output
    if k.shape[0] != 1 or k.shape != v.shape or positions.numel() != k.shape[2]:
        raise ValueError("packed prefill must be [1, heads, total_tokens, head_dim]")
    _write_packed_kv[(k.shape[2], k.shape[1])](
        k, v, ok, ov, positions, token_slots,
        *k.stride()[1:3], *v.stride()[1:3], *ok.stride()[:3],
        k.shape[1], k.shape[-1], triton.next_power_of_2(k.shape[-1]))


@triton.jit
def _accept_count(DRAFT, Q, P, U, COUNTS,
                  VOCAB: tl.constexpr, GAMMA: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.arange(0, BLOCK)
    token = tl.load(DRAFT + row * GAMMA + pos, pos < GAMMA, 0)
    q = tl.load(Q + (row * GAMMA + pos) * VOCAB + token, pos < GAMMA, 1)
    p = tl.load(P + (row * (GAMMA + 1) + pos) * VOCAB + token, pos < GAMMA, 1)
    u = tl.load(U + row * GAMMA + pos, pos < GAMMA, 0)
    reject = (pos < GAMMA) & (u >= tl.minimum(p / tl.maximum(q, 1.17549435e-38), 1.))
    tl.store(COUNTS + row, tl.min(tl.where(reject, pos, GAMMA), 0))


@triton.jit
def _residual_race(P, Q, COUNTS, NOISE, SCORES, IDS,
                   VOCAB: tl.constexpr, GAMMA: tl.constexpr,
                   TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    token = tile * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(COUNTS + row)
    p = tl.load(P + (row * (GAMMA + 1) + count) * VOCAB + token, token < VOCAB, 0)
    q = tl.load(Q + (row * GAMMA + count) * VOCAB + token,
                (token < VOCAB) & (count < GAMMA), 0)
    weight = tl.maximum(p - q, 0.)
    noise = tl.load(NOISE + row * VOCAB + token, token < VOCAB, 1.)
    score = tl.where((token < VOCAB) & (weight > 0),
                     tl.log(weight) - tl.log(tl.maximum(noise, 1.17549435e-38)), -float("inf"))
    best = tl.max(score, 0)
    winner = tl.min(tl.where(score == best, token, 2147483647), 0)
    tl.store(SCORES + row * TILES + tile, best)
    tl.store(IDS + row * TILES + tile, winner)


@triton.jit
def _finish_batch(DRAFT, COUNTS, SCORES, IDS, PREFIX, COMMIT, BONUS, OUTPUT, LENGTHS,
                  GAMMA: tl.constexpr, TILES: tl.constexpr,
                  REDUCE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, REDUCE)
    scores = tl.load(SCORES + row * TILES + tile, tile < TILES, -float("inf"))
    ids = tl.load(IDS + row * TILES + tile, tile < TILES, 2147483647)
    best = tl.max(scores, 0)
    winner = tl.min(tl.where(scores == best, ids, 2147483647), 0)
    winner = tl.where(best > -float("inf"), winner, -1)
    count = tl.load(COUNTS + row)
    pos = tl.arange(0, BLOCK)
    draft = tl.load(DRAFT + row * GAMMA + pos, pos < GAMMA, 0)
    tl.store(OUTPUT + row * (GAMMA + 1) + pos,
             tl.where(pos < count, draft, tl.where(pos == count, winner, 0)), pos < GAMMA + 1)
    tl.store(COMMIT + row, count + 1)
    tl.store(BONUS + row, winner)
    tl.store(LENGTHS + row, tl.load(PREFIX + row) + count + 1)


def batched_leviathan(draft, q, p, prefix_lengths, *, generator=None,
                      uniforms=None, replacement_noise=None):
    """Three launches for the whole batch, with no CPU read of accept lengths.

    Rejection samples max(P-Q, 0); an entirely accepted block samples its bonus
    from P. Supplied randomness permits bitwise comparison with the scalar path.
    """
    bs, gamma, vocab = q.shape
    if (draft.shape != (bs, gamma) or p.shape != (bs, gamma + 1, vocab)
            or prefix_lengths.shape != (bs,)):
        raise ValueError("batched Leviathan shapes disagree")
    if any(not x.is_cuda or not x.is_contiguous() for x in (draft, q, p, prefix_lengths)):
        raise ValueError("batched Leviathan expects contiguous CUDA inputs")
    if q.dtype != torch.float32 or p.dtype != torch.float32:
        raise ValueError("acceptance distributions must be float32")
    if uniforms is None:
        uniforms = torch.rand((bs, gamma), device=q.device, generator=generator)
    if replacement_noise is None:
        replacement_noise = torch.empty((bs, vocab), device=q.device).exponential_(generator=generator)
    counts = torch.empty(bs, device=q.device, dtype=torch.int32)
    commit = torch.empty_like(counts)
    bonus = torch.empty(bs, device=q.device, dtype=torch.int64)
    output = torch.empty((bs, gamma + 1), device=q.device, dtype=torch.int64)
    lengths = torch.empty_like(prefix_lengths)
    tiles = triton.cdiv(vocab, 1024)
    scores = torch.empty((bs, tiles), device=q.device)
    ids = torch.empty((bs, tiles), device=q.device, dtype=torch.int32)
    _accept_count[(bs,)](draft, q, p, uniforms, counts, vocab, gamma,
                         triton.next_power_of_2(gamma), enable_fp_fusion=False)
    _residual_race[(bs, tiles)](p, q, counts, replacement_noise, scores, ids,
                               vocab, gamma, tiles, 1024, enable_fp_fusion=False)
    _finish_batch[(bs,)](draft, counts, scores, ids, prefix_lengths, commit, bonus,
                         output, lengths, gamma, tiles, triton.next_power_of_2(tiles),
                         triton.next_power_of_2(gamma + 1))
    return counts, commit, bonus, output, lengths, None
