"""Small serving-only Triton kernels; BF16 rounding matches the eager conv."""
import torch
import triton
import triton.language as tl

MIXED_HEAD_ROWS = 16


@triton.jit
def _rmsnorm(X, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / WIDTH
    normalized = (x * tl.rsqrt(variance + EPS)).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    tl.store(Y + row * WIDTH + col, normalized * weight, col < WIDTH)


def rmsnorm(hidden, weight, eps):
    hidden = hidden.contiguous()
    output = torch.empty_like(hidden)
    width = hidden.shape[-1]
    _rmsnorm[(hidden.numel() // width,)](hidden, weight, output, width, eps,
                                       triton.next_power_of_2(width), enable_fp_fusion=False)
    return output


@triton.jit
def _silu_mul(GU, OUT, WIDTH: tl.constexpr, ROW_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(GU + row * ROW_STRIDE + col, col < WIDTH, 0).to(tl.float32)
    up = tl.load(GU + row * ROW_STRIDE + WIDTH + col, col < WIDTH, 0).to(tl.float32)
    activated = (gate / (1 + tl.exp(-gate))).to(GU.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * WIDTH + col, activated * up, col < WIDTH)


def silu_mul(packed):
    if not packed.is_contiguous() or packed.shape[-1] % 2:
        raise ValueError("expected a contiguous packed gate/up tensor")
    width = packed.shape[-1] // 2
    output = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=packed.dtype)
    _silu_mul[(output.numel() // width, triton.cdiv(width, 256))](
        packed, output, width, 2 * width, 256, enable_fp_fusion=False)
    return output


@triton.jit
def _conv2(X, D, B, Y, WIDTH: tl.constexpr, GROUP: tl.constexpr, LENGTH: tl.constexpr,
           X0: tl.constexpr, X1: tl.constexpr, D0: tl.constexpr, D1: tl.constexpr,
           D2: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    sample, pos = row // LENGTH, row % LENGTH
    dtype = X.dtype.element_ty
    current = tl.load(X + sample * X0 + pos * X1 + col, col < WIDTH, 0).to(tl.float32)
    previous = tl.load(X + sample * X0 + (pos - 1) * X1 + col, (pos > 0) & (col < WIDTH), 0).to(tl.float32)
    d0 = tl.load(D + sample * D0 + pos * D1 + col // GROUP, col < WIDTH, 0).to(tl.float32)
    d1 = tl.load(D + sample * D0 + pos * D1 + D2 + col // GROUP, col < WIDTH, 0).to(tl.float32)
    b0 = tl.load(B + col, col < WIDTH, 0).to(tl.float32)
    b1 = tl.load(B + WIDTH + col, col < WIDTH, 0).to(tl.float32)
    # Preserve eager's cast after add, multiply, and accumulation. Disabling
    # FMA also prevents a supposedly faster kernel from changing the conv.
    c0, c1 = (b0 + d0).to(dtype).to(tl.float32), (b1 + d1).to(dtype).to(tl.float32)
    y = (current * c0).to(dtype).to(tl.float32) + (previous * c1).to(dtype).to(tl.float32)
    tl.store(Y + row * WIDTH + col, y, col < WIDTH)


def conv2(hidden, dynamic, base, group_size):
    if hidden.ndim != 3 or dynamic.shape != (*hidden.shape[:2], 2, hidden.shape[-1] // group_size):
        raise ValueError("invalid two-tap convolution layout")
    if hidden.stride(-1) != 1 or dynamic.stride(-1) != 1 or base.shape != (2, hidden.shape[-1]):
        raise ValueError("invalid convolution strides/base")
    out = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    batch, length, width = hidden.shape
    _conv2[(batch * length, triton.cdiv(width, 256))](
        hidden, dynamic, base, out, width, group_size, length,
        hidden.stride(0), hidden.stride(1), dynamic.stride(0), dynamic.stride(1), dynamic.stride(2),
        256, enable_fp_fusion=False,
    )
    return out


def prepare(conv, hidden, segments):
    if segments != (hidden.shape[1],):
        raise ValueError("serving convolution expects exactly one draft block")
    dynamic = conv.kernel_projection(hidden).view(*hidden.shape[:-1], 2, 2, hidden.shape[-1] // conv.group_size)
    return conv2(hidden, dynamic[..., 0, :, :], conv.base_kernel[0], conv.group_size), dynamic[..., 1, :, :]


def finish(conv, hidden, dynamic, segments):
    if segments != (hidden.shape[1],):
        raise ValueError("serving convolution cannot touch rejection memory")
    return conv2(hidden, dynamic, conv.base_kernel[1], conv.group_size)


@triton.jit
def _mixed_head(BASE, HIDDEN, PREV, DE, DW, SE, SW, A, B, OUT, TAGS, CACHE,
                POSITION, VOCAB: tl.constexpr, RANK: tl.constexpr, ROWS: tl.constexpr,
                CACHE_SLOTS: tl.constexpr, NOISE, SCORES, INDICES,
                TEMPERATURE: tl.constexpr, SAMPLE: tl.constexpr):
    sample = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    rank = tl.arange(0, RANK)
    dtype = BASE.dtype.element_ty
    previous = tl.load(PREV + sample)
    u = tl.load(DE + previous * RANK + rank).to(tl.float32)
    s = tl.load(SE + previous * RANK + rank).to(tl.float32)
    hidden = tl.load(HIDDEN + sample * RANK + rank).to(tl.float32)
    v = (s * hidden).to(dtype).to(tl.float32)
    w2 = tl.load(SW + rows[:, None] * RANK + rank[None], rows[:, None] < VOCAB, 0).to(tl.float32)
    if CACHE_SLOTS:
        slot = previous % CACHE_SLOTS
        cached = tl.load(TAGS + slot) == previous
        if cached:
            d1 = tl.load(CACHE + slot * VOCAB + rows, rows < VOCAB, 0).to(tl.float32)
        else:
            w1 = tl.load(DW + rows[:, None] * RANK + rank[None], rows[:, None] < VOCAB, 0).to(tl.float32)
            d1 = tl.sum(w1 * u[None], axis=1).to(dtype).to(tl.float32)
            tl.store(CACHE + slot * VOCAB + rows, d1, rows < VOCAB)
    else:
        w1 = tl.load(DW + rows[:, None] * RANK + rank[None], rows[:, None] < VOCAB, 0).to(tl.float32)
        d1 = tl.sum(w1 * u[None], axis=1).to(dtype).to(tl.float32)
    d2 = tl.sum(w2 * v[None], axis=1).to(dtype).to(tl.float32)
    a, b = tl.load(A + POSITION).to(tl.float32), tl.load(B + POSITION).to(tl.float32)
    base = tl.load(BASE + sample * VOCAB + rows, rows < VOCAB, 0).to(tl.float32)
    scale = (a + b).to(dtype).to(tl.float32)
    combined = ((scale * base).to(dtype).to(tl.float32) + (a * d1).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    combined = combined + (b * d2).to(dtype).to(tl.float32)
    tl.store(OUT + sample * VOCAB + rows, combined, rows < VOCAB)
    if SAMPLE:
        noise = tl.load(NOISE + sample * VOCAB + rows, rows < VOCAB, 1.)
        # Sample from the STORED BF16 logits, exactly the logits later normalized
        # into q. Exponential racing avoids normalization on the serial path.
        scores = combined.to(dtype).to(tl.float32) / TEMPERATURE - tl.log(tl.maximum(noise, 1.17549435e-38))
        scores = tl.where(rows < VOCAB, scores, -float("inf"))
        best = tl.max(scores, 0)
        winner = tl.min(tl.where(scores == best, rows, 2147483647), 0)
        tiles = tl.cdiv(VOCAB, ROWS)
        offset = sample * tiles + tl.program_id(1)
        tl.store(SCORES + offset, best)
        tl.store(INDICES + offset, winner)


@triton.jit
def _sample_winner(SCORES, INDICES, OUT, TILES: tl.constexpr, BLOCK: tl.constexpr):
    sample = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    score = tl.load(SCORES + sample * TILES + tile, tile < TILES, -float("inf"))
    ids = tl.load(INDICES + sample * TILES + tile, tile < TILES, 2147483647)
    best = tl.max(score, 0)
    winner = tl.min(tl.where(score == best, ids, 2147483647), 0)
    tl.store(OUT + sample, winner)


@triton.jit
def _commit_dspark_cache(PREV, TAGS, STATS, SLOTS: tl.constexpr):
    previous = tl.load(PREV)
    slot = previous % SLOTS
    hit = tl.load(TAGS + slot) == previous
    tl.store(STATS, tl.load(STATS) + hit.to(tl.int32))
    tl.store(STATS + 1, tl.load(STATS + 1) + 1)
    tl.store(TAGS + slot, previous)


def mixed_head(base, hidden, previous, position, head, cache=None):
    if base.ndim != 2 or not hidden.is_contiguous() or not base.is_contiguous():
        raise ValueError("fused mixed head expects contiguous per-position inputs")
    output = torch.empty_like(base)
    rank = head.dspark_token_embed.weight.shape[-1]
    if rank & (rank - 1):
        raise ValueError("fused mixed-head rank must be a power of two")
    tags, values, stats = (None, None, None) if cache is None else cache
    if cache is not None and base.shape[0] != 1:
        raise ValueError("token correction cache requires single-request rollout")
    rows = MIXED_HEAD_ROWS
    _mixed_head[(base.shape[0], triton.cdiv(base.shape[1], rows))](
        base, hidden, previous, head.dspark_token_embed.weight, head.dspark_out_proj.weight,
        head.selector.predecessor_codebook.weight, head.selector.successor_codebook.weight,
        head.dspark_logit_scale, head.df2_logit_scale, output, tags, values, position, base.shape[1], rank, rows,
        0 if cache is None else tags.numel(),
        None, None, None, 1., False,
        enable_fp_fusion=False,
    )
    if cache is not None:
        # Publish the tag only AFTER every vocabulary tile has finished writing.
        _commit_dspark_cache[(1,)](previous, tags, stats, tags.numel())
    return output


def mixed_head_sample_fused_matrix(base, hidden, previous, position, head, noise, temperature=1., cache=None):
    """Fused full-vocabulary correction plus exponential-race categorical draw."""
    if temperature <= 0 or base.ndim != 2 or not base.is_contiguous() or not noise.is_contiguous():
        raise ValueError("fused sampling requires contiguous matrices and positive temperature")
    if noise.shape != base.shape or hidden.stride(-1) != 1:
        raise ValueError("sampling inputs are misaligned")
    rank = head.dspark_token_embed.weight.shape[-1]
    if rank & (rank - 1):
        raise ValueError("fused head requires power-of-two rank")
    rows = 16
    batch, vocab = base.shape
    tiles = triton.cdiv(vocab, rows)
    output = torch.empty_like(base)
    scores = torch.empty((batch, tiles), device=base.device, dtype=torch.float32)
    indices = torch.empty((batch, tiles), device=base.device, dtype=torch.int32)
    sampled = torch.empty((batch,), device=base.device, dtype=torch.long)
    tags, values, stats = (None, None, None) if cache is None else cache
    if cache is not None and batch != 1:
        raise ValueError("token correction cache requires a single request")
    _mixed_head[(batch, tiles)](
        base, hidden, previous, head.dspark_token_embed.weight, head.dspark_out_proj.weight,
        head.selector.predecessor_codebook.weight, head.selector.successor_codebook.weight,
        head.dspark_logit_scale, head.df2_logit_scale, output, tags, values, position,
        vocab, rank, rows, 0 if cache is None else tags.numel(), noise, scores, indices,
        temperature, True, enable_fp_fusion=False,
    )
    if cache is not None:
        _commit_dspark_cache[(1,)](previous, tags, stats, tags.numel())
    _sample_winner[(batch,)](scores, indices, sampled, tiles, triton.next_power_of_2(tiles))
    return output, sampled


@triton.jit
def _race_tiles(LOGITS, NOISE, SCORES, INDICES, VOCAB: tl.constexpr, TEMPERATURE: tl.constexpr,
                BLOCK: tl.constexpr):
    sample = tl.program_id(0)
    tile = tl.program_id(1)
    ids = tile * BLOCK + tl.arange(0, BLOCK)
    logits = tl.load(LOGITS + sample * VOCAB + ids, ids < VOCAB, -float("inf")).to(tl.float32)
    noise = tl.load(NOISE + sample * VOCAB + ids, ids < VOCAB, 1.)
    scores = logits / TEMPERATURE - tl.log(tl.maximum(noise, 1.17549435e-38))
    best = tl.max(scores, 0)
    winner = tl.min(tl.where(scores == best, ids, 2147483647), 0)
    offset = sample * tl.cdiv(VOCAB, BLOCK) + tile
    tl.store(SCORES + offset, best)
    tl.store(INDICES + offset, winner)


def race_sample(logits, noise, temperature=1.):
    """Fused log-noise/temperature/argmax, outside the register-heavy GEMV."""
    if temperature <= 0 or logits.ndim != 2 or logits.shape != noise.shape:
        raise ValueError("race sampling requires aligned matrices and positive temperature")
    if not logits.is_contiguous() or not noise.is_contiguous():
        raise ValueError("race sampling requires contiguous matrices")
    batch, vocab = logits.shape
    tiles = triton.cdiv(vocab, 1024)
    scores = torch.empty((batch, tiles), device=logits.device, dtype=torch.float32)
    indices = torch.empty((batch, tiles), device=logits.device, dtype=torch.int32)
    sampled = torch.empty((batch,), device=logits.device, dtype=torch.long)
    _race_tiles[(batch, tiles)](logits, noise, scores, indices, vocab, temperature, 1024)
    _sample_winner[(batch,)](scores, indices, sampled, tiles, triton.next_power_of_2(tiles))
    return sampled


@triton.jit
def _probability_race_tiles(LOGITS, NOISE, TILE_MAX, SCORES, INDICES,
                            VOCAB: tl.constexpr, TEMPERATURE: tl.constexpr,
                            BLOCK: tl.constexpr):
    sample = tl.program_id(0)
    tile = tl.program_id(1)
    ids = tile * BLOCK + tl.arange(0, BLOCK)
    valid = ids < VOCAB
    scaled = tl.load(
        LOGITS + sample * VOCAB + ids, valid, -float("inf")
    ).to(tl.float32) / TEMPERATURE
    local_max = tl.max(scaled, 0)
    noise = tl.load(NOISE + sample * VOCAB + ids, valid, 1.)
    keys = tl.exp(scaled - local_max) / noise
    keys = tl.where(valid, keys, -1.)
    best = tl.max(keys, 0)
    winner = tl.min(tl.where(keys == best, ids, 2147483647), 0)
    offset = sample * tl.cdiv(VOCAB, BLOCK) + tile
    tl.store(TILE_MAX + offset, local_max)
    tl.store(SCORES + offset, best)
    tl.store(INDICES + offset, winner)


@triton.jit
def _probability_race_winner(TILE_MAX, SCORES, INDICES, OUT,
                             TILES: tl.constexpr, BLOCK: tl.constexpr):
    sample = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    valid = tile < TILES
    local_max = tl.load(
        TILE_MAX + sample * TILES + tile, valid, -float("inf")
    )
    keys = tl.load(SCORES + sample * TILES + tile, valid, -1.)
    ids = tl.load(INDICES + sample * TILES + tile, valid, 2147483647)
    global_max = tl.max(local_max, 0)
    keys *= tl.exp(local_max - global_max)
    keys = tl.where(valid, keys, -1.)
    best = tl.max(keys, 0)
    winner = tl.min(tl.where(keys == best, ids, 2147483647), 0)
    tl.store(OUT + sample, tl.where(winner == 2147483647, 0, winner))


def probability_race_sample(logits, noise, temperature=1.):
    """SGLang-equivalent online categorical sampler for corrected logits."""
    if temperature <= 0 or logits.ndim != 2 or logits.shape != noise.shape:
        raise ValueError("probability-race sampling requires aligned matrices")
    if not logits.is_contiguous() or not noise.is_contiguous():
        raise ValueError("probability-race sampling requires contiguous matrices")
    batch, vocab = logits.shape
    tiles = triton.cdiv(vocab, 1024)
    tile_max = torch.empty((batch, tiles), device=logits.device, dtype=torch.float32)
    scores = torch.empty_like(tile_max)
    indices = torch.empty((batch, tiles), device=logits.device, dtype=torch.int32)
    sampled = torch.empty((batch,), device=logits.device, dtype=torch.long)
    _probability_race_tiles[(batch, tiles)](
        logits, noise, tile_max, scores, indices, vocab, temperature, 1024,
    )
    _probability_race_winner[(batch,)](
        tile_max, scores, indices, sampled, tiles, triton.next_power_of_2(tiles),
    )
    return sampled


@triton.jit
def _full_vocab_softmax(LOGITS, OUT, VOCAB: tl.constexpr,
                        TEMPERATURE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_max = -float("inf")
    for start in range(0, VOCAB, BLOCK):
        ids = start + tl.arange(0, BLOCK)
        valid = ids < VOCAB
        value = tl.load(
            LOGITS + row * VOCAB + ids, valid, -float("inf")
        ).to(tl.float32) / TEMPERATURE
        row_max = tl.maximum(row_max, tl.max(value, 0))
    total = 0.0
    for start in range(0, VOCAB, BLOCK):
        ids = start + tl.arange(0, BLOCK)
        valid = ids < VOCAB
        value = tl.load(
            LOGITS + row * VOCAB + ids, valid, -float("inf")
        ).to(tl.float32) / TEMPERATURE
        total += tl.sum(tl.where(valid, tl.exp(value - row_max), 0.), 0)
    for start in range(0, VOCAB, BLOCK):
        ids = start + tl.arange(0, BLOCK)
        valid = ids < VOCAB
        value = tl.load(
            LOGITS + row * VOCAB + ids, valid, -float("inf")
        ).to(tl.float32) / TEMPERATURE
        tl.store(OUT + row * VOCAB + ids, tl.exp(value - row_max) / total, valid)


@triton.jit
def _vocab_partial_stats(LOGITS, STATS, VOCAB: tl.constexpr, TILES: tl.constexpr,
                         TEMPERATURE: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    ids = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(LOGITS + row * VOCAB + ids, ids < VOCAB, -float("inf")).to(tl.float32) / TEMPERATURE
    maximum = tl.max(value, 0)
    # Grammar/top-k masks may eliminate an entire vocabulary tile.
    safe_maximum = tl.where(maximum == -float("inf"), 0., maximum)
    total = tl.sum(tl.exp(value - safe_maximum), 0)
    tl.store(STATS + (row * TILES + tile) * 2, maximum)
    tl.store(STATS + (row * TILES + tile) * 2 + 1, total)


@triton.jit
def _vocab_normalize(LOGITS, STATS, OUT, VOCAB: tl.constexpr, TILES: tl.constexpr,
                     TEMPERATURE: tl.constexpr, REDUCE: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    t = tl.arange(0, REDUCE)
    maxima = tl.load(STATS + (row * TILES + t) * 2, t < TILES, -float("inf"))
    totals = tl.load(STATS + (row * TILES + t) * 2 + 1, t < TILES, 0.)
    maximum = tl.max(maxima, 0)
    total = tl.sum(totals * tl.exp(maxima - maximum), 0)
    ids = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(LOGITS + row * VOCAB + ids, ids < VOCAB, -float("inf")).to(tl.float32) / TEMPERATURE
    tl.store(OUT + row * VOCAB + ids, tl.exp(value - maximum) / total, ids < VOCAB)


def full_vocab_probabilities(logits, temperature=1., *, out=None):
    """Normalize all corrected rows with the serving softmax kernel."""
    if temperature <= 0 or not logits.is_contiguous():
        raise ValueError("full-vocabulary softmax requires contiguous logits")
    vocab = logits.shape[-1]
    if out is not None and (out.shape != logits.shape or out.device != logits.device
                            or out.dtype != torch.float32 or not out.is_contiguous()):
        raise ValueError("full-vocabulary softmax output must be contiguous FP32 with matching shape/device")
    output = (torch.empty(logits.shape, device=logits.device, dtype=torch.float32)
              if out is None else out)
    rows, tiles = logits.numel() // vocab, triton.cdiv(vocab, 4096)
    if tiles <= 2:
        _full_vocab_softmax[(rows,)](logits, output, vocab, temperature, 4096)
    else:
        # Spread a 150K-token vocabulary across SMs instead of serializing three
        # complete vocabulary sweeps in a single CTA for each draft position.
        stats = torch.empty((rows, tiles, 2), device=logits.device, dtype=torch.float32)
        _vocab_partial_stats[(rows, tiles)](logits, stats, vocab, tiles, temperature, 4096,
                                           enable_fp_fusion=False)
        _vocab_normalize[(rows, tiles)](logits, stats, output, vocab, tiles,
                                       temperature, triton.next_power_of_2(tiles), 4096,
                                       enable_fp_fusion=False)
    return output


@triton.jit
def _block_keep(DRAFT, Q, P, KEEP, VOCAB: tl.constexpr, GAMMA: tl.constexpr):
    value = 1.0
    tl.store(KEEP, value)
    for position in tl.static_range(0, GAMMA):
        token = tl.load(DRAFT + position)
        q = tl.load(Q + position * VOCAB + token).to(tl.float32)
        p = tl.load(P + position * VOCAB + token).to(tl.float32)
        value = tl.minimum(value * p / tl.maximum(q, 1.17549435e-38), 1.0)
        tl.store(KEEP + position + 1, value)


@triton.jit
def _block_residual_partials(P, Q, KEEP, PARTIAL, VOCAB: tl.constexpr,
                             GAMMA: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    position, tile = tl.program_id(0), tl.program_id(1)
    token = tile * BLOCK + tl.arange(0, BLOCK)
    valid = token < VOCAB
    scale = tl.load(KEEP + position)
    p = tl.load(P + position * VOCAB + token, valid, 0).to(tl.float32)
    q = tl.load(Q + position * VOCAB + token, valid, 0).to(tl.float32)
    residual = tl.maximum(scale * p - q, 0.0)
    tl.store(PARTIAL + position * TILES + tile, tl.sum(residual, axis=0))


@triton.jit
def _block_residual_reduce(PARTIAL, MASS, TILES: tl.constexpr, BLOCK: tl.constexpr):
    position = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    values = tl.load(PARTIAL + position * TILES + tile, tile < TILES, 0.0)
    tl.store(MASS + position, tl.sum(values, axis=0))


@triton.jit
def _block_select_tau(KEEP, MASS, UNIFORMS, TAU, GAMMA: tl.constexpr):
    tau = 0
    for index in tl.static_range(1, GAMMA + 1):
        keep = tl.load(KEEP + index)
        if index == GAMMA:
            h = keep
        else:
            residual = tl.load(MASS + index)
            denominator = residual + 1.0 - keep
            h = tl.where(denominator > 0, residual / denominator, 1.0)
        tau = tl.where(tl.load(UNIFORMS + index - 1) <= h, index, tau)
    tl.store(TAU, tau)


@triton.jit
def _block_residual_race(P, Q, KEEP, TAU, NOISE, SCORES, INDICES,
                         VOCAB: tl.constexpr, GAMMA: tl.constexpr,
                         TILES: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    token = tile * BLOCK + tl.arange(0, BLOCK)
    valid = token < VOCAB
    tau = tl.load(TAU)
    p = tl.load(P + tau * VOCAB + token, valid, 0).to(tl.float32)
    q = tl.load(Q + tau * VOCAB + token, valid & (tau < GAMMA), 0).to(tl.float32)
    scale = tl.load(KEEP + tau)
    weight = tl.maximum(scale * p - q, 0.0)
    noise = tl.load(NOISE + token, valid, 1.0)
    score = tl.where(valid & (weight > 0),
                     tl.log(weight) - tl.log(tl.maximum(noise, 1.17549435e-38)), -float("inf"))
    best = tl.max(score, axis=0)
    winner = tl.min(tl.where(score == best, token, 2147483647), axis=0)
    tl.store(SCORES + tile, best)
    tl.store(INDICES + tile, winner)


def block_verify_cuda(draft_ids, q, p, generator=None, uniforms=None, replacement_noise=None):
    """Allocation-light CUDA implementation of lossless Block Verification."""
    if draft_ids.ndim != 1 or q.ndim != 2 or p.ndim != 2 or q.shape != p[:-1].shape:
        raise ValueError("verification dimensions do not align")
    if not q.is_contiguous() or not p.is_contiguous() or not draft_ids.is_contiguous():
        raise ValueError("block verification expects contiguous inputs")
    gamma, vocab = q.shape
    if gamma < 1:
        raise ValueError("draft must contain at least one token")
    if uniforms is None:
        uniforms = torch.rand(gamma, device=q.device, generator=generator)
    if uniforms.shape != (gamma,) or uniforms.device != q.device:
        raise ValueError("uniforms must have one device-local value per draft token")
    if replacement_noise is None:
        replacement_noise = torch.empty(vocab, device=q.device, dtype=torch.float32).exponential_(generator=generator)
    if replacement_noise.shape != (vocab,) or replacement_noise.device != q.device:
        raise ValueError("replacement noise must have one device-local value per vocabulary item")
    block = 1024
    tiles = triton.cdiv(vocab, block)
    keep = torch.empty(gamma + 1, device=q.device, dtype=torch.float32)
    partial = torch.empty((gamma, tiles), device=q.device, dtype=torch.float32)
    mass = torch.empty(gamma, device=q.device, dtype=torch.float32)
    tau = torch.empty((), device=q.device, dtype=torch.long)
    scores = torch.empty((1, tiles), device=q.device, dtype=torch.float32)
    indices = torch.empty((1, tiles), device=q.device, dtype=torch.int32)
    replacement = torch.empty((1,), device=q.device, dtype=torch.long)
    _block_keep[(1,)](draft_ids, q, p, keep, vocab, gamma, enable_fp_fusion=False)
    _block_residual_partials[(gamma, tiles)](
        p, q, keep, partial, vocab, gamma, tiles, block, enable_fp_fusion=False)
    _block_residual_reduce[(gamma,)](
        partial, mass, tiles, triton.next_power_of_2(tiles), enable_fp_fusion=False)
    _block_select_tau[(1,)](keep, mass, uniforms, tau, gamma, enable_fp_fusion=False)
    _block_residual_race[(tiles,)](
        p, q, keep, tau, replacement_noise, scores, indices, vocab, gamma, tiles, block,
        enable_fp_fusion=False)
    _sample_winner[(1,)](scores, indices, replacement, tiles, triton.next_power_of_2(tiles))
    count = int(tau.item())
    return torch.cat((draft_ids[:count], replacement)), count


@triton.jit
def _leviathan_count(DRAFT, Q, P, U, META, VOCAB: tl.constexpr,
                     GAMMA: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.arange(0, BLOCK)
    token = tl.load(DRAFT + pos, pos < GAMMA, 0)
    q = tl.load(Q + pos * VOCAB + token, pos < GAMMA, 1).to(tl.float32)
    p = tl.load(P + pos * VOCAB + token, pos < GAMMA, 1).to(tl.float32)
    u = tl.load(U + pos, pos < GAMMA, 0)
    reject = (pos < GAMMA) & (u >= tl.minimum(p / tl.maximum(q, 1.17549435e-38), 1.0))
    tl.store(META, tl.min(tl.where(reject, pos, GAMMA), axis=0))


@triton.jit
def _leviathan_race(P, Q, META, NOISE, SCORES, INDICES, VOCAB: tl.constexpr,
                    GAMMA: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    token = tile * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(META)
    p = tl.load(P + count * VOCAB + token, token < VOCAB, 0).to(tl.float32)
    q = tl.load(Q + count * VOCAB + token, (token < VOCAB) & (count < GAMMA), 0).to(tl.float32)
    weight = tl.maximum(p - q, 0.0)
    noise = tl.load(NOISE + token, token < VOCAB, 1).to(tl.float32)
    score = tl.where((token < VOCAB) & (weight > 0),
                     tl.log(weight) - tl.log(tl.maximum(noise, 1.17549435e-38)), -float("inf"))
    best = tl.max(score, axis=0)
    winner = tl.min(tl.where(score == best, token, 2147483647), axis=0)
    tl.store(SCORES + tile, best)
    tl.store(INDICES + tile, winner)


@triton.jit
def _leviathan_winner(SCORES, INDICES, META, TILES: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.arange(0, BLOCK)
    scores = tl.load(SCORES + tile, tile < TILES, -float("inf"))
    ids = tl.load(INDICES + tile, tile < TILES, 2147483647)
    best = tl.max(scores, axis=0)
    winner = tl.min(tl.where(scores == best, ids, 2147483647), axis=0)
    tl.store(META + 1, tl.where(best > -float("inf"), winner, -1))


def leviathan_verify_cuda(draft_ids, q, p, generator=None, uniforms=None, replacement_noise=None):
    """Exact token verification with one host sync for count and residual audit."""
    if (draft_ids.ndim != 1 or q.ndim != 2 or p.shape != (q.shape[0] + 1, q.shape[1])
            or draft_ids.numel() != q.shape[0] or q.shape[0] < 1):
        raise ValueError("verification dimensions do not align")
    if any(not value.is_cuda or not value.is_contiguous() or value.device != q.device
           for value in (draft_ids, q, p)):
        raise ValueError("verification requires contiguous tensors on one CUDA device")
    if q.dtype != torch.float32 or p.dtype != torch.float32:
        raise ValueError("verification probabilities must be float32")
    gamma, vocab = q.shape
    if uniforms is None:
        uniforms = torch.rand(gamma, device=q.device, generator=generator)
    if replacement_noise is None:
        replacement_noise = torch.empty(vocab, device=q.device, dtype=torch.float32).exponential_(generator=generator)
    for value, size in ((uniforms, gamma), (replacement_noise, vocab)):
        if value.shape != (size,) or value.device != q.device or not value.is_contiguous():
            raise ValueError("invalid verification randomness shape/device/layout")
    tiles = triton.cdiv(vocab, 1024)
    metadata = torch.empty(2, device=q.device, dtype=torch.long)
    scores = torch.empty(tiles, device=q.device)
    ids = torch.empty(tiles, device=q.device, dtype=torch.int32)
    _leviathan_count[(1,)](draft_ids, q, p, uniforms, metadata, vocab, gamma,
                          triton.next_power_of_2(gamma), enable_fp_fusion=False)
    _leviathan_race[(tiles,)](p, q, metadata, replacement_noise, scores, ids,
                             vocab, gamma, 1024, enable_fp_fusion=False)
    _leviathan_winner[(1,)](scores, ids, metadata, tiles, triton.next_power_of_2(tiles))
    count, winner = metadata.tolist()
    if winner < 0:
        raise RuntimeError("zero replacement residual after rejection")
    return torch.cat((draft_ids[:count], metadata[1:])), count


def mixed_head_sample(base, hidden, previous, position, head, noise, temperature=1., cache=None):
    logits = mixed_head(base, hidden, previous, position, head, cache)
    return logits, race_sample(logits, noise, temperature)


def mixed_head_sample_torch(base, hidden, previous, position, head, noise, temperature=1., cache=None):
    if cache is not None:
        return mixed_head_sample(base, hidden, previous, position, head, noise, temperature, cache)
    a = head.dspark_logit_scale[position].to(base.dtype)
    b = head.df2_logit_scale[position].to(base.dtype)
    u = head.dspark_token_embed(previous)
    v = head.selector.predecessor_codebook(previous) * hidden
    logits = (a+b)*base + a*torch.nn.functional.linear(u, head.dspark_out_proj.weight) + b*torch.nn.functional.linear(
        v, head.selector.successor_codebook.weight)
    return logits, race_sample(logits, noise, temperature)


@triton.jit
def _folded_df2_head(BASE, LATENT, OUTW, A, B, OUT, POSITION,
                     VOCAB: tl.constexpr, RANK: tl.constexpr, ROWS: tl.constexpr,
                     NOISE, SCORES, INDICES, TEMPERATURE: tl.constexpr,
                     SAMPLE: tl.constexpr):
    sample = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    rank = tl.arange(0, RANK)
    dtype = BASE.dtype.element_ty
    latent = tl.load(LATENT + sample * RANK + rank).to(tl.float32)
    weight = tl.load(
        OUTW + rows[:, None] * RANK + rank[None],
        rows[:, None] < VOCAB,
        0,
    ).to(tl.float32)
    correction = tl.sum(weight * latent[None], axis=1).to(dtype).to(tl.float32)
    a = tl.load(A + POSITION).to(tl.float32)
    b = tl.load(B + POSITION).to(tl.float32)
    base = tl.load(BASE + sample * VOCAB + rows, rows < VOCAB, 0).to(tl.float32)
    scale = (a + b).to(dtype).to(tl.float32)
    combined = (scale * base).to(dtype).to(tl.float32) + correction
    tl.store(OUT + sample * VOCAB + rows, combined, rows < VOCAB)
    if SAMPLE:
        noise = tl.load(NOISE + sample * VOCAB + rows, rows < VOCAB, 1.)
        scores = combined.to(dtype).to(tl.float32) / TEMPERATURE - tl.log(
            tl.maximum(noise, 1.17549435e-38)
        )
        scores = tl.where(rows < VOCAB, scores, -float("inf"))
        best = tl.max(scores, 0)
        winner = tl.min(tl.where(scores == best, rows, 2147483647), 0)
        tiles = tl.cdiv(VOCAB, ROWS)
        offset = sample * tiles + tl.program_id(1)
        tl.store(SCORES + offset, best)
        tl.store(INDICES + offset, winner)


def _folded_df2_launch(base, latent, position, head, noise=None, temperature=1.):
    if base.ndim != 2 or not base.is_contiguous() or not latent.is_contiguous():
        raise ValueError("folded DF2 kernel expects contiguous matrices")
    rank = head.folded_out_proj.weight.shape[-1]
    if latent.shape != (base.shape[0], rank) or rank & (rank - 1):
        raise ValueError("folded DF2 latent shape/rank is invalid")
    sample = noise is not None
    if sample and (temperature <= 0 or noise.shape != base.shape or not noise.is_contiguous()):
        raise ValueError("folded DF2 sampling inputs are invalid")
    batch, vocab = base.shape
    rows = getattr(head, "_inference_head_rows", MIXED_HEAD_ROWS)
    warps = getattr(head, "_inference_head_warps", 4)
    tiles = triton.cdiv(vocab, rows)
    output = torch.empty_like(base)
    scores = torch.empty((batch, tiles), device=base.device, dtype=torch.float32) if sample else None
    indices = torch.empty((batch, tiles), device=base.device, dtype=torch.int32) if sample else None
    _folded_df2_head[(batch, tiles)](
        base, latent, head.folded_out_proj.weight,
        head.dspark_logit_scale, head.df2_logit_scale, output, position,
        vocab, rank, rows, noise, scores, indices, temperature, sample,
        enable_fp_fusion=False, num_warps=warps,
    )
    if not sample:
        return output
    sampled = torch.empty((batch,), device=base.device, dtype=torch.long)
    _sample_winner[(batch,)](
        scores, indices, sampled, tiles, triton.next_power_of_2(tiles)
    )
    return output, sampled


def folded_df2_head(base, latent, position, head):
    return _folded_df2_launch(base, latent, position, head)


@triton.jit
def _folded_df2_tensor_head(BASE, LATENT, WEIGHT, A, B, NOISE, OUT, SCORES, IDS,
                           POSITION, BS: tl.constexpr,
                           VOCAB: tl.constexpr, RANK: tl.constexpr,
                           TEMPERATURE: tl.constexpr, BM: tl.constexpr,
                           BN: tl.constexpr, BASE_ROW: tl.constexpr,
                           NOISE_ROW: tl.constexpr, OUT_ROW: tl.constexpr):
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    token = tl.program_id(1) * BN + tl.arange(0, BN)
    rank = tl.arange(0, RANK)
    latent = tl.load(LATENT + batch[:, None] * RANK + rank[None, :], batch[:, None] < BS, 0)
    weight = tl.load(WEIGHT + token[None, :] * RANK + rank[:, None], token[None, :] < VOCAB, 0)
    dtype = BASE.dtype.element_ty
    correction = tl.dot(latent, weight).to(dtype).to(tl.float32)
    scale = (tl.load(A + POSITION).to(tl.float32) + tl.load(B + POSITION).to(tl.float32)).to(dtype).to(tl.float32)
    offset = batch[:, None] * VOCAB + token[None, :]
    valid = (batch[:, None] < BS) & (token[None, :] < VOCAB)
    base = tl.load(BASE + batch[:, None] * BASE_ROW + token[None, :], valid, 0).to(tl.float32)
    combined = ((scale * base).to(dtype).to(tl.float32) + correction).to(dtype)
    tl.store(OUT + batch[:, None] * OUT_ROW + token[None, :], combined, valid)
    noise = tl.load(NOISE + batch[:, None] * NOISE_ROW + token[None, :], valid, 1.)
    race = combined.to(tl.float32) / TEMPERATURE - tl.log(tl.maximum(noise, 1.17549435e-38))
    race = tl.where(valid, race, -float("inf"))
    best = tl.max(race, 1)
    winner = tl.min(tl.where(race == best[:, None], token[None, :], 2147483647), 1)
    result = batch * tl.cdiv(VOCAB, BN) + tl.program_id(1)
    tl.store(SCORES + result, best, batch < BS)
    tl.store(IDS + result, winner, batch < BS)


def folded_head_backend(device):
    # The Hopper-tuned tile rereads too much weight data on A10/SM86.
    return "cublas" if torch.cuda.get_device_capability(device) == (8, 6) else "tensor"


def folded_df2_head_sample_fused_matrix(base, latent, position, head, noise, temperature=1., out=None):
    """Share vocabulary weight reads across requests using tensor cores."""
    if base.shape[0] == 1 and out is None:
        return _folded_df2_launch(base, latent, position, head, noise, temperature)
    batch, vocab = base.shape
    rank = head.folded_out_proj.weight.shape[-1]
    if (base.stride(-1) != 1 or not latent.is_contiguous() or noise.stride(-1) != 1
            or latent.shape != (batch, rank) or noise.shape != base.shape
            or rank & (rank - 1) or temperature <= 0):
        raise ValueError("invalid batched folded head inputs")
    if out is not None and (out.shape != base.shape or out.dtype != base.dtype
                            or out.device != base.device or out.stride(-1) != 1
                            or out.stride(0) < vocab):
        raise ValueError("invalid folded head output slice")
    backend = getattr(head, "_inference_head_backend", "auto")
    if backend == "auto":
        backend = folded_head_backend(base.device)
    if backend not in ("tensor", "cublas"):
        raise ValueError(f"unknown folded head backend: {backend}")
    if backend == "cublas" or rank < 16:
        correction = torch.nn.functional.linear(latent, head.folded_out_proj.weight)
        scale = (head.dspark_logit_scale[position].float()
                 + head.df2_logit_scale[position].float()).to(base.dtype)
        output = scale * base + correction
        sampled = probability_race_sample(output, noise.contiguous(), temperature)
        if out is not None:
            out.copy_(output)
            output = out
        return output, sampled
    bm, bn = 16, 128
    tiles = triton.cdiv(vocab, bn)
    output = torch.empty_like(base) if out is None else out
    scores = torch.empty((batch, tiles), device=base.device, dtype=torch.float32)
    ids = torch.empty((batch, tiles), device=base.device, dtype=torch.int32)
    _folded_df2_tensor_head[(triton.cdiv(batch, bm), tiles)](
        base, latent, head.folded_out_proj.weight, head.dspark_logit_scale,
        head.df2_logit_scale, noise, output, scores, ids, position, batch,
        vocab, rank, temperature, bm, bn, base.stride(0), noise.stride(0), output.stride(0),
        num_warps=4, num_stages=1,
        enable_fp_fusion=False)
    sampled = torch.empty(batch, device=base.device, dtype=torch.long)
    _sample_winner[(batch,)](scores, ids, sampled, tiles, triton.next_power_of_2(tiles))
    return output, sampled


def folded_df2_head_sample(base, latent, position, head, noise, temperature=1.):
    """Batch-1 path: keep projection and SGLang-style sampling register-light."""
    logits = folded_df2_head(base, latent, position, head)
    return logits, probability_race_sample(logits, noise, temperature)


@triton.jit
def _folded_df2_latent(PROJECTED, PREVIOUS, PREDECESSOR, DF2_PROJECTION,
                       TOKEN_EMBED, A, B, OUT, POSITION,
                       RANK: tl.constexpr, ROWS: tl.constexpr, PROJECTED_ROW: tl.constexpr):
    sample = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, RANK)
    dtype = PROJECTED.dtype.element_ty
    previous = tl.load(PREVIOUS + sample)
    projected = tl.load(PROJECTED + sample * PROJECTED_ROW + cols).to(tl.float32)
    predecessor = tl.load(PREDECESSOR + previous * RANK + cols).to(tl.float32)
    # Match eager BF16 rounding after the Hadamard interaction and GEMV.
    interaction = (projected * predecessor).to(dtype).to(tl.float32)
    weight = tl.load(
        DF2_PROJECTION + rows[:, None] * RANK + cols[None],
        rows[:, None] < RANK,
        0,
    ).to(tl.float32)
    df2 = tl.sum(weight * interaction[None], axis=1).to(dtype).to(tl.float32)
    token = tl.load(
        TOKEN_EMBED + previous * RANK + rows,
        rows < RANK,
        0,
    ).to(tl.float32)
    a = tl.load(A + POSITION).to(dtype).to(tl.float32)
    b = tl.load(B + POSITION).to(dtype).to(tl.float32)
    left = (a * token).to(dtype).to(tl.float32)
    right = (b * df2).to(dtype).to(tl.float32)
    latent = (left + right).to(dtype)
    tl.store(OUT + sample * RANK + rows, latent, rows < RANK)


def folded_df2_latent(projected, previous, position, head):
    """Fuse token lookup, Hadamard state, rank GEMV, scaling, and addition."""
    rank = head.folded_out_proj.weight.shape[-1]
    if (
        projected.ndim != 2
        or projected.shape[-1] != rank
        or previous.shape != (projected.shape[0],)
        or projected.stride(-1) != 1
        or rank & (rank - 1)
    ):
        raise ValueError("folded DF2 latent inputs are invalid")
    output = torch.empty_like(projected)
    rows = 16
    _folded_df2_latent[(projected.shape[0], triton.cdiv(rank, rows))](
        projected,
        previous,
        head.predecessor_codebook.weight,
        head.df2_latent_projection.weight,
        head.folded_token_embed.weight,
        head.dspark_logit_scale,
        head.df2_logit_scale,
        output,
        position,
        rank,
        rows,
        projected.stride(0),
        enable_fp_fusion=False,
    )
    return output
