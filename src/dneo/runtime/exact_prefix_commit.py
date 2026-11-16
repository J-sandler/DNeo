"""Grouped commit with PyTorch-compatible vectorized RMS reduction."""
import torch
import triton
import triton.language as tl

from dneo.runtime.prefix_commit import GroupedPrefixCommit


@triton.jit
def _torch_norm_commit(KV, W, COS, SIN, POS, SLOTS, DEST,
                       TOKENS: tl.constexpr, HEADS: tl.constexpr,
                       C0: tl.constexpr, C1: tl.constexpr,
                       P0: tl.constexpr, P1: tl.constexpr,
                       O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
                       BATCH: tl.constexpr, EPS: tl.constexpr, INDEXED: tl.constexpr):
    lane, token = tl.program_id(0), tl.program_id(1)
    layer = lane // (BATCH * HEADS)
    batch_head = lane % (BATCH * HEADS)
    batch, head = batch_head // HEADS, batch_head % HEADS
    dim = tl.arange(0, 128)
    row = (layer * BATCH + batch) * TOKENS + token
    offset = row * (2 * HEADS * 128) + head * 128
    dtype = KV.dtype.element_ty
    raw = tl.load(KV + offset + dim).to(tl.float32)
    # CUDA Reduce.cuh combines four adjacent accumulators before the warp tree.
    even, odd = tl.split(tl.reshape(raw, (64, 2)))
    x0, x2 = tl.split(tl.reshape(even, (32, 2)))
    x1, x3 = tl.split(tl.reshape(odd, (32, 2)))
    partial = ((x0 * x0 + x1 * x1) + x2 * x2) + x3 * x3
    variance = tl.sum(partial, 0) * (1.0 / 128)
    norm = (raw * tl.rsqrt(variance + EPS)).to(dtype).to(tl.float32)
    weight = tl.load(W + layer * 128 + dim).to(tl.float32)
    key = (norm * weight).to(dtype)
    rotate = (dim + 64) % 128
    rotated = tl.gather(key, rotate, 0).to(tl.float32)
    cos = tl.load(COS + batch * C0 + token * C1 + dim).to(tl.float32)
    sin = tl.load(SIN + batch * C0 + token * C1 + dim).to(tl.float32)
    result = (key.to(tl.float32) * cos).to(dtype).to(tl.float32) + (
        tl.where(dim < 64, -rotated, rotated) * sin).to(dtype).to(tl.float32)
    value = tl.load(KV + offset + HEADS * 128 + dim)
    slot = tl.load(SLOTS + batch) if INDEXED else batch
    position = tl.load(POS + batch * P0 + token * P1)
    out_k = tl.load(DEST + 2 * layer).to(tl.pointer_type(dtype))
    out_v = tl.load(DEST + 2 * layer + 1).to(tl.pointer_type(dtype))
    address = slot * O0 + head * O1 + position * O2 + dim
    tl.store(out_k + address, result)
    tl.store(out_v + address, value)


class TorchNormPrefixCommit(GroupedPrefixCommit):
    """Use the exact vectorized 128-wide norm; retain other-layout fallbacks."""

    @torch.inference_mode()
    def __call__(self, contexts, cos, sin, positions, slots=None, *, fused_norm=False):
        if fused_norm or self.dim != 128:
            return super().__call__(contexts, cos, sin, positions, slots,
                                    fused_norm=fused_norm)
        batch, tokens, heads, width = contexts[0].shape
        if (len(contexts) != len(self.layers) or heads != 1 or width != self.weights.shape[1]
                or any(c.shape != contexts[0].shape or not c.is_contiguous()
                       for c in contexts)
                or positions.shape != (batch, tokens)
                or cos.shape != (batch, tokens, self.dim)
                or sin.shape != cos.shape or sin.stride() != cos.stride()
                or cos.stride(-1) != 1
                or (slots is not None and (slots.shape != (batch,)
                                          or not slots.is_contiguous()))):
            return super().__call__(contexts, cos, sin, positions, slots,
                                    fused_norm=fused_norm)
        storage = contexts[0].untyped_storage().data_ptr()
        elements = batch * tokens * width
        if all(c.untyped_storage().data_ptr() == storage
               and c.storage_offset() == contexts[0].storage_offset() + i * elements
               for i, c in enumerate(contexts)):
            inputs = contexts[0].as_strided(
                (len(contexts), batch * tokens, width), (elements, width, 1))
        else:
            inputs = torch.stack(contexts).view(len(contexts), batch * tokens, width)
        kv = torch.bmm(inputs, self.weights)
        _torch_norm_commit[(len(contexts) * batch * self.heads, tokens)](
            kv, self.norm_weights, cos, sin, positions, slots, self.destinations,
            tokens, self.heads, *cos.stride()[:2], *positions.stride(),
            *self.outputs[0][0].stride()[:3], batch, self.eps, slots is not None,
            num_warps=4, enable_fp_fusion=False)
