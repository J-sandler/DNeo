"""Experimental DF2 projection + pre-convolution epilogue for local draft rows."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(X, W, BASE, OUT, POST, ROWS: tl.constexpr, D: tl.constexpr,
             B: tl.constexpr, G: tl.constexpr, BM: tl.constexpr,
             GG: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    groups = tl.program_id(1) * GG + tl.arange(0, 4*GG) % GG
    tap = tl.arange(0, 4*GG) // GG
    ng = D // G
    cols = tap * ng + groups
    kk = tl.arange(0, BK)
    accum = tl.full((BM, 4*GG), 0., tl.float32)
    for start in range(tl.cdiv(D, BK)):
        inner = start * BK + kk
        x = tl.load(X + rows[:, None] * D + inner[None, :],
                    (rows[:, None] < ROWS) & (inner[None, :] < D), 0)
        w = tl.load(W + cols[None, :] * D + inner[:, None],
                    (groups[None, :] < ng) & (inner[:, None] < D), 0)
        accum += tl.dot(x, w, input_precision='ieee')
    delta = accum.to(X.dtype.element_ty).to(tl.float32)
    tl.store(POST + (rows[:, None] * 2 + tap[None, :] - 2) * ng + groups[None, :],
             delta, (rows[:, None] < ROWS) & (groups[None, :] < ng) & (tap[None, :] >= 2))
    local_cols = tl.arange(0, GG*G)
    channels = tl.program_id(1) * GG * G + local_cols
    group_index = tl.broadcast_to((local_cols // G)[None, :], (BM, GG*G))
    a = tl.gather(delta, group_index, 1)
    c = tl.gather(delta, group_index + GG, 1)
    dtype = X.dtype.element_ty
    a = (a + tl.load(BASE + channels, channels < D, 0)[None, :]).to(dtype).to(tl.float32)
    c = (c + tl.load(BASE + D + channels, channels < D, 0)[None, :]).to(dtype).to(tl.float32)
    current = tl.load(X + rows[:, None] * D + channels[None, :],
                      (rows[:, None] < ROWS) & (channels[None, :] < D), 0).to(tl.float32)
    previous = tl.load(X + (rows[:, None]-1) * D + channels[None, :],
                       (rows[:, None] < ROWS) & (rows[:, None] % B > 0) & (channels[None, :] < D), 0).to(tl.float32)
    result = (current*a).to(dtype).to(tl.float32) + (previous*c).to(dtype).to(tl.float32)
    tl.store(OUT + rows[:, None] * D + channels[None, :], result,
             (rows[:, None] < ROWS) & (channels[None, :] < D))


def fused_prepare(conv, hidden, block_size, row_tile=16):
    batch, n, d = hidden.shape
    if (n != block_size or not hidden.is_contiguous() or conv.group_size != 16
            or conv.kernel_projection.bias is not None or hidden.dtype != torch.bfloat16):
        raise ValueError('fused DF2 prepare requires contiguous BF16 local rows, G=16, no bias')
    out = torch.empty_like(hidden)
    post = torch.empty(batch, n, 2, d//16, device=hidden.device, dtype=hidden.dtype)
    _prepare[(triton.cdiv(batch*n,row_tile),triton.cdiv(d//16,16))](
        hidden,conv.kernel_projection.weight,conv.base_kernel[0],out,post,batch*n,d,block_size,16,
        row_tile,16,64,num_warps=4,num_stages=2,enable_fp_fusion=False,
    )
    return out,post
