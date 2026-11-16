"""Joint FA3 attention for a dense native prefix/rejection-cache suffix.

Experimental until trained-checkpoint numerical and latency qualification passes.
Only the first unused cache slot is scratch; live prefix/memory K/V stay intact.
"""
import torch
import triton
import triton.language as tl


def validate_dense_layout(prefix_lengths, positions, valid, capacity, request_slots=None):
    if positions.shape != valid.shape or positions.shape[0] != prefix_lengths.numel():
        raise ValueError("dense memory metadata has incompatible dimensions")
    if valid.dtype != torch.bool or prefix_lengths.ndim != 1:
        raise ValueError("dense memory needs boolean validity and one prefix length per row")
    count = valid.sum(-1)
    expected = prefix_lengths[:, None] + valid.long().cumsum(-1) - 1
    if bool(((positions != expected) & valid).any()):
        raise ValueError("valid rejection memory must be an ordered contiguous suffix")
    if bool(((prefix_lengths < 0) | (prefix_lengths + count >= capacity)).any()):
        raise ValueError("dense memory cache has no spare anchor slot")
    if request_slots is not None and request_slots.unique().numel() != request_slots.numel():
        raise ValueError("concurrent dense attention requests need distinct cache slots")


@triton.jit
def _stage_anchor(AK, AV, PK, PV, LENGTH, VALID, SLOTS, TOTAL,
                  A0: tl.constexpr, A1: tl.constexpr,
                  AV0: tl.constexpr, AV1: tl.constexpr,
                  K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                  V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                  M0: tl.constexpr, M1: tl.constexpr,
                  MEMORY: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr,
                  INDEXED: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    memory = tl.arange(0, TILE)
    live = tl.load(VALID + row * M0 + memory * M1, memory < MEMORY, 0)
    end = tl.load(LENGTH + row) + tl.sum(live.to(tl.int32), 0)
    slot = tl.load(SLOTS + row) if INDEXED else row
    dim = tl.arange(0, D)
    key = tl.load(AK + row * A0 + head * A1 + dim)
    value = tl.load(AV + row * AV0 + head * AV1 + dim)
    tl.store(PK + slot * K0 + head * K1 + end * K2 + dim, key)
    tl.store(PV + slot * V0 + head * V1 + end * V2 + dim, value)
    if head == 0:
        tl.store(TOTAL + row, end + 1)


def dense_cached_attention(q, prefix_kv, anchor_k, anchor_v, lengths,
                           positions, valid, request_slots=None, *, validate=True):
    """Attend jointly to prefix, contiguous memory, and one scratch anchor.

    Native commit guarantees dense valid memory beginning at the current anchor.
    Other callers must validate that invariant; arbitrary sparse memory is not
    supported by this specialized kernel. The scratch slot is overwritten by
    subsequent verifier-feature commit before it can become live prefix/memory.
    """
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

    pk, pv = prefix_kv
    batch, _, _, width = q.shape
    if (anchor_k.shape != anchor_v.shape or anchor_k.shape[0] != batch
            or anchor_k.shape[2] != 1 or pk.shape != pv.shape
            or anchor_k.shape[1] != pk.shape[1] or width != pk.shape[-1]
            or lengths.shape != (batch,) or positions.shape != valid.shape
            or valid.shape[0] != batch or lengths.dtype != torch.int32
            or any(x.stride(-1) != 1 for x in (q, pk, pv, anchor_k, anchor_v))):
        raise ValueError("unsupported joint-cache attention layout")
    if validate:
        validate_dense_layout(lengths, positions, valid, pk.shape[2], request_slots)
        if request_slots is not None and bool(((request_slots < 0) | (request_slots >= pk.shape[0])).any()):
            raise ValueError("cache request slot is outside allocated storage")
    total = torch.empty(batch, device=q.device, dtype=torch.int32)
    _stage_anchor[(batch, anchor_k.shape[1])](
        anchor_k, anchor_v, pk, pv, lengths, valid, request_slots, total,
        *anchor_k.stride()[:2], *anchor_v.stride()[:2],
        *pk.stride()[:3], *pv.stride()[:3], *valid.stride(),
        valid.shape[1], width, triton.next_power_of_2(max(1, valid.shape[1])),
        request_slots is not None, num_warps=4)
    output = flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=pk.transpose(1, 2), v_cache=pv.transpose(1, 2),
        cache_seqlens=total, cache_batch_idx=request_slots,
        causal=False, return_softmax_lse=True, num_splits=1, ver=3)
    return output[0].transpose(1, 2)
