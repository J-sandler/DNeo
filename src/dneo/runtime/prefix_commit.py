"""Inference-only prefix/rejection K/V projection and cache commit."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_scatter(K, V, COS, SIN, POS, SLOTS, OK, OV,
                       K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                       V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                       C0: tl.constexpr, C1: tl.constexpr,
                       P0: tl.constexpr, P1: tl.constexpr,
                       O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
                       OV0: tl.constexpr, OV1: tl.constexpr, OV2: tl.constexpr,
                       HEADS: tl.constexpr, DIM: tl.constexpr,
                       INDEXED: tl.constexpr):
    batch_head, token = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // HEADS, batch_head % HEADS
    dim = tl.arange(0, DIM)
    rotate = (dim + DIM // 2) % DIM
    offset = batch * K0 + token * K1 + head * K2
    x = tl.load(K + offset + dim).to(tl.float32)
    xr = tl.load(K + offset + rotate).to(tl.float32)
    # Keep eager rotary's BF16 product rounding before the final sum.
    cos = tl.load(COS + batch*C0 + token*C1 + dim)
    sin = tl.load(SIN + batch*C0 + token*C1 + dim)
    key = (x*cos).to(K.dtype.element_ty).to(tl.float32) + (tl.where(dim<DIM//2,-xr,xr)*sin).to(K.dtype.element_ty).to(tl.float32)
    value = tl.load(V + batch*V0 + token*V1 + head*V2 + dim)
    slot = tl.load(SLOTS+batch) if INDEXED else batch
    pos = tl.load(POS+batch*P0+token*P1)
    tl.store(OK+slot*O0+head*O1+pos*O2+dim,key)
    tl.store(OV+slot*OV0+head*OV1+pos*OV2+dim,value)


def project_prefix_into_cache(layer, context, cos, sin, positions, output, slots=None):
    """Same raw projections; cache positions/slots must be valid and unique."""
    if layer.expanded_prefix_projection is not None:
        raise ValueError('fused commit does not support expanded prefix projection')
    k, v = layer._project_prefix_raw(context)
    # Native and fused RMSNorm can differ at BF16 rounding boundaries. Preserve
    # the caller's exact normalization implementation rather than replacing it.
    k = layer.k_norm(k)
    batch, tokens, heads, dim = k.shape
    ok, ov = output
    if (dim & (dim-1) or k.stride(-1)!=1 or v.stride(-1)!=1
            or positions.shape!=(batch,tokens) or cos.shape!=(batch,tokens,dim)
            or sin.shape!=cos.shape or sin.stride()!=cos.stride()
            or cos.stride(-1)!=1 or ok.stride(-1)!=1 or ov.stride(-1)!=1
            or ok.shape!=ov.shape or ok.shape[1]!=heads or ok.shape[-1]!=dim
            or (slots is not None and (slots.shape!=(batch,) or not slots.is_contiguous()))):
        raise ValueError('unsupported prefix commit layout')
    _rope_scatter[(batch*heads,tokens)](
        k,v,cos,sin,positions,slots,ok,ov,
        *k.stride()[:3],*v.stride()[:3],*cos.stride()[:2],*positions.stride(),
        *ok.stride()[:3],*ov.stride()[:3],heads,dim,
        slots is not None,num_warps=4,enable_fp_fusion=False)


@triton.jit
def _grouped_commit(KV, NORMALIZED, W, COS, SIN, POS, SLOTS, DEST,
                    TOKENS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                    C0: tl.constexpr, C1: tl.constexpr,
                    P0: tl.constexpr, P1: tl.constexpr,
                    O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
                    BATCH: tl.constexpr, EPS: tl.constexpr,
                    INDEXED: tl.constexpr, FUSED_NORM: tl.constexpr):
    layer_batch_head, token = tl.program_id(0), tl.program_id(1)
    layer = layer_batch_head // (BATCH * HEADS)
    batch_head = layer_batch_head % (BATCH * HEADS)
    batch, head = batch_head // HEADS, batch_head % HEADS
    dim = tl.arange(0, DIM)
    rotate = (dim + DIM // 2) % DIM
    row = (layer * BATCH + batch) * TOKENS + token
    offset = row * (2 * HEADS * DIM) + head * DIM
    dtype = KV.dtype.element_ty
    if FUSED_NORM:
        raw = tl.load(KV + offset + dim).to(tl.float32)
        variance = tl.sum(raw * raw, 0) / DIM
        norm = (raw * tl.rsqrt(variance + EPS)).to(dtype).to(tl.float32)
        weight = tl.load(W + layer * DIM + dim).to(tl.float32)
        key = (norm * weight).to(dtype)
    else:
        key = tl.load(NORMALIZED + row * HEADS * DIM + head * DIM + dim)
    rotated = tl.gather(key, rotate, 0).to(tl.float32)
    x = key.to(tl.float32)
    cos = tl.load(COS + batch * C0 + token * C1 + dim).to(tl.float32)
    sin = tl.load(SIN + batch * C0 + token * C1 + dim).to(tl.float32)
    key = (x * cos).to(dtype).to(tl.float32) + (
        tl.where(dim < DIM // 2, -rotated, rotated) * sin
    ).to(dtype).to(tl.float32)
    value = tl.load(KV + offset + HEADS * DIM + dim)
    slot = tl.load(SLOTS + batch) if INDEXED else batch
    position = tl.load(POS + batch * P0 + token * P1)
    out_k = tl.load(DEST + 2 * layer).to(tl.pointer_type(dtype))
    out_v = tl.load(DEST + 2 * layer + 1).to(tl.pointer_type(dtype))
    out_offset = slot * O0 + head * O1 + position * O2 + dim
    tl.store(out_k + out_offset, key)
    tl.store(out_v + out_offset, value)


class GroupedPrefixCommit:
    """Pack frozen layer K/V weights once; leave routed BF16 features unchanged."""

    @classmethod
    def supports(cls, layers, outputs):
        if not layers or len(layers) != len(outputs):
            return False
        first = layers[0]
        if first.k_norm.__class__.__name__ != 'Qwen3RMSNorm':
            return False
        signature = (first.num_key_value_heads, first.head_dim,
                     first.k_norm.variance_epsilon)
        shape, strides = outputs[0][0].shape, outputs[0][0].stride()
        device = first.k_proj.weight.device
        if len(shape) != 4 or shape[1] != signature[0] or shape[-1] != signature[1]:
            return False
        for layer, output in zip(layers, outputs):
            if (layer.expanded_prefix_projection is not None
                    or layer.k_norm.__class__.__name__ != 'Qwen3RMSNorm'
                    or (layer.num_key_value_heads, layer.head_dim,
                        layer.k_norm.variance_epsilon) != signature
                    or signature[1] & (signature[1] - 1)):
                return False
            projections = (layer.prefix_k_proj, layer.prefix_v_proj) if (
                layer.separate_prefix_postfix_kv) else (layer.k_proj, layer.v_proj)
            for projection in projections:
                if (projection.bias is not None or not projection.weight.is_cuda
                        or projection.weight.dtype != torch.bfloat16
                        or projection.weight.device != device
                        or projection.weight.shape != first.k_proj.weight.shape):
                    return False
            if (layer.k_norm.weight.shape != (signature[1],)
                    or layer.k_norm.weight.dtype != torch.bfloat16
                    or layer.k_norm.weight.device != device):
                return False
            if len(output) != 2 or any(x.shape != shape or x.stride() != strides
                   or x.dtype != torch.bfloat16 or not x.is_cuda
                   or x.device != device
                   for x in output):
                return False
        return strides[-1] == 1

    @torch.inference_mode()
    def __init__(self, layers, outputs):
        if not self.supports(layers, outputs):
            raise ValueError('unsupported grouped prefix commit architecture')
        self.layers, self.outputs = tuple(layers), tuple(outputs)
        packed = []
        for layer in layers:
            projections = (layer.prefix_k_proj, layer.prefix_v_proj) if (
                layer.separate_prefix_postfix_kv) else (layer.k_proj, layer.v_proj)
            packed.append(torch.cat([p.weight for p in projections], dim=0))
        self.weights = torch.stack(packed).transpose(1, 2)
        self.norm_weights = torch.stack([layer.k_norm.weight for layer in layers])
        self.destinations = torch.tensor(
            [x.data_ptr() for output in outputs for x in output],
            device=self.weights.device, dtype=torch.int64)
        self.heads, self.dim = layers[0].num_key_value_heads, layers[0].head_dim
        self.eps = layers[0].k_norm.variance_epsilon

    @torch.inference_mode()
    def __call__(self, contexts, cos, sin, positions, slots=None, *, fused_norm=False):
        batch, tokens, router_heads, width = contexts[0].shape
        if (len(contexts) != len(self.layers) or router_heads != 1
                or any(c.shape != contexts[0].shape or not c.is_contiguous()
                       for c in contexts)
                or width != self.weights.shape[1]
                or positions.shape != (batch, tokens)
                or cos.shape != (batch, tokens, self.dim)
                or sin.shape != cos.shape or sin.stride() != cos.stride()
                or cos.stride(-1) != 1
                or (slots is not None and (slots.shape != (batch,)
                                          or not slots.is_contiguous()))):
            raise ValueError('unsupported grouped prefix commit layout')
        # MoFA already emits one contiguous layered bank. View it without a copy.
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
        normalized = None
        if not fused_norm:
            key = kv[..., :self.heads * self.dim].reshape(
                len(contexts), batch * tokens, self.heads, self.dim)
            floating = key.float()
            variance = floating.pow(2).mean(-1, keepdim=True)
            normalized = ((floating * torch.rsqrt(variance + self.eps)).to(key.dtype)
                          * self.norm_weights[:, None, None]).contiguous()
        _grouped_commit[(len(contexts) * batch * self.heads, tokens)](
            kv, normalized, self.norm_weights, cos, sin, positions, slots,
            self.destinations, tokens, self.heads, self.dim,
            *cos.stride()[:2], *positions.stride(), *self.outputs[0][0].stride()[:3],
            batch, self.eps, slots is not None, fused_norm,
            num_warps=4, enable_fp_fusion=False)
