"""Verbatim inference arithmetic extracted from the trained reference."""
from __future__ import annotations
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (x * cos.unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(1))


def rejection_memory_attention_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    input_attention_mask: torch.Tensor,
    rejection_keep_mask: torch.Tensor,
    block_size: int,
    prefix_only: bool = False,
    bidirectional: bool = False,
) -> torch.Tensor:
    """Prefix-causal, block-local bidirectional mask/rejection attention."""
    batch, blocks = anchor_positions.shape
    if rejection_keep_mask.ndim != 3 or rejection_keep_mask.shape[:2] != (
        batch,
        blocks,
    ):
        raise ValueError("rejection validity must be [batch, blocks, memory]")
    memory = rejection_keep_mask.shape[2]
    expanded_size = int(block_size) + memory
    query_length = blocks * expanded_size
    context_length = input_attention_mask.shape[1]
    device = anchor_positions.device

    query_index = torch.arange(query_length, device=device).view(1, 1, -1, 1)
    query_block = query_index // expanded_size
    query_offset = query_index % expanded_size
    context_index = torch.arange(context_length, device=device).view(1, 1, 1, -1)
    expanded_anchors = anchor_positions.view(batch, 1, blocks, 1).repeat_interleave(
        expanded_size,
        dim=2,
    )
    context_mask = (context_index < expanded_anchors) & input_attention_mask[
        :, None, None, :
    ].bool()

    key_index = torch.arange(query_length, device=device).view(1, 1, 1, -1)
    key_block = key_index // expanded_size
    key_offset = key_index % expanded_size
    same_block = query_block == key_block
    query_is_local = query_offset < int(block_size)
    key_is_local = key_offset < int(block_size)
    if prefix_only and bidirectional:
        raise ValueError("prefix-only and bidirectional attention conflict")
    local_causal = (
        key_offset == 0 if prefix_only
        else torch.ones_like(key_offset, dtype=torch.bool) if bidirectional
        else key_offset <= query_offset
    )
    expanded_valid = torch.cat(
        (
            block_keep_mask[:, :, None].expand(-1, -1, int(block_size)),
            rejection_keep_mask.bool(),
        ),
        dim=2,
    ).reshape(batch, query_length)
    key_valid = expanded_valid[:, None, None, :]
    local_mask = same_block & (
        (query_is_local & key_is_local & local_causal)
        | (query_is_local & ~key_is_local)
        | (~query_is_local & key_is_local)
        | (~query_is_local & ~key_is_local)
    ) & key_valid
    return torch.cat((context_mask, local_mask), dim=-1) & expanded_valid[
        :, None, :, None
    ]


def target_attention_input_features(
    backbone: nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    *,
    detach: bool = True,
    num_layers: Optional[int] = None,
) -> torch.Tensor:
    """Return the normalized residual consumed by each verifier attention layer."""
    layers = tuple(backbone.layers)
    if num_layers is not None:
        if not 0 < int(num_layers) <= len(layers):
            raise ValueError(
                f"requested {num_layers} features from {len(layers)} verifier layers"
            )
        layers = layers[: int(num_layers)]
    if len(hidden_states) < len(layers):
        raise ValueError(
            "verifier did not return an input residual for every attention layer"
        )
    features = torch.stack(
        tuple(
            layer.input_layernorm(hidden_states[layer_index])
            for layer_index, layer in enumerate(layers)
        ),
        dim=2,
    )
    return features.detach() if detach else features


def _grouped_dynamic_causal_convolution(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel: torch.Tensor,
    group_size: int,
    segments: tuple[int, ...],
    coefficient_gate: str = "linear",
) -> torch.Tensor:
    """Apply a depthwise causal kernel without crossing packed block streams."""
    if hidden.ndim != 3 or dynamic.ndim != 4:
        raise ValueError("DFlash2 convolution expects [batch, tokens, hidden]")
    batch, length, hidden_size = hidden.shape
    if base_kernel.ndim == 2:
        kernel_size = int(base_kernel.shape[0])
    elif base_kernel.ndim == 3:
        kernel_size = int(base_kernel.shape[1])
    else:
        raise ValueError("DFlash2 base kernel must be [taps, hidden] or [positions, taps, hidden]")
    groups = hidden_size // int(group_size)
    if dynamic.shape != (batch, length, kernel_size, groups):
        raise ValueError("DFlash2 dynamic kernels do not align with hidden states")
    block_width = sum(int(width) for width in segments)
    if block_width <= 0 or length % block_width:
        raise ValueError("DFlash2 convolution layout does not tile packed tokens")
    blocks = length // block_width
    hidden_blocks = hidden.view(batch, blocks, block_width, groups, group_size)
    dynamic_blocks = dynamic.view(batch, blocks, block_width, kernel_size, groups)
    output = torch.zeros_like(hidden_blocks)
    start = 0
    for width in segments:
        width = int(width)
        if width <= 0:
            continue
        values = hidden_blocks[:, :, start : start + width]
        corrections = dynamic_blocks[:, :, start : start + width]
        for offset in range(kernel_size):
            if offset == 0:
                shifted = values
            elif offset >= width:
                shifted = torch.zeros_like(values)
            else:
                shifted = F.pad(
                    values[:, :, : width - offset],
                    (0, 0, 0, 0, offset, 0),
                )
            if base_kernel.ndim == 2:
                base = base_kernel[offset].view(1, 1, 1, groups, group_size)
            else:
                if int(base_kernel.shape[0]) != block_width:
                    raise ValueError(
                        "position-specific DFlash2 kernel does not match packed block width"
                    )
                base = base_kernel[start : start + width, offset].view(
                    1, 1, width, groups, group_size
                )
            correction = corrections[..., offset, :]
            if coefficient_gate == "tap-local-sigmoid-gated" and offset:
                if offset >= width:
                    correction = torch.zeros_like(correction)
                else:
                    correction = F.pad(
                        correction[:, :, : width - offset],
                        (0, 0, offset, 0),
                    )
            correction = correction.unsqueeze(-1)
            coefficient = base + correction
            if coefficient_gate in {
                "sigmoid-gated",
                "tap-local-sigmoid-gated",
            }:
                coefficient = torch.sigmoid(coefficient)
            elif coefficient_gate == "relu-gated":
                coefficient = F.relu(coefficient)
            elif coefficient_gate == "pre-relu-gated":
                coefficient = base + F.relu(correction)
            elif coefficient_gate != "linear":
                raise ValueError(
                    f"unsupported DFlash2 coefficient gate: {coefficient_gate}"
                )
            output[:, :, start : start + width] += shifted * coefficient
        start += width
    return output.view(batch, length, hidden_size)

