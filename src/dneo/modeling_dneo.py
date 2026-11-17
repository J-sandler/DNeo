"""Inference model for the released folded-head, prefix-only, KV-only DNeo.

The state layout is identical to the trained checkpoint. Frozen target vocabulary
modules are bound explicitly and are never serialized into the draft checkpoint.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, Qwen3PreTrainedModel

from .configuration_dneo import DNeoConfig
from .reference import apply_rope, rejection_memory_attention_mask, target_attention_input_features


class DynamicConv(nn.Module):
    def __init__(self, hidden, group_size):
        super().__init__()
        self.hidden_size, self.group_size = hidden, group_size
        self.kernel_size, self.stack_depth, self.variant = 2, 1, "linear"
        self.post_delta_source, self.position_count = "pre", 0
        self.position_kernel_projection = None
        self.is_low_rank = self.is_dual_dependent = self.is_static_linear = False
        self.is_tap_local_successive = self.is_gated_hadamard = self.is_full_gated_mlp = False
        self.anchor_to_first_only = False
        self.base_kernel = nn.Parameter(torch.zeros(2, 2, hidden))
        self.kernel_projection = nn.Linear(hidden, 4 * (hidden // group_size), bias=False)

    def _convolve(self, hidden, dynamic, base, segments):
        from .reference import _grouped_dynamic_causal_convolution
        return _grouped_dynamic_causal_convolution(hidden, dynamic, base, self.group_size, segments, "linear")

    def prepare(self, hidden, segments):
        paired = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, 2, self.hidden_size // self.group_size)
        return self._convolve(hidden, paired[..., 0, :, :], self.base_kernel[0], segments), paired[..., 1, :, :]

    def finish(self, hidden, dynamic, segments):
        return self._convolve(hidden, dynamic, self.base_kernel[1], segments)


class DraftLayer(nn.Module):
    def __init__(self, config, block_size, group_size):
        super().__init__()
        self.num_attention_heads, self.num_key_value_heads = config.num_attention_heads, config.num_key_value_heads
        self.prefix_num_key_value_heads = self.num_key_value_heads
        self.head_dim, self.groups = config.head_dim, config.num_attention_heads // config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)
        self.attention_conv = DynamicConv(config.hidden_size, group_size)
        self.mlp_conv = DynamicConv(config.hidden_size, group_size)
        self.block_size, self.backend, self.dropout = block_size, "sdpa", 0.
        self.dflash2_convolutions, self.dflash2_conv_prefix_taps = True, 0
        self.dual_attention_normalization = self.separate_prefix_postfix_kv = False
        self.expanded_prefix_projection = self.mlp_intermediate_hadamard = self.postfix_attention = None
        self.prefix_k_proj = self.prefix_v_proj = self.grouped_online_router = None

    def _project_prefix_raw(self, hidden):
        k = self.k_proj.weight.view(self.num_key_value_heads, self.head_dim, hidden.shape[-1])
        v = self.v_proj.weight.view(self.num_key_value_heads, self.head_dim, hidden.shape[-1])
        if hidden.shape[2] == 1:
            return torch.einsum("bsd,hkd->bshk", hidden[:, :, 0], k), torch.einsum("bsd,hkd->bshk", hidden[:, :, 0], v)
        return torch.einsum("bshd,hkd->bshk", hidden, k), torch.einsum("bshd,hkd->bshk", hidden, v)

    def _project_prefix(self, hidden, cos, sin, **kwargs):
        key, value = self._project_prefix_raw(hidden)
        return apply_rope(self.k_norm(key).permute(0, 2, 1, 3), cos, sin), value.permute(0, 2, 1, 3)

    def _attention_once(self, query, key, value, mask):
        if key.shape[1] != query.shape[1]:
            key, value = key.repeat_interleave(self.groups, 1), value.repeat_interleave(self.groups, 1)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

    def _finish_attention_and_mlp(self, residual, attended, attention_kernel, segments, conv_draft_only=False):
        if conv_draft_only:
            raise ValueError("KV-only memory never enters draft convolution streams")
        hidden = residual + self.attention_conv.finish(self.o_proj(attended), attention_kernel, segments)
        normalized, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(hidden), segments)
        return hidden + self.mlp_conv.finish(self.mlp(normalized), kernel, segments)

    def forward(self, hidden, context, cos, sin, mask, context_cos=None, context_sin=None, dflash2_conv_segments=None, **kwargs):
        residual = hidden
        normalized, dynamic = self.attention_conv.prepare(self.input_layernorm(hidden), dflash2_conv_segments or (self.block_size,))
        batch, width, _ = hidden.shape
        query = self.q_norm(self.q_proj(normalized).view(batch, width, self.num_attention_heads, self.head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(normalized).view(batch, width, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(normalized).view(batch, width, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        pk, pv = self._project_prefix(context, context_cos, context_sin)
        attended = self._attention_once(query, torch.cat((pk, key), 2), torch.cat((pv, value), 2), mask)
        attended = attended.transpose(1, 2).reshape(batch, width, -1)
        return self._finish_attention_and_mlp(residual, attended, dynamic, dflash2_conv_segments or (self.block_size,))


class Conditioner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_target_layers, self.num_draft_layers = config.num_target_layers, config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.per_layer_mofa = True
        self.shared_layer_mofa = config.shared_layer_mofa
        router_layers = 1 if self.shared_layer_mofa else config.num_hidden_layers
        self.router_logits = nn.Parameter(torch.zeros(router_layers, 1, config.num_target_layers))
        self.register_buffer("fixed_layer_indices", torch.empty(0, dtype=torch.long))
        candidate_mask = torch.zeros(config.num_target_layers, dtype=torch.bool)
        candidate_mask[config.target_layer_ids] = True
        self.register_buffer("shared_mofa_candidate_mask", candidate_mask)
        self.register_buffer("shared_mofa_router_dual", torch.zeros(1))
        self.register_buffer("shared_mofa_router_dual_fixed", torch.zeros(1, dtype=torch.int64))
        self.rejection_router_logits = None

    def forward(self, target_hidden, **kwargs):
        values = self.router_logits.float().masked_fill(~self.shared_mofa_candidate_mask, torch.finfo(torch.float32).min)
        probabilities = values.softmax(-1)
        if self.shared_layer_mofa:
            context = torch.einsum("hl,bsld->bshd", probabilities[0].expand(self.num_kv_heads, -1).to(target_hidden.dtype), target_hidden)
            return (context,) * self.num_draft_layers
        return tuple(torch.einsum("hl,bsld->bshd", weights.expand(self.num_kv_heads, -1).to(target_hidden.dtype), target_hidden) for weights in probabilities)


class FoldedDenseDF2Head(nn.Module):
    folded_dense_df2 = dense_output = True

    def __init__(self, config):
        super().__init__()
        v, h, r, f = config.vocab_size, config.hidden_size, config.head_rank, config.folded_rank
        self.rank, self.folded_rank, self.positions = r, f, config.block_size - 1
        self.folded_token_embed = nn.Embedding(v, f)
        self.predecessor_codebook = nn.Embedding(v, r)
        self.hidden_projection = nn.Linear(h, r, bias=False)
        self.df2_latent_projection = nn.Linear(r, f, bias=False)
        self.folded_out_proj = nn.Linear(f, v, bias=False)
        self.dspark_logit_scale = nn.Parameter(torch.ones(self.positions))
        self.df2_logit_scale = nn.Parameter(torch.zeros(self.positions))
        self.teacher = None


class DraftBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        qwen = Qwen3Config(**config.target_config)
        self.layers = nn.ModuleList(DraftLayer(qwen, config.block_size, config.conv_group_size) for _ in range(config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=qwen.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(qwen)
        self.hidden_size, self.block_size = config.hidden_size, config.block_size
        self.rejection_memory = self.rejection_memory_kv_only = self.prefix_only_mask_attention = self.dflash2_convolutions = True
        self.is_mla = self.prefix_drafter_prefill = self.internal_mofa = self.intermediate_sequential_markov = False
        self.dual_attention_normalization = self.rejection_memory_layer_residual = self.bidirectional_mask_attention = False
        self.rejection_memory_fusion_mlp_hidden_size = self.dflash2_conv_prefix_taps = 0
        self.rejection_source_embeddings = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))

    def forward(self, block_embeddings, contexts, anchors, keep, prefix_valid, context_cos=None, context_sin=None,
                rejection_contexts=None, rejection_keep_mask=None, rejection_position_ids=None, **kwargs):
        batch, blocks, b, hidden = block_embeddings.shape
        width = 0 if rejection_keep_mask is None else rejection_keep_mask.shape[-1]
        device, prefix_length = block_embeddings.device, prefix_valid.shape[-1]
        queries = torch.arange(blocks * b, device=device) // b
        prefix_mask = (torch.arange(prefix_length, device=device)[None, None] < anchors[:, queries, None]) & prefix_valid[:, None].bool()
        if rejection_contexts is None:
            rejection_keep_mask = torch.zeros(batch, blocks, 0, device=device, dtype=torch.bool)
            rejection_position_ids = torch.zeros(batch, blocks, 0, device=device, dtype=torch.long)
            rejection_contexts = tuple(x.new_empty(batch, blocks, 0, x.shape[2], hidden) for x in contexts)
        memory_blocks = torch.arange(blocks * width, device=device) // max(width, 1)
        memory_mask = ((queries[:, None] == memory_blocks[None])[None]
                       & rejection_keep_mask.reshape(batch, 1, -1))
        local_index = torch.arange(blocks * b, device=device)
        local_mask = ((queries[:, None] == queries[None]) & ((local_index % b)[None] == 0))[None, None].expand(batch, -1, -1, -1)
        mask = torch.cat((prefix_mask[:, None], memory_mask[:, None], local_mask), -1)
        mask = mask & keep[:, queries, None][:, None]
        flat = block_embeddings.reshape(batch, blocks * b, hidden)
        positions = (anchors[..., None] + torch.arange(b, device=device)).reshape(batch, -1)
        cos, sin = self.rotary_emb(flat, positions)
        mc, ms = self.rotary_emb(flat, rejection_position_ids.reshape(batch, -1))
        for index, layer in enumerate(self.layers):
            memory = rejection_contexts[index].masked_fill(~rejection_keep_mask[..., None, None], 0).flatten(1, 2)
            context = torch.cat((contexts[index], memory), 1)
            flat = layer(flat, context, cos, sin, mask, context_cos=torch.cat((context_cos, mc), -2),
                         context_sin=torch.cat((context_sin, ms), -2), dflash2_conv_segments=(b,))
        flat = flat + 0.0 * self.rejection_source_embeddings.sum()
        return self.norm(flat).reshape(batch, blocks, b, hidden)


class DNeoDraftModel(Qwen3PreTrainedModel):
    config_class = DNeoConfig
    base_model_prefix = ""
    _no_split_modules = ["DraftLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.draft = DraftBackbone(config)
        self.conditioner = Conditioner(config)
        self.ar_head = FoldedDenseDF2Head(config)
        self.block_size, self.mask_token_id = config.block_size, config.mask_token_id
        object.__setattr__(self, "embed_tokens", None)
        object.__setattr__(self, "lm_head", None)
        self.post_init()
        # The trained loader casts these non-persistent buffers too. HF's
        # parameter dtype alone does not reproduce their rounded frequencies.
        if config.rotary_frequency_dtype is not None:
            self.draft.rotary_emb.to(dtype=getattr(torch, config.rotary_frequency_dtype))

    def bind_target(self, target):
        embedding = target.get_input_embeddings()
        head = target.get_output_embeddings() if hasattr(target, "get_output_embeddings") else target.lm_head
        if embedding.weight.shape != (self.config.vocab_size, self.config.hidden_size) or head.weight.shape != embedding.weight.shape:
            raise ValueError("target vocabulary/width does not match the trained DNeo interface")
        embedding_view = nn.Embedding.from_pretrained(embedding.weight.detach(), freeze=True)
        head_view = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False,
                              device="meta", dtype=head.weight.dtype)
        head_view.weight = nn.Parameter(head.weight.detach(), requires_grad=False)
        object.__setattr__(self, "embed_tokens", embedding_view)
        object.__setattr__(self, "lm_head", head_view)
        return self

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, block_embeddings, prefix_features, anchor_positions, memory_features=None,
                memory_valid=None, memory_positions=None):
        batch, blocks, b, _ = block_embeddings.shape
        length = prefix_features.shape[1]
        positions = torch.arange(length, device=prefix_features.device)[None].expand(batch, -1)
        cos, sin = self.draft.rotary_emb(prefix_features[:, :, 0], positions)
        contexts = self.conditioner(prefix_features)
        memory = None
        if memory_features is not None:
            flat = memory_features.flatten(1, 2)
            memory = tuple(x.reshape(batch, blocks, memory_valid.shape[-1], *x.shape[2:]) for x in self.conditioner(flat))
        return self.draft(block_embeddings, contexts, anchor_positions,
                          torch.ones(batch, blocks, device=block_embeddings.device, dtype=torch.bool),
                          torch.ones(batch, length, device=block_embeddings.device, dtype=torch.bool),
                          context_cos=cos, context_sin=sin, rejection_contexts=memory,
                          rejection_keep_mask=memory_valid, rejection_position_ids=memory_positions)
