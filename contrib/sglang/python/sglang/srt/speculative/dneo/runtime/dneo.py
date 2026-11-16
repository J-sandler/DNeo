"""Exact single-request dNEO speculative decoding, separate from training.

The target cache ends immediately BEFORE the next anchor. Verification processes
the anchor plus all proposals once; accepted states extend the prefix, while the
rejected suffix is retained as memory for the next draft. See the packaged source_manifest.json for
the pinned DFlash/DF2/dSparK implementations used to audit this protocol.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def probabilities(logits, temperature=1.0, top_p=1.0):
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid sampling parameters")
    if temperature == 0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    p = (logits.float() / temperature).softmax(-1)
    if top_p < 1:
        ordered, ids = p.sort(descending=True, dim=-1)
        ordered = ordered.masked_fill(ordered.cumsum(-1) - ordered >= top_p, 0)
        p = torch.zeros_like(p).scatter(-1, ids, ordered)
        p = p / p.sum(-1, keepdim=True)
    return p


def categorical(p, generator=None):
    return torch.multinomial(p, 1, generator=generator).squeeze(-1)


def block_keep_probabilities(draft_ids, q, p):
    """Probability that Block Verification retains each sampled sub-block."""
    qt = q.gather(-1, draft_ids[:, None]).squeeze(-1)
    pt = p[:-1].gather(-1, draft_ids[:, None]).squeeze(-1)
    keep = [torch.ones((), device=q.device, dtype=torch.float32)]
    for ratio in pt.float() / qt.float().clamp_min(torch.finfo(torch.float32).tiny):
        keep.append((keep[-1] * ratio).clamp_max(1.0))
    return torch.stack(keep)


def expected_accepted_tokens(draft_ids, q, p, method="leviathan"):
    """Conditional expectation for one already-sampled draft path."""
    if method == "block":
        return block_keep_probabilities(draft_ids, q, p)[1:].sum()
    if method != "leviathan":
        raise ValueError(f"unknown verification method {method}")
    qt = q.gather(-1, draft_ids[:, None]).squeeze(-1)
    pt = p[:-1].gather(-1, draft_ids[:, None]).squeeze(-1)
    token_accept = (pt.float() / qt.float().clamp_min(torch.finfo(torch.float32).tiny)).clamp_max(1.0)
    return token_accept.cumprod(0).sum()


def _block_verify_reference(draft_ids, q, p, generator=None, uniforms=None, replacement_noise=None):
    """Paper-faithful reference used on CPU and to audit the CUDA kernel."""
    gamma, vocab = q.shape
    keep = block_keep_probabilities(draft_ids, q, p)
    residual_mass = (keep[:-1, None] * p[:-1].float() - q.float()).clamp_min(0).sum(-1)
    denominator = residual_mass[1:] + 1.0 - keep[1:-1]
    h = torch.where(denominator > 0, residual_mass[1:] / denominator, torch.ones_like(denominator))
    h = torch.cat((h, keep[-1:]))
    if uniforms is None:
        uniforms = torch.rand(gamma, device=q.device, generator=generator)
    accepted = uniforms <= h
    accepted_indices = torch.arange(1, gamma + 1, device=q.device).masked_fill(~accepted, 0)
    count = int(accepted_indices.max().item())
    if count == gamma:
        weights = p[-1].float()
    else:
        weights = (keep[count] * p[count].float() - q[count].float()).clamp_min(0)
    mass = weights.sum()
    if not bool(mass > 0):
        raise RuntimeError("zero replacement residual in block verification")
    if replacement_noise is None:
        replacement = categorical(weights / mass, generator).reshape(1)
    else:
        if replacement_noise.shape != (vocab,):
            raise ValueError("replacement noise must have one value per vocabulary item")
        score = torch.where(weights > 0, weights.log() - replacement_noise.float().clamp_min(
            torch.finfo(torch.float32).tiny).log(), -torch.inf)
        replacement = score.argmax().reshape(1)
    return torch.cat((draft_ids[:count], replacement)), count


def block_verify(draft_ids, q, p, generator=None, uniforms=None, replacement_noise=None):
    """Lossless Block Verification from Sun et al., arXiv:2403.10444."""
    gamma = draft_ids.numel()
    if q.shape != p[:-1].shape or q.shape[0] != gamma:
        raise ValueError("verification dimensions do not align")
    if q.is_cuda:
        from sglang.srt.speculative.dneo.runtime.kernels import block_verify_cuda
        return block_verify_cuda(draft_ids, q, p, generator, uniforms, replacement_noise)
    return _block_verify_reference(draft_ids, q, p, generator, uniforms, replacement_noise)


def verify(draft_ids, q, p, generator=None, uniforms=None, method="leviathan", replacement_noise=None):
    """Leviathan verification, including the positive residual and bonus.

    q: [gamma, vocab], p: [gamma+1, vocab]. Returned count excludes the bonus
    or replacement. This operation is independent of how drafts were obtained.
    """
    if method == "block":
        return block_verify(draft_ids, q, p, generator, uniforms, replacement_noise)
    if method != "leviathan":
        raise ValueError(f"unknown verification method {method}")
    gamma = draft_ids.numel()
    if q.shape != p[:-1].shape or q.shape[0] != gamma:
        raise ValueError("verification dimensions do not align")
    qt = q.gather(-1, draft_ids[:, None]).squeeze(-1)
    pt = p[:-1].gather(-1, draft_ids[:, None]).squeeze(-1)
    if uniforms is None:
        uniforms = torch.rand(gamma, device=q.device, generator=generator)
    accepted = uniforms < (pt / qt.clamp_min(torch.finfo(q.dtype).tiny)).clamp_max(1)
    count = int(accepted.long().cumprod(0).sum().item())
    if count == gamma:
        next_p = p[-1]
    else:
        residual = (p[count] - q[count]).clamp_min(0)
        # A rejection has positive residual mass mathematically. A zero mass
        # indicates corrupt/inconsistent distributions, not a reason to use p.
        mass = residual.sum()
        if not bool(mass > 0):
            raise RuntimeError("zero replacement residual after rejection")
        next_p = residual / mass
    replacement = categorical(next_p, generator).reshape(1)
    return torch.cat((draft_ids[:count], replacement)), count


class Timer:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.events = []

    @contextlib.contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        yield
        end.record()
        self.events.append((name, begin, end))

    def milliseconds(self):
        torch.cuda.synchronize()
        result = {}
        for name, begin, end in self.events:
            result[name] = result.get(name, 0.0) + begin.elapsed_time(end)
        return result


from sglang.srt.speculative.dneo.loading import load_checkpoint

class HFTarget:
    """Incremental Hugging Face verifier with attention-input feature capture."""

    def __init__(self, target, capture_features=True, feature_layer_ids=None):
        self.target = target
        self.backbone = target.model
        self.cache = None
        self.capture_features = capture_features
        self.feature_capture = None
        if capture_features and feature_layer_ids is not None:
            from sglang.srt.speculative.dneo.runtime.feature_layout import NormalizedFeatureCapture
            self.feature_capture = NormalizedFeatureCapture(self.backbone, feature_layer_ids)

    def close(self):
        if self.feature_capture is not None:
            self.feature_capture.close()

    def reset(self):
        from transformers import DynamicCache
        self.cache = DynamicCache()

    @torch.inference_mode()
    def forward(self, ids, last_logit_only=False):
        from sglang.srt.speculative.dneo.modeling_dneo import target_attention_input_features
        output = self.backbone(
            input_ids=ids, past_key_values=self.cache, use_cache=True,
            output_hidden_states=self.capture_features and self.feature_capture is None, return_dict=True,
        )
        self.cache = output.past_key_values
        features = (self.feature_capture.stack() if self.feature_capture is not None else
                    target_attention_input_features(self.backbone, output.hidden_states)
                    if self.capture_features else None)
        hidden = output.last_hidden_state[:, -1:] if last_logit_only else output.last_hidden_state
        logits = self.target.lm_head(hidden)
        return logits, features

    def crop(self, length):
        self.cache.crop(length)


HEAD_MODES = {"auto", "folded-df2", "dense-mixed", "weighted-dspark-only", "sparse-top16-mixed"}


class MixedHead(torch.nn.Module):
    """Serving-only dSparK/DF2 ablations, with no training metric caches."""

    def __init__(self, head, fused=False, cache_slots=0, mode="dense-mixed"):
        super().__init__()
        if mode not in HEAD_MODES:
            raise ValueError(f"unknown dNEO head mode {mode}")
        self.head = head
        self.fused = fused
        self.mode = mode
        self.correction_cache = None
        if cache_slots and mode == "dense-mixed":
            w = head.dspark_out_proj.weight
            self.correction_cache = (
                torch.full((cache_slots,), -1, device=w.device, dtype=torch.int32),
                torch.empty((cache_slots, w.shape[0]), device=w.device, dtype=w.dtype),
                torch.zeros(2, device=w.device, dtype=torch.int32),
            )

    def forward(self, base, hidden, previous, position):
        h = self.head
        if self.mode == "sparse-top16-mixed":
            raise ValueError("sparse top-16 mode returns candidate scores, not dense logits")
        if self.mode == "dense-mixed" and self.fused and base.is_cuda:
            from sglang.srt.speculative.dneo.runtime.kernels import mixed_head
            return mixed_head(
                base.contiguous(), hidden.contiguous(), previous, position,
                h, self.correction_cache,
            )
        a = h.dspark_logit_scale[position].to(base.dtype)
        u = h.dspark_token_embed(previous)
        dspark = F.linear(u, h.dspark_out_proj.weight)
        if self.mode == "weighted-dspark-only":
            return a * (base + dspark)
        b = h.df2_logit_scale[position].to(base.dtype)
        v = h.selector.predecessor_codebook(previous) * hidden
        return (a + b) * base + a * dspark + b * F.linear(
            v, h.selector.successor_codebook.weight)

    def project_hidden(self, hidden):
        return self.head.selector.hidden_projection(hidden)

    def sparse_scores(self, base, hidden, previous, position):
        """Score the marginal top-K only, matching the DF2 candidate support."""
        if self.mode != "sparse-top16-mixed":
            raise ValueError("candidate scoring requires sparse-top16-mixed mode")
        h = self.head
        candidates = torch.topk(base, h.selector.top_k, dim=-1, sorted=False).indices
        unary = base.gather(-1, candidates)
        successor = F.embedding(candidates, h.selector.successor_codebook.weight)
        dspark_out = F.embedding(candidates, h.dspark_out_proj.weight)
        dspark_state = h.dspark_token_embed(previous)
        selector_state = h.selector.predecessor_codebook(previous) * hidden
        dspark_scores = torch.einsum("br,bkr->bk", dspark_state, dspark_out)
        selector_scores = torch.einsum("br,bkr->bk", selector_state, successor)
        a = h.dspark_logit_scale[position].to(base.dtype)
        b = h.df2_logit_scale[position].to(base.dtype)
        scores = (a + b) * unary + a * dspark_scores + b * selector_scores
        return candidates, scores


class FoldedDF2Head(torch.nn.Module):
    """Serving wrapper for the adapted single-projection dense DF2 head."""

    def __init__(self, head, fused=False):
        super().__init__()
        if not getattr(head, "folded_dense_df2", False):
            raise ValueError("folded-df2 serving requires a FoldedDenseDF2Head")
        if getattr(head, "teacher", None) is not None:
            raise ValueError("serving checkpoint must not contain the distillation teacher")
        self.head = head
        self.fused = fused
        self.correction_cache = None

    def project_hidden(self, hidden):
        return self.head.hidden_projection(hidden)

    def latent(self, projected, previous, position):
        h = self.head
        interaction = h.predecessor_codebook(previous) * projected
        df2 = F.linear(interaction, h.df2_latent_projection.weight)
        a = h.dspark_logit_scale[position].to(projected.dtype)
        b = h.df2_logit_scale[position].to(projected.dtype)
        return a * h.folded_token_embed(previous) + b * df2

    def forward(self, base, projected, previous, position):
        h = self.head
        latent = self.latent(projected, previous, position)
        a = h.dspark_logit_scale[position].to(base.dtype)
        b = h.df2_logit_scale[position].to(base.dtype)
        if self.fused and base.is_cuda:
            from sglang.srt.speculative.dneo.runtime.kernels import folded_df2_head
            return folded_df2_head(base.contiguous(), latent.contiguous(), position, h)
        return (a + b) * base + F.linear(latent, h.folded_out_proj.weight)


class DNeoRuntime:
    def __init__(self, target, model, variant="reference", profile=False, target_backend=None, context_capacity=1024,
                 optimize_backbone=False, optimize_head=None, split_memory_attention=False,
                 verification_method="leviathan", head_mode="auto", batch_size=1,
                 target_graph_widths=None, draft_graph_capacities=None,
                 draft_kernel="auto", attention_splits=1, fused_verification=False,
                 prepare_prefix_graphs=True):
        if draft_kernel in {"auto", "hardware-auto"}:
            from sglang.srt.speculative.dneo.runtime.structural_draft import default_draft_kernel
            draft_kernel = default_draft_kernel(model.embed_tokens.weight.device) if variant == "static-prefix" else "reference"
        if optimize_head is None:
            optimize_head = variant == "static-prefix"
        if variant not in {"reference", "cached", "fused-head", "compiled", "triton", "target-graph", "full-graph", "pruned", "head-graph", "head-cache", "io-graph", "fused-draft", "static-prefix"}:
            raise ValueError(f"unknown variant {variant}")
        self.model = model
        self.fused_verification = bool(fused_verification)
        self.optimize_backbone = optimize_backbone
        self.optimize_head = optimize_head
        self.split_memory_attention = split_memory_attention
        if verification_method not in {"leviathan", "block"}:
            raise ValueError(f"unknown verification method {verification_method}")
        self.verification_method = verification_method
        if head_mode not in HEAD_MODES:
            raise ValueError(f"unknown dNEO head mode {head_mode}")
        folded_head = bool(getattr(model.ar_head, "folded_dense_df2", False))
        if head_mode == "auto":
            head_mode = "folded-df2" if folded_head else "dense-mixed"
        if folded_head != (head_mode == "folded-df2"):
            raise ValueError(
                f"head mode {head_mode!r} is incompatible with "
                f"{'folded' if folded_head else 'legacy joint'} checkpoint"
            )
        self.head_mode = head_mode
        if split_memory_attention and not optimize_backbone:
            raise ValueError("split memory attention requires backbone optimization")
        if (optimize_backbone or optimize_head) and variant != "static-prefix":
            raise ValueError("optimized serving requires the static-prefix variant")
        from sglang.srt.speculative.dneo.runtime.feature_layout import feature_layer_ids, router_weights
        conditioner = model.conditioner
        extra_modes = ("kv_reuse", "mla", "fixed_mofa_layers", "simple_grouped_mofa",
                       "global_grouped_mofa", "grouped_online_mofa", "mofa_mlp",
                       "dflash_feature_projection", "fused_activation_mofa",
                       "dflash_feature_sum", "temporal_feature_conditioning", "gru_augmentation")
        shared = (getattr(conditioner, "per_layer_mofa", False)
                  and not any(getattr(conditioner, name, False) for name in extra_modes))
        self.feature_layer_ids = feature_layer_ids(conditioner, compact=shared and variant == "static-prefix")
        self.feature_count = len(self.feature_layer_ids)
        self._feature_indices = torch.tensor(self.feature_layer_ids, device=model.embed_tokens.weight.device)
        self._shared_mofa_weights = None
        if shared:
            prefix = router_weights(conditioner, conditioner.router_logits, self.feature_layer_ids,
                                    model.embed_tokens.weight.dtype)
            memory_logits = getattr(conditioner, "rejection_router_logits", None)
            memory = prefix if memory_logits is None else router_weights(
                conditioner, memory_logits, self.feature_layer_ids, prefix.dtype)
            self._shared_mofa_weights = prefix, memory
        self.draft = model.draft
        self.structural_draft = None
        if draft_kernel != "reference":
            if draft_kernel not in {"structural-attention", "structural", "structural-fused", "structural-precise", "structural-safe", "fused-sdpa", "fused-sdpa-pruned", "fused-sdpa-gqa", "fused-sdpa-pack", "fused-sdpa-pack-pruned", "fused-sdpa-cachedkv", "fused-sdpa-native"} or variant != "static-prefix":
                raise ValueError("structural draft kernels require the static-prefix runtime")
            from sglang.srt.speculative.dneo.runtime.structural_draft import StructuralDraft
            self.structural_draft = StructuralDraft(self.draft, draft_kernel, attention_splits)
            self.structural_draft.attention_capacity = context_capacity
        self.draft_kernel = draft_kernel
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.context_capacity = context_capacity
        capture_ids = self.feature_layer_ids if variant == "static-prefix" else None
        self.target = target_backend or (None if variant == "static-prefix" else HFTarget(target))
        if variant in {"target-graph", "full-graph", "pruned", "head-graph", "head-cache", "io-graph", "fused-draft", "static-prefix"} and target_backend is None:
            from sglang.srt.speculative.dneo.runtime.graph_target import GraphHFTarget
            self.target = GraphHFTarget(target, context_capacity, batch_size=self.batch_size,
                                        feature_layer_ids=capture_ids)
            self.target.prepare(
                range(1, model.block_size + 1)
                if target_graph_widths is None else target_graph_widths
            )
        self.variant = variant
        self.timer = Timer(profile)
        fused_head = self.batch_size == 1 and variant in {
            "fused-head", "compiled", "triton", "target-graph", "full-graph", "pruned",
            "head-graph", "head-cache", "io-graph", "fused-draft", "static-prefix",
        }
        if folded_head:
            self.head = FoldedDF2Head(model.ar_head, fused=fused_head)
        else:
            self.head = MixedHead(
                model.ar_head,
                fused=fused_head,
                cache_slots=512 if variant in {"head-cache", "io-graph", "fused-draft", "static-prefix"} else 0,
                mode=head_mode,
            )
        if variant == "compiled":
            self.head = torch.compile(self.head, fullgraph=True, mode="reduce-overhead")
        self.prefix_context = None
        self.prefix_kv = None
        self.raw_features = None
        self.prefix_length = 0
        self.memory = None
        self.memory_positions = None
        self.draft_graphs = None
        self.prefix_buffers = None
        if variant == "static-prefix":
            capacity = max(64, 2 ** (context_capacity - 1).bit_length())
            weight = model.embed_tokens.weight
            self.prefix_buffers = tuple(tuple(torch.zeros(
                (self.batch_size, layer.num_key_value_heads, capacity, layer.head_dim),
                device=weight.device, dtype=weight.dtype) for _ in range(2)) for layer in self.draft.layers)
        self.operators = None
        if variant in {"fused-draft", "static-prefix"}:
            from sglang.srt.speculative.dneo.runtime.draft_operators import DraftOperators
            self.operators = DraftOperators(self.draft)
        self.grouped_prefix_commit = None
        self._compact_rope_tables = None
        if (self.prefix_buffers is not None
                and os.environ.get('DNEO_GROUPED_PREFIX_COMMIT', '0') == '1'
                and getattr(self, '_shared_mofa_weights', None) is not None):
            from sglang.srt.speculative.dneo.runtime.prefix_commit import GroupedPrefixCommit
            if GroupedPrefixCommit.supports(self.draft.layers, self.prefix_buffers):
                commit_type = GroupedPrefixCommit
                if os.environ.get('DNEO_GROUPED_COMMIT_NORM', 'original') == 'torch-fused':
                    from sglang.srt.speculative.dneo.runtime.exact_prefix_commit import TorchNormPrefixCommit
                    commit_type = TorchNormPrefixCommit
                self.grouped_prefix_commit = commit_type(
                    self.draft.layers, self.prefix_buffers)
        if (variant == 'static-prefix' and self.prefix_buffers is not None
                and os.environ.get('DNEO_COMPACT_ROUTER', 'original') == 'exact'
                and self.draft.rotary_emb.__class__.__name__ == 'Qwen3RotaryEmbedding'
                and getattr(self.draft.rotary_emb, 'rope_type', None) == 'default'):
            capacity = self.prefix_buffers[0][0].shape[2]
            positions = torch.arange(capacity, device=model.embed_tokens.weight.device)[None]
            cos, sin = self.draft.rotary_emb(model.embed_tokens.weight[:1][None], positions)
            self._compact_rope_tables = (cos[0].contiguous(), sin[0].contiguous())
            if self.structural_draft is not None:
                self.structural_draft.rotary_tables = self._compact_rope_tables
        if variant in {"full-graph", "pruned", "head-graph", "head-cache", "io-graph", "fused-draft", "static-prefix"}:
            from sglang.srt.speculative.dneo.runtime.graph_draft import DraftGraphs
            self.draft_graphs = DraftGraphs(self)
            maximum = max(64, 2 ** (context_capacity - 1).bit_length())
            capacities = (
                tuple(2 ** exponent for exponent in range(6, maximum.bit_length()))
                if draft_graph_capacities is None else tuple(draft_graph_capacities)
            )
            self.draft_graphs.warmup(capacities)
        self.generator = torch.Generator(device=model.embed_tokens.weight.device)
        self.head_graphs = None
        if variant in {"head-graph", "head-cache", "io-graph", "fused-draft", "static-prefix"}:
            from sglang.srt.speculative.dneo.runtime.graph_head import HeadGraphs
            self.head_graphs = HeadGraphs(self)
        self.prefix_graphs = None
        if prepare_prefix_graphs and variant in {"io-graph", "fused-draft", "static-prefix"}:
            from sglang.srt.speculative.dneo.runtime.graph_prefix import PrefixGraphs
            self.prefix_graphs = PrefixGraphs(self)
            self.prefix_graphs.prepare()

    def close(self):
        """Release graph cycles and native distributed state without touching training."""
        for name in ("draft_graphs", "head_graphs", "prefix_graphs"):
            setattr(self, name, None)
        if hasattr(self.target, "close"):
            self.target.close()

    def compact_features(self, features):
        if features.shape[2] != self.feature_count:
            if features.shape[2] != self.model.conditioner.num_target_layers:
                raise ValueError("target features do not match checkpoint support")
            features = features.index_select(2, self._feature_indices)
        return features

    def route(self, features, positions, memory=False):
        features = self.compact_features(features)
        if (self.variant == "static-prefix" and features.is_cuda
                and getattr(self, "_shared_mofa_weights", None) is not None):
            if self.feature_count < self.model.conditioner.num_target_layers:
                if os.environ.get('DNEO_COMPACT_ROUTER', 'original') == 'exact':
                    from sglang.srt.speculative.dneo.runtime.compact_mofa_exact import compact_mofa_exact, supports_exact_mofa
                    weights = self._shared_mofa_weights[int(memory)]
                    if supports_exact_mofa(features, weights, self.feature_layer_ids,
                                           self.model.conditioner.num_target_layers):
                        if self._compact_rope_tables is not None:
                            return compact_mofa_exact(features, weights, self.feature_layer_ids,
                                self.model.conditioner.num_target_layers,
                                rotary=(*self._compact_rope_tables, positions))
                        contexts = compact_mofa_exact(features, weights, self.feature_layer_ids,
                            self.model.conditioner.num_target_layers)
                        return contexts, self.draft.rotary_emb(features[:, :, 0], positions)
                cos, sin = self.draft.rotary_emb(features[:, :, 0], positions)
                if (os.environ.get('DNEO_COMPACT_ROUTER', 'original') == 'sparse'
                        and self.feature_count == 7 and features.dtype == torch.bfloat16):
                    from sglang.srt.speculative.dneo.runtime.compact_mofa_sparse import compact_mofa_sparse
                    return compact_mofa_sparse(
                        features, self._shared_mofa_weights[int(memory)],
                        self.feature_layer_ids, self.model.conditioner.num_target_layers,
                        reuse=features.shape[0] * features.shape[1] >= 32), (cos, sin)
                from sglang.srt.speculative.dneo.runtime.compact_mofa import compact_mofa
                reuse = (os.environ.get('DNEO_MOFA_REUSE_LOADS', '0') == '1'
                         and features.shape[0] * features.shape[1] >= 32
                         and self._shared_mofa_weights[int(memory)].shape[0] <= 8)
                return compact_mofa(features, self._shared_mofa_weights[int(memory)],
                    self.feature_layer_ids, self.model.conditioner.num_target_layers, reuse=reuse), (cos, sin)
            from sglang.srt.speculative.dneo.runtime.serving_kernels import shared_mofa
            cos, sin = self.draft.rotary_emb(features[:, :, 0], positions)
            if (os.environ.get('DNEO_MOFA_REUSE_LOADS', '0') == '1'
                    and features.shape[0] * features.shape[1] >= 32
                    and self._shared_mofa_weights[int(memory)].shape[0] <= 8):
                from sglang.srt.speculative.dneo.runtime.mofa_reuse import shared_mofa_reuse
                return shared_mofa_reuse(features, self._shared_mofa_weights[int(memory)]), (cos, sin)
            return shared_mofa(features, self._shared_mofa_weights[int(memory)]), (cos, sin)
        cos, sin = self.draft.rotary_emb(features[:, :, 0], positions)
        return self.model.conditioner(
            target_hidden=features, target_kv=None, context_cos=cos,
            context_sin=sin, context_position_ids=positions, rejection_memory=memory,
        ), (cos, sin)

    def extend_prefix(self, features):
        if not features.shape[1]:
            return
        features = self.compact_features(features)
        start = self.prefix_length
        self.prefix_length += features.shape[1]
        if self.variant == "reference":
            self.raw_features = features if self.raw_features is None else torch.cat(
                (self.raw_features, features), dim=1)
            return
        positions = torch.arange(start, self.prefix_length, device=features.device)[None].expand(
            features.shape[0], -1
        )
        if self.prefix_graphs is not None:
            contexts, new_kv = self.prefix_graphs(features, positions)
        else:
            contexts, (cos, sin) = self.route(features, positions)
            new_kv = tuple(layer._project_prefix(c, cos, sin) for layer, c in zip(self.draft.layers, contexts))
        if self.prefix_buffers is not None:
            committed = (self.prefix_graphs is not None
                         and self.prefix_graphs.commits_prefix
                         and features.shape[1] in self.prefix_graphs.graphs)
            if not committed:
                for destinations, sources in zip(self.prefix_buffers, new_kv):
                    for destination, source in zip(destinations, sources):
                        destination[:, :, start:self.prefix_length].copy_(source)
            if self.draft_graphs is not None and self.draft_graphs.expanded_kv:
                self.draft_graphs.update_prefix(new_kv, start)
            self.prefix_kv = tuple(tuple(x[:, :, :self.prefix_length] for x in kv) for kv in self.prefix_buffers)
            return
        if self.prefix_context is None:
            # Graph outputs alias a shared pool; persistent caches own storage.
            self.prefix_context = tuple(c.clone() for c in contexts)
            self.prefix_kv = tuple(tuple(x.clone() for x in kv) for kv in new_kv)
        else:
            self.prefix_context = tuple(torch.cat((a, b), dim=1) for a, b in zip(self.prefix_context, contexts))
            self.prefix_kv = tuple(tuple(torch.cat((a, b), dim=2) for a, b in zip(old, new))
                                   for old, new in zip(self.prefix_kv, new_kv))

    @contextlib.contextmanager
    def cached_projection(self):
        if self.variant == "reference":
            yield
            return
        originals = []
        for layer, kv in zip(self.draft.layers, self.prefix_kv):
            originals.append(layer._project_prefix)
            if getattr(self.draft, "rejection_memory_kv_only", False):
                length = self.prefix_length
                original = layer._project_prefix
                def project(context, cos, sin, _kv=kv, _original=original, _length=length):
                    if context.shape[1] == _length:
                        return _kv
                    memory = _original(context[:, _length:], cos[:, _length:], sin[:, _length:])
                    return tuple(torch.cat((prefix, suffix), dim=2) for prefix, suffix in zip(_kv, memory))
                layer._project_prefix = project
            else:
                layer._project_prefix = types.MethodType(lambda self, *args, _kv=kv, **kwargs: _kv, layer)
        try:
            yield
        finally:
            for layer, method in zip(self.draft.layers, originals):
                layer._project_prefix = method

    @contextlib.contextmanager
    def convolution_kernels(self):
        if self.variant not in {"triton", "target-graph", "full-graph", "pruned", "head-graph", "head-cache", "io-graph", "fused-draft", "static-prefix"}:
            yield
            return
        from sglang.srt.speculative.dneo.runtime.kernels import prepare, finish
        originals = []
        for layer in self.draft.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                originals.append((conv, conv.prepare, conv.finish))
                conv.prepare = types.MethodType(prepare, conv)
                conv.finish = types.MethodType(finish, conv)
        try:
            yield
        finally:
            for conv, pre, post in originals:
                conv.prepare, conv.finish = pre, post

    @contextlib.contextmanager
    def last_layer_pruning(self, memory_width=None):
        if getattr(self.draft, "rejection_memory_kv_only", False):
            yield
            return
        if self.variant != "pruned" and not self.optimize_backbone:
            yield
            return
        from sglang.srt.speculative.dneo.runtime.pruned_layer import forward, split_attention
        if memory_width is None:
            memory_width = 0 if self.memory is None else self.memory.shape[1]
        originals = []
        for index, layer in enumerate(self.draft.layers):
            if index == len(self.draft.layers) - 1 or (self.optimize_backbone and memory_width == 0):
                originals.append((layer, "forward", layer.forward))
                layer.forward = types.MethodType(forward, layer)
            elif self.optimize_backbone and self.split_memory_attention:
                originals.append((layer, "_attention", layer._attention))
                layer._attention = types.MethodType(split_attention, layer)
        try:
            yield
        finally:
            for layer, name, method in originals:
                setattr(layer, name, method)

    def operator_kernels(self):
        reference = getattr(self.structural_draft, "reference_arithmetic", False)
        return contextlib.nullcontext() if self.operators is None or reference else self.operators.activate()

    def draft_hidden(self, anchor):
        if self.draft_graphs is not None:
            return self.draft_graphs(anchor)
        b = self.model.block_size
        batch = anchor.numel()
        if batch != self.batch_size:
            raise ValueError(f"runtime expected batch {self.batch_size}, got {batch}")
        ids = torch.full((batch, 1, b), self.model.mask_token_id, device=anchor.device, dtype=torch.long)
        ids[:, :, 0] = anchor.reshape(batch, 1)
        embeddings = self.model.embed_tokens(ids)
        anchor_pos = torch.full((batch, 1), self.prefix_length, device=anchor.device)
        prefix_pos = torch.arange(self.prefix_length, device=anchor.device)[None].expand(batch, -1)
        if self.variant == "reference":
            contexts, (cos, sin) = self.route(self.raw_features, prefix_pos)
        else:
            contexts = self.prefix_context
            cos, sin = self.draft.rotary_emb(contexts[0][:, :, 0], prefix_pos)
        kwargs = {}
        if self.memory is not None:
            routed, _ = self.route(self.memory, self.memory_positions, memory=True)
            kwargs = dict(
                rejection_contexts=tuple(c[:, None] for c in routed),
                rejection_keep_mask=torch.ones((batch, 1, self.memory.shape[1]), device=anchor.device, dtype=torch.bool),
                rejection_position_ids=self.memory_positions[:, None],
            )
        with self.cached_projection(), self.convolution_kernels(), self.last_layer_pruning():
            output = self.draft(
                embeddings, contexts, anchor_pos, torch.ones_like(anchor_pos, dtype=torch.bool),
                torch.ones((batch, self.prefix_length), device=anchor.device, dtype=torch.long),
                context_cos=cos, context_sin=sin, **kwargs,
            )
        return output[:, 0, 1:]

    def rollout(self, hidden, anchor, gamma, temperature, top_p, generator):
        if self.head_graphs is not None and generator is self.generator:
            return self.head_graphs(hidden, anchor, gamma, temperature, top_p)
        return self.rollout_eager(hidden, anchor, gamma, temperature, top_p, generator)

    def rollout_eager(self, hidden, anchor, gamma, temperature, top_p, generator):
        if (
            self.head_mode in {"dense-mixed", "folded-df2"}
            and self.optimize_head
            and temperature > 0
            and top_p == 1.
        ):
            return self.rollout_fused_sampling(hidden, anchor, gamma, temperature, generator)
        base = self.model.lm_head(hidden[:, :gamma])
        projected = (
            self.head.project_hidden(hidden[:, :gamma])
            if self.head_mode == "folded-df2"
            else self.model.ar_head.selector.hidden_projection(hidden[:, :gamma])
        )
        previous = anchor.reshape(-1)
        proposals, distributions = [], []
        for position in range(gamma):
            if self.head_mode == "sparse-top16-mixed":
                candidates, scores = self.head.sparse_scores(
                    base[:, position], projected[:, position], previous, position
                )
                candidate_q = probabilities(scores, temperature, top_p)
                selected = categorical(candidate_q, generator)
                previous = candidates.gather(-1, selected[:, None]).squeeze(-1)
                q = torch.zeros(
                    base.shape[0], base.shape[-1], device=base.device, dtype=candidate_q.dtype
                ).scatter(-1, candidates, candidate_q)
                proposals.append(previous)
                distributions.append(q)
                continue
            logits = self.head(base[:, position], projected[:, position], previous, position)
            q = probabilities(logits, temperature, top_p)
            previous = categorical(q, generator)
            proposals.append(previous)
            distributions.append(q)
        proposed = torch.stack(proposals, dim=1)
        q = torch.stack(distributions, dim=1)
        if proposed.shape[0] == 1:
            return proposed[0], q[0]
        return proposed, q

    def rollout_fused_sampling(self, hidden, anchor, gamma, temperature, generator, *, probabilities_out=None):
        from sglang.srt.speculative.dneo.runtime.kernels import (
            folded_df2_latent,
            folded_df2_head_sample,
            folded_df2_head_sample_fused_matrix,
            full_vocab_probabilities,
            mixed_head_sample,
            mixed_head_sample_fused_matrix,
        )
        events = getattr(self, "head_profile_events", None)
        if events:
            events[0].record()
        # Dropping the anchor leaves a gap between request rows. Without this
        # packing, F.linear can dispatch a batched GEMM that rereads the 150K
        # vocabulary matrix for every request instead of one flattened GEMM.
        hidden = hidden[:, :gamma].contiguous()
        base = self.model.lm_head(hidden)
        if events:
            events[1].record()
        projected = (
            self.head.project_hidden(hidden)
            if self.head_mode == "folded-df2"
            else self.model.ar_head.selector.hidden_projection(hidden)
        )
        noise = torch.empty(base.shape, device=base.device, dtype=torch.float32).exponential_(generator=generator)
        if events:
            events[2].record()
        previous = anchor.reshape(-1)
        proposals, logits = [], []
        strided_io = (self.head_mode == "folded-df2" and base.shape[0] > 1
                      and os.environ.get("DNEO_HEAD_STRIDED_IO", "0") == "1")
        corrected_block = torch.empty_like(base) if strided_io else None
        for position in range(gamma):
            if self.head_mode == "folded-df2":
                latent = folded_df2_latent(
                    projected[:, position] if strided_io else projected[:, position].contiguous(), previous, position,
                    self.model.ar_head,
                )
                sample = (
                    folded_df2_head_sample_fused_matrix
                    if base.shape[0] > 1 else folded_df2_head_sample
                )
                if strided_io:
                    corrected, previous = sample(
                        base[:, position], latent, position, self.model.ar_head,
                        noise[:, position], temperature, out=corrected_block[:, position])
                else:
                    corrected, previous = sample(
                        base[:, position].contiguous(), latent.contiguous(), position,
                        self.model.ar_head, noise[:, position].contiguous(), temperature,
                    )
            else:
                sample = (
                    mixed_head_sample_fused_matrix
                    if base.shape[0] > 1 else mixed_head_sample
                )
                corrected, previous = sample(
                    base[:, position].contiguous(), projected[:, position].contiguous(), previous, position,
                    self.model.ar_head, noise[:, position].contiguous(), temperature, self.head.correction_cache)
            proposals.append(previous)
            if not strided_io:
                logits.append(corrected)
        if events:
            events[3].record()
        proposed = torch.stack(proposals, dim=1)
        stacked_logits = corrected_block if strided_io else torch.stack(logits, dim=1)
        q = (
            full_vocab_probabilities(stacked_logits, temperature, out=probabilities_out)
            if self.head_mode == "folded-df2"
            else probabilities(stacked_logits, temperature, 1.)
        )
        if events:
            events[4].record()
        if proposed.shape[0] == 1:
            return proposed[0], q[0]
        return proposed, q

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=128, temperature=1., top_p=1., seed=42,
                 eos_ids=(), trace=False, gamma=None, token_callback=None, observer=None,
                 target_temperature=None, draft_temperature=None):
        """Decode with independent target/proposal temperatures; unset overrides use temperature."""
        target_temperature = temperature if target_temperature is None else target_temperature
        draft_temperature = temperature if draft_temperature is None else draft_temperature
        if any(not math.isfinite(t) or t < 0 for t in (target_temperature, draft_temperature)):
            raise ValueError("target and draft temperatures must be finite and nonnegative")
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
            raise ValueError("this latency harness requires a nonempty single prompt")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if prompt_ids.shape[1] + max_new_tokens > self.context_capacity:
            raise ValueError("prompt plus output budget exceeds context_capacity")
        gamma_limit = self.model.block_size - 1 if gamma is None else gamma
        if not 1 <= gamma_limit < self.model.block_size:
            raise ValueError("gamma must be between 1 and the trained block_size - 1")
        self.prefix_context = self.prefix_kv = self.raw_features = self.memory = None
        self.memory_positions = None
        self.prefix_length = 0
        self.timer.events.clear()
        self.target.reset()
        if self.head_graphs is not None:
            self.head_graphs.prepare(draft_temperature, top_p)
        if self.head.correction_cache is not None:
            self.head.correction_cache[2].zero_()
        generator = self.generator.manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with self.timer.stage("prefill"):
            logits, features = self.target.forward(prompt_ids, last_logit_only=True)
            self.extend_prefix(features)
            anchor = categorical(probabilities(logits[:, -1], target_temperature, top_p), generator)
        if observer is not None:
            observer("prefill", logits=logits[:, -1], emitted=anchor)
        generated = [anchor]
        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        calls, accepted_total, records = 0, 0, []
        accepted_histogram = [0] * (gamma_limit + 1)
        reached_positions, accepted_positions = [0] * gamma_limit, [0] * gamma_limit
        first_eos = int(anchor.item()) in eos_ids
        if token_callback is not None:
            token_callback(anchor.tolist())
        while sum(x.numel() for x in generated) < max_new_tokens and not first_eos:
            remaining = max_new_tokens - sum(x.numel() for x in generated)
            gamma = min(gamma_limit, remaining)
            old_prefix = self.prefix_length
            if getattr(self.target, "native_graph", False):
                # Stay on the fixed-width target graph at response tails. The
                # emitted stream is still truncated to the requested budget.
                gamma = min(gamma_limit, self.context_capacity - old_prefix - 1)
            with self.timer.stage("draft"):
                hidden = self.draft_hidden(anchor)
            with self.timer.stage("head_rollout"):
                proposed, q = self.rollout(hidden, anchor, gamma, draft_temperature, top_p, generator)
            with self.timer.stage("verify"):
                logits, features = self.target.forward(torch.cat((anchor, proposed))[None])
                if self.fused_verification and target_temperature > 0 and top_p == 1:
                    from sglang.srt.speculative.dneo.runtime.kernels import full_vocab_probabilities
                    p = full_vocab_probabilities(logits[0].contiguous(), target_temperature)
                else:
                    p = probabilities(logits[0], target_temperature, top_p)
            with self.timer.stage("accept"):
                if self.fused_verification and self.verification_method == "leviathan":
                    from sglang.srt.speculative.dneo.runtime.kernels import leviathan_verify_cuda
                    emitted, count = leviathan_verify_cuda(proposed, q, p, generator)
                else:
                    emitted, count = verify(proposed, q, p, generator, method=self.verification_method)
            if observer is not None:
                observer("verify", logits=logits[0], p=p, q=q, proposed=proposed,
                         accepted=count, emitted=emitted[:remaining], anchor_index=old_prefix)
            with self.timer.stage("commit"):
                # The input anchor and accepted proposals join the true prefix.
                self.target.crop(old_prefix + 1 + count)
                self.extend_prefix(features[:, :1 + count])
                if count < gamma:
                    self.memory = features[:, 1 + count:].clone()
                    self.memory_positions = torch.arange(
                        old_prefix + 1 + count, old_prefix + 1 + gamma, device=prompt_ids.device)[None]
                else:
                    self.memory = self.memory_positions = None
                anchor = emitted[-1:]
            calls += 1
            accepted_total += count
            accepted_histogram[count] += 1
            for position in range(min(count + 1, gamma)):
                reached_positions[position] += 1
                accepted_positions[position] += int(position < count)
            if trace:
                records.append(dict(anchor_index=old_prefix, proposed=proposed.tolist(), accepted=count,
                                    emitted=emitted.tolist(), memory_positions=[] if self.memory_positions is None
                                    else self.memory_positions[0].tolist(),
                                    target_top2_ids=logits[0].topk(2, dim=-1).indices.tolist(),
                                    target_top2_logits=logits[0].topk(2, dim=-1).values.tolist()))
            emitted = emitted[:remaining]
            if eos_ids:
                for index, token in enumerate(emitted.tolist()):
                    if token in eos_ids:
                        emitted = emitted[:index + 1]
                        first_eos = True
                        break
            generated.append(emitted)
            if token_callback is not None:
                token_callback(emitted.tolist())
        torch.cuda.synchronize()
        end = time.perf_counter()
        ids = torch.cat(generated)
        cache_stats = None if self.head.correction_cache is None else self.head.correction_cache[2].tolist()
        return dict(token_ids=ids.tolist(), tokens=ids.numel(), verify_calls=calls,
                    gamma=gamma_limit, context_capacity=self.context_capacity,
                    verification_method=self.verification_method,
                    dspark_cache_hits=None if cache_stats is None else cache_stats[0],
                    dspark_cache_lookups=None if cache_stats is None else cache_stats[1],
                    accepted_drafts=accepted_total, tpf=1 + accepted_total / max(calls, 1),
                    accepted_length_histogram=accepted_histogram,
                    per_position_reached=reached_positions, per_position_accepted=accepted_positions,
                    prefill_seconds=decode_start - start, decode_seconds=end - decode_start,
                    elapsed_seconds=end - start,
                    decode_tps=(ids.numel() - 1) / max(end - decode_start, 1e-9),
                    end_to_end_tps=ids.numel() / max(end - start, 1e-9),
                    stage_ms=self.timer.milliseconds() if self.timer.enabled else {}, trace=records)
