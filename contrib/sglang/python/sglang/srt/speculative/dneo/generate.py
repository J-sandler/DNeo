"""Single-request Transformers inference with the optimized shared DNeo engine."""
import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from .loading import load_checkpoint
from .runtime.dneo import categorical, probabilities
from .runtime.dneo_production import activate_production_runtime
from .runtime.hf_cohort import CohortHFTarget, DNeoCohortDraft


@torch.inference_mode()
def decode(target, draft, input_ids, max_new_tokens=128, target_temperature=1., seed=42, context_length=1024, eos_token_id=None, trace=None):
    if target_temperature not in (0, 1):
        raise ValueError("optimized acceptance graphs currently support target temperatures 0 and 1")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("use a nonempty single prompt")
    if max_new_tokens < 1 or input_ids.shape[1] + max_new_tokens > context_length:
        raise ValueError("prompt plus output budget exceeds context length")
    eos_ids = frozenset(() if eos_token_id is None else
                        (eos_token_id,) if isinstance(eos_token_id, int) else eos_token_id)
    captures = draft.config.target_layer_ids
    verifier = CohortHFTarget(target, max_batch=1, capacity=context_length + draft.block_size, feature_layers=captures)
    engine = DNeoCohortDraft(draft, capacity=context_length + draft.block_size, max_batch=1)
    verifier.prepare(batches=(1,), widths=(draft.block_size,))
    torch.manual_seed(seed)
    engine.worker.runtime.generator.manual_seed(seed)
    engine.reset(1)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
    torch.cuda.synchronize()
    start = time.perf_counter()
    logits, features = verifier.forward_at(input_ids, positions, last_logit_only=True)
    engine.commit(features.to(draft.embed_tokens.weight.dtype) if features is not None else features, positions)
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    anchor = categorical(probabilities(logits[:, -1], target_temperature), generator)
    output = anchor.tolist()
    if trace is not None:
        trace.append(dict(stage="prefill", emitted=output.copy()))
    lengths = torch.tensor([input_ids.shape[1]], device=input_ids.device)
    torch.cuda.synchronize()
    decode_start = time.perf_counter()
    calls = 0
    while len(output) < max_new_tokens and output[-1] not in eos_ids:
        proposed_ids, draft_q = engine.propose(anchor, lengths)
        block = torch.cat((anchor[:, None], proposed_ids), dim=1)
        at = lengths[:, None] + torch.arange(draft.block_size, device=input_ids.device)[None]
        logits, features = verifier.forward_at(block, at)
        greedy = logits.argmax(-1).tolist()[0] if trace is not None else None
        if trace is not None:
            proposal = proposed_ids[0].tolist()
            top_values, top_ids = logits[0].float().topk(2, dim=-1)
            details = dict(proposal=proposal, target_top2_ids=top_ids.tolist(),
                           target_top2_logits=top_values.tolist(), target_finite=bool(logits.isfinite().all()),
                           q_finite=bool(draft_q.isfinite().all()),
                           q_normalization_error=float((draft_q.sum(-1) - 1).abs().max()))
        accepted = engine.accept(logits, target_temperature)
        count, commit, bonus, emitted, new_lengths, _ = accepted
        # Captured graphs share a pool. Own outputs before commit can reuse it.
        bonus, commit = bonus.clone(), commit.clone()
        tokens = emitted[0, :int(commit[0])].tolist()
        if trace is not None:
            trace.append(dict(stage="verify", output_start=len(output), prefix_length=int(lengths[0]),
                              anchor=int(anchor[0]), **details,
                              accepted_drafts=int(count[0]), commit_length=int(commit[0]),
                              emitted=tokens.copy(), target_greedy=greedy))
        engine.commit(features.to(draft.embed_tokens.weight.dtype) if features is not None else features, at, commit)
        tokens = tokens[:max_new_tokens - len(output)]
        stop = next((index + 1 for index, token in enumerate(tokens) if token in eos_ids), len(tokens))
        tokens = tokens[:stop]
        output.extend(tokens)
        anchor, lengths = bonus.long(), lengths + commit.long()
        calls += 1
    torch.cuda.synchronize()
    end = time.perf_counter()
    return dict(token_ids=output, tokens=len(output), verify_calls=calls,
                tpf=(len(output) - 1) / calls if calls else None,
                request_seconds=end - start, decode_seconds=end - decode_start,
                request_tps=len(output) / (end - start),
                decode_tps=(len(output) - 1) / (end - decode_start) if calls else None,
                prefill_seconds=decode_start - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--target-temperature", type=float, choices=(0., 1.), default=1.)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    activate_production_runtime()
    target, draft, _, _, _ = load_checkpoint(args.draft_model)
    tokenizer = AutoTokenizer.from_pretrained(draft.config.target_model_name_or_path, revision=draft.config.target_revision)
    ids = tokenizer.apply_chat_template([dict(role="user", content=args.prompt)], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=False, enable_thinking=True).to(target.device)
    eos = getattr(getattr(target, "generation_config", None), "eos_token_id", None)
    result = decode(target, draft, ids, args.max_new_tokens, args.target_temperature, args.seed,
                    args.context_length, tokenizer.eos_token_id if eos is None else eos)
    result["text"] = tokenizer.decode(result["token_ids"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
