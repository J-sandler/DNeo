"""Matched HF cohort timing, preserved from the internal benchmark harness."""
import json
import time
import torch
from dneo.runtime.dneo import probabilities, categorical
from dneo.runtime.hf_cohort import DNeoCohortDraft
@torch.inference_mode()
def generate_cohort(native, draft, records, budget, temperature, seed, audit=False,
                    acceptance_histograms=None):
    batch = len(records)
    lengths = torch.tensor([len(row["token_ids"]) for row in records], device="cuda")
    width = int(lengths.max())
    ids = torch.zeros(batch, width, device="cuda", dtype=torch.long)
    for row, record in enumerate(records):
        ids[row, :len(record["token_ids"])] = torch.tensor(record["token_ids"], device="cuda")
    positions = torch.arange(width, device="cuda")[None].repeat(batch, 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if draft is not None:
        draft.reset(batch)
        torch.manual_seed(seed)
        if hasattr(draft, "generator"):
            draft.generator.manual_seed(seed)
        if isinstance(draft, DNeoCohortDraft):
            draft.worker.runtime.generator.manual_seed(seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    if getattr(native, "prefill_logit_policy", "last") == "full":
        logits, features = native.forward_at(ids, positions)
        last = logits[torch.arange(batch, device=ids.device), lengths - 1]
    else:
        logits, features = native.prefill_at(ids, positions, logit_positions=lengths - 1)
        last = logits[:, 0]
    if draft is not None:
        draft.commit(features, positions)
    anchor = categorical(probabilities(last, temperature), generator)
    torch.cuda.synchronize()
    first = time.perf_counter()
    produced = [1] * batch
    calls = [0] * batch
    while min(produced) < budget:
        active = torch.tensor([count < budget for count in produced], device="cuda")
        if draft is None:
            logits, _ = native.forward_at(anchor[:, None], lengths[:, None])
            anchor = categorical(probabilities(logits[:, 0], temperature), generator)
            lengths.add_(active.long())
            for row in range(batch):
                if produced[row] < budget:
                    produced[row] += 1
                    calls[row] += 1
        else:
            proposed, q = draft.propose(anchor, lengths)
            if audit and isinstance(draft, DNeoCohortDraft):
                diagnostic = dict(proposal_min=int(proposed.min()), proposal_max=int(proposed.max()),
                    q_finite=bool(q.isfinite().all()),
                    prefix_lengths=lengths.tolist())
                print("hf_dneo_cohort_audit=" + json.dumps(diagnostic), flush=True)
                if (not diagnostic["q_finite"]
                        or diagnostic["proposal_min"] < 0
                        or diagnostic["proposal_max"] >= native.target.config.vocab_size):
                    raise RuntimeError("invalid DNeo cohort graph output")
            block = torch.cat((anchor[:, None], proposed), dim=1)
            positions = lengths[:, None] + torch.arange(block.shape[1], device="cuda")
            logits, features = native.forward_at(block, positions)
            accepted, commit, anchor, emitted, _, _ = draft.accept(logits, temperature)
            # Production scheduling copies graph outputs into request-owned
            # state before another graph can reuse the shared capture pool.
            anchor, commit = anchor.clone(), commit.clone()
            if temperature == 0:
                valid = torch.arange(block.shape[1], device="cuda")[None] < commit[:, None]
                if not torch.equal(logits.argmax(-1)[valid], emitted[valid]):
                    raise RuntimeError("HF greedy-target acceptance audit failed")
            commit = torch.where(active, commit, 0)
            draft.commit(features, positions, commit)
            lengths.add_(commit.long())
            for row, count in enumerate(commit.tolist()):
                if produced[row] < budget:
                    if acceptance_histograms is not None:
                        if not 1 <= count <= len(acceptance_histograms[row]):
                            raise RuntimeError(f"Invalid speculative commit length: {count}")
                        acceptance_histograms[row][count - 1] += 1
                    produced[row] = min(budget, produced[row] + count)
                    calls[row] += 1
    torch.cuda.synchronize()
    ended = time.perf_counter()
    return calls, ended - started, ended - first, first - started

