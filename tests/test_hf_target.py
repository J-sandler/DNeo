import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from dneo.runtime.hf_cohort import CohortHFTarget
from dneo import generate
from dneo.runtime.dneo import probabilities, verify
from types import SimpleNamespace


def target_fixture(monkeypatch, dtype=torch.float32):
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    torch.manual_seed(31)
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        vocab_size=64, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(config).to(dtype).eval()
    return target, CohortHFTarget(target, max_batch=2, capacity=32)


@torch.inference_mode()
def test_explicit_target_cache_rewind_ignores_rejected_suffix(monkeypatch):
    target, verifier = target_fixture(monkeypatch)
    prefix = torch.tensor([[1, 4, 7, 8, 3]])
    verifier.forward_at(prefix, torch.arange(5)[None])
    rejected = torch.tensor([[9, 10, 11, 12]])
    verifier.forward_at(rejected, torch.arange(5, 9)[None])
    replacement = torch.tensor([[21, 22, 23, 24]])
    actual, _ = verifier.forward_at(replacement, torch.arange(6, 10)[None])
    sequence = torch.cat((prefix, rejected[:, :1], replacement), dim=1)
    expected = target(sequence, use_cache=False).logits[:, -4:]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


@torch.inference_mode()
def test_target_cache_uses_independent_positions_per_row(monkeypatch):
    target, verifier = target_fixture(monkeypatch)
    prefix = torch.tensor([[1, 4, 7, 8, 3], [2, 9, 5, 6, 4]])
    verifier.forward_at(prefix, torch.arange(5)[None].expand(2, -1))
    block = torch.tensor([[21, 22, 23], [31, 32, 33]])
    positions = torch.tensor([[3, 4, 5], [5, 6, 7]])
    actual, _ = verifier.forward_at(block, positions)
    for row, length in enumerate((3, 5)):
        sequence = torch.cat((prefix[row:row + 1, :length], block[row:row + 1]), dim=1)
        expected = target(sequence, use_cache=False).logits[:, -3:]
        torch.testing.assert_close(actual[row:row + 1], expected, rtol=1e-5, atol=1e-7)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_speculative_decode_matches_hf_greedy_through_accepts_and_rewinds(monkeypatch, dtype):
    target, _ = target_fixture(monkeypatch, dtype)
    ids = torch.tensor([[1, 4, 7, 8, 3]])
    expected = target.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20,
                               do_sample=False)[:, ids.shape[1]:].tolist()[0]

    class Verifier(CohortHFTarget):
        def prepare(self, **kwargs):
            pass

    class ScriptedDraft:
        def __init__(self, *args, **kwargs):
            self.generator = torch.Generator()
            self.worker = SimpleNamespace(runtime=SimpleNamespace(generator=self.generator))
            self.steps = 0

        def reset(self, batch):
            pass

        def commit(self, *args):
            assert args[0].dtype == torch.bfloat16

        def propose(self, anchor, lengths):
            self.prefix = lengths
            offset = int(lengths[0]) - ids.shape[1] + 1
            proposal = (expected + [0] * 3)[offset:offset + 3]
            accepted = (0, 1, 3)[self.steps % 3]
            if accepted < 3:
                proposal[accepted] = (proposal[accepted] + 1) % target.config.vocab_size
            self.proposed = torch.tensor(proposal)
            self.q = torch.nn.functional.one_hot(self.proposed, target.config.vocab_size).float()
            self.steps += 1
            return self.proposed[None], self.q[None]

        def accept(self, logits, temperature):
            emitted, count = verify(self.proposed, self.q, probabilities(logits[0], temperature), self.generator)
            return (torch.tensor([count]), torch.tensor([count + 1]), emitted[-1:].clone(),
                    emitted[None], self.prefix + count + 1, None)

    monkeypatch.setattr(generate, "CohortHFTarget", Verifier)
    monkeypatch.setattr(generate, "DNeoCohortDraft", ScriptedDraft)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    draft = SimpleNamespace(block_size=4, config=SimpleNamespace(target_layer_ids=[0]),
                            embed_tokens=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)))
    trace = []
    result = generate.decode(target, draft, ids, max_new_tokens=20, target_temperature=0,
                             context_length=64, trace=trace)
    assert result["token_ids"] == expected
    assert [row["accepted_drafts"] for row in trace if row["stage"] == "verify"][:3] == [0, 1, 3]
