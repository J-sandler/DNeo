import sys
from types import SimpleNamespace

import torch

from dneo import generate
import pytest


def test_chat_cli_explicitly_requests_token_tensor(monkeypatch, capsys):
    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            assert kwargs["return_tensors"] == "pt"
            return torch.tensor([[1, 3]])

        def decode(self, token_ids):
            return "ok"

    target = SimpleNamespace(device="cpu")
    draft = SimpleNamespace(config=SimpleNamespace(target_model_name_or_path="example", target_revision="pinned"))
    monkeypatch.setattr(generate, "activate_production_runtime", lambda: None)
    monkeypatch.setattr(generate, "load_checkpoint", lambda _: (target, draft, None, None, None))
    monkeypatch.setattr(generate.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(generate, "decode", lambda *args: {"token_ids": [4, 2]})
    monkeypatch.setattr(sys, "argv", ["dneo-generate", "--draft-model", "example", "--prompt", "hello"])
    generate.main()
    assert '"text": "ok"' in capsys.readouterr().out


@pytest.mark.parametrize("eos,initial,expected", [([2, 3], 7, [7, 8, 9, 3]),
                                                ([2, 3], 3, [3]),
                                                (3, 7, [7, 8, 9, 3]),
                                                (None, 7, [7, 8, 9, 3, 8])])
@pytest.mark.parametrize("feature_dtype", [None, torch.float32])
def test_decode_stop_ids_and_graph_output_ownership(monkeypatch, eos, initial, expected, feature_dtype):
    class Target:
        def prepare(self, **kwargs):
            pass

        def forward_at(self, ids, positions, **kwargs):
            logits = torch.full((1, 1, 16), -10.)
            logits[..., initial] = 10.
            return logits, None if feature_dtype is None else torch.zeros(1, ids.shape[1], 1, 4, dtype=feature_dtype)

    class Draft:
        def __init__(self, *args, **kwargs):
            self.worker = SimpleNamespace(runtime=SimpleNamespace(generator=torch.Generator()))
            self.live = None

        def reset(self, batch):
            pass

        def propose(self, anchor, lengths):
            return torch.tensor([[8, 9, 10]]), None

        def accept(self, logits, temperature):
            self.live = (torch.tensor([3]), torch.tensor([3]), torch.tensor([[8, 9, 3, 0]]))
            commit, bonus, emitted = self.live
            return torch.tensor([2]), commit, bonus, emitted, None, None

        def commit(self, *args):
            if args[0] is not None:
                assert args[0].dtype == torch.bfloat16
            if self.live is not None:
                for value in self.live:
                    value.fill_(999)

    monkeypatch.setattr(generate, "CohortHFTarget", lambda *args, **kwargs: Target())
    monkeypatch.setattr(generate, "DNeoCohortDraft", Draft)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    draft = SimpleNamespace(block_size=4, config=SimpleNamespace(target_layer_ids=[0]),
                            embed_tokens=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)))
    result = generate.decode(None, draft, torch.tensor([[1, 5]]), max_new_tokens=5,
                             target_temperature=0, eos_token_id=eos)
    assert result["token_ids"] == expected
