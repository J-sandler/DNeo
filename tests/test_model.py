import torch
from transformers import Qwen3Config

from dneo import DNeoConfig, DNeoDraftModel


def tiny_config():
    target = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         vocab_size=64, max_position_embeddings=128)
    return DNeoConfig(target_config=target, num_hidden_layers=2, target_layer_ids=[0, 3],
                     block_size=4, head_rank=8, folded_rank=8, conv_group_size=8, mask_token_id=5)


def test_save_load_strict_and_shared_target_weights_not_serialized(tmp_path):
    model = DNeoDraftModel(tiny_config())
    model.conditioner.shared_mofa_candidate_mask.zero_()
    model.conditioner.shared_mofa_candidate_mask[[0, 3]] = True
    target = torch.nn.Module()
    target.embed = torch.nn.Embedding(64, 32)
    target.head = torch.nn.Linear(32, 64, bias=False)
    target.get_input_embeddings = lambda: target.embed
    target.get_output_embeddings = lambda: target.head
    model.bind_target(target)
    assert model.embed_tokens.weight.data_ptr() == target.embed.weight.data_ptr()
    assert model.lm_head.weight.data_ptr() == target.head.weight.data_ptr()
    assert not any(key.startswith(("embed_tokens", "lm_head")) for key in model.state_dict())
    model.save_pretrained(tmp_path)
    loaded = DNeoDraftModel.from_pretrained(tmp_path)
    assert set(loaded.state_dict()) == set(model.state_dict())
    assert all(torch.equal(value, loaded.state_dict()[key]) and value.dtype == loaded.state_dict()[key].dtype
               for key, value in model.state_dict().items())


def test_reference_memory_is_kv_only_and_invalid_slots_do_not_change_output():
    torch.manual_seed(3)
    model = DNeoDraftModel(tiny_config()).eval()
    # Avoid zero-initialized synthetic convs hiding memory sensitivity.
    for layer in model.draft.layers:
        for conv in (layer.attention_conv, layer.mlp_conv):
            conv.base_kernel.data[:, 0].fill_(1.)
    embeddings = torch.randn(1, 1, 4, 32)
    prefix = torch.randn(1, 6, 4, 32)
    memory = torch.randn(1, 1, 3, 4, 32)
    valid = torch.tensor([[[True, False, False]]])
    positions = torch.tensor([[[7, 8, 9]]])
    actual = model(embeddings, prefix, torch.tensor([[6]]), memory, valid, positions)
    memory[:, :, 1:] = 10000
    expected = model(embeddings, prefix, torch.tensor([[6]]), memory, valid, positions)
    assert torch.equal(actual, expected)
    assert actual.shape == (1, 1, 4, 32)


def test_defaults_are_serializable():
    from typing import get_type_hints
    from dneo.reference import rejection_memory_attention_mask
    assert DNeoConfig().to_json_string()
    assert get_type_hints(rejection_memory_attention_mask)


def test_shared_router_save_load_and_exact_broadcast_compatibility(tmp_path):
    from dneo.runtime.feature_layout import router_weights
    torch.manual_seed(17)
    config = tiny_config()
    config.shared_layer_mofa = True
    shared = DNeoDraftModel(config).eval()
    shared.conditioner.router_logits.data.normal_()
    assert shared.conditioner.router_logits.shape == (1, 1, 4)
    shared.save_pretrained(tmp_path)
    restored = DNeoDraftModel.from_pretrained(tmp_path)
    assert restored.config.shared_layer_mofa
    assert torch.equal(restored.conditioner.router_logits, shared.conditioner.router_logits)
    broadcast = DNeoDraftModel(tiny_config()).eval()
    state = dict(shared.state_dict())
    state["conditioner.router_logits"] = state["conditioner.router_logits"].expand(2, -1, -1).clone()
    broadcast.load_state_dict(state, strict=True)
    features = torch.randn(2, 6, 4, 32)
    for actual, expected in zip(shared.conditioner(features), broadcast.conditioner(features)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert shared.conditioner(features)[0].shape == (2, 6, 2, 32)
    for dtype in (torch.float32, torch.bfloat16):
        actual = router_weights(shared.conditioner, shared.conditioner.router_logits, [0, 3], dtype)
        expected = router_weights(broadcast.conditioner, broadcast.conditioner.router_logits, [0, 3], dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_router_rejects_multiple_stored_routers():
    from dneo.runtime.feature_layout import router_weights
    import pytest
    config = tiny_config()
    config.shared_layer_mofa = True
    model = DNeoDraftModel(config)
    with pytest.raises(ValueError, match="exactly one router"):
        router_weights(model.conditioner, torch.zeros(2, 1, 4), [0, 3], torch.float32)


def test_fp32_verifier_conversion_preserves_bf16_draft_vocabulary_views():
    model = DNeoDraftModel(tiny_config()).eval()
    target = torch.nn.Module()
    target.embed = torch.nn.Embedding(64, 32, dtype=torch.bfloat16)
    target.head = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)
    target.get_input_embeddings = lambda: target.embed
    target.get_output_embeddings = lambda: target.head
    model.bind_target(target)
    embedding, head = model.embed_tokens.weight.clone(), model.lm_head.weight.clone()
    target.float()
    assert target.embed.weight.dtype == target.head.weight.dtype == torch.float32
    assert model.embed_tokens.weight.dtype == model.lm_head.weight.dtype == torch.bfloat16
    assert torch.equal(embedding, model.embed_tokens.weight)
    assert torch.equal(head, model.lm_head.weight)


def test_native_hook_adapter_uses_explicit_worker_dependencies(monkeypatch):
    from types import SimpleNamespace as NS
    from dneo.runtime import native_hooks, feature_layout
    hooks = []
    projection = NS(register_forward_pre_hook=lambda hook: hooks.append(hook))
    runner = NS(model=NS(model=NS(layers=[NS(self_attn=NS(qkv_proj=projection))])))
    target = NS(forward_batch_generation=lambda *args, **kwargs: None)
    worker = NS(server_args=NS(max_running_requests=1, disable_cuda_graph=True),
                _target_worker=target, model_runner=runner, block_size=4)
    model = NS(conditioner=NS(per_layer_mofa=True), draft=NS(rejection_memory_kv_only=True),
               mask_token_id=5, embed_tokens=NS(weight=NS(shape=(64, 32))))
    model.cuda = model.eval = lambda: model
    model.requires_grad_ = lambda _: model
    monkeypatch.setattr(feature_layout, "feature_layer_ids", lambda _: [0])
    monkeypatch.setattr(native_hooks.torch, "empty", lambda *args, **kwargs: None)
    monkeypatch.setattr(native_hooks.gc, "collect", lambda: None)
    native_hooks.install_hooks(worker, model, {})
    assert worker._max_batch == 1 and worker._disable_graph
    assert len(hooks) == 1 and worker.dneo_model is model


def test_package_import_does_not_import_trainer_or_native_server(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import dneo
    code = ("import dneo, sys; "
            "assert 'scripts.train_sdit_mofa' not in sys.modules; "
            "assert 'dneo.runtime.dneo_worker' not in sys.modules")
    environment = dict(os.environ, PYTHONPATH=str(Path(dneo.__file__).resolve().parents[1]))
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment, check=True, timeout=180)
def test_installed_source_integrity_manifest_covers_current_code():
    import hashlib
    import json
    from pathlib import Path
    import dneo
    root = Path(dneo.__file__).parent
    records = json.loads((root / "runtime/source_manifest.json").read_text())
    for relative, record in records.items():
        assert hashlib.sha256((root / relative.removeprefix("src/dneo/")).read_bytes()).hexdigest() == record["packaged_sha256"], relative

