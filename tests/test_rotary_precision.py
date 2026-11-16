import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from dneo import DNeoConfig, DNeoDraftModel


def configuration(precision):
    target = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        vocab_size=64, max_position_embeddings=128, rope_theta=1000000.)
    return DNeoConfig(target_config=target, num_hidden_layers=2, target_layer_ids=[0, 3],
                      block_size=4, head_rank=8, folded_rank=8, conv_group_size=8,
                      mask_token_id=5, rotary_frequency_dtype=precision)


def test_explicit_rotary_precision_matches_original_whole_model_cast():
    config = configuration("bfloat16")
    expected = Qwen3RotaryEmbedding(Qwen3Config(**config.target_config)).to(torch.bfloat16)
    model = DNeoDraftModel(config)
    for name, value in expected.named_buffers():
        actual = getattr(model.draft.rotary_emb, name)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, value)
    assert not any("inv_freq" in name for name in model.state_dict())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hf_loading_preserves_checkpoint_rotary_precision(tmp_path, dtype):
    model = DNeoDraftModel(configuration("bfloat16")).to(torch.bfloat16)
    expected = model.draft.rotary_emb.inv_freq.clone()
    model.save_pretrained(tmp_path)
    loaded, info = DNeoDraftModel.from_pretrained(tmp_path, dtype=dtype, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert loaded.config.rotary_frequency_dtype == "bfloat16"
    assert loaded.draft.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert torch.equal(loaded.draft.rotary_emb.inv_freq, expected)


def test_unspecified_precision_keeps_fp32_reference_construction():
    model = DNeoDraftModel(configuration(None))
    assert model.draft.rotary_emb.inv_freq.dtype == torch.float32


def test_invalid_rotary_precision_fails_closed():
    with pytest.raises(ValueError, match="rotary_frequency_dtype"):
        configuration("fp8")
