"""Portable inference-only configuration. No training arguments or local paths."""
from transformers import PretrainedConfig, Qwen3Config


class DNeoConfig(PretrainedConfig):
    model_type = "dneo"

    def __init__(self, target_model_name_or_path="Qwen/Qwen3-4B",
                 target_revision="1cfa9a7208912126459214e8b04321603b3df60c",
                 target_config=None, block_size=16, num_hidden_layers=5,
                 target_layer_ids=None, head_rank=256, folded_rank=256,
                 conv_group_size=16, mask_token_id=151669,
                 rotary_frequency_dtype=None, shared_layer_mofa=False, **kwargs):
        super().__init__(**kwargs)
        self.target_model_name_or_path = target_model_name_or_path
        self.target_revision = target_revision
        self.target_config = (target_config.to_dict() if hasattr(target_config, "to_dict") else target_config) or Qwen3Config(
            hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
            num_attention_heads=32, num_key_value_heads=8, head_dim=128,
            max_position_embeddings=40960, rope_theta=1000000., vocab_size=151936,
            tie_word_embeddings=True).to_dict()
        self.block_size = int(block_size)
        self.num_hidden_layers = int(num_hidden_layers)
        if not isinstance(shared_layer_mofa, bool):
            raise ValueError("shared_layer_mofa must be a boolean")
        self.shared_layer_mofa = shared_layer_mofa
        self.target_layer_ids = list(target_layer_ids if target_layer_ids is not None else [0, 17, 21, 24, 28, 31, 35])
        self.head_rank, self.folded_rank = int(head_rank), int(folded_rank)
        self.conv_group_size, self.mask_token_id = int(conv_group_size), int(mask_token_id)
        if rotary_frequency_dtype not in (None, "float32", "bfloat16"):
            raise ValueError("rotary_frequency_dtype must be float32 or bfloat16")
        self.rotary_frequency_dtype = rotary_frequency_dtype
        config = Qwen3Config(**self.target_config)
        for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                    "rms_norm_eps", "vocab_size", "max_position_embeddings", "hidden_act"):
            setattr(self, key, getattr(config, key))
        self.rope_theta = getattr(config, "rope_theta", (getattr(config, "rope_parameters", None) or {}).get("rope_theta", 1000000.))
        self.num_target_layers = config.num_hidden_layers
        self.dflash_config = dict(block_size=self.block_size, mask_token_id=self.mask_token_id,
                                  target_layer_ids=self.target_layer_ids, num_target_layers=self.num_target_layers)
        self.architectures = ["DNeoDraftModel"]
        if self.block_size < 2 or self.num_hidden_layers < 1 or self.head_rank < 1 or self.folded_rank < 1:
            raise ValueError("invalid DNeo dimensions")
        if self.conv_group_size < 1 or self.hidden_size % self.conv_group_size:
            raise ValueError("conv group size must divide hidden width")
        if self.num_hidden_layers > self.num_target_layers or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("invalid draft depth or grouped-query attention dimensions")
        if getattr(config, "attention_bias", False):
            raise ValueError("this optimized DNeo format requires bias-free Qwen3 projections")
        if not self.target_layer_ids or sorted(set(self.target_layer_ids)) != self.target_layer_ids:
            raise ValueError("target layers must be sorted and unique")
        if self.target_layer_ids[0] < 0 or self.target_layer_ids[-1] >= self.num_target_layers:
            raise ValueError("target capture outside target depth")
