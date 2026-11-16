"""DNeo public model and configuration; optimized runtime is loaded lazily."""
from transformers import AutoConfig, AutoModel
from .configuration_dneo import DNeoConfig
from .modeling_dneo import DNeoDraftModel

AutoConfig.register("dneo", DNeoConfig, exist_ok=True)
AutoModel.register(DNeoConfig, DNeoDraftModel, exist_ok=True)

__all__ = ["DNeoConfig", "DNeoDraftModel"]
