"""ITH scoring models."""

from .bih import BIHModel
from .dhi import DHIModel
from .hab import HABModel
from .pth import PTHModel
from .shi import SHIModel
from .thi import THIModel
from .ith_fusion import ITHFusionModel

__all__ = [
    "BIHModel", "DHIModel", "HABModel", "PTHModel",
    "SHIModel", "THIModel", "ITHFusionModel",
]

MODEL_REGISTRY = {
    "bih": BIHModel,
    "dhi": DHIModel,
    "hab": HABModel,
    "pth": PTHModel,
    "shi": SHIModel,
    "thi": THIModel,
    "ith_fusion": ITHFusionModel,
}


def get_model(name: str):
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
