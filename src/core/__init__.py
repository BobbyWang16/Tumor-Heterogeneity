"""ITH core module: base classes, I/O, preprocessing, configuration."""

from .base import BaseHeterogeneityModel
from .config import (
    DataConfig,
    ModelConfig,
    BIHConfig,
    DHIConfig,
    HABConfig,
    PTHConfig,
    SHIConfig,
    THIConfig,
    ITHFusionConfig,
)
from .io import (
    load_image,
    load_seg,
    load_nifti,
    save_nifti,
    save_nifti_affine,
    match_image_mask_pairs,
)
from .preprocess import (
    keep_largest_cc,
    locate_tumor_3d,
    clip_hu,
    crop_to_mask,
)
from .preprocess_pipeline import preprocess_ct, batch_preprocess
from .dataset import DatasetLoader

__all__ = [
    "BaseHeterogeneityModel",
    "DataConfig",
    "ModelConfig",
    "BIHConfig",
    "DHIConfig",
    "HABConfig",
    "PTHConfig",
    "SHIConfig",
    "THIConfig",
    "ITHFusionConfig",
    "load_image",
    "load_seg",
    "load_nifti",
    "save_nifti",
    "save_nifti_affine",
    "match_image_mask_pairs",
    "keep_largest_cc",
    "locate_tumor_3d",
    "clip_hu",
    "crop_to_mask",
    "preprocess_ct",
    "batch_preprocess",
    "DatasetLoader",
]
