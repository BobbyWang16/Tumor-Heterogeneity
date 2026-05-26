"""
ITH 项目工具模块

公共I/O和预处理工具，被各评分模型共享。
"""

from .io import (
    load_image,
    load_nifti,
    load_seg,
    save_nifti,
    save_nifti_from_affine,
    match_image_mask_pairs,
)

from .preprocessing import (
    keep_largest_cc,
    locate_tumor_3d,
    clip_hu,
    crop_to_mask,
)

__all__ = [
    "load_image", "load_nifti", "load_seg",
    "save_nifti", "save_nifti_from_affine",
    "match_image_mask_pairs",
    "keep_largest_cc", "locate_tumor_3d",
    "clip_hu", "crop_to_mask",
]
