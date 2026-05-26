"""
Unified medical image I/O operations.

All image loading/saving goes through this module to ensure consistent
orientation, spacing handling, and format support.
"""

import os
import glob
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import SimpleITK as sitk
import nibabel as nib


def load_image(path: str) -> Tuple[np.ndarray, sitk.Image]:
    """Load CT image volume from NIfTI or DICOM series."""
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(names)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def load_seg(path: str) -> Tuple[np.ndarray, sitk.Image]:
    """Load binary segmentation mask."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    arr = (arr > 0).astype(np.uint8)
    return arr, img


def load_nifti(path: str) -> Tuple[sitk.Image, np.ndarray, Tuple[float, ...]]:
    """Load NIfTI with full metadata."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return img, arr, spacing


def save_nifti(array: np.ndarray, reference: sitk.Image, path: str) -> None:
    """Save array as NIfTI preserving spatial metadata."""
    out = sitk.GetImageFromArray(array.astype(np.int16))
    out.CopyInformation(reference)
    sitk.WriteImage(out, path)


def save_nifti_affine(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """Save using nibabel with explicit affine."""
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), path)


def match_image_mask_pairs(
    image_dir: str,
    mask_dir: str,
    image_suffix: str = ".nii.gz",
    mask_suffix: str = ".nii.gz",
) -> List[Tuple[str, str, str]]:
    """Match image and mask files by case ID.

    Returns list of (case_id, image_path, mask_path).
    """
    images = sorted(glob.glob(os.path.join(image_dir, f"*{image_suffix}")))
    masks = sorted(glob.glob(os.path.join(mask_dir, f"*{mask_suffix}")))

    def _extract_id(p: str, suffix: str) -> str:
        name = Path(p).name
        if name.endswith(suffix):
            name = name[:-len(suffix)]
        for tag in ["_ct", "_resample", "_mask", "_image", "_seg", "_label", "_tumor"]:
            name = name.replace(tag, "")
        return name.strip("_")

    id2img = {_extract_id(p, image_suffix): p for p in images}
    id2msk = {_extract_id(p, mask_suffix): p for p in masks}

    pairs = []
    for cid in sorted(set(id2img.keys()) & set(id2msk.keys())):
        pairs.append((cid, id2img[cid], id2msk[cid]))
    return pairs
