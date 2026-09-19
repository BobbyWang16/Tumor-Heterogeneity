"""
CT and mask preprocessing pipeline.

Steps:
1. HU windowing: clip to [-1200, 800]
2. Resample to 1x1x1 mm isotropic (CT: Bspline, Mask: NearestNeighbor)
3. Re-orient to RAI (direction: 1 0 0 0 1 0 0 0 1)
4. Save to image_process / mask_process
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from .io import match_image_mask_pairs


def _get_rai_orienter():
    """Create RAI orientation filter (direction = 1 0 0 0 1 0 0 0 1)."""
    orienter = sitk.OrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAI")
    return orienter


def preprocess_ct(
    image_path: str,
    mask_path: str,
    output_image_path: str,
    output_mask_path: str,
    hu_min: float = -1200.0,
    hu_max: float = 800.0,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """
    Preprocess a single CT + mask pair.

    Args:
        image_path: input CT path
        mask_path: input mask path
        output_image_path: output CT path
        output_mask_path: output mask path
        hu_min: HU lower bound for windowing
        hu_max: HU upper bound for windowing
        target_spacing: target voxel spacing (mm)
    """
    # Read
    ct = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # Ensure same physical space
    mask.CopyInformation(ct)

    # 1. HU windowing
    ct = sitk.Clamp(ct, sitk.sitkInt16, int(hu_min), int(hu_max))

    # 2. Resample to isotropic 1mm
    orig_spacing = np.array(ct.GetSpacing())
    orig_size = np.array(ct.GetSize())
    new_size = np.round(orig_size * orig_spacing / np.array(target_spacing)).astype(int).tolist()

    # CT: Bspline interpolation
    resampler_ct = sitk.ResampleImageFilter()
    resampler_ct.SetInterpolator(sitk.sitkBSpline)
    resampler_ct.SetOutputSpacing(target_spacing)
    resampler_ct.SetSize(new_size)
    resampler_ct.SetOutputDirection(ct.GetDirection())
    resampler_ct.SetOutputOrigin(ct.GetOrigin())
    ct_resampled = resampler_ct.Execute(ct)

    # Mask: NearestNeighbor interpolation
    resampler_mask = sitk.ResampleImageFilter()
    resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_mask.SetOutputSpacing(target_spacing)
    resampler_mask.SetSize(new_size)
    resampler_mask.SetOutputDirection(mask.GetDirection())
    resampler_mask.SetOutputOrigin(mask.GetOrigin())
    mask_resampled = resampler_mask.Execute(mask)

    # 3. Re-orient to RAI
    orienter = _get_rai_orienter()
    ct_rai = orienter.Execute(ct_resampled)
    mask_rai = orienter.Execute(mask_resampled)

    # Ensure mask is binary
    mask_rai = sitk.Cast(sitk.Clamp(mask_rai, sitk.sitkUInt8, 0, 1), sitk.sitkUInt8)

    # 4. Save
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    sitk.WriteImage(ct_rai, output_image_path)
    sitk.WriteImage(mask_rai, output_mask_path)


def batch_preprocess(
    dataset: str,
    data_root: str = "./data",
    hu_min: float = -1200.0,
    hu_max: float = 800.0,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    image_suffix: str = ".nii.gz",
    mask_suffix: str = ".nii.gz",
) -> None:
    """
    Batch preprocess all cases in a dataset.

    Output:
        data_root/dataset/image_process/
        data_root/dataset/mask_process/
    """
    base = Path(data_root) / dataset
    image_dir = base / "image"
    mask_dir = base / "mask"
    out_image_dir = base / "image_process"
    out_mask_dir = base / "mask_process"

    pairs = match_image_mask_pairs(str(image_dir), str(mask_dir), image_suffix, mask_suffix)
    if not pairs:
        raise RuntimeError(f"No pairs found in {image_dir} / {mask_dir}")

    print(f"[{dataset}] Preprocessing {len(pairs)} cases...")
    print(f"  HU window: [{hu_min}, {hu_max}]")
    print(f"  Target spacing: {target_spacing}")
    print(f"  Output: {out_image_dir}, {out_mask_dir}")

    failures = []
    for case_id, img_path, msk_path in tqdm(pairs, desc=f"[{dataset}]"):
        out_img = out_image_dir / (case_id + image_suffix)
        out_msk = out_mask_dir / (case_id + mask_suffix)

        # Skip if already processed
        if out_img.exists() and out_msk.exists():
            continue

        try:
            preprocess_ct(
                img_path, msk_path,
                str(out_img), str(out_msk),
                hu_min, hu_max, target_spacing,
            )
        except Exception as e:
            tqdm.write(f"  [ERR] {case_id}: {e}")
            failures.append(case_id)

    if failures:
        raise RuntimeError(f"{dataset}: preprocessing failed for {len(failures)} case(s): {', '.join(failures)}")
    print(f"[{dataset}] Done.")
