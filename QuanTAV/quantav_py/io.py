from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from .config import Spacing


def load_volume(path: str | Path) -> tuple[np.ndarray, Spacing, Any]:
    """Load a 3D image or mask.

    Returns ``(array, spacing, metadata)``. Spacing is always in array axis
    order. SimpleITK arrays are returned as ``(z, y, x)`` and their spacing is
    reversed to match that order.
    """

    p = Path(path)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".npy"):
        arr = np.load(p)
        return arr, (1.0, 1.0, 1.0), {"format": "npy"}

    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(image)
        spacing_xyz = tuple(float(v) for v in image.GetSpacing())
        spacing = spacing_xyz[::-1]
        return arr, spacing, {"format": "sitk", "image": image}
    except Exception as sitk_error:
        try:
            import nibabel as nib

            img = nib.load(str(p))
            arr = np.asanyarray(img.dataobj)
            zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
            return arr, zooms, {"format": "nibabel", "image": img}
        except Exception as nib_error:
            raise RuntimeError(
                f"Could not load volume {p}. SimpleITK error: {sitk_error}; "
                f"nibabel error: {nib_error}"
            ) from nib_error


def save_features_csv(
    path: str | Path,
    feature_names: list[str],
    feature_values: np.ndarray,
    case_id: str = "case",
) -> None:
    """Save one feature vector as a CSV file with column names."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    values = [case_id] + [float(v) for v in feature_values]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", *feature_names])
        writer.writerow(values)


def save_volume_like(
    path: str | Path,
    array: np.ndarray,
    reference_metadata: Optional[Any] = None,
) -> None:
    """Save a volume, using SimpleITK reference metadata when available."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if reference_metadata and reference_metadata.get("format") == "sitk":
        import SimpleITK as sitk

        ref = reference_metadata["image"]
        image = sitk.GetImageFromArray(array)
        image.CopyInformation(ref)
        sitk.WriteImage(image, str(p))
        return

    if "".join(p.suffixes).lower().endswith(".npy"):
        np.save(p, array)
        return

    try:
        import nibabel as nib

        affine = np.eye(4)
        if reference_metadata and reference_metadata.get("format") == "nibabel":
            affine = reference_metadata["image"].affine
        nib.Nifti1Image(array, affine).to_filename(str(p))
    except Exception as exc:
        raise RuntimeError(f"Could not save volume to {p}: {exc}") from exc

