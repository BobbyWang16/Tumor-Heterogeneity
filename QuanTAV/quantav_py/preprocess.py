from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage as ndi

from .config import Spacing


def as_bool_mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got shape {mask.shape}")
    return np.asarray(mask > 0, dtype=bool)


def ensure_same_shape(*arrays: np.ndarray) -> None:
    shapes = {tuple(a.shape) for a in arrays if a is not None}
    if len(shapes) > 1:
        raise ValueError(f"All arrays must have the same shape, got {sorted(shapes)}")


def resample_to_spacing(
    array: np.ndarray,
    spacing: Spacing,
    target_spacing: Spacing,
    order: int,
) -> tuple[np.ndarray, Spacing]:
    """Resample a 3D array to target spacing using scipy zoom."""

    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {array.shape}")
    zoom = tuple(float(s) / float(t) for s, t in zip(spacing, target_spacing))
    result = ndi.zoom(array, zoom=zoom, order=order)
    if order == 0:
        result = result > 0
    return result, target_spacing


def largest_component(mask: np.ndarray, connectivity: int = 3) -> np.ndarray:
    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labeled, n_labels = ndi.label(mask, structure=structure)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labeled == keep


def remove_small_components(
    mask: np.ndarray,
    min_size: int,
    connectivity: int = 1,
) -> np.ndarray:
    if min_size <= 1:
        return mask.astype(bool, copy=False)
    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labeled, n_labels = ndi.label(mask, structure=structure)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labeled]


def bbox_from_mask(mask: np.ndarray, margin_voxels: Tuple[int, int, int] = (0, 0, 0)):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, s) for s in mask.shape)
    lo = coords.min(axis=0) - np.asarray(margin_voxels)
    hi = coords.max(axis=0) + np.asarray(margin_voxels) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

