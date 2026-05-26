"""
Shared preprocessing utilities for tumor analysis.
"""

from typing import Tuple

import numpy as np
from scipy import ndimage as ndi


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    lab, n = ndi.label(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def locate_tumor_3d(
    vol: np.ndarray, msk: np.ndarray, padding: int = 2
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Crop volume and mask to tumor bounding box."""
    coords = np.argwhere(msk > 0)
    if coords.size == 0:
        return None, None, None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)

    z0 = max(z0 - padding, 0)
    y0 = max(y0 - padding, 0)
    x0 = max(x0 - padding, 0)
    z1 = min(z1 + padding + 1, vol.shape[0])
    y1 = min(y1 + padding + 1, vol.shape[1])
    x1 = min(x1 + padding + 1, vol.shape[2])

    return vol[z0:z1, y0:y1, x0:x1], msk[z0:z1, y0:y1, x0:x1], (z0, y0, x0)


def clip_hu(image: np.ndarray, mask: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    """Clip HU values and set background to hu_min."""
    clipped = np.clip(image, hu_min, hu_max)
    clipped[mask == 0] = hu_min
    return clipped


def crop_to_mask(
    img: np.ndarray, mask: np.ndarray, margin: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop image to mask extent with margin."""
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return img, mask
    z0 = max(idx[0].min() - margin, 0)
    z1 = min(idx[0].max() + margin + 1, img.shape[0])
    y0 = max(idx[1].min() - margin, 0)
    y1 = min(idx[1].max() + margin + 1, img.shape[1])
    x0 = max(idx[2].min() - margin, 0)
    x1 = min(idx[2].max() + margin + 1, img.shape[2])
    return img[z0:z1, y0:y1, x0:x1], mask[z0:z1, y0:y1, x0:x1]
