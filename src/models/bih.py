"""
BIH (Box-counting Heterogeneity Index) model.

Signal-weighted box-counting fractal analysis on the maximum axial slice.
"""

import numpy as np
from scipy import ndimage as ndi

from src.core import BaseHeterogeneityModel, BIHConfig, keep_largest_cc


def _signal_weighted_boxcount_2d(image_2d, mask_2d, box_sizes):
    """Signal-weighted Box-Counting (2D)."""
    mask_2d = mask_2d.astype(bool)
    if mask_2d.sum() == 0:
        return [], []

    img_masked = image_2d[mask_2d]
    img_min, img_max = img_masked.min(), img_masked.max()
    if img_max - img_min < 1e-6:
        return [], []

    image_norm = ((image_2d - img_min) / (img_max - img_min)) * 999 + 1
    NLs, valid_sizes = [], []

    for L in box_sizes:
        L = int(round(L))
        if L < 1:
            continue
        H, W = image_2d.shape
        pad_h, pad_w = (-H) % L, (-W) % L
        img_pad = np.pad(image_norm, ((0, pad_h), (0, pad_w)), mode="constant")
        mask_pad = np.pad(mask_2d, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)

        n_boxes_h, n_boxes_w = img_pad.shape[0] // L, img_pad.shape[1] // L
        img_boxes = img_pad.reshape(n_boxes_h, L, n_boxes_w, L).transpose(0, 2, 1, 3).reshape(-1, L, L)
        mask_boxes = mask_pad.reshape(n_boxes_h, L, n_boxes_w, L).transpose(0, 2, 1, 3).reshape(-1, L, L)

        valid = mask_boxes.any(axis=(1, 2))
        if not valid.any():
            continue

        # Vectorized masked mean per box
        masked_sums = (img_boxes * mask_boxes).sum(axis=(1, 2))
        mask_counts = mask_boxes.sum(axis=(1, 2))
        box_means = masked_sums[valid] / mask_counts[valid]
        NLs.append(float(box_means.mean()))
        valid_sizes.append(int(L))

    return NLs, valid_sizes


def _compute_fractal_features(image_2d, mask_2d, max_L_ratio=0.25, step=2):
    """Compute signal-weighted fractal dimension (FD) and fill coefficient (FA)."""
    mask_2d = mask_2d.astype(bool)
    if mask_2d.sum() == 0:
        return np.nan, np.nan, np.nan, [], [], [], [], np.nan

    min_dim = min(mask_2d.shape)
    max_L = max(int(min_dim * max_L_ratio), step * 3)
    Ls = list(range(step, max_L + 1, step))
    if len(Ls) < 3:
        max_L = min(min_dim // 2, step * 10)
        Ls = list(range(step, max_L + 1, step))
    if len(Ls) < 3:
        return np.nan, np.nan, np.nan, [], [], Ls, [], np.nan

    NLs, valid_Ls = _signal_weighted_boxcount_2d(image_2d, mask_2d, Ls)
    if len(NLs) < 3:
        return np.nan, np.nan, np.nan, [], [], Ls, [], np.nan

    NLs = np.array(NLs, dtype=np.float64)
    valid_Ls = np.array(valid_Ls, dtype=np.float64)
    valid_mask = NLs > 0
    if valid_mask.sum() < 3:
        return np.nan, np.nan, np.nan, [], [], valid_Ls.tolist(), NLs.tolist(), np.nan

    NLs, valid_Ls = NLs[valid_mask], valid_Ls[valid_mask]
    log_Ls, log_NLs = np.log(valid_Ls), np.log(NLs)

    coeffs = np.polyfit(log_Ls, log_NLs, 1)
    FD, log_K = -coeffs[0], coeffs[1]
    FA = np.exp(log_K)

    log_NLs_pred = np.polyval(coeffs, log_Ls)
    residual = np.sqrt(np.mean((log_NLs - log_NLs_pred) ** 2))
    ss_res = np.sum((log_NLs - log_NLs_pred) ** 2)
    ss_tot = np.sum((log_NLs - np.mean(log_NLs)) ** 2) + 1e-10
    r2 = 1.0 - ss_res / ss_tot

    return float(FD), float(FA), float(r2), log_Ls.tolist(), log_NLs.tolist(), valid_Ls.tolist(), NLs.tolist(), float(residual)


def _max_axial_slice(mask_3d):
    """Find the axial slice with maximum tumor area."""
    areas = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    z_max = int(np.argmax(areas))
    return z_max, int(areas[z_max])


class BIHModel(BaseHeterogeneityModel):
    """Box-counting Heterogeneity Index (BIH) model."""

    name = "bih"
    config_class = BIHConfig

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_names = ["fd", "logA", "r2", "fa", "n_scales", "residual"]

    def compute(self, image, mask):
        mask = keep_largest_cc(mask)
        z_max, area_max = _max_axial_slice(mask)
        fd, fa, r2, log_Ls, log_NLs, Ls, NLs, residual = _compute_fractal_features(
            image[z_max], mask[z_max]
        )

        features = {
            "fd": fd,
            "logA": np.log(fa) if fa > 0 and np.isfinite(fa) else np.nan,
            "r2": r2, "fa": fa, "n_scales": len(Ls), "residual": residual,
        }
        return {
            "score": fd if np.isfinite(fd) else np.nan,
            "features": features,
            "details": {"z_max": z_max, "area_max": area_max, "Ls": Ls, "NLs": NLs},
        }
