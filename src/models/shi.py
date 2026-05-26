"""
SHI (Shape Heterogeneity Index) model.

Geometric features, convex hull, fractal dimension, and lacunarity from 3D mask.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull

from src.core import BaseHeterogeneityModel, SHIConfig


EPS = 1e-8


def _surface_area_voxel(mask, spacing):
    if mask.sum() == 0:
        return 0.0
    m = mask.astype(np.uint8)
    s0, s1, s2 = spacing
    area = 0.0
    m0 = np.pad(m, ((1, 1), (0, 0), (0, 0)), mode="constant")
    area += float(np.abs(m0[1:] - m0[:-1]).sum()) * (s1 * s2)
    m1 = np.pad(m, ((0, 0), (1, 1), (0, 0)), mode="constant")
    area += float(np.abs(m1[:, 1:] - m1[:, :-1]).sum()) * (s0 * s2)
    m2 = np.pad(m, ((0, 0), (0, 0), (1, 1)), mode="constant")
    area += float(np.abs(m2[:, :, 1:] - m2[:, :, :-1]).sum()) * (s0 * s1)
    return area


def _geometric_features(mask, spacing):
    s0, s1, s2 = spacing
    vox_vol = s0 * s1 * s2
    vol_mm3 = float(mask.sum()) * float(vox_vol)
    sa_mm2 = _surface_area_voxel(mask, spacing)
    sv_ratio = float(sa_mm2 / (vol_mm3 + EPS)) if vol_mm3 > 0 else np.nan
    sphericity = float((np.pi ** (1 / 3)) * ((6.0 * vol_mm3) ** (2 / 3)) / (sa_mm2 + EPS)) if vol_mm3 > 0 else np.nan
    iq = float(36.0 * np.pi * (vol_mm3 ** 2) / ((sa_mm2 + EPS) ** 3)) if vol_mm3 > 0 else np.nan
    compactness1 = float((sa_mm2 ** 1.5) / (vol_mm3 + EPS)) if vol_mm3 > 0 else np.nan
    coords = np.argwhere(mask)
    if coords.size > 0:
        mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
        bbox_mm = (maxs - mins).astype(int)
        bbox_vol = (bbox_mm[0] * s0) * (bbox_mm[1] * s1) * (bbox_mm[2] * s2)
        extent = float(vol_mm3 / (bbox_vol + EPS))
    else:
        extent = np.nan
    return dict(volume_mm3=vol_mm3, surface_mm2=sa_mm2, SV_ratio=sv_ratio,
                sphericity=sphericity, isoperimetric_quotient=iq,
                compactness1=compactness1, extent=extent)


def _convex_hull_features(mask, spacing, max_points=100000):
    coords = np.argwhere(mask)
    if coords.shape[0] < 4:
        return dict(convex_volume_mm3=np.nan, convex_surface_mm2=np.nan, solidity=np.nan, surf_convexity=np.nan)
    pts = coords.astype(np.float64) * np.array(spacing, dtype=np.float64)
    if pts.shape[0] > max_points:
        sel = np.random.RandomState(2025).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[sel]
    try:
        hull = ConvexHull(pts)
        conv_vol, conv_area = float(hull.volume), float(hull.area)
    except Exception:
        return dict(convex_volume_mm3=np.nan, convex_surface_mm2=np.nan, solidity=np.nan, surf_convexity=np.nan)
    geom = _geometric_features(mask, spacing)
    solidity = float(geom["volume_mm3"] / (conv_vol + EPS)) if conv_vol > 0 else np.nan
    surf_convexity = float((conv_area + EPS) / (geom["surface_mm2"] + EPS)) if np.isfinite(conv_area) else np.nan
    return dict(convex_volume_mm3=conv_vol, convex_surface_mm2=conv_area, solidity=solidity, surf_convexity=surf_convexity)


def _auto_scales(mask, min_box=1, max_box=None):
    dmin = int(np.min(mask.shape))
    if max_box is None:
        max_box = max(2, dmin // 2)
    scales, s = [], max(1, min_box)
    while s <= max_box:
        scales.append(s)
        s *= 2
    if len(scales) < 3 and dmin >= 4:
        scales = [1, 2, 4][:min(3, dmin)]
    return scales


def _box_counting(mask, scales):
    Ns, logs = [], []
    for s in scales:
        if s <= 0:
            continue
        pad = [(0, (-d) % s) for d in mask.shape]
        m = np.pad(mask, pad, mode="constant")
        try:
            boxes = m.reshape(m.shape[0] // s, s, m.shape[1] // s, s, m.shape[2] // s, s).swapaxes(1, 2).swapaxes(3, 4).reshape(-1, s, s, s)
        except Exception:
            continue
        hit = boxes.sum(axis=(1, 2, 3))
        N = int(np.count_nonzero(hit))
        if N <= 0:
            continue
        Ns.append(N)
        logs.append(np.log2(s + EPS))
    return Ns, logs


def _lacunarity(hit_counts, add_one=True):
    mu = float(hit_counts.mean())
    if mu <= 0:
        return np.nan
    val = float(hit_counts.var() / (mu * mu))
    return val + 1.0 if add_one else val


def _fractal_and_lacunarity(mask, scales=None, connectivity=1, add_one_in_lac=True, compute_surface_fd=True):
    if mask.sum() == 0:
        return dict(FD=np.nan, FD_R2=np.nan, LAC_mean=np.nan, LAC_slope=np.nan,
                    LAC_small=np.nan, LAC_large=np.nan, LAC_AUC=np.nan, LAC_R2=np.nan,
                    FD_surface=np.nan, FD_surface_R2=np.nan)

    mask = ndi.binary_closing(mask.astype(bool), iterations=1, structure=ndi.generate_binary_structure(3, connectivity))
    scales = _auto_scales(mask) if scales is None else list(scales)

    # Volume FD
    Ns, logs = _box_counting(mask, scales)
    if len(Ns) >= 2:
        x, y = np.array(logs, dtype=np.float64), np.log2(np.array(Ns, dtype=np.float64) + EPS)
        coef = np.polyfit(x, y, 1)
        yhat = np.polyval(coef, x)
        FD, FD_R2 = float(-coef[0]), float(1.0 - np.sum((y - yhat) ** 2) / (np.sum((y - y.mean()) ** 2) + EPS))
    else:
        FD, FD_R2 = np.nan, np.nan

    # Lacunarity
    lacs, x_lac = [], []
    for s in scales:
        if s <= 0:
            continue
        pad = [(0, (-d) % s) for d in mask.shape]
        m = np.pad(mask, pad, mode="constant", constant_values=0)
        try:
            boxes = m.reshape(m.shape[0] // s, s, m.shape[1] // s, s, m.shape[2] // s, s).swapaxes(1, 2).swapaxes(3, 4).reshape(-1, s, s, s)
        except Exception:
            continue
        hit = boxes.sum(axis=(1, 2, 3)).astype(np.float64)
        lac = _lacunarity(hit, add_one=add_one_in_lac)
        if np.isfinite(lac):
            lacs.append(float(lac))
            x_lac.append(np.log2(s + EPS))

    if len(lacs) >= 2:
        x_l, y_l = np.array(x_lac, dtype=np.float64), np.log2(np.array(lacs, dtype=np.float64) + EPS)
        coef_l = np.polyfit(x_l, y_l, 1)
        yhat_l = np.polyval(coef_l, x_l)
        LAC_slope = float(coef_l[0])
        LAC_R2 = float(1.0 - np.sum((y_l - yhat_l) ** 2) / (np.sum((y_l - y_l.mean()) ** 2) + EPS))
        LAC_mean, LAC_small, LAC_large, LAC_AUC = float(np.mean(lacs)), float(lacs[0]), float(lacs[-1]), float(np.trapz(y_l, x_l))
    else:
        LAC_mean = LAC_slope = LAC_small = LAC_large = LAC_AUC = LAC_R2 = np.nan

    # Surface FD
    FD_surface = FD_surface_R2 = np.nan
    if compute_surface_fd:
        surface = mask & (~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, connectivity)))
        Ns_s, logs_s = _box_counting(surface, scales)
        if len(Ns_s) >= 2:
            x_s, y_s = np.array(logs_s, dtype=np.float64), np.log2(np.array(Ns_s, dtype=np.float64) + EPS)
            coef_s = np.polyfit(x_s, y_s, 1)
            yhat_s = np.polyval(coef_s, x_s)
            FD_surface = float(-coef_s[0])
            FD_surface_R2 = float(1.0 - np.sum((y_s - yhat_s) ** 2) / (np.sum((y_s - y_s.mean()) ** 2) + EPS))

    return dict(FD=FD, FD_R2=FD_R2, LAC_mean=LAC_mean, LAC_slope=LAC_slope,
                LAC_small=LAC_small, LAC_large=LAC_large, LAC_AUC=LAC_AUC, LAC_R2=LAC_R2,
                FD_surface=FD_surface, FD_surface_R2=FD_surface_R2)


class SHIModel(BaseHeterogeneityModel):
    """Shape Heterogeneity Index model."""

    name = "shi"
    config_class = SHIConfig

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_names = [
            "volume_mm3", "surface_mm2", "SV_ratio", "sphericity", "isoperimetric_quotient",
            "compactness1", "extent", "convex_volume_mm3", "convex_surface_mm2", "solidity",
            "surf_convexity", "FD", "FD_R2", "LAC_mean", "LAC_slope", "LAC_small",
            "LAC_large", "LAC_AUC", "LAC_R2", "FD_surface", "FD_surface_R2",
        ]

    def compute(self, image, mask):
        spacing = (1.0, 1.0, 1.0)
        geom = _geometric_features(mask, spacing)
        convex = _convex_hull_features(mask, spacing)
        frac = _fractal_and_lacunarity(mask, scales=self.cfg.scales)
        features = {**geom, **convex, **frac}

        score, n_terms = 0.0, 0
        if np.isfinite(features.get("sphericity", np.nan)):
            score += 1.0 - features["sphericity"]; n_terms += 1
        if np.isfinite(features.get("solidity", np.nan)):
            score += 1.0 - features["solidity"]; n_terms += 1
        if np.isfinite(features.get("FD_surface", np.nan)):
            score += features["FD_surface"] / 3.0; n_terms += 1

        return {"score": float(score / n_terms) if n_terms > 0 else np.nan, "features": features}
