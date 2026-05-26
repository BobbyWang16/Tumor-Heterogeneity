"""
THI (Texture Heterogeneity Index) model.

Per-voxel radiomic features, KMeans clustering, ITH score from spatial fragmentation.
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor

from src.core import BaseHeterogeneityModel, THIConfig, locate_tumor_3d


EPS = 1e-8


def _single_voxel_feats(vol, z, y, x, r=1):
    """Extract 3x3x3 local features for a single voxel."""
    z0, z1 = max(0, z - r), min(vol.shape[0], z + r + 1)
    y0, y1 = max(0, y - r), min(vol.shape[1], y + r + 1)
    x0, x1 = max(0, x - r), min(vol.shape[2], x + r + 1)
    patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float32).ravel()
    center = float(vol[z, y, x])

    mean = float(patch.mean()) if patch.size else 0.0
    std = float(patch.std()) if patch.size else 0.0
    vmin = float(patch.min()) if patch.size else 0.0
    vmax = float(patch.max()) if patch.size else 0.0
    p_range = vmax - vmin

    if p_range <= 0 or patch.size == 0:
        energy, entropy = 0.0, 0.0
    else:
        p_norm = (patch - vmin) / (p_range + EPS)
        hist, _ = np.histogram(p_norm, bins=8, range=(0, 1), density=True)
        hist = hist.astype(np.float32)
        energy = float((hist ** 2).sum())
        entropy = float(-(hist * np.log2(hist + EPS)).sum())

    diff = patch - center
    contrast = float((diff ** 2).mean()) if patch.size else 0.0
    homogeneity = 1.0 / (1.0 + contrast)
    grad = float(np.abs(diff).mean()) if patch.size else 0.0

    return [mean, std, vmin, vmax, p_range, energy, entropy, contrast, homogeneity, grad]


def _extract_voxel_features(sub_vol, sub_msk, workers=8):
    """Extract per-voxel radiomic features within ROI."""
    coords = np.argwhere(sub_msk > 0)
    if coords.shape[0] == 0:
        return None, None

    def worker(idx):
        return _single_voxel_feats(sub_vol, int(idx[0]), int(idx[1]), int(idx[2]))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        feats = list(ex.map(worker, coords))

    feats = np.asarray(feats, dtype=np.float32)
    return MinMaxScaler().fit_transform(feats).astype(np.float32), coords


def _voxel_clustering(features, coords, shape, n_clusters=3, random_state=2025):
    """KMeans clustering of voxel-wise features."""
    if features is None or coords is None or features.shape[0] == 0:
        return np.zeros(shape, np.uint8)

    labels = KMeans(n_clusters=int(n_clusters), n_init=20, random_state=random_state).fit_predict(features)
    lab_vol = np.zeros(shape, np.uint8)
    lab_vol[coords[:, 0], coords[:, 1], coords[:, 2]] = labels.astype(np.uint8) + 1
    return lab_vol


def _cal_ith_score(label_vol):
    """Compute ITH score: 1 - mean(max_component_fraction_per_cluster)."""
    if label_vol is None or label_vol.size == 0:
        return np.nan

    uniq = np.unique(label_vol)
    uniq = uniq[uniq > 0]
    if uniq.size == 0:
        return 0.0

    accum, total = 0.0, 0
    struct = ndimage.generate_binary_structure(3, 3)
    for val in uniq:
        lab, n = ndimage.label(label_vol == val, structure=struct)
        if n == 0:
            continue
        sizes = np.array(ndimage.sum(np.ones_like(lab, dtype=np.uint8), labels=lab, index=np.arange(1, n + 1)))
        if sizes.size == 0:
            continue
        total += 1
        accum += float(sizes.max() / (sizes.sum() + EPS))

    return float(1.0 - accum / total) if total > 0 else 0.0


class THIModel(BaseHeterogeneityModel):
    """Texture Heterogeneity Index model."""

    name = "thi"
    config_class = THIConfig

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_names = ["ith_score", "n_voxels"]

    def compute(self, image, mask):
        sub_vol, sub_msk, offset = locate_tumor_3d(image, mask, padding=self.cfg.padding)
        if sub_vol is None:
            return {"score": np.nan, "features": {"ith_score": np.nan, "n_voxels": 0},
                    "label_map": np.zeros_like(mask, dtype=np.uint8)}

        n_voxels = int((sub_msk > 0).sum())
        if n_voxels < 20:
            return {"score": np.nan, "features": {"ith_score": np.nan, "n_voxels": n_voxels},
                    "label_map": np.zeros_like(mask, dtype=np.uint8)}

        sub_vol = self._preprocess(sub_vol, sub_msk)
        feats, coords = _extract_voxel_features(sub_vol, sub_msk, workers=self.cfg.n_workers)
        sub_label_vol = _voxel_clustering(feats, coords, sub_msk.shape, n_clusters=self.cfg.k)

        z0, y0, x0 = offset
        full_label = np.zeros_like(mask, np.uint8)
        full_label[z0:z0 + sub_label_vol.shape[0], y0:y0 + sub_label_vol.shape[1], x0:x0 + sub_label_vol.shape[2]] = sub_label_vol

        ith_score = _cal_ith_score(full_label)
        return {"score": float(ith_score), "features": {"ith_score": float(ith_score), "n_voxels": int((mask > 0).sum())},
                "label_map": full_label}

    def _preprocess(self, sub_vol, sub_msk):
        roi = sub_vol[sub_msk > 0].astype(np.float32)
        if roi.size == 0:
            return sub_vol
        clip = self.cfg.clip
        if clip and len(clip) == 2:
            lo, hi = np.percentile(roi, clip)
            sub_vol[sub_msk > 0] = np.clip(sub_vol[sub_msk > 0], lo, hi)
        roi = sub_vol[sub_msk > 0].astype(np.float32)
        mean, std = float(roi.mean()), float(roi.std(ddof=1) + EPS)
        sub_vol[sub_msk > 0] = (sub_vol[sub_msk > 0] - mean) / std
        return sub_vol
