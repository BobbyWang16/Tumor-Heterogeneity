"""
PTH (Peritumoral Heterogeneity) model.

Creates a peritumoral ring, segments it with SLIC supervoxels,
computes block-level features, then measures inter-block heterogeneity.
"""

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import entropy
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops

from src.core import BaseHeterogeneityModel, PTHConfig


class PTHModel(BaseHeterogeneityModel):
    """Peritumoral heterogeneity model."""

    name = "pth"
    config_class = PTHConfig

    def compute(self, image, mask):
        ring_mask = self._create_ring(mask, self.cfg.ring_mm)
        if ring_mask.sum() < 100:
            return {"score": 0.0, "features": {}, "label_map": np.zeros_like(mask, dtype=np.int16), "details": {"n_blocks": 0}}

        seg_labels, df_blocks, n_seg, crop_box = self._compute_block_features(
            image, ring_mask, self.cfg.target_block_size_mm3, self.cfg.compactness
        )

        # Map cropped seg back to full image size
        if seg_labels is not None and crop_box is not None:
            z_min, y_min, x_min, z_max, y_max, x_max = crop_box
            seg_full = np.zeros_like(mask, dtype=np.int16)
            seg_full[z_min:z_max, y_min:y_max, x_min:x_max] = seg_labels
            seg_fallback = seg_full
        else:
            seg_fallback = np.zeros_like(mask, dtype=np.int16)

        if df_blocks.empty:
            return {"score": 0.0, "features": {}, "label_map": seg_fallback, "details": {"n_blocks": 0}}

        pth_features = self._compute_pth_features(df_blocks)
        self.feature_names = list(pth_features.keys())
        cv_keys = [k for k in pth_features.keys() if k.startswith("PTH_CV_")]
        score = float(np.mean([pth_features[k] for k in cv_keys])) if cv_keys else 0.0

        return {
            "score": score, "features": pth_features,
            "label_map": seg_fallback,
            "details": {"n_blocks": n_seg, "mean_block_size": float(df_blocks["n_vox"].mean())},
        }

    @staticmethod
    def _create_ring(tumor_mask, ring_mm=5.0, spacing=(1.0, 1.0, 1.0)):
        tumor_mask = tumor_mask.astype(bool)
        sx, sy, sz = spacing
        it_x, it_y, it_z = int(np.round(ring_mm / sx)), int(np.round(ring_mm / sy)), int(np.round(ring_mm / sz))
        rz, ry, rx = max(it_z, 0), max(it_y, 0), max(it_x, 0)
        if rx == 0 and ry == 0 and rz == 0:
            return np.zeros_like(tumor_mask, dtype=bool)

        zz, yy, xx = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
        selem = (xx / max(rx, 1)) ** 2 + (yy / max(ry, 1)) ** 2 + (zz / max(rz, 1)) ** 2 <= 1
        dilated = ndi.binary_dilation(tumor_mask, structure=selem)
        return dilated & (~tumor_mask)

    @staticmethod
    def _kurtosis(x):
        x = x.astype(np.float64)
        if x.size < 4:
            return 0.0
        m, v = x.mean(), x.var()
        return np.mean((x - m) ** 4) / (v ** 2) - 3.0 if v > 1e-12 else 0.0

    @classmethod
    def _compute_block_features(cls, image_hu, roi_mask, target_block_size_mm3=5000, compactness=15, sigma=1.0):
        roi_mask = roi_mask.astype(bool)
        roi_voxels = int(roi_mask.sum())
        if roi_voxels < 100:
            return None, pd.DataFrame(), 0, None

        # Crop to bounding box to reduce memory usage for large volumes
        coords = np.argwhere(roi_mask)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0) + 1
        crop_box = (int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max))
        img_crop = image_hu[z_min:z_max, y_min:y_max, x_min:x_max].astype(np.float32, copy=False)
        mask_crop = roi_mask[z_min:z_max, y_min:y_max, x_min:x_max]

        n_segments = max(20, min(int(roi_voxels / target_block_size_mm3), 100))
        seg = slic(img_crop, n_segments=n_segments, compactness=compactness,
                   sigma=sigma, mask=mask_crop, start_label=1, channel_axis=None)

        gx = ndi.sobel(img_crop, axis=2)
        gy = ndi.sobel(img_crop, axis=1)
        gz = ndi.sobel(img_crop, axis=0)
        grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9

        q = np.clip(((img_crop + 1000) / 1400) * 31, 0, 31).astype(np.uint8)
        rows = []

        for lab in np.unique(seg[mask_crop]):
            m = seg == lab
            vals = img_crop[m]
            if vals.size < 30:
                continue

            mean_hu = float(vals.mean())
            std_hu = float(vals.std())
            p10, p90 = float(np.percentile(vals, 10)), float(np.percentile(vals, 90))
            kurt = float(cls._kurtosis(vals))
            vol = int(vals.size)

            gm = grad_mag[m]
            grad_mean, grad_std = float(gm.mean()), float(gm.std())

            gxx = float(np.mean(gx[m] * gx[m]))
            gyy = float(np.mean(gy[m] * gy[m]))
            gzz = float(np.mean(gz[m] * gz[m]))
            gxy = float(np.mean(gx[m] * gy[m]))
            gxz = float(np.mean(gx[m] * gz[m]))
            gyz = float(np.mean(gy[m] * gz[m]))
            w = np.sort(np.linalg.eigvalsh(np.array([[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]], dtype=np.float64)))[::-1]
            grad_anis = float((w[0] - w[2]) / (w.sum() + 1e-9))

            glcm_contrast, glcm_homogeneity = 0.0, 0.0
            if vol >= 50:
                z_ids = np.unique(np.where(m)[0])
                contrasts, homos = [], []
                for z in z_ids:
                    m2 = m[z]
                    if m2.sum() < 20:
                        continue
                    try:
                        glcm = graycomatrix(q[z], distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                            levels=32, symmetric=True, normed=True)
                        contrasts.append(graycoprops(glcm, "contrast").mean())
                        homos.append(graycoprops(glcm, "homogeneity").mean())
                    except Exception:
                        pass
                if contrasts:
                    glcm_contrast = float(np.mean(contrasts))
                    glcm_homogeneity = float(np.mean(homos))

            rows.append({
                "block_id": int(lab), "n_vox": vol, "mean_hu": mean_hu, "std_hu": std_hu,
                "p10_hu": p10, "p90_hu": p90, "kurtosis_hu": kurt,
                "grad_mean": grad_mean, "grad_std": grad_std, "grad_anisotropy": grad_anis,
                "glcm_contrast": glcm_contrast, "glcm_homogeneity": glcm_homogeneity,
            })

        return seg, pd.DataFrame(rows), n_segments, crop_box

    @staticmethod
    def _compute_pth_features(df_blocks):
        if df_blocks.empty:
            return {}

        feature_cols = ["mean_hu", "std_hu", "p10_hu", "p90_hu", "kurtosis_hu",
                        "grad_mean", "grad_std", "grad_anisotropy", "glcm_contrast", "glcm_homogeneity"]
        pth = {}
        for col in feature_cols:
            if col not in df_blocks.columns:
                continue
            x = df_blocks[col].values
            if len(x) < 2:
                continue
            m = x.mean()
            pth[f"PTH_CV_{col}"] = float(x.std() / (abs(m) + 1e-9)) if abs(m) > 1e-9 else 0.0
            p25, p75 = np.percentile(x, [25, 75])
            pth[f"PTH_IQR_{col}"] = float(p75 - p25)
            p10, p90 = np.percentile(x, [10, 90])
            pth[f"PTH_P90P10_{col}"] = float(p90 - p10)
            pth[f"PTH_range_{col}"] = float(x.max() - x.min())
            if x.std() > 1e-9:
                hist, _ = np.histogram(x, bins=min(10, len(x) // 5 + 1))
                hist = hist + 1e-9
                hist = hist / hist.sum()
                pth[f"PTH_entropy_{col}"] = float(entropy(hist, base=2))

        pth["PTH_n_blocks"] = len(df_blocks)
        pth["PTH_mean_block_size"] = float(df_blocks["n_vox"].mean())
        pth["PTH_std_block_size"] = float(df_blocks["n_vox"].std())
        return pth
