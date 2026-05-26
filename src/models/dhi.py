"""
DHI (Density Heterogeneity Index) model.

First-order histogram features + 3D local coefficient of variation (LCV).
"""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy import stats as st

from src.core import BaseHeterogeneityModel, DHIConfig


EPS = 1e-8


def _local_cv_3d(img, msk, win=3):
    """Compute 3D local coefficient of variation (LCV)."""
    if img.ndim != 3 or msk.ndim != 3 or img.shape != msk.shape:
        return dict(lcv_mean=np.nan, lcv_median=np.nan, lcv_iqr=np.nan, lcv_p95=np.nan)

    win = max(3, win if win % 2 == 1 else win + 1)
    img = img.astype(np.float32, copy=False)
    msk_f = (msk > 0).astype(np.float32, copy=False)

    m1 = uniform_filter(img, size=win, mode="nearest")
    m2 = uniform_filter(img * img, size=win, mode="nearest")
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    lcv = sd / (m1 + EPS)

    cover = uniform_filter(msk_f, size=win, mode="nearest")
    valid = (cover >= 1.0 - 1e-6) & (msk > 0)
    vals = lcv[valid]

    if vals.size < 5:
        return dict(lcv_mean=np.nan, lcv_median=np.nan, lcv_iqr=np.nan, lcv_p95=np.nan)

    q25, q75 = np.percentile(vals, [25, 75])
    return dict(
        lcv_mean=float(vals.mean()),
        lcv_median=float(np.median(vals)),
        lcv_iqr=float(q75 - q25),
        lcv_p95=float(np.percentile(vals, 95)),
    )


def _calc_hist_features(img_arr, msk_arr, clip=(1, 99), nbins=64, win_lcv=3):
    """Extract first-order histogram + 3D local heterogeneity features."""
    if img_arr.shape != msk_arr.shape:
        raise ValueError("Image and mask must have the same shape.")

    roi = img_arr[msk_arr > 0].astype(np.float32)
    n_vox = int(roi.size)
    if n_vox < 5:
        return _empty_features(n_vox)

    lo, hi = np.percentile(roi, clip)
    roi = roi[(roi >= lo) & (roi <= hi)]
    if roi.size == 0:
        return _empty_features(n_vox)

    mean = float(roi.mean())
    median = float(np.median(roi))
    sd = float(roi.std(ddof=1)) if roi.size > 1 else 0.0
    var = sd ** 2
    skew = float(st.skew(roi)) if roi.size > 2 else 0.0
    kurt = float(st.kurtosis(roi, fisher=True)) if roi.size > 3 else 0.0
    q01, q05, q10, q25, q75, q90, q95, q99 = np.percentile(roi, [1, 5, 10, 25, 75, 90, 95, 99]).tolist()
    iqr = q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else np.nan
    rng = float(roi.max() - roi.min())
    mad = float(np.median(np.abs(roi - median)))

    hist, _ = np.histogram(roi, bins=nbins, density=False)
    prob = hist.astype(np.float32) / float(hist.sum()) if hist.sum() > 0 else np.zeros_like(hist, dtype=np.float32)
    entropy = float(-(prob * np.log2(prob + EPS)).sum())
    uniformity = float((prob ** 2).sum())
    energy = float((roi ** 2).sum())

    cv = sd / (mean + EPS)
    rcv = iqr / (median + EPS) if np.isfinite(iqr) else np.nan
    qcd = iqr / (q75 + q25 + EPS) if np.isfinite(q75) and np.isfinite(q25) else np.nan
    p95p5 = (q95 + EPS) / (q05 + EPS) if np.isfinite(q95) and np.isfinite(q05) else np.nan

    lcv_stats = _local_cv_3d(img_arr, msk_arr, win=win_lcv)

    return dict(
        n_voxels=n_vox, mean=mean, median=median, sd=sd, variance=var,
        skewness=skew, kurtosis=kurt, p01=q01, p05=q05, p10=q10, p25=q25,
        p75=q75, p90=q90, p95=q95, p99=q99, iqr=iqr, value_range=rng, mad=mad,
        entropy=entropy, uniformity=uniformity, energy=energy, CV=cv, RCV=rcv,
        QCD=qcd, P95_div_P5=p95p5, **lcv_stats,
    )


def _empty_features(n_vox):
    keys = ["mean", "median", "sd", "variance", "skewness", "kurtosis",
            "p01", "p05", "p10", "p25", "p75", "p90", "p95", "p99",
            "iqr", "value_range", "mad", "entropy", "uniformity", "energy",
            "CV", "RCV", "QCD", "P95_div_P5",
            "lcv_mean", "lcv_median", "lcv_iqr", "lcv_p95"]
    return dict(n_voxels=n_vox, **{k: np.nan for k in keys})


class DHIModel(BaseHeterogeneityModel):
    """Density Heterogeneity Index (DHI) model."""

    name = "dhi"
    config_class = DHIConfig

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_names = [
            "mean", "median", "sd", "variance", "skewness", "kurtosis",
            "entropy", "uniformity", "energy", "CV", "RCV", "QCD",
            "lcv_mean", "lcv_median", "lcv_iqr", "lcv_p95",
        ]

    def compute(self, image, mask):
        feats = _calc_hist_features(
            image, mask,
            clip=self.cfg.clip_percentiles,
            nbins=self.cfg.nbins,
            win_lcv=self.cfg.win_lcv,
        )
        score = feats.get("CV", np.nan)
        if not np.isfinite(score):
            score = np.nan
        features = {k: feats.get(k, np.nan) for k in self.feature_names}
        return {"score": score, "features": features, "details": {"n_voxels": feats.get("n_voxels", 0)}}
