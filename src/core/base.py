"""
Base model abstraction for all heterogeneity scoring models.

All models inherit from BaseHeterogeneityModel and implement:
  - compute(image, mask) -> dict: score a single case
  - get_feature_names() -> list[str]: names of extracted features

Batch processing, I/O, and result saving are handled by the base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io import load_image, load_seg, match_image_mask_pairs
from .config import DataConfig, ModelConfig


class BaseHeterogeneityModel(ABC):
    """Abstract base class for all ITH scoring models."""

    name: str = "base"
    config_class = ModelConfig

    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or self.config_class()
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------ #
    #  Subclass hooks
    # ------------------------------------------------------------------ #

    @abstractmethod
    def compute(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Compute heterogeneity score for a single case.

        Args:
            image: 3D CT volume (Z, Y, X)
            mask:  3D binary mask (Z, Y, X)

        Returns:
            dict with at least:
                {
                    "score": float,          # primary heterogeneity score
                    "features": dict,        # extracted feature dict
                }
            May also include:
                "label_map", "confidence_map", "details"
        """
        ...

    def get_feature_names(self) -> List[str]:
        """Return list of feature names extracted by this model."""
        return self.feature_names

    # ------------------------------------------------------------------ #
    #  Batch processing (provided by base class)
    # ------------------------------------------------------------------ #

    def batch_compute(
        self,
        data_cfg: DataConfig,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run model on all cases in a dataset.

        Args:
            data_cfg: dataset configuration
            limit:    if set, only process first N cases

        Returns:
            DataFrame with one row per case
        """
        pairs = match_image_mask_pairs(
            data_cfg.image_dir,
            data_cfg.mask_dir,
            data_cfg.image_suffix,
            data_cfg.mask_suffix,
        )
        if not pairs:
            warnings.warn(f"No image-mask pairs found in {data_cfg.image_dir}")
            return pd.DataFrame()

        if limit:
            pairs = pairs[:limit]

        results = []
        iterator = tqdm(pairs, desc=f"[{self.name.upper()}]") if self.cfg.verbose else pairs

        for case_id, img_path, msk_path in iterator:
            vol = msk = result = None
            try:
                vol, _ = load_image(img_path)
                msk, _ = load_seg(msk_path)
                result = self.compute(vol, msk)
                results.append(self._result_to_row({"case_id": case_id, **result}))
            except Exception as exc:
                if self.cfg.verbose:
                    tqdm.write(f"  [ERR] {case_id}: {exc}")
                results.append(self._result_to_row({"case_id": case_id, "error": str(exc)}))
            finally:
                # Keep only table fields; do not retain per-case 3D maps.
                del vol, msk, result

        return pd.DataFrame(results)

    def _results_to_dataframe(self, results: List[dict]) -> pd.DataFrame:
        """Convert list of result dicts to a flat DataFrame."""
        return pd.DataFrame([self._result_to_row(r) for r in results])

    @staticmethod
    def _result_to_row(r: dict) -> dict:
        """Flatten one result without retaining spatial maps."""
        row = {"case_id": r.get("case_id", ""), "score": r.get("score", np.nan)}
        feats = r.get("features", {})
        if isinstance(feats, (dict, pd.DataFrame)):
            for key, value in feats.items():
                row[f"feat_{key}"] = value
        details = r.get("details", {})
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, (int, float, str)):
                    row[f"detail_{key}"] = value
        if "error" in r:
            row["error"] = r["error"]
        return row

    # ------------------------------------------------------------------ #
    #  Convenience: run + save
    # ------------------------------------------------------------------ #

    def run_and_save(self, data_cfg: DataConfig, csv_name: Optional[str] = None) -> str:
        """Run batch_compute and save results to CSV."""
        df = self.batch_compute(data_cfg)
        if df.empty:
            raise ValueError(f"No cases processed for {data_cfg.dataset}")

        out_dir = Path(data_cfg.output_dir) / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_name = csv_name or f"{self.name}_{data_cfg.dataset}.csv"
        out_path = out_dir / csv_name
        df.to_csv(out_path, index=False)
        if not np.isfinite(pd.to_numeric(df["score"], errors="coerce")).any():
            raise ValueError(f"No finite scores for {self.name}; see {out_path}")

        if self.cfg.verbose:
            print(f"[{self.name.upper()}] Saved {len(df)} rows to {out_path}")
        return str(out_path)
