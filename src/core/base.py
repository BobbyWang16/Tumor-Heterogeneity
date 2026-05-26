"""
Base model abstraction for all heterogeneity scoring models.

All models inherit from BaseHeterogeneityModel and implement:
  - compute(image, mask) -> dict: score a single case
  - get_feature_names() -> list[str]: names of extracted features

Batch processing, I/O, and result saving are handled by the base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import gc
import os
import time
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
            try:
                vol, _ = load_image(img_path)
                msk, _ = load_seg(msk_path)
                result = self.compute(vol, msk)
                row = {"case_id": case_id, **result}
                results.append(row)
            except Exception as exc:
                if self.cfg.verbose:
                    tqdm.write(f"  [ERR] {case_id}: {exc}")
                results.append({"case_id": case_id, "error": str(exc)})
            finally:
                # Explicitly free large arrays to prevent memory accumulation
                for var in ("vol", "msk", "result", "row"):
                    if var in locals():
                        del locals()[var]
                gc.collect()

        return self._results_to_dataframe(results)

    def _results_to_dataframe(self, results: List[dict]) -> pd.DataFrame:
        """Convert list of result dicts to a flat DataFrame."""
        rows = []
        for r in results:
            row = {"case_id": r.get("case_id", "")}
            row["score"] = r.get("score", np.nan)

            # flatten features dict
            feats = r.get("features", {})
            if isinstance(feats, dict):
                for k, v in feats.items():
                    row[f"feat_{k}"] = v
            elif isinstance(feats, pd.DataFrame):
                for k, v in feats.items():
                    row[f"feat_{k}"] = v

            # details dict (optional)
            details = r.get("details", {})
            if isinstance(details, dict):
                for k, v in details.items():
                    if isinstance(v, (int, float, str)):
                        row[f"detail_{k}"] = v

            if "error" in r:
                row["error"] = r["error"]

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Convenience: run + save
    # ------------------------------------------------------------------ #

    def run_and_save(self, data_cfg: DataConfig, csv_name: Optional[str] = None) -> str:
        """Run batch_compute and save results to CSV."""
        df = self.batch_compute(data_cfg)

        out_dir = Path(data_cfg.output_dir) / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_name = csv_name or f"{self.name}_{data_cfg.dataset}.csv"
        out_path = out_dir / csv_name
        df.to_csv(out_path, index=False)

        if self.cfg.verbose:
            print(f"[{self.name.upper()}] Saved {len(df)} rows to {out_path}")
        return str(out_path)
