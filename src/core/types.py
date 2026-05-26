"""
Shared type definitions for the ITH project.
"""

from typing import Protocol, TypedDict, Optional
import numpy as np
import pandas as pd

# Numpy array aliases for clarity
Image3D = np.ndarray  # (Z, Y, X) CT image volume
Mask3D = np.ndarray   # (Z, Y, X) binary segmentation mask
Features = np.ndarray  # (N, D) feature matrix


class ScoreResult(TypedDict, total=False):
    """Standard result format for heterogeneity scores."""
    case_id: str
    score: float
    features: pd.DataFrame
    label_map: np.ndarray
    details: dict


class BatchConfig(Protocol):
    """Protocol for batch processing configuration."""
    dataset: str
    image_dir: str
    mask_dir: str
    output_dir: str
    image_suffix: str
    mask_suffix: str


class HeterogeneityModel(Protocol):
    """Protocol for all heterogeneity scoring models."""

    def compute(self, image: Image3D, mask: Mask3D) -> ScoreResult:
        """Compute heterogeneity score for a single case."""
        ...

    def batch_compute(self, image_dir: str, mask_dir: str,
                      output_dir: str) -> pd.DataFrame:
        """Compute scores for all cases in a dataset."""
        ...
