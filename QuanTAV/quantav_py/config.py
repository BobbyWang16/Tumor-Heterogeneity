from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


Spacing = Tuple[float, float, float]


@dataclass(slots=True)
class QuantavParams:
    """Runtime settings for tumor-associated vessel feature extraction.

    The library treats array axes and spacing in the same order. If the input
    array has shape ``(z, y, x)``, pass spacing as ``(z_mm, y_mm, x_mm)``.
    ``load_volume`` returns spacing in this convention.
    """

    spacing: Spacing = (1.0, 1.0, 1.0)
    target_spacing: Optional[Spacing] = None
    peritumoral_radius_mm: float = 30.0
    crop_to_roi: bool = True
    crop_margin_mm: float = 2.0
    clean_tumor: bool = True
    fill_tumor_holes: bool = True
    min_vessel_component_size: int = 20
    skeletonize_tumor_union: bool = True
    min_branch_points: int = 11
    organization_distance_range_mm: Tuple[float, float] = (0.0, 11.0)
    organization_window_size: int = 35
    organization_window_step: Optional[int] = None
    spherical_projection_size: int = 400
    spherical_interpolation_points: int = 50
    hough_num_peaks: int = 5
    nan_fill: float = 0.0
    create_visualizations: bool = True
    visualization_dpi: int = 180
    visualization_downsample: int = 1

    def normalized_spacing(self) -> Spacing:
        return _as_spacing(self.spacing, "spacing")

    def normalized_target_spacing(self) -> Optional[Spacing]:
        if self.target_spacing is None:
            return None
        return _as_spacing(self.target_spacing, "target_spacing")

    def window_step(self) -> int:
        if self.organization_window_step is not None:
            return max(1, int(self.organization_window_step))
        return max(1, int(round(self.organization_window_size / 3)))


def _as_spacing(value: Sequence[float], name: str) -> Spacing:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    spacing = tuple(float(v) for v in value)
    if any(v <= 0 for v in spacing):
        raise ValueError(f"{name} values must be positive")
    return spacing  # type: ignore[return-value]

