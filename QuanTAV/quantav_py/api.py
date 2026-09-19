from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import QuantavParams
from .features import FeatureResult, compute_quantav_features
from .io import load_volume, save_features_csv
from .preprocess import ensure_same_shape, resample_to_spacing
from .roi import prepare_roi


def analyze_case(
    image: np.ndarray | str | Path | None,
    tumor_mask: np.ndarray | str | Path,
    vessel_mask: np.ndarray | str | Path,
    params: QuantavParams | None = None,
    output_dir: str | Path | None = None,
    case_id: str = "case",
) -> FeatureResult:
    """Run the complete Python QuanTAV pipeline.

    Parameters can be numpy arrays or paths to NIfTI/NRRD/NumPy files.
    The image is optional because the current feature set only needs tumor and
    vessel masks; it is accepted for future extensions and visual QA workflows.
    """

    params = params or QuantavParams()
    image_arr, tumor_arr, vessel_arr, params = _resolve_inputs(image, tumor_mask, vessel_mask, params)
    roi = prepare_roi(image_arr, tumor_arr, vessel_arr, params)
    result = compute_quantav_features(roi.tumor_mask, roi.vessel_mask, params)
    result.metadata.update(
        {
            "case_id": case_id,
            "crop_slices": _slices_to_tuples(roi.crop_slices),
            "peritumoral_radius_mm": params.peritumoral_radius_mm,
            "spacing": params.normalized_spacing(),
        }
    )

    if output_dir is not None:
        out_dir = Path(output_dir)
        save_features_csv(out_dir / f"{case_id}_features.csv", result.feature_names, result.features, case_id)
        if params.create_visualizations:
            from .visualization import create_visualizations

            paths = create_visualizations(
                roi.image,
                roi.tumor_mask,
                roi.vessel_mask,
                roi.peritumoral_mask,
                result.skeleton_mask,
                result.branches,
                result.raw_branches,
                out_dir,
                case_id,
                params,
            )
            result.metadata["visualization_paths"] = paths

    return result


def _resolve_inputs(
    image: np.ndarray | str | Path | None,
    tumor_mask: np.ndarray | str | Path,
    vessel_mask: np.ndarray | str | Path,
    params: QuantavParams,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, QuantavParams]:
    image_arr, image_spacing = _load_optional(image)
    tumor_arr, tumor_spacing = _load_required(tumor_mask)
    vessel_arr, vessel_spacing = _load_required(vessel_mask)

    spacing = tumor_spacing or vessel_spacing or image_spacing or params.normalized_spacing()
    for name, source_spacing in (("image", image_spacing), ("tumor_mask", tumor_spacing), ("vessel_mask", vessel_spacing)):
        if source_spacing is not None and not np.allclose(source_spacing, spacing):
            raise ValueError(f"{name} spacing {source_spacing} does not match {spacing}; align inputs first")
    params = replace(params, spacing=spacing)

    ensure_same_shape(*(a for a in [image_arr, tumor_arr, vessel_arr] if a is not None))

    target_spacing = params.normalized_target_spacing()
    if target_spacing is not None:
        tumor_arr, _ = resample_to_spacing(tumor_arr, params.normalized_spacing(), target_spacing, order=0)
        vessel_arr, _ = resample_to_spacing(vessel_arr, params.normalized_spacing(), target_spacing, order=0)
        if image_arr is not None:
            image_arr, _ = resample_to_spacing(image_arr, params.normalized_spacing(), target_spacing, order=1)
        params = replace(params, spacing=target_spacing)

    return image_arr, tumor_arr, vessel_arr, params


def _load_optional(value: np.ndarray | str | Path | None) -> tuple[np.ndarray | None, tuple[float, float, float] | None]:
    if value is None:
        return None, None
    return _load_required(value)


def _load_required(value: np.ndarray | str | Path) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    if isinstance(value, (str, Path)):
        arr, spacing, metadata = load_volume(value)
        if isinstance(metadata, dict) and metadata.get("format") == "npy":
            spacing = None  # NumPy files have no physical spacing metadata.
        return np.asarray(arr), spacing
    return np.asarray(value), None


def _slices_to_tuples(slices: tuple[slice, slice, slice]) -> tuple[tuple[int | None, int | None], ...]:
    return tuple((s.start, s.stop) for s in slices)
