from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage.transform import hough_line, hough_line_peaks

from .config import QuantavParams
from .skeleton import skeletonize_and_extract_branches


@dataclass(slots=True)
class BranchFeature:
    points_vox: np.ndarray
    points_mm: np.ndarray
    curvature: np.ndarray
    torsion: float
    enters_tumor: bool
    length_mm: float = 0.0
    chord_mm: float = 0.0
    mean_curvature: float = 0.0
    max_curvature: float = 0.0


@dataclass(slots=True)
class FeatureResult:
    features: np.ndarray
    feature_names: list[str]
    morphology: np.ndarray
    morphology_names: list[str]
    organization: np.ndarray
    organization_names: list[str]
    branches: list[BranchFeature] = field(default_factory=list)
    raw_branches: list[np.ndarray] = field(default_factory=list)
    skeleton_mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.feature_names, self.features)}

    def morphology_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.morphology_names, self.morphology)}

    def organization_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.organization_names, self.organization)}


def compute_quantav_features(
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    params: QuantavParams | None = None,
) -> FeatureResult:
    """Compute QuanTAV-style morphology and organization features."""

    params = params or QuantavParams()
    tumor = np.asarray(tumor_mask > 0, dtype=bool)
    vessel = np.asarray(vessel_mask > 0, dtype=bool)
    if tumor.ndim != 3 or vessel.ndim != 3:
        raise ValueError(f"tumor and vessel masks must be 3D, got {tumor.shape} and {vessel.shape}")
    if tumor.shape != vessel.shape:
        raise ValueError(f"tumor and vessel masks differ: {tumor.shape} vs {vessel.shape}")
    if not np.any(tumor):
        raise ValueError("tumor_mask is empty; QuanTAV features require a tumor ROI")

    skeleton_input = (vessel | tumor) if params.skeletonize_tumor_union else vessel
    skel_result = skeletonize_and_extract_branches(skeleton_input)

    morphology, morphology_names, branch_features = compute_morphology_features(
        vessel, tumor, skel_result.branches, params
    )
    organization, organization_names = compute_organization_features(
        vessel, tumor, skel_result.branches, params
    )

    features = np.concatenate([morphology, organization]).astype(float)
    names = morphology_names + organization_names
    features = _replace_nan(features, params.nan_fill)
    voxel_volume_mm3 = float(np.prod(params.normalized_spacing()))
    skeleton_length_mm = float(sum(branch.length_mm for branch in branch_features))
    vessel_no_tumor = vessel & ~tumor

    return FeatureResult(
        features=features,
        feature_names=names,
        morphology=_replace_nan(morphology, params.nan_fill),
        morphology_names=morphology_names,
        organization=_replace_nan(organization, params.nan_fill),
        organization_names=organization_names,
        branches=branch_features,
        raw_branches=skel_result.branches,
        skeleton_mask=skel_result.skeleton_mask,
        metadata={
            "n_raw_branches": len(skel_result.branches),
            "n_analyzed_branches": len(branch_features),
            "vessel_voxels": int(vessel.sum()),
            "peritumoral_vessel_voxels": int(vessel_no_tumor.sum()),
            "vessel_volume_mm3": float(vessel_no_tumor.sum() * voxel_volume_mm3),
            "tumor_voxels": int(tumor.sum()),
            "tumor_volume_mm3": float(tumor.sum() * voxel_volume_mm3),
            "skeleton_voxels": int(skel_result.skeleton_mask.sum()),
            "analyzed_skeleton_length_mm": skeleton_length_mm,
            "feeding_vessel_count": int(sum(branch.enters_tumor for branch in branch_features)),
        },
    )


def compute_morphology_features(
    vessel_mask: np.ndarray,
    tumor_mask: np.ndarray,
    skeleton_branches: Sequence[np.ndarray],
    params: QuantavParams,
) -> tuple[np.ndarray, list[str], list[BranchFeature]]:
    spacing = np.asarray(params.normalized_spacing(), dtype=float)
    vessel_no_tumor = vessel_mask.copy()
    vessel_no_tumor[tumor_mask] = False

    torsions: list[float] = []
    curv_mean: list[float] = []
    curv_std: list[float] = []
    curv_max: list[float] = []
    curv_skew: list[float] = []
    curv_kurt: list[float] = []
    all_curvature: list[float] = []
    branch_features: list[BranchFeature] = []
    feeding_vessel_count = 0
    total_points = 0

    for branch in skeleton_branches:
        vox = np.rint(branch).astype(int)
        vox = _clip_points(vox, tumor_mask.shape)
        if len(vox) == 0:
            continue
        inside_tumor = tumor_mask[tuple(vox.T)]
        enters_tumor = bool(np.any(inside_tumor))
        vox = vox[~inside_tumor]
        vox = _remove_adjacent_duplicates(vox, spacing)
        if len(vox) < params.min_branch_points:
            continue

        pts_mm = vox.astype(float) * spacing
        length = _polyline_length(pts_mm)
        chord = float(np.linalg.norm(pts_mm[-1] - pts_mm[0]))
        if length <= 0:
            continue
        torsion = float(1.0 - chord / length)
        curvature = _curvature_from_triplets(pts_mm)
        interior_curv = curvature[1:-1]
        interior_curv = interior_curv[np.isfinite(interior_curv)]
        if len(interior_curv) <= 2 or len(np.unique(interior_curv)) <= 1:
            continue

        torsions.append(torsion)
        curv_mean.append(float(np.mean(interior_curv)))
        curv_std.append(float(np.std(interior_curv, ddof=1)) if len(interior_curv) > 1 else 0.0)
        curv_max.append(float(np.max(interior_curv)))
        curv_skew.append(_safe_skew(interior_curv))
        curv_kurt.append(_safe_kurtosis(interior_curv))
        all_curvature.extend(float(v) for v in interior_curv)
        total_points += len(vox)
        if enters_tumor:
            feeding_vessel_count += 1

        branch_features.append(
            BranchFeature(
                points_vox=vox.astype(float),
                points_mm=pts_mm,
                curvature=curvature,
                torsion=torsion,
                enters_tumor=enters_tumor,
                length_mm=length,
                chord_mm=chord,
                mean_curvature=float(np.mean(interior_curv)),
                max_curvature=float(np.max(interior_curv)),
            )
        )

    torsion_arr = np.asarray(torsions, dtype=float)
    curv_mean_arr = np.asarray(curv_mean, dtype=float)
    curv_std_arr = np.asarray(curv_std, dtype=float)
    curv_max_arr = np.asarray(curv_max, dtype=float)
    curv_skew_arr = np.asarray(curv_skew, dtype=float)
    curv_kurt_arr = np.asarray(curv_kurt, dtype=float)
    all_curv_arr = np.asarray(all_curvature, dtype=float)

    values: list[float] = []
    names: list[str] = []
    for prefix, arr in [
        ("morph_torsion", torsion_arr),
        ("morph_branch_curvature_mean", curv_mean_arr),
        ("morph_branch_curvature_std", curv_std_arr),
        ("morph_branch_curvature_max", curv_max_arr),
        ("morph_branch_curvature_skewness", curv_skew_arr),
        ("morph_branch_curvature_kurtosis", curv_kurt_arr),
        ("morph_point_curvature", all_curv_arr),
    ]:
        stat_values, stat_names = _five_stats(arr, prefix)
        values.extend(stat_values)
        names.extend(stat_names)

    hist_curv, hist_curv_names = _hist10(all_curv_arr, "morph_point_curvature_hist")
    hist_tors, hist_tors_names = _hist10(torsion_arr, "morph_torsion_hist")
    values.extend(hist_curv)
    names.extend(hist_curv_names)
    values.extend(hist_tors)
    names.extend(hist_tors_names)

    vessel_vol = float(vessel_no_tumor.sum())
    tumor_vol = float(tumor_mask.sum())
    roi_vol = float(np.prod(vessel_mask.shape))
    values.extend(
        [
            vessel_vol,
            _safe_div(vessel_vol, roi_vol),
            _safe_div(vessel_vol, tumor_vol),
            float(total_points),
            float(feeding_vessel_count),
            _safe_div(float(feeding_vessel_count), float(len(torsion_arr))),
        ]
    )
    names.extend(
        [
            "morph_vessel_volume_voxels",
            "morph_vessel_density",
            "morph_vessel_to_tumor_volume_ratio",
            "morph_skeleton_points_total",
            "morph_feeding_vessel_count",
            "morph_feeding_vessel_fraction",
        ]
    )

    assert len(values) == 61, len(values)
    return np.asarray(values, dtype=float), names, branch_features


def compute_organization_features(
    vessel_mask: np.ndarray,
    tumor_mask: np.ndarray,
    skeleton_branches: Sequence[np.ndarray],
    params: QuantavParams,
) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    names: list[str] = []
    for prefix, image, pad_dim in build_organization_projection_images(
        vessel_mask, tumor_mask, skeleton_branches, params
    ):
        angles = _process_hough_2d(
            image,
            window_size=params.organization_window_size,
            step=params.window_step(),
            pad_dim=pad_dim,
            num_peaks=params.hough_num_peaks,
        )
        stat_values, stat_names = _five_orientation_stats(np.asarray(angles), prefix)
        values.extend(stat_values)
        names.extend(stat_names)

    assert len(values) == 30, len(values)
    return np.asarray(values, dtype=float), names


def build_organization_projection_images(
    vessel_mask: np.ndarray,
    tumor_mask: np.ndarray,
    skeleton_branches: Sequence[np.ndarray],
    params: QuantavParams,
) -> list[tuple[str, np.ndarray, int]]:
    """Create the six binary projection images used by organization features.

    The first three projections use the peritumoral vessel volume. The last
    three use interpolated centerline points in tumor-centered spherical
    coordinates. ``pad_dim`` tells the Hough window scanner which dimension is
    angular and should wrap around.
    """

    spacing = params.normalized_spacing()
    distance_map = ndi.distance_transform_edt(~tumor_mask, sampling=spacing)
    distance_range = params.organization_distance_range_mm

    coords = np.argwhere(vessel_mask)
    keep = np.zeros(len(coords), dtype=bool)
    if len(coords):
        d = distance_map[tuple(coords.T)]
        outside = ~tumor_mask[tuple(coords.T)]
        keep = outside & (d > distance_range[0]) & (d <= distance_range[1])

    projected_volume = np.zeros_like(vessel_mask, dtype=bool)
    if len(coords):
        projected_volume[tuple(coords[keep].T)] = True

    projections = [
        ("org_cartesian_axis0_axis1", np.max(projected_volume, axis=2), 0),
        ("org_cartesian_axis0_axis2", np.max(projected_volume, axis=1), 0),
        ("org_cartesian_axis1_axis2", np.max(projected_volume, axis=0), 0),
    ]
    projections.extend(_spherical_projections(skeleton_branches, tumor_mask, params))
    return projections


def _spherical_projections(
    skeleton_branches: Sequence[np.ndarray],
    tumor_mask: np.ndarray,
    params: QuantavParams,
) -> list[tuple[str, np.ndarray, int]]:
    size = int(params.spherical_projection_size)
    bw_phitheta = np.zeros((size + 1, size + 1), dtype=bool)
    bw_rphi = np.zeros((size + 1, size + 1), dtype=bool)
    bw_rtheta = np.zeros((size + 1, size + 1), dtype=bool)
    if not skeleton_branches or not np.any(tumor_mask):
        return [
            ("org_spherical_elevation_rotation", bw_phitheta, 2),
            ("org_spherical_distance_rotation", bw_rtheta.T, 1),
            ("org_spherical_distance_elevation", bw_rphi.T, 0),
        ]

    spacing = np.asarray(params.normalized_spacing(), dtype=float)
    centroid_mm = np.argwhere(tumor_mask).mean(axis=0) * spacing
    distance_map = ndi.distance_transform_edt(~tumor_mask, sampling=spacing)
    points_vox = _interpolate_branches(skeleton_branches, params.spherical_interpolation_points)
    if len(points_vox) == 0:
        return [
            ("org_spherical_elevation_rotation", bw_phitheta, 2),
            ("org_spherical_distance_rotation", bw_rtheta.T, 1),
            ("org_spherical_distance_elevation", bw_rphi.T, 0),
        ]

    rounded = _clip_points(np.rint(points_vox).astype(int), tumor_mask.shape)
    outside = ~tumor_mask[tuple(rounded.T)]
    distances = distance_map[tuple(rounded.T)]
    points_mm = points_vox * spacing
    rel = points_mm - centroid_mm
    x = rel[:, 2]
    y = rel[:, 1]
    z = rel[:, 0]
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x * x + y * y))

    dist_min, dist_max = params.organization_distance_range_mm
    dist_width = max(float(dist_max - dist_min), np.finfo(float).eps)
    theta_idx = np.rint(0.5 * size * (theta + np.pi) / np.pi).astype(int)
    phi_idx = np.rint(0.5 * size * (phi + np.pi / 2) / (np.pi / 2)).astype(int)
    r_idx = np.rint((distances - dist_min) * size / dist_width).astype(int)

    valid = outside & (r_idx > 0) & (r_idx <= size)
    valid &= (theta_idx >= 0) & (theta_idx <= size) & (phi_idx >= 0) & (phi_idx <= size)
    for ri, pi, ti in zip(r_idx[valid], phi_idx[valid], theta_idx[valid]):
        bw_phitheta[pi, ti] = True
        bw_rphi[ri, pi] = True
        bw_rtheta[ri, ti] = True

    shift = round(0.75 * size)
    bw_phitheta = np.concatenate([bw_phitheta[:, shift + 1 :], bw_phitheta[:, : shift + 1]], axis=1)
    bw_rtheta = np.concatenate([bw_rtheta[:, shift + 1 :], bw_rtheta[:, : shift + 1]], axis=1)

    return [
        ("org_spherical_elevation_rotation", bw_phitheta, 2),
        ("org_spherical_distance_rotation", bw_rtheta.T, 1),
        ("org_spherical_distance_elevation", bw_rphi.T, 0),
    ]


def _process_hough_2d(
    image: np.ndarray,
    window_size: int,
    step: int,
    pad_dim: int,
    num_peaks: int,
) -> list[float]:
    image = np.asarray(image > 0, dtype=bool)
    if pad_dim == 1:
        image = np.concatenate([image, image[: window_size - 1, :]], axis=0)
    elif pad_dim == 2:
        image = np.concatenate([image, image[:, : window_size - 1]], axis=1)

    if not np.any(image):
        return []

    angles_all: list[float] = []
    h, w = image.shape
    if h < window_size or w < window_size:
        return _hough_angles(image, num_peaks)

    for i in range(0, h - window_size + 1, step):
        for j in range(0, w - window_size + 1, step):
            roi = image[i : i + window_size, j : j + window_size]
            if np.count_nonzero(roi) == 0:
                continue
            angles_all.extend(_hough_angles(roi, num_peaks))
    return angles_all


def _hough_angles(image: np.ndarray, num_peaks: int) -> list[float]:
    if np.count_nonzero(image) < 2:
        return []
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 361, endpoint=True)
    hspace, angles, dists = hough_line(image, theta=tested_angles)
    accums, peak_angles, _ = hough_line_peaks(
        hspace,
        angles,
        dists,
        num_peaks=num_peaks,
    )
    if len(accums) == 0:
        return []
    return [float(np.rad2deg(a)) for a in peak_angles]


def _interpolate_branches(branches: Sequence[np.ndarray], n_insert: int) -> np.ndarray:
    all_points: list[np.ndarray] = []
    for branch in branches:
        pts = np.asarray(branch, dtype=float)
        if len(pts) == 0:
            continue
        all_points.append(pts[:1])
        for p0, p1 in zip(pts[:-1], pts[1:]):
            steps = np.linspace(0.0, 1.0, n_insert + 2, endpoint=True)[1:]
            all_points.append(p0[None, :] + steps[:, None] * (p1 - p0)[None, :])
    if not all_points:
        return np.empty((0, 3), dtype=float)
    return np.vstack(all_points)


def _curvature_from_triplets(points_mm: np.ndarray) -> np.ndarray:
    curv = np.zeros(len(points_mm), dtype=float)
    if len(points_mm) < 3:
        return curv
    p1 = points_mm[:-2]
    p2 = points_mm[1:-1]
    p3 = points_mm[2:]
    a = p2 - p1
    b = p3 - p2
    c = p3 - p1
    cross_norm = np.linalg.norm(np.cross(a, c), axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 2.0 * cross_norm / denom
    k[~np.isfinite(k)] = 0.0
    curv[1:-1] = k
    return curv


def _polyline_length(points_mm: np.ndarray) -> float:
    if len(points_mm) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points_mm, axis=0), axis=1).sum())


def _remove_adjacent_duplicates(points: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    pts_mm = points.astype(float) * spacing
    d = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    keep = np.r_[True, d >= 1e-6]
    return points[keep]


def _clip_points(points: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 3)
    clipped = points.copy()
    for axis, size in enumerate(shape):
        clipped[:, axis] = np.clip(clipped[:, axis], 0, size - 1)
    return clipped


def _five_stats(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    names = [f"{prefix}_{suffix}" for suffix in ("mean", "std", "max", "skewness", "kurtosis")]
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [np.nan] * 5, names
    return [
        float(np.mean(arr)),
        float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        float(np.max(arr)),
        _safe_skew(arr),
        _safe_kurtosis(arr),
    ], names


def _five_orientation_stats(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    names = [f"{prefix}_{suffix}" for suffix in ("mean", "median", "std", "skewness", "kurtosis")]
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [np.nan] * 5, names
    return [
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.std(arr, ddof=0)),
        _safe_skew(arr),
        _safe_kurtosis(arr),
    ], names


def _hist10(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    names = [f"{prefix}_bin_{i:02d}" for i in range(1, 11)]
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0] * 10, names
    counts, _ = np.histogram(arr, bins=10)
    return [float(v) for v in counts], names


def _safe_skew(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3 or np.allclose(arr, arr[0]):
        return 0.0
    out = float(skew(arr, bias=True))
    return out if np.isfinite(out) else 0.0


def _safe_kurtosis(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4 or np.allclose(arr, arr[0]):
        return 0.0
    out = float(kurtosis(arr, fisher=False, bias=True))
    return out if np.isfinite(out) else 0.0


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else np.nan


def _replace_nan(values: np.ndarray, fill: float) -> np.ndarray:
    return np.nan_to_num(values.astype(float), nan=float(fill), posinf=float(fill), neginf=float(fill))
