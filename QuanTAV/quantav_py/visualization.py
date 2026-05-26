from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from .config import QuantavParams
from .features import BranchFeature, build_organization_projection_images


def create_visualizations(
    image: np.ndarray | None,
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    peritumoral_mask: np.ndarray,
    skeleton_mask: np.ndarray | None,
    branches: list[BranchFeature],
    raw_branches: Sequence[np.ndarray],
    output_dir: str | Path,
    case_id: str,
    params: QuantavParams,
) -> dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, object] = {}
    projection_branches = raw_branches or [branch.points_vox for branch in branches]

    paths["tumor_vessels"] = str(
        visualize_tumor_vessels(
            tumor_mask,
            vessel_mask,
            out_dir / f"{case_id}_tumor_vessels.png",
            params,
            branches=branches,
        )
    )
    paths["skeleton_branches"] = str(
        visualize_skeleton_branches(
            tumor_mask,
            branches,
            out_dir / f"{case_id}_skeleton_branches.png",
            params,
        )
    )
    paths["central_slices"] = str(
        visualize_slice_overlays(
            image,
            tumor_mask,
            vessel_mask,
            peritumoral_mask,
            skeleton_mask,
            out_dir / f"{case_id}_central_slices.png",
            params,
        )
    )
    paths["mip_overview"] = str(
        visualize_mip_overview(
            image,
            tumor_mask,
            vessel_mask,
            peritumoral_mask,
            skeleton_mask,
            out_dir / f"{case_id}_mip_overview.png",
            params,
        )
    )
    paths["curvature"] = str(
        visualize_branch_metric(
            tumor_mask,
            branches,
            out_dir / f"{case_id}_curvature.png",
            params,
            metric="curvature",
        )
    )
    paths["torsion"] = str(
        visualize_branch_metric(
            tumor_mask,
            branches,
            out_dir / f"{case_id}_torsion.png",
            params,
            metric="torsion",
        )
    )
    paths["feature_distributions"] = str(
        visualize_feature_distributions(
            branches,
            out_dir / f"{case_id}_feature_distributions.png",
            params,
        )
    )
    paths["organization_projections"] = visualize_organization_projections(
        vessel_mask,
        tumor_mask,
        projection_branches,
        out_dir,
        case_id,
        params,
    )
    paths["branch_metrics_csv"] = str(
        save_branch_metrics_csv(branches, out_dir / f"{case_id}_branch_metrics.csv")
    )
    return paths


def visualize_tumor_vessels(
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    output_path: str | Path,
    params: QuantavParams,
    branches: Sequence[BranchFeature] | None = None,
) -> Path:
    factor = max(1, int(params.visualization_downsample))
    tumor = _maybe_downsample(tumor_mask, factor)
    vessel = _maybe_downsample(vessel_mask, factor)
    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    _add_surface(ax, tumor, color=(0.0, 0.8, 0.85), alpha=0.45)
    _add_surface(ax, vessel, color=(1.0, 0.15, 0.1), alpha=0.62)
    for branch in branches or []:
        pts = branch.points_vox / factor
        if len(pts) >= 2:
            ax.plot(pts[:, 2], pts[:, 1], pts[:, 0], color="white", linewidth=1.1, alpha=0.82)
    _style_3d_axis(ax, tumor.shape)
    _add_legend(
        ax,
        [
            ("Tumor", "#00cbd6"),
            ("Peritumoral vessels", "#ff2b1f"),
            ("Centerline", "#ffffff"),
        ],
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=params.visualization_dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def visualize_skeleton_branches(
    tumor_mask: np.ndarray,
    branches: Sequence[BranchFeature],
    output_path: str | Path,
    params: QuantavParams,
) -> Path:
    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    _add_surface(ax, tumor_mask, color=(0.0, 0.8, 0.85), alpha=0.28)

    lengths = np.asarray([branch.length_mm for branch in branches], dtype=float)
    vmax = float(np.nanpercentile(lengths, 95)) if lengths.size else 1.0
    norm = colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-9))

    for branch in branches:
        pts = branch.points_vox
        if len(pts) < 2:
            continue
        color = "#ffd166" if branch.enters_tumor else cm.plasma(norm(branch.length_mm))
        linewidth = 2.6 if branch.enters_tumor else 1.7
        ax.plot(pts[:, 2], pts[:, 1], pts[:, 0], color=color, linewidth=linewidth, alpha=0.95)

    if branches:
        sm = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Branch length (mm)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
    else:
        ax.text2D(0.5, 0.5, "No analyzed branches", color="white", ha="center", transform=ax.transAxes)

    _style_3d_axis(ax, tumor_mask.shape)
    _add_legend(ax, [("Feeding branch", "#ffd166"), ("Other branch", "#c05bff")])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=params.visualization_dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def visualize_slice_overlays(
    image: np.ndarray | None,
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    peritumoral_mask: np.ndarray,
    skeleton_mask: np.ndarray | None,
    output_path: str | Path,
    params: QuantavParams,
) -> Path:
    center = _tumor_center_index(tumor_mask)
    source = image if image is not None else np.zeros_like(tumor_mask, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    planes = [
        ("Axial", 0, center[0]),
        ("Coronal", 1, center[1]),
        ("Sagittal", 2, center[2]),
    ]
    for ax, (title, axis, index) in zip(axes, planes):
        bg = _normalize_slice(_slice_along(source, axis, index))
        ax.imshow(bg, cmap="gray", interpolation="nearest")
        _overlay_mask(ax, _slice_along(peritumoral_mask, axis, index), "#ffd166", 0.18)
        _overlay_mask(ax, _slice_along(tumor_mask, axis, index), "#00cbd6", 0.44)
        _overlay_mask(ax, _slice_along(vessel_mask, axis, index), "#ff2b1f", 0.72)
        if skeleton_mask is not None:
            _overlay_mask(ax, _slice_along(skeleton_mask, axis, index), "#ffffff", 0.95)
        _add_contour(ax, _slice_along(peritumoral_mask, axis, index), "#ffd166", 0.9)
        _add_contour(ax, _slice_along(tumor_mask, axis, index), "#00f5ff", 1.2)
        ax.set_title(f"{title} slice {index}", color="white")
        ax.set_axis_off()
    _figure_legend(
        fig,
        [
            ("Peritumoral ROI", "#ffd166"),
            ("Tumor", "#00cbd6"),
            ("Vessel", "#ff2b1f"),
            ("Skeleton", "#ffffff"),
        ],
    )
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=params.visualization_dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def visualize_mip_overview(
    image: np.ndarray | None,
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    peritumoral_mask: np.ndarray,
    skeleton_mask: np.ndarray | None,
    output_path: str | Path,
    params: QuantavParams,
) -> Path:
    source = image if image is not None else np.zeros_like(tumor_mask, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    projections = [
        ("XY MIP", 0),
        ("XZ MIP", 1),
        ("YZ MIP", 2),
    ]
    for ax, (title, axis) in zip(axes, projections):
        base = _normalize_slice(np.max(source, axis=axis))
        rgb = np.dstack([base, base, base])
        rgb = _blend(rgb, np.max(peritumoral_mask, axis=axis), "#ffd166", 0.18)
        rgb = _blend(rgb, np.max(tumor_mask, axis=axis), "#00cbd6", 0.46)
        rgb = _blend(rgb, np.max(vessel_mask, axis=axis), "#ff2b1f", 0.78)
        if skeleton_mask is not None:
            rgb = _blend(rgb, np.max(skeleton_mask, axis=axis), "#ffffff", 0.95)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title, color="white")
        ax.set_axis_off()
    _figure_legend(
        fig,
        [
            ("Peritumoral ROI", "#ffd166"),
            ("Tumor", "#00cbd6"),
            ("Vessel", "#ff2b1f"),
            ("Skeleton", "#ffffff"),
        ],
    )
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=params.visualization_dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def visualize_branch_metric(
    tumor_mask: np.ndarray,
    branches: list[BranchFeature],
    output_path: str | Path,
    params: QuantavParams,
    metric: str,
) -> Path:
    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    _add_surface(ax, tumor_mask, color=(0.0, 0.8, 0.85), alpha=0.25)

    if metric == "curvature":
        value_chunks = [b.curvature[np.isfinite(b.curvature)] for b in branches if len(b.curvature)]
        values = np.concatenate(value_chunks) if value_chunks else np.asarray([], dtype=float)
        vmax = float(np.nanpercentile(values, 95)) if values.size else 1.0
        norm = colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-9))
        for branch in branches:
            pts = branch.points_vox
            curv = branch.curvature
            for i in range(1, len(pts) - 1):
                seg = pts[i - 1 : i + 2]
                ax.plot(seg[:, 2], seg[:, 1], seg[:, 0], color=cm.viridis(norm(curv[i])), linewidth=1.8)
        sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        label = "Curvature (1/mm)"
    elif metric == "torsion":
        vals = np.asarray([b.torsion for b in branches], dtype=float)
        vmax = float(np.nanpercentile(vals, 95)) if vals.size else 1.0
        norm = colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-9))
        for branch in branches:
            pts = branch.points_vox
            if len(pts) >= 2:
                ax.plot(pts[:, 2], pts[:, 1], pts[:, 0], color=cm.inferno(norm(branch.torsion)), linewidth=2.0)
        sm = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        label = "Torsion / tortuosity"
    else:
        raise ValueError("metric must be 'curvature' or 'torsion'")

    if branches:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(label, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
    else:
        ax.text2D(0.5, 0.5, "No analyzed branches", color="white", ha="center", transform=ax.transAxes)

    _style_3d_axis(ax, tumor_mask.shape)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=params.visualization_dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def visualize_feature_distributions(
    branches: Sequence[BranchFeature],
    output_path: str | Path,
    params: QuantavParams,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    metrics = [
        ("Branch torsion", [branch.torsion for branch in branches], "#ef476f"),
        ("Branch length (mm)", [branch.length_mm for branch in branches], "#118ab2"),
        ("Mean curvature (1/mm)", [branch.mean_curvature for branch in branches], "#06d6a0"),
        ("Max curvature (1/mm)", [branch.max_curvature for branch in branches], "#7b2cbf"),
    ]
    for ax, (title, values, color) in zip(axes.ravel(), metrics):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            bins = min(24, max(6, int(np.sqrt(arr.size)) + 3))
            ax.hist(arr, bins=bins, color=color, alpha=0.82, edgecolor="white")
            ax.axvline(float(np.mean(arr)), color="black", linewidth=1.2, linestyle="--", label="mean")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No analyzed branches", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(alpha=0.18)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=params.visualization_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def visualize_organization_projections(
    vessel_mask: np.ndarray,
    tumor_mask: np.ndarray,
    skeleton_branches: Sequence[np.ndarray],
    output_dir: str | Path,
    case_id: str,
    params: QuantavParams,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    projections = build_organization_projection_images(vessel_mask, tumor_mask, skeleton_branches, params)
    labels = {
        "org_cartesian_axis0_axis1": "Cartesian axis0-axis1",
        "org_cartesian_axis0_axis2": "Cartesian axis0-axis2",
        "org_cartesian_axis1_axis2": "Cartesian axis1-axis2",
        "org_spherical_elevation_rotation": "Spherical elevation-rotation",
        "org_spherical_distance_rotation": "Spherical distance-rotation",
        "org_spherical_distance_elevation": "Spherical distance-elevation",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
    individual: dict[str, str] = {}
    for ax, (name, image, _) in zip(axes.ravel(), projections):
        ax.imshow(image, cmap="magma", interpolation="nearest", origin="lower")
        ax.set_title(labels.get(name, name), fontsize=10)
        ax.set_axis_off()

        single_path = out_dir / f"{case_id}_{name}.png"
        single_fig, single_ax = plt.subplots(figsize=(5, 5), facecolor="white")
        single_ax.imshow(image, cmap="magma", interpolation="nearest", origin="lower")
        single_ax.set_title(labels.get(name, name))
        single_ax.set_axis_off()
        single_fig.tight_layout()
        single_fig.savefig(single_path, dpi=params.visualization_dpi, bbox_inches="tight")
        plt.close(single_fig)
        individual[name] = str(single_path)

    fig.tight_layout()
    summary_path = out_dir / f"{case_id}_organization_projections.png"
    fig.savefig(summary_path, dpi=params.visualization_dpi, bbox_inches="tight")
    plt.close(fig)
    return {"summary": str(summary_path), **individual}


def save_branch_metrics_csv(branches: Sequence[BranchFeature], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "branch_id",
                "n_points",
                "length_mm",
                "chord_mm",
                "torsion",
                "mean_curvature",
                "max_curvature",
                "enters_tumor",
            ]
        )
        for i, branch in enumerate(branches, start=1):
            writer.writerow(
                [
                    i,
                    len(branch.points_vox),
                    branch.length_mm,
                    branch.chord_mm,
                    branch.torsion,
                    branch.mean_curvature,
                    branch.max_curvature,
                    int(branch.enters_tumor),
                ]
            )
    return output


def _add_surface(ax, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
    if np.count_nonzero(mask) == 0:
        return
    if min(mask.shape) < 2:
        return
    try:
        verts, faces, _, _ = marching_cubes(mask.astype(float), level=0.5)
    except ValueError:
        return
    mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)


def _style_3d_axis(ax, shape: tuple[int, int, int]) -> None:
    ax.set_xlim(0, shape[2])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[0])
    ax.set_box_aspect((shape[2], shape[1], shape[0]))
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()


def _maybe_downsample(mask: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    if factor <= 1:
        return mask
    return mask[::factor, ::factor, ::factor]


def _tumor_center_index(tumor_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(tumor_mask)
    if coords.size == 0:
        return np.asarray(tumor_mask.shape) // 2
    center = np.rint(coords.mean(axis=0)).astype(int)
    return np.minimum(np.maximum(center, 0), np.asarray(tumor_mask.shape) - 1)


def _slice_along(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.asarray(array[index, :, :])
    if axis == 1:
        return np.asarray(array[:, index, :])
    if axis == 2:
        return np.asarray(array[:, :, index])
    raise ValueError("axis must be 0, 1, or 2")


def _normalize_slice(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=float)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _overlay_mask(ax, mask: np.ndarray, color: str, alpha: float) -> None:
    mask = np.asarray(mask > 0, dtype=bool)
    if not np.any(mask):
        return
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask] = colors.to_rgba(color, alpha=alpha)
    ax.imshow(rgba, interpolation="nearest")


def _add_contour(ax, mask: np.ndarray, color: str, linewidth: float) -> None:
    mask = np.asarray(mask > 0, dtype=bool)
    if not np.any(mask) or np.all(mask):
        return
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def _blend(rgb: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    out = np.asarray(rgb, dtype=float).copy()
    mask = np.asarray(mask > 0, dtype=bool)
    if not np.any(mask):
        return out
    color_rgb = np.asarray(colors.to_rgb(color), dtype=float)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_rgb
    return np.clip(out, 0.0, 1.0)


def _add_legend(ax, items: Sequence[tuple[str, str]]) -> None:
    handles = [Line2D([0], [0], color=color, lw=3, label=label) for label, color in items]
    legend = ax.legend(handles=handles, loc="upper left", frameon=False)
    for text in legend.get_texts():
        text.set_color("white")


def _figure_legend(fig, items: Sequence[tuple[str, str]]) -> None:
    handles = [Line2D([0], [0], color=color, lw=5, label=label) for label, color in items]
    legend = fig.legend(handles=handles, loc="lower center", ncol=len(items), frameon=False)
    for text in legend.get_texts():
        text.set_color("white")
