"""Reusable cohort-level and spatial diagnostics for ITH algorithms.

The cohort plots operate on the standardized CSV files emitted by
``BaseHeterogeneityModel.run_and_save`` and always align models by ``case_id``.
The spatial plot operates on in-memory model outputs and is intended for
single-case algorithm quality control.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import save_and_close


MODEL_DISPLAY = {
    "bih": "BIH",
    "dhi": "DHI",
    "hab": "HAB",
    "pth": "PTH",
    "shi": "SHI",
    "thi": "THI",
    "ith_fusion": "ITH-FS",
}

MODEL_COLORS = {
    "BIH": "#2563EB",
    "DHI": "#DC2626",
    "HAB": "#059669",
    "PTH": "#7C3AED",
    "SHI": "#D97706",
    "THI": "#0D9488",
    "ITH-FS": "#BE185D",
}


def collect_score_tables(
    dataset_name: str,
    results_dir: str | Path,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load standardized model result CSVs.

    The preferred filename is ``{model}_{dataset}.csv``. A conservative
    fallback accepts another dataset-matching CSV only when it contains both
    ``case_id`` and ``score``. Per-case auxiliary CSV files are ignored.
    """

    root = Path(results_dir)
    model_names = list(models) if models is not None else list(MODEL_DISPLAY)
    tables: Dict[str, pd.DataFrame] = {}

    for model in model_names:
        model_dir = root / model
        if not model_dir.exists():
            continue

        preferred = model_dir / f"{model}_{dataset_name}.csv"
        candidates = [preferred] if preferred.exists() else []
        if not candidates:
            candidates = sorted(model_dir.glob(f"*{dataset_name}*.csv"))

        selected: Optional[pd.DataFrame] = None
        selected_path: Optional[Path] = None
        for csv_path in candidates:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                warnings.warn(f"Cannot read {csv_path}: {exc}")
                continue
            if {"case_id", "score"}.issubset(df.columns):
                selected = df
                selected_path = csv_path
                break

        if selected is None:
            continue

        selected = selected.copy()
        selected["case_id"] = selected["case_id"].astype(str)
        selected["score"] = pd.to_numeric(selected["score"], errors="coerce")
        selected.attrs["source_path"] = str(selected_path)
        tables[model] = selected

    return tables


def build_score_matrix(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an outer-joined case-by-model score matrix."""

    score_series = []
    for model, df in tables.items():
        if not {"case_id", "score"}.issubset(df.columns):
            continue
        duplicate_count = int(df["case_id"].duplicated(keep=False).sum())
        if duplicate_count:
            warnings.warn(
                f"{model}: {duplicate_count} rows have duplicate case_id; "
                "the first score per case is used."
            )
        series = (
            df.drop_duplicates("case_id", keep="first")
            .set_index("case_id")["score"]
            .rename(MODEL_DISPLAY.get(model, model.upper()))
        )
        score_series.append(series)

    if not score_series:
        return pd.DataFrame()
    matrix = pd.concat(score_series, axis=1, join="outer")
    matrix.index.name = "case_id"
    return matrix.sort_index()


def summarize_score_quality(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize valid, invalid, error, and duplicate rows per model."""

    rows = []
    for model, df in tables.items():
        score = pd.to_numeric(df.get("score"), errors="coerce")
        finite = np.isfinite(score.to_numpy(dtype=float, na_value=np.nan))
        if "error" in df.columns:
            errors = df["error"].fillna("").astype(str).str.strip().ne("")
        else:
            errors = pd.Series(False, index=df.index)

        n_total = len(df)
        n_error = int(errors.sum())
        n_valid = int((finite & ~errors.to_numpy()).sum())
        n_invalid = int(n_total - n_valid - n_error)
        rows.append(
            {
                "model": MODEL_DISPLAY.get(model, model.upper()),
                "n_rows": n_total,
                "n_cases": int(df["case_id"].nunique()) if "case_id" in df else 0,
                "n_valid": n_valid,
                "n_invalid": max(0, n_invalid),
                "n_error": n_error,
                "n_duplicate_rows": int(df["case_id"].duplicated(keep=False).sum())
                if "case_id" in df
                else 0,
                "valid_fraction": float(n_valid / n_total) if n_total else 0.0,
                "source_path": df.attrs.get("source_path", ""),
            }
        )
    return pd.DataFrame(rows)


def robust_zscore(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Column-wise robust z-score using median and scaled MAD.

    Constant columns fall back to the standard deviation. A completely
    constant valid column maps to zero rather than producing infinities.
    """

    out = pd.DataFrame(index=score_matrix.index)
    for col in score_matrix.columns:
        values = pd.to_numeric(score_matrix[col], errors="coerce")
        median = values.median()
        mad = (values - median).abs().median()
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            scale = values.std(ddof=0)
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            out[col] = np.where(values.notna(), 0.0, np.nan)
        else:
            out[col] = (values - median) / scale
    return out


def plot_score_quality_dashboard(
    quality: pd.DataFrame,
    out_path: str | Path,
    dataset_name: str = "Dataset",
) -> str:
    """Plot valid/invalid/error composition for every available model."""

    if quality.empty:
        raise ValueError("quality table is empty")
    q = quality.set_index("model")
    fig, ax = plt.subplots(figsize=(max(8, 1.35 * len(q)), 5.5))
    bottom = np.zeros(len(q), dtype=float)
    for column, label, color in [
        ("n_valid", "Valid finite score", "#22C55E"),
        ("n_invalid", "Missing / non-finite", "#F59E0B"),
        ("n_error", "Runtime error", "#DC2626"),
    ]:
        values = q[column].to_numpy(dtype=float)
        ax.bar(q.index, values, bottom=bottom, label=label, color=color, width=0.72)
        bottom += values

    for idx, (_, row) in enumerate(q.iterrows()):
        ax.text(
            idx,
            row["n_rows"] + max(q["n_rows"].max() * 0.015, 0.5),
            f"{100 * row['valid_fraction']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0, max(q["n_rows"].max() * 1.13, 1))
    ax.set_ylabel("Number of rows")
    ax.set_xlabel("Model")
    ax.set_title(f"{dataset_name} · Algorithm Output Quality")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    save_and_close(fig, str(out_path))
    return str(out_path)


def plot_score_distributions(
    score_matrix: pd.DataFrame,
    out_path: str | Path,
    dataset_name: str = "Dataset",
    bins: int = 30,
    display_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> str:
    """Plot raw score distributions without implying shared scale.

    The histogram display is limited to the requested percentile interval so
    a few numerical extremes do not hide the cohort body. The annotation still
    reports the full raw range and the number of values outside the display.
    """

    if score_matrix.empty:
        raise ValueError("score matrix is empty")
    n_models = len(score_matrix.columns)
    n_cols = min(3, n_models)
    n_rows = int(math.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, score_matrix.columns):
        values = score_matrix[col].dropna().to_numpy(dtype=float)
        color = MODEL_COLORS.get(col, "#2563EB")
        if values.size:
            median = float(np.median(values))
            q25, q75 = np.percentile(values, [25, 75])
            lo, hi = np.percentile(values, display_percentiles)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = float(np.min(values)), float(np.max(values))
            displayed = values[(values >= lo) & (values <= hi)]
            if displayed.size == 0:
                displayed = values
            ax.hist(
                displayed,
                bins=bins,
                range=(lo, hi) if lo < hi else None,
                color=color,
                alpha=0.82,
                edgecolor="white",
            )
            ax.axvline(median, color="#111827", linestyle="--", linewidth=1.5)
            ax.axvspan(q25, q75, color="#111827", alpha=0.08)
            n_outside = int(values.size - displayed.size)
            ax.text(
                0.98,
                0.94,
                f"n={values.size}\nmedian={median:.3g}\nIQR={q75-q25:.3g}\n"
                f"full range=[{values.min():.3g}, {values.max():.3g}]\n"
                f"outside P{display_percentiles[0]:g}–P{display_percentiles[1]:g}: {n_outside}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
            )
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Raw score")
        ax.set_ylabel("Cases")
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[n_models:]:
        ax.axis("off")
    fig.suptitle(f"{dataset_name} · Raw Score Distributions", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_and_close(fig, str(out_path), tight=False)
    return str(out_path)


def plot_model_agreement(
    score_matrix: pd.DataFrame,
    out_path: str | Path,
    dataset_name: str = "Dataset",
    method: str = "spearman",
    min_pairs: int = 3,
) -> str:
    """Plot pairwise model correlation with pairwise sample counts."""

    if score_matrix.shape[1] < 2:
        raise ValueError("at least two model score columns are required")
    corr = score_matrix.corr(method=method, min_periods=min_pairs)
    counts = score_matrix.notna().astype(int).T.dot(score_matrix.notna().astype(int))
    annotations = corr.copy().astype(object)
    for row in corr.index:
        for col in corr.columns:
            rho = corr.loc[row, col]
            n_pair = int(counts.loc[row, col])
            annotations.loc[row, col] = (
                f"{rho:.2f}\n(n={n_pair})" if np.isfinite(rho) else f"NA\n(n={n_pair})"
            )

    fig, ax = plt.subplots(figsize=(max(7, 1.25 * len(corr)), max(6, 1.05 * len(corr))))
    sns.heatmap(
        corr,
        annot=annotations,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": f"{method.title()} correlation"},
        ax=ax,
    )
    ax.set_title(f"{dataset_name} · Cross-Model Agreement", pad=14)
    fig.tight_layout()
    save_and_close(fig, str(out_path), tight=False)
    return str(out_path)


def plot_robust_score_heatmap(
    score_matrix: pd.DataFrame,
    out_path: str | Path,
    dataset_name: str = "Dataset",
    top_n: int = 100,
    clip: float = 3.0,
) -> str:
    """Plot case-level robust z-scores, prioritizing unusual cases."""

    if score_matrix.empty:
        raise ValueError("score matrix is empty")
    z = robust_zscore(score_matrix)
    unusualness = z.abs().mean(axis=1, skipna=True)
    selected = unusualness.sort_values(ascending=False).head(max(1, int(top_n))).index
    z_selected = z.loc[selected].clip(-clip, clip)

    height = min(20, max(5, 0.18 * len(z_selected)))
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(z.columns)), height))
    show_case_labels = len(z_selected) <= 50
    sns.heatmap(
        z_selected,
        cmap="RdBu_r",
        center=0,
        vmin=-clip,
        vmax=clip,
        yticklabels=show_case_labels,
        cbar_kws={"label": "Robust z-score (clipped)"},
        ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Cases ordered by cross-model unusualness")
    ax.set_title(
        f"{dataset_name} · Case-Level Score Pattern\n"
        f"Top {len(z_selected)} most unusual cases"
    )
    fig.tight_layout()
    save_and_close(fig, str(out_path), tight=False)
    return str(out_path)


def plot_rank_profiles(
    score_matrix: pd.DataFrame,
    out_path: str | Path,
    dataset_name: str = "Dataset",
    n_cases: int = 12,
) -> str:
    """Show consensus-high cases and cases with the largest model disagreement."""

    if score_matrix.shape[1] < 2:
        raise ValueError("at least two model score columns are required")
    ranks = score_matrix.rank(axis=0, method="average", pct=True)
    valid_models = ranks.notna().sum(axis=1)
    eligible = ranks.loc[valid_models >= 2]
    if eligible.empty:
        raise ValueError("no cases have at least two valid model scores")

    consensus = eligible.mean(axis=1, skipna=True)
    disagreement = eligible.std(axis=1, skipna=True)
    n_each = max(1, int(n_cases) // 2)
    consensus_ids = consensus.nlargest(n_each).index.tolist()
    disagreement_ids = [
        case_id
        for case_id in disagreement.nlargest(n_each * 2).index
        if case_id not in consensus_ids
    ][:n_each]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, case_ids, title in [
        (axes[0], consensus_ids, "Highest mean percentile"),
        (axes[1], disagreement_ids, "Largest model disagreement"),
    ]:
        for case_id in case_ids:
            values = eligible.loc[case_id]
            ax.plot(
                range(len(values)),
                values,
                marker="o",
                linewidth=1.6,
                markersize=4,
                alpha=0.82,
                label=str(case_id),
            )
        ax.set_xticks(range(len(eligible.columns)))
        ax.set_xticklabels(eligible.columns, rotation=30, ha="right")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if case_ids:
            ax.legend(fontsize=8, frameon=False, loc="best")
    axes[0].set_ylabel("Within-model score percentile")
    fig.suptitle(f"{dataset_name} · Cross-Model Rank Profiles", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_and_close(fig, str(out_path), tight=False)
    return str(out_path)


def plot_spatial_model_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    model_results: Mapping[str, Mapping[str, Any]],
    out_path: str | Path,
    case_id: str = "case",
    include_confidence: bool = True,
) -> str:
    """Compare label maps and confidence maps in three orthogonal views.

    Parameters
    ----------
    image, mask:
        Co-registered 3D arrays in ``(Z, Y, X)`` order.
    model_results:
        Mapping from model name to the dictionary returned by ``compute``.
        Entries without a shape-compatible ``label_map`` are skipped.
    out_path:
        Destination PNG path.
    include_confidence:
        Add an extra row for shape-compatible ``confidence_map`` outputs.
    """

    image = np.asarray(image)
    mask = np.asarray(mask) > 0
    if image.ndim != 3 or mask.ndim != 3 or image.shape != mask.shape:
        raise ValueError("image and mask must be shape-compatible 3D arrays")
    if not np.any(mask):
        raise ValueError("mask is empty")

    rows: list[Tuple[str, str, Optional[np.ndarray]]] = [("CT + tumor", "base", None)]
    for model, result in model_results.items():
        label_map = result.get("label_map")
        if isinstance(label_map, np.ndarray) and label_map.shape == image.shape:
            score = result.get("score", np.nan)
            score_text = f"{float(score):.3g}" if np.isscalar(score) and np.isfinite(score) else "NA"
            rows.append((f"{MODEL_DISPLAY.get(model, model.upper())} · score={score_text}", "label", label_map))
        confidence_map = result.get("confidence_map")
        if (
            include_confidence
            and isinstance(confidence_map, np.ndarray)
            and confidence_map.shape == image.shape
        ):
            rows.append((f"{MODEL_DISPLAY.get(model, model.upper())} · confidence", "confidence", confidence_map))

    indices = _maximum_mask_indices(mask)
    views = ["Axial", "Coronal", "Sagittal"]
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12.5, max(3.2, 3.05 * len(rows))),
        squeeze=False,
    )
    roi_values = image[mask]
    vmin, vmax = np.percentile(roi_values, [1, 99])

    for row_idx, (row_name, kind, values) in enumerate(rows):
        for col_idx, (view, index) in enumerate(zip(views, indices)):
            ax = axes[row_idx, col_idx]
            image_slice = _orthogonal_slice(image, col_idx, index)
            mask_slice = _orthogonal_slice(mask, col_idx, index)
            ax.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax)

            if kind == "label" and values is not None:
                value_slice = _orthogonal_slice(values, col_idx, index)
                positive = np.asarray(value_slice) > 0
                if np.any(positive):
                    max_label = max(1, int(np.nanmax(value_slice)))
                    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, max_label)))
                    overlay = np.ma.masked_where(~positive, value_slice)
                    ax.imshow(
                        overlay,
                        cmap=cmap,
                        vmin=1,
                        vmax=max_label,
                        alpha=0.62,
                        interpolation="nearest",
                    )
            elif kind == "confidence" and values is not None:
                value_slice = _orthogonal_slice(values, col_idx, index)
                overlay = np.ma.masked_where(~mask_slice, value_slice)
                ax.imshow(overlay, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.72)

            if np.any(mask_slice) and min(mask_slice.shape) >= 2:
                ax.contour(mask_slice.astype(float), levels=[0.5], colors="#FDE047", linewidths=0.8)
            if row_idx == 0:
                ax.set_title(f"{view} · index {index}", fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{case_id} · Spatial Algorithm Diagnostics", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_and_close(fig, str(out_path), tight=False)
    return str(out_path)


def generate_algorithm_diagnostics(
    dataset_name: str,
    results_dir: str | Path,
    out_dir: str | Path,
    models: Optional[Sequence[str]] = None,
    top_n_heatmap: int = 100,
) -> Dict[str, str]:
    """Generate all available cohort-level diagnostic artifacts."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = collect_score_tables(dataset_name, results_dir, models=models)
    matrix = build_score_matrix(tables)
    quality = summarize_score_quality(tables)
    if matrix.empty:
        warnings.warn(f"No standardized score tables found in {results_dir}")
        return {}

    paths: Dict[str, str] = {}
    matrix_path = output / "00_case_model_score_matrix.csv"
    quality_path = output / "00_algorithm_quality_summary.csv"
    matrix.to_csv(matrix_path)
    quality.to_csv(quality_path, index=False)
    paths["score_matrix"] = str(matrix_path)
    paths["quality_summary"] = str(quality_path)

    paths["quality_dashboard"] = plot_score_quality_dashboard(
        quality, output / "01_algorithm_output_quality.png", dataset_name
    )
    paths["score_distributions"] = plot_score_distributions(
        matrix, output / "02_raw_score_distributions.png", dataset_name
    )
    if matrix.shape[1] >= 2:
        paths["model_agreement"] = plot_model_agreement(
            matrix, output / "03_cross_model_agreement.png", dataset_name
        )
        paths["robust_heatmap"] = plot_robust_score_heatmap(
            matrix,
            output / "04_case_robust_zscore_heatmap.png",
            dataset_name,
            top_n=top_n_heatmap,
        )
        paths["rank_profiles"] = plot_rank_profiles(
            matrix, output / "05_cross_model_rank_profiles.png", dataset_name
        )

    descriptive = matrix.describe().T
    descriptive["missing"] = matrix.isna().sum()
    descriptive["finite_fraction"] = matrix.notna().mean()
    stats_path = output / "06_score_statistical_summary.csv"
    descriptive.to_csv(stats_path)
    paths["statistical_summary"] = str(stats_path)
    return paths


def _maximum_mask_indices(mask: np.ndarray) -> Tuple[int, int, int]:
    axial = int(np.argmax(mask.sum(axis=(1, 2))))
    coronal = int(np.argmax(mask.sum(axis=(0, 2))))
    sagittal = int(np.argmax(mask.sum(axis=(0, 1))))
    return axial, coronal, sagittal


def _orthogonal_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return array[index, :, :]
    if axis == 1:
        return np.flipud(array[:, index, :])
    if axis == 2:
        return np.flipud(array[:, :, index])
    raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
