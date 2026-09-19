"""Case-aligned survival analysis for standardized ITH model scores."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core import DatasetLoader
from src.visualization.algorithm_diagnostics import build_score_matrix, collect_score_tables


def generate_survival_report(
    dataset_name: str,
    data_root: str | Path,
    results_dir: str | Path,
    out_dir: str | Path,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Fit one-variable Cox models and draw median-split KM curves.

    Scores and clinical records are joined by ``case_id``. Hazard ratios are
    reported per one standard deviation so differently scaled models remain
    comparable.
    """
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Survival analysis requires: pip install -e '.[survival]'"
        ) from exc

    survival = DatasetLoader(dataset_name, str(data_root)).get_survival_data()
    if survival is None:
        raise ValueError(
            f"{dataset_name}/clinical.xlsx must contain PatientID (or case_id), time, and event"
        )

    scores = build_score_matrix(collect_score_tables(dataset_name, results_dir, models=models))
    if scores.empty:
        raise ValueError(f"No standardized model score files found in {results_dir}")

    merged = survival.merge(scores.reset_index(), on="case_id", how="inner")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged_path = output / "00_survival_score_table.csv"
    merged.to_csv(merged_path, index=False)

    valid_models = [column for column in scores if merged[column].nunique(dropna=True) > 1]
    if not valid_models:
        raise ValueError("No model has non-constant scores for survival analysis")

    n_cols = min(3, len(valid_models))
    n_rows = math.ceil(len(valid_models) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)
    summaries = []

    for ax, model in zip(axes.flat, valid_models):
        frame = merged[["time", "event", model]].replace([np.inf, -np.inf], np.nan).dropna()
        frame = frame[(frame["time"] > 0) & frame["event"].isin([0, 1])]
        if len(frame) < 10 or frame["event"].nunique() < 2:
            ax.text(0.5, 0.5, "Insufficient survival data", ha="center", va="center")
            ax.set_title(model)
            ax.axis("off")
            continue

        score_sd = frame[model].std(ddof=0)
        frame = frame.assign(score_z=(frame[model] - frame[model].mean()) / score_sd)
        cph = CoxPHFitter()
        cph.fit(frame[["time", "event", "score_z"]], duration_col="time", event_col="event")
        row = cph.summary.loc["score_z"]

        median = frame[model].median()
        # A rank-based median split keeps groups balanced even when many cases
        # have exactly the same score.
        high = frame[model].rank(method="first") > (len(frame) / 2)
        kmf = KaplanMeierFitter()
        for label, selector, color in (
            ("Low score", ~high, "#2563EB"),
            ("High score", high, "#DC2626"),
        ):
            kmf.fit(
                frame.loc[selector, "time"], frame.loc[selector, "event"],
                label=f"{label} (n={selector.sum()})",
            )
            kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=2)

        test = logrank_test(
            frame.loc[high, "time"], frame.loc[~high, "time"],
            frame.loc[high, "event"], frame.loc[~high, "event"],
        )
        summaries.append({
            "model": model,
            "n": len(frame),
            "events": int(frame["event"].sum()),
            "hazard_ratio_per_sd": float(row["exp(coef)"]),
            "hr_ci_lower_95": float(np.exp(row["coef lower 95%"])),
            "hr_ci_upper_95": float(np.exp(row["coef upper 95%"])),
            "cox_p": float(row["p"]),
            "logrank_p": float(test.p_value),
            "median_cutoff": float(median),
        })
        ax.set_title(f"{model} · HR/SD {row['exp(coef)']:.2f} · log-rank p={test.p_value:.3g}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes.flat[len(valid_models):]:
        ax.axis("off")
    fig.suptitle(f"{dataset_name} · ITH score survival overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = output / "01_kaplan_meier_overview.png"
    pdf_path = output / "01_kaplan_meier_overview.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = output / "00_survival_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    return {
        "merged_data": str(merged_path),
        "summary": str(summary_path),
        "km_overview_png": str(png_path),
        "km_overview_pdf": str(pdf_path),
    }
