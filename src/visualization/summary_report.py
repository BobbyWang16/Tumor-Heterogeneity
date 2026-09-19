"""Cross-model cohort report.

This compatibility wrapper delegates to the case-aligned diagnostics module.
Unlike the legacy implementation, it reads the standardized result filenames
and joins scores by ``case_id`` instead of truncating arrays by row count.
"""

from __future__ import annotations

from typing import Dict

from .algorithm_diagnostics import generate_algorithm_diagnostics


def generate_summary_report(dataset_name, results_dir, out_dir) -> Dict[str, str]:
    """Generate case-aligned cross-model diagnostic figures and tables."""

    print(f"\n{'=' * 60}")
    print(f"Generating algorithm diagnostics for {dataset_name}")
    print(f"{'=' * 60}")
    paths = generate_algorithm_diagnostics(
        dataset_name=dataset_name,
        results_dir=results_dir,
        out_dir=out_dir,
    )
    if not paths:
        print("[WARN] No standardized score data found for summary report.")
        return {}
    print(f"[Summary] Generated {len(paths)} artifacts in {out_dir}")
    return paths
