from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.algorithm_diagnostics import (
    build_score_matrix,
    generate_algorithm_diagnostics,
    plot_spatial_model_comparison,
    robust_zscore,
)


class AlgorithmDiagnosticsTests(unittest.TestCase):
    def test_score_matrix_aligns_by_case_id(self):
        tables = {
            "bih": pd.DataFrame(
                {"case_id": ["case-b", "case-a"], "score": [2.0, 1.0]}
            ),
            "dhi": pd.DataFrame(
                {"case_id": ["case-a", "case-b"], "score": [10.0, 20.0]}
            ),
        }
        matrix = build_score_matrix(tables)
        self.assertEqual(matrix.loc["case-a", "BIH"], 1.0)
        self.assertEqual(matrix.loc["case-a", "DHI"], 10.0)
        self.assertEqual(matrix.loc["case-b", "BIH"], 2.0)
        self.assertEqual(matrix.loc["case-b", "DHI"], 20.0)

    def test_robust_zscore_handles_constant_and_missing_columns(self):
        matrix = pd.DataFrame(
            {
                "constant": [2.0, 2.0, np.nan],
                "varying": [1.0, 2.0, 3.0],
            },
            index=["a", "b", "c"],
        )
        z = robust_zscore(matrix)
        self.assertEqual(z.loc["a", "constant"], 0.0)
        self.assertTrue(np.isnan(z.loc["c", "constant"]))
        self.assertAlmostEqual(z.loc["b", "varying"], 0.0)

    def test_generate_diagnostics_creates_case_aligned_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            out = Path(tmp) / "diagnostics"
            for model, scores in {
                "bih": [0.1, 0.2, 0.3, 0.4],
                "dhi": [4.0, 3.0, 2.0, 1.0],
            }.items():
                model_dir = root / model
                model_dir.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "case_id": ["c1", "c2", "c3", "c4"],
                        "score": scores,
                    }
                ).to_csv(model_dir / f"{model}_demo.csv", index=False)

            paths = generate_algorithm_diagnostics("demo", root, out, top_n_heatmap=4)
            self.assertGreaterEqual(len(paths), 8)
            self.assertTrue(all(Path(path).exists() for path in paths.values()))
            matrix = pd.read_csv(paths["score_matrix"], index_col="case_id")
            self.assertEqual(matrix.shape, (4, 2))

    def test_spatial_comparison_accepts_label_and_confidence_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape = (12, 14, 16)
            z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
            mask = ((z - 6) ** 2 + (y - 7) ** 2 + (x - 8) ** 2) < 20
            image = np.where(mask, 50.0, -800.0)
            labels = np.where(mask, (x > 8).astype(np.uint8) + 1, 0)
            confidence = np.where(mask, np.clip(x / 15.0, 0, 1), 0.0)
            out_path = Path(tmp) / "spatial.png"

            plot_spatial_model_comparison(
                image,
                mask,
                {
                    "ith_fusion": {
                        "score": 0.25,
                        "label_map": labels,
                        "confidence_map": confidence,
                    }
                },
                out_path,
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
