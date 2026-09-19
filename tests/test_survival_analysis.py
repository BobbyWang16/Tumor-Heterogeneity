from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.survival import generate_survival_report


class SurvivalAnalysisTests(unittest.TestCase):
    def test_report_joins_by_case_id_and_creates_visuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "demo"
            results_dir = root / "results" / "demo"
            model_dir = results_dir / "bih"
            data_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)

            n = 24
            case_ids = [f"c{i:02d}" for i in range(n)]
            rng = np.random.default_rng(2026)
            pd.DataFrame({
                "PatientID": case_ids,
                "time": rng.integers(30, 900, size=n),
                "event": [0, 1] * (n // 2),
            }).to_excel(data_dir / "clinical.xlsx", index=False)
            scores = np.linspace(0.1, 1.0, n)
            rng.shuffle(scores)
            pd.DataFrame({
                "case_id": list(reversed(case_ids)),
                "score": scores,
            }).to_csv(model_dir / "bih_demo.csv", index=False)

            paths = generate_survival_report(
                "demo", root / "data", results_dir, results_dir / "survival"
            )
            self.assertEqual(set(paths), {
                "merged_data", "summary", "km_overview_png", "km_overview_pdf"
            })
            self.assertTrue(all(Path(path).exists() for path in paths.values()))
            merged = pd.read_csv(paths["merged_data"])
            self.assertEqual(len(merged), n)
            expected = scores[-1]
            self.assertAlmostEqual(
                merged.loc[merged["case_id"] == "c00", "BIH"].iloc[0], expected
            )


if __name__ == "__main__":
    unittest.main()
