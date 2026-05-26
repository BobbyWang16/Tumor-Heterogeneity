"""
Pipeline runner for ITH model execution and visualization.

Provides a unified interface to run any model, generate visualizations,
and create summary reports.
"""

import os
from pathlib import Path
from typing import List, Optional

from src.core import DataConfig
from src.models import MODEL_REGISTRY
from src.visualization import model_visualizers, summary_report


class ModelRunner:
    """Execute ITH models and generate outputs."""

    def __init__(self, data_cfg: DataConfig, models: Optional[List[str]] = None):
        self.data_cfg = data_cfg
        self.models = models or list(MODEL_REGISTRY.keys())

    def run(self, visualize: bool = False) -> dict:
        """Run all configured models."""
        results = {}
        for name in self.models:
            ModelClass = MODEL_REGISTRY[name]
            model = ModelClass()
            csv_path = model.run_and_save(self.data_cfg)
            results[name] = csv_path

            if visualize and csv_path:
                viz_dir = Path(self.data_cfg.output_dir) / name / "viz"
                model_visualizers.visualize_model_results(name, csv_path, str(viz_dir), self.data_cfg.dataset)

        if visualize:
            summary_dir = Path(self.data_cfg.output_dir) / "summary"
            summary_report.generate_summary_report(
                self.data_cfg.dataset,
                self.data_cfg.output_dir,
                str(summary_dir)
            )

        return results

    def run_single(self, model_name: str, visualize: bool = False) -> str:
        """Run a single model."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass()
        csv_path = model.run_and_save(self.data_cfg)

        if visualize and csv_path:
            viz_dir = Path(self.data_cfg.output_dir) / model_name / "viz"
            model_visualizers.visualize_model_results(model_name, csv_path, str(viz_dir), self.data_cfg.dataset)

        return csv_path
