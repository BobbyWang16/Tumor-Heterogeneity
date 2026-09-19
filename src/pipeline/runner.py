"""
Pipeline runner for ITH model execution and visualization.

Provides a unified interface to run any model, generate visualizations,
and create summary reports.
"""

from pathlib import Path
from typing import List, Optional

from src.core import DataConfig
from src.models import MODEL_REGISTRY


class ModelRunner:
    """Execute ITH models and generate outputs."""

    def __init__(self, data_cfg: DataConfig, models: Optional[List[str]] = None):
        self.data_cfg = data_cfg
        self.models = list(MODEL_REGISTRY) if models is None else list(dict.fromkeys(models))
        invalid = [name for name in self.models if name not in MODEL_REGISTRY]
        if invalid:
            raise ValueError(f"Unknown models: {invalid}")

    def run(self, visualize: bool = False) -> dict:
        """Run all configured models."""
        results = {}
        for name in self.models:
            results[name] = self.run_single(name, visualize=visualize)

        if visualize:
            from src.visualization import generate_summary_report

            summary_dir = Path(self.data_cfg.output_dir) / "summary"
            generate_summary_report(
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
            from src.visualization import model_visualizers

            viz_dir = Path(self.data_cfg.output_dir) / model_name / "viz"
            model_visualizers.visualize_model_results(model_name, csv_path, str(viz_dir), self.data_cfg.dataset)

        return csv_path
