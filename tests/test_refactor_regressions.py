"""Regression coverage for batch memory, dataset paths and numerical edge cases."""

import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

from src.core import BaseHeterogeneityModel, DataConfig, DatasetLoader, ModelConfig
from src.core import base
from src.pipeline import ModelRunner
from DualCT.ith_analysis import NSCLCHeterogeneityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "QuanTAV"))
from quantav_py.config import QuantavParams
from quantav_py.api import _resolve_inputs


@pytest.mark.parametrize("directories", [[], ["image_process"], ["mask_process"], ["image_process", "mask_process"]])
def test_processed_directories_are_selected_as_a_pair(tmp_path, directories):
    for name in directories:
        (tmp_path / "cohort" / name).mkdir(parents=True)
    cfg = DataConfig("cohort", data_root=str(tmp_path))
    loader = DatasetLoader("cohort", data_root=str(tmp_path))
    processed = len(directories) == 2
    assert Path(cfg.image_dir).name == ("image_process" if processed else "image")
    assert Path(cfg.mask_dir).name == ("mask_process" if processed else "mask")
    assert loader.image_dir == Path(cfg.image_dir)
    assert loader.mask_dir == Path(cfg.mask_dir)


def test_observer_layouts(tmp_path):
    for name in ("mask_process", "mask_a", "multimask/b"):
        (tmp_path / "cohort" / name).mkdir(parents=True)
    loader = DatasetLoader("cohort", data_root=str(tmp_path))
    assert loader.has_multimask
    assert set(loader.get_observer_mask_dirs()) == {"a", "b"}


def test_batch_releases_maps_before_next_case(monkeypatch):
    refs = []

    class Dummy(BaseHeterogeneityModel):
        def compute(self, image, mask):
            assert all(ref() is None for ref in refs)
            labels = np.zeros((4, 4, 4))
            refs.append(weakref.ref(labels))
            return {"score": 2.0, "features": {"mean": 3.0}, "label_map": labels}

    monkeypatch.setattr(base, "match_image_mask_pairs", lambda *args: [(str(i), "image", "mask") for i in range(3)])
    monkeypatch.setattr(base, "load_image", lambda path: (np.ones((4, 4, 4)), None))
    monkeypatch.setattr(base, "load_seg", lambda path: (np.ones((4, 4, 4)), None))
    result = Dummy(ModelConfig(verbose=False)).batch_compute(DataConfig("cohort"))
    assert list(result.columns) == ["case_id", "score", "feat_mean"]
    assert result["score"].tolist() == [2.0] * 3
    assert all(ref() is None for ref in refs)


def test_runner_validates_and_deduplicates():
    cfg = DataConfig("cohort")
    with pytest.raises(ValueError, match="Unknown models"):
        ModelRunner(cfg, ["bih", "missing"])
    assert ModelRunner(cfg, ["bih", "bih"]).models == ["bih"]
    assert ModelRunner(cfg, []).run() == {}


def test_noncontiguous_cluster_labels():
    labels = np.array([2, 2, 5, 5])
    features = np.zeros((4, 4))
    features[:, 2] = [20, 30, 0, 10]
    actual = NSCLCHeterogeneityAnalyzer._reorder_labels_by_enhancement(labels, features)
    np.testing.assert_array_equal(actual, [1, 1, 0, 0])


def test_masked_smoothing_has_no_divide_by_zero():
    data = np.ones((7, 7, 7), dtype=np.float32)
    mask = np.zeros_like(data, dtype=bool)
    mask[3, 3, 3] = True
    analyzer = NSCLCHeterogeneityAnalyzer(data, data, mask)
    with np.errstate(divide="raise", invalid="raise"):
        result = analyzer._kernel_smooth_masked(data, mask)
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_spacing_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite and positive"):
        QuantavParams(spacing=(1, value, 1)).normalized_spacing()


def test_quantav_rejects_mismatched_file_spacing(monkeypatch):
    from quantav_py import api

    def load(path):
        return np.ones((3, 3, 3)), ((2, 1, 1) if path == "vessel" else (1, 1, 1)), None

    monkeypatch.setattr(api, "load_volume", load)
    with pytest.raises(ValueError, match="align inputs first"):
        _resolve_inputs(None, "tumor", "vessel", QuantavParams())


def test_numpy_inputs_use_explicit_spacing(tmp_path):
    path = tmp_path / "mask.npy"
    np.save(path, np.ones((3, 3, 3)))
    _, _, _, params = _resolve_inputs(None, path, path, QuantavParams(spacing=(2, 3, 4)))
    assert params.spacing == (2, 3, 4)


def test_preprocess_reports_partial_failure(monkeypatch, tmp_path):
    from src.core import preprocess_pipeline

    monkeypatch.setattr(preprocess_pipeline, "match_image_mask_pairs", lambda *args: [("bad", "i", "m")])
    def fail(*args):
        raise ValueError("broken volume")
    monkeypatch.setattr(preprocess_pipeline, "preprocess_ct", fail)
    with pytest.raises(RuntimeError, match="failed for 1 case"):
        preprocess_pipeline.batch_preprocess("cohort", str(tmp_path))


def test_empty_batch_does_not_write_success_csv(monkeypatch, tmp_path):
    class Dummy(BaseHeterogeneityModel):
        def compute(self, image, mask):
            return {"score": 1}
    monkeypatch.setattr(base, "match_image_mask_pairs", lambda *args: [])
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="No cases processed"):
        Dummy().run_and_save(DataConfig("cohort", output_root=str(tmp_path)))
    assert not list(tmp_path.rglob("*.csv"))


def test_gene_inputs_align_only_shared_cases(monkeypatch):
    import pandas as pd
    from analysis import nsclc_gene_enrichment as genes

    expression = pd.DataFrame([[1, 2, 3]], columns=["b", "missing", "a"], index=["gene1"])
    clinical = pd.DataFrame({"Case ID": ["a", "b", "extra"], "value": [10, 20, 30]})
    monkeypatch.setattr(genes.pd, "read_csv", lambda *args, **kwargs: expression)
    monkeypatch.setattr(genes.pd, "read_excel", lambda *args, **kwargs: clinical)
    expr, matched = genes.load_inputs()
    assert expr.columns.tolist() == matched.index.tolist() == ["b", "a"]
    assert matched["value"].tolist() == [20, 10]


def test_icc_passes_data_root_and_checks_exit_status(monkeypatch, tmp_path):
    import subprocess
    import run_all_scores

    for name in ("a", "b"):
        (tmp_path / "cohort" / "multimask" / name).mkdir(parents=True)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    run_all_scores.run_icc_test("cohort", data_dir=str(tmp_path), models=["bih"])
    cmd, kwargs = calls[0]
    assert cmd[cmd.index("--data-dir") + 1] == str(tmp_path)
    assert kwargs["check"] is True
