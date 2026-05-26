#!/usr/bin/env python
"""
ITH (Intratumor Heterogeneity) Scoring Pipeline

使用方式 (两种均可):

  方式1: 直接修改下方 CONFIG 配置，然后运行
      python run_all_scores.py

  方式2: 通过命令行参数覆盖配置
      python run_all_scores.py --dataset NSCLC-Radiomics --models bih dhi hab --visualize
      python run_all_scores.py --dataset immunotherapy --icc-test
      python run_all_scores.py --all-datasets
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import DataConfig, DatasetLoader
from src.models import MODEL_REGISTRY
from src.pipeline import ModelRunner


# ============================================================================
#  USER CONFIGURATION - 修改这里的参数即可运行
# ============================================================================

CONFIG = {
    # 数据集设置
    "dataset": "NSCLC-Radiomics",       # 数据集名称，设为 None 则处理所有数据集
    "data_dir": "./data",                # 数据根目录
    "out_dir": "./results",              # 结果输出目录

    # 模型设置
    "models": ["bih", "dhi", "hab", "pth", "shi", "thi", "ith_fusion"],
    # "models": ["bih", "dhi"],          # 只运行指定模型

    # 功能开关
    "visualize": True,                   # 是否生成可视化图表
    "icc_test": False,                   # 是否运行ICC inter-observer测试
    "skip_preprocess_check": False,      # 是否跳过预处理目录检查
}

# ============================================================================


def check_preprocess(dataset: str, data_dir: str = "./data") -> bool:
    """Check if image_process / mask_process exist."""
    base = Path(data_dir) / dataset
    has_process = (base / "image_process").exists() and (base / "mask_process").exists()
    if not has_process:
        print(f"[WARN] {dataset}: image_process / mask_process not found.")
        print(f"       Run: python scripts/preprocess_all.py --datasets {dataset}")
    return has_process


def run_models(
    dataset: str,
    models: List[str],
    data_dir: str = "./data",
    out_dir: str = "./results",
    visualize: bool = False,
) -> dict:
    """Run ITH scoring models."""
    data_cfg = DataConfig(dataset=dataset, data_root=data_dir, output_root=out_dir)
    runner = ModelRunner(data_cfg, models=models)
    return runner.run(visualize=visualize)


def run_icc_test(
    dataset: str,
    data_dir: str = "./data",
    out_dir: str = "./results",
    models: Optional[List[str]] = None,
) -> None:
    """Run ICC inter-observer test if multimask data exists."""
    loader = DatasetLoader(dataset, data_root=data_dir)
    if not loader.has_multimask:
        print(f"[ICC] No multimask directory for {dataset}, skipping.")
        return

    observers = list(loader.get_observer_mask_dirs().keys())
    if len(observers) < 2:
        print(f"[ICC] Only {len(observers)} observer(s) found, need >= 2.")
        return

    print(f"\n[ICC] Running inter-observer ICC test for {dataset}...")
    print(f"      Observers: {', '.join(observers)}")

    import subprocess
    cmd = [
        sys.executable, "-m", "src.evaluation.inter_observer_test",
        "--dataset", dataset,
        "--observers", *observers,
        "--output", str(Path(out_dir) / "ICC"),
    ]
    if models:
        cmd.extend(["--models", *models])
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="ITH Scoring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_scores.py --dataset NSCLC-Radiomics
  python run_all_scores.py --dataset NSCLC-Radiomics --models bih dhi hab --visualize
  python run_all_scores.py --dataset immunotherapy --icc-test
  python run_all_scores.py --all-datasets --visualize
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dataset", type=str, help="Dataset name to process")
    group.add_argument("--all-datasets", action="store_true", help="Process all available datasets")

    parser.add_argument("--models", type=str, nargs="+",
                        default=None,
                        help=f"Models to run. Available: {', '.join(MODEL_REGISTRY.keys())}")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--visualize", action="store_true", default=None,
                        help="Generate visualization plots")
    parser.add_argument("--survival", action="store_true", help="Run survival analysis")
    parser.add_argument("--icc-test", action="store_true", default=None,
                        help="Run ICC inter-observer test")
    parser.add_argument("--skip-preprocess-check", action="store_true", default=None)

    args = parser.parse_args()

    # Merge CONFIG defaults with CLI overrides
    dataset = args.dataset or CONFIG.get("dataset")
    all_datasets = args.all_datasets or CONFIG.get("all_datasets", False)
    if not dataset and not all_datasets:
        dataset = CONFIG.get("dataset")

    models = args.models or CONFIG.get("models", list(MODEL_REGISTRY.keys()))
    data_dir = args.data_dir or CONFIG.get("data_dir", "./data")
    out_dir = args.out_dir or CONFIG.get("out_dir", "./results")
    visualize = args.visualize if args.visualize is not None else CONFIG.get("visualize", False)
    icc_test = args.icc_test if args.icc_test is not None else CONFIG.get("icc_test", False)
    skip_preprocess = args.skip_preprocess_check if args.skip_preprocess_check is not None else CONFIG.get("skip_preprocess_check", False)

    invalid = [m for m in models if m not in MODEL_REGISTRY]
    if invalid:
        print(f"[ERROR] Unknown models: {invalid}")
        print(f"Available: {list(MODEL_REGISTRY.keys())}")
        return 1

    if all_datasets:
        datasets = [d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")]
    else:
        datasets = [dataset]

    overall_success = True
    for dataset in datasets:
        print("\n" + "=" * 70)
        print(f"Processing: {dataset}")
        print("=" * 70)

        if not skip_preprocess:
            if not check_preprocess(dataset, data_dir):
                overall_success = False
                continue

        try:
            print(f"\n[Step 1] Running ITH scoring models...")
            results = run_models(dataset, models, data_dir, out_dir, visualize)
            for name, path in results.items():
                status = "OK" if path else "FAIL"
                print(f"  [{status}] {name.upper()}: {path}")

            if icc_test:
                run_icc_test(dataset, data_dir, out_dir, models)

        except Exception as e:
            print(f"\n[ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
            overall_success = False

    print("\n" + "=" * 70)
    print("Done." if overall_success else "Some datasets failed.")
    print(f"Output: {Path(out_dir).absolute()}")
    print("=" * 70)
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
