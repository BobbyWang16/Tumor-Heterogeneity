#!/usr/bin/env python
"""
Inter-Observer Variability Test for ITH Features

检验异质性特征是否受不同勾画者分割差异的影响。

用法:
    # 1. 准备数据：每个病例有多个observer的掩膜
    #    data/
    #    └── NSCLC-Radiomics/
    #        ├── image/           # CT图像
    #        ├── mask_observer1/  # 勾画者1的掩膜
    #        ├── mask_observer2/  # 勾画者2的掩膜
    #        └── mask_observer3/  # 勾画者3的掩膜

    # 2. 运行测试
    python -m src.evaluation.inter_observer_test \\
        --dataset NSCLC-Radiomics \\
        --observers observer1 observer2 observer3 \\
        --models bih dhi hab pth shi thi \\
        --output ./results/ICC

输出:
    - ICC结果CSV
    - ICC可视化图表
    - 通过/失败的特征列表
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core import DatasetLoader, load_image, load_seg, match_image_mask_pairs
from src.models import MODEL_REGISTRY
from src.evaluation.icc_analysis import run_icc_workflow


def extract_features_for_observer(
    dataset_name: str,
    observer: str,
    models: list,
    data_root: str = "./data",
) -> pd.DataFrame:
    """
    对单个勾画者的所有病例提取异质性特征。

    Returns:
        DataFrame: 每行一个病例，每列一个特征
    """
    image_dir = Path(data_root) / dataset_name / "image"
    loader = DatasetLoader(dataset_name, data_root=data_root, use_processed=False)
    mask_dir = loader.get_observer_mask_dirs().get(observer)
    if mask_dir is None:
        raise FileNotFoundError(f"Mask directory not found for observer: {observer}")

    pairs = match_image_mask_pairs(str(image_dir), str(mask_dir))
    if not pairs:
        raise ValueError(f"No image-mask pairs found in {image_dir} & {mask_dir}")

    print(f"[{observer}] Found {len(pairs)} cases")

    rows = []
    for case_id, img_path, msk_path in tqdm(pairs, desc=f"[{observer}]"):
        try:
            vol, _ = load_image(img_path)
            msk, _ = load_seg(msk_path)

            row = {"case_id": case_id, "observer": observer}

            for model_name in models:
                ModelClass = MODEL_REGISTRY[model_name]
                model = ModelClass()
                result = model.compute(vol, msk)

                # 记录主分数
                row[f"{model_name}_score"] = result.get("score", np.nan)

                # 记录详细特征
                features = result.get("features", {})
                for feat_name, feat_val in features.items():
                    if isinstance(feat_val, (int, float, np.floating)):
                        col_name = f"{model_name}_{feat_name}"
                        row[col_name] = float(feat_val)

            rows.append(row)

        except Exception as e:
            print(f"  [WARN] Failed {case_id}: {e}")
            rows.append({"case_id": case_id, "observer": observer, "error": str(e)})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Inter-Observer ICC Test for ITH Features")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--observers", type=str, nargs="+", required=True,
                        help="勾画者标识列表，如 observer1 observer2 observer3")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["bih", "dhi", "hab", "pth", "shi", "thi"],
                        help="要测试的模型")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="数据根目录")
    parser.add_argument("--output", type=str, default="./results/ICC",
                        help="输出目录")
    parser.add_argument("--icc-type", type=str, default="ICC(2,1)",
                        choices=["ICC(1,1)", "ICC(2,1)", "ICC(3,1)", "ICC(2,k)", "ICC(3,k)"],
                        help="ICC类型（默认ICC(2,1)适用于inter-observer）")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="ICC可靠性阈值（默认0.75）")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="跳过特征提取，直接读取已有CSV")
    parser.add_argument("--features-csv", type=str,
                        help="已有的特征CSV路径（用于--skip-extraction）")
    args = parser.parse_args()

    print("=" * 60)
    print("Inter-Observer ICC Test for ITH Features")
    print("=" * 60)
    print(f"Dataset:     {args.dataset}")
    print(f"Observers:   {', '.join(args.observers)}")
    print(f"Models:      {', '.join(args.models)}")
    print(f"ICC Type:    {args.icc_type}")
    print(f"Threshold:   {args.threshold}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Step 1: 特征提取（或读取已有数据）
    # ------------------------------------------------------------------ #
    if args.skip_extraction and args.features_csv:
        print(f"\n[Step 1] Loading features from {args.features_csv}")
        all_data = pd.read_csv(args.features_csv)
    else:
        print(f"\n[Step 1] Extracting features for {len(args.observers)} observers...")
        all_data_list = []
        for observer in args.observers:
            df = extract_features_for_observer(
                args.dataset, observer, args.models, args.data_dir
            )
            all_data_list.append(df)

        all_data = pd.concat(all_data_list, ignore_index=True)

        # 保存原始特征
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = out_dir / f"{args.dataset}_features_all_observers.csv"
        all_data.to_csv(raw_csv, index=False)
        print(f"[Step 1] Raw features saved to {raw_csv}")

    # ------------------------------------------------------------------ #
    #  Step 2: 识别可分析的特征列
    # ------------------------------------------------------------------ #
    # 排除非特征列
    exclude_cols = ["case_id", "observer", "error"]
    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    # 去除全为NaN的特征
    valid_features = []
    for col in feature_cols:
        if all_data[col].notna().sum() >= len(args.observers) * 2:
            valid_features.append(col)

    print(f"\n[Step 2] {len(valid_features)} valid features for ICC analysis")

    # ------------------------------------------------------------------ #
    #  Step 3: ICC分析
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3] Running ICC analysis...")
    results_df, summary = run_icc_workflow(
        data=all_data,
        feature_cols=valid_features,
        output_dir=args.output,
        subject_col="case_id",
        observer_col="observer",
        icc_type=args.icc_type,
        icc_threshold=args.threshold,
        dataset_name=args.dataset,
    )

    # ------------------------------------------------------------------ #
    #  Step 4: 按模型汇总
    # ------------------------------------------------------------------ #
    print(f"\n[Step 4] Model-level summary:")
    print("-" * 60)
    for model_name in args.models:
        model_features = [c for c in valid_features if c.startswith(f"{model_name}_")]
        if not model_features:
            continue
        model_results = results_df[results_df["feature"].isin(model_features)]
        valid_results = model_results[model_results["icc"].notna()]
        if len(valid_results) == 0:
            continue

        pass_count = (valid_results["icc"] >= args.threshold).sum()
        mean_icc = valid_results["icc"].mean()
        print(f"  {model_name.upper():12s}: {pass_count}/{len(valid_results)} pass  |  Mean ICC: {mean_icc:.3f}")

    print(f"\n{'='*60}")
    print("Inter-Observer ICC Test Complete")
    print(f"{'='*60}")
    print(f"Output directory: {Path(args.output).absolute()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
