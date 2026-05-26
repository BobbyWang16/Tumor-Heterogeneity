#!/usr/bin/env python
"""
批量预处理所有数据集的CT和mask。

处理步骤：
1. HU调窗：clip到 [-1200, 800]
2. 重采样到 1x1x1 mm（CT: Bspline, Mask: NearestNeighbor）
3. 重orientation到 RAI (1 0 0 0 1 0 0 0 1)
4. 保存到 image_process / mask_process

用法：
    python scripts/preprocess_all.py --datasets NSCLC-Radiomics NSCLC-Radiogenomics
    python scripts/preprocess_all.py --all
"""

import argparse
import sys

from src.core import batch_preprocess


def main():
    parser = argparse.ArgumentParser(description="Batch preprocess CT and mask")
    parser.add_argument("--datasets", type=str, nargs="+", help="数据集名称列表")
    parser.add_argument("--all", action="store_true", help="处理所有数据集")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--hu-min", type=float, default=-1200.0)
    parser.add_argument("--hu-max", type=float, default=800.0)
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    args = parser.parse_args()

    if args.all:
        import os
        datasets = [d for d in os.listdir(args.data_dir)
                    if os.path.isdir(os.path.join(args.data_dir, d))
                    and not d.startswith(".")]
    else:
        datasets = args.datasets

    if not datasets:
        print("[ERROR] 请指定 --datasets 或 --all")
        return 1

    for dataset in datasets:
        try:
            batch_preprocess(
                dataset=dataset,
                data_root=args.data_dir,
                hu_min=args.hu_min,
                hu_max=args.hu_max,
                target_spacing=tuple(args.spacing),
            )
        except Exception as e:
            print(f"[ERROR] {dataset}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
