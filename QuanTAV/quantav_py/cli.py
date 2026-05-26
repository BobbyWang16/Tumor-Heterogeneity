from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import analyze_case
from .config import QuantavParams


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute QuanTAV-style tumor vessel features.")
    parser.add_argument("--image", default=None, help="Optional CT image path.")
    parser.add_argument("--tumor-mask", required=True, help="Tumor mask path.")
    parser.add_argument("--vessel-mask", required=True, help="Vessel mask path.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV and figures.")
    parser.add_argument("--case-id", default="case", help="Case identifier.")
    parser.add_argument("--spacing", nargs=3, type=float, default=None, help="Array-axis spacing, e.g. z y x.")
    parser.add_argument("--target-spacing", nargs=3, type=float, default=None, help="Optional resampling spacing.")
    parser.add_argument("--radius-mm", type=float, default=30.0, help="Peritumoral radius in mm.")
    parser.add_argument("--min-vessel-component-size", type=int, default=20)
    parser.add_argument("--no-visualization", action="store_true")
    args = parser.parse_args()

    params = QuantavParams(
        spacing=tuple(args.spacing) if args.spacing else (1.0, 1.0, 1.0),
        target_spacing=tuple(args.target_spacing) if args.target_spacing else None,
        peritumoral_radius_mm=args.radius_mm,
        min_vessel_component_size=args.min_vessel_component_size,
        create_visualizations=not args.no_visualization,
    )

    result = analyze_case(
        image=args.image,
        tumor_mask=args.tumor_mask,
        vessel_mask=args.vessel_mask,
        params=params,
        output_dir=args.output_dir,
        case_id=args.case_id,
    )
    print(
        json.dumps(
            {
                "case_id": args.case_id,
                "n_features": len(result.features),
                "n_analyzed_branches": result.metadata.get("n_analyzed_branches"),
                "features_csv": str(Path(args.output_dir) / f"{args.case_id}_features.csv"),
                "visualization_paths": result.metadata.get("visualization_paths", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

