from __future__ import annotations

import numpy as np

from example_api import synthetic_case
from quantav_py import QuantavParams, analyze_case


def main() -> None:
    image, tumor, vessel = synthetic_case()
    params = QuantavParams(
        spacing=(1.0, 1.0, 1.0),
        peritumoral_radius_mm=30.0,
        min_vessel_component_size=3,
        organization_window_size=21,
        create_visualizations=False,
    )
    result = analyze_case(image, tumor, vessel, params=params, case_id="smoke")
    assert result.features.shape == (91,)
    assert np.isfinite(result.features).all()
    assert len(result.feature_names) == 91
    print("smoke ok", result.metadata)


if __name__ == "__main__":
    main()

