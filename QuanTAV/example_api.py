from __future__ import annotations

import numpy as np

from quantav_py import QuantavParams, analyze_case


def synthetic_case(shape=(64, 64, 64)):
    zz, yy, xx = np.indices(shape)
    center = np.asarray(shape) / 2
    tumor = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) < 7**2
    vessel = np.zeros(shape, dtype=bool)
    for radius, phase in ((15, 0.0), (19, 1.1), (23, 2.2)):
        t = np.linspace(-2.5, 2.5, 120)
        z = center[0] + 0.35 * radius * np.sin(1.5 * t + phase)
        y = center[1] + radius * np.cos(t + phase)
        x = center[2] + radius * np.sin(t)
        for p in np.rint(np.c_[z, y, x]).astype(int):
            z0, y0, x0 = p
            vessel[
                max(0, z0 - 1) : min(shape[0], z0 + 2),
                max(0, y0 - 1) : min(shape[1], y0 + 2),
                max(0, x0 - 1) : min(shape[2], x0 + 2),
            ] = True
    vessel[tumor] = False
    image = np.zeros(shape, dtype=np.float32)
    image[tumor] = 80
    image[vessel] = 300
    return image, tumor, vessel


if __name__ == "__main__":
    image, tumor, vessel = synthetic_case()
    params = QuantavParams(
        spacing=(1.0, 1.0, 1.0),
        peritumoral_radius_mm=30.0,
        min_vessel_component_size=3,
        organization_window_size=21,
        create_visualizations=True,
    )
    result = analyze_case(
        image=image,
        tumor_mask=tumor,
        vessel_mask=vessel,
        params=params,
        output_dir="./python_output",
        case_id="synthetic",
    )
    print(result.features.shape)
    print(result.metadata)
