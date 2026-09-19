"""Run after installing the package: python examples/quickstart.py."""
import numpy as np
from src.core import DHIConfig
from src.models import DHIModel


def main():
    rng = np.random.default_rng(42)
    image = rng.normal(60, 15, size=(32, 32, 32)).astype(np.float32)
    z, y, x = np.indices(image.shape)
    mask = ((z - 16)**2 + (y - 16)**2 + (x - 16)**2 <= 9**2)
    result = DHIModel(DHIConfig(nbins=64, win_lcv=3)).compute(image, mask)
    assert np.isfinite(result["score"])
    print("DHI:", round(result["score"], 4))
    print("Features:", len(result["features"]))
    print("ROI voxels:", result["details"]["n_voxels"])


if __name__ == "__main__":
    main()
