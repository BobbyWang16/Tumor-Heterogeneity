# Tumor Heterogeneity

[简体中文](README.md) · **English**

**Describe tumor heterogeneity from CT images and segmentation masks.**

Calculate intensity, shape, texture, habitat, and peritumoral descriptors through a shared Python interface. Export case-level feature tables and explore cohort diagnostics and survival associations.

[Getting Started](docs/en/getting-started.md) · [Installation](docs/en/installation.md) · [User guide](docs/en/index.md) · [API Reference](docs/en/api.md) · [Package design](docs/en/package-design.md)

## What problem does it address?

Two tumors can have the same mean intensity but very different internal structures. One may be relatively uniform; another may contain interleaved regions. This toolkit describes those differences from several perspectives: **intensity distributions, spatial organization, geometry, and surrounding tissue**.

Each model returns a summary score and supporting features; some also return spatial maps. Start with DHI to understand intensity variability, then add models that address your research question. See [Basic concepts](docs/en/concepts.md) and [Design principles](docs/en/design-principles.md).

## Installation

Source installation is currently supported. The core declares Python ≥ 3.9; Python 3.11 is recommended. QuanTAV requires Python ≥ 3.10.

```bash
git clone https://github.com/BobbyWang16/Tumor-Heterogeneity.git
cd Tumor-Heterogeneity
python -m pip install -e .
```

The distribution name is `tumor-heterogeneity`; imports currently use `src`. Migration to `tumor_heterogeneity` is proposed, not implemented. These instructions do not assume a PyPI release exists.

## Your first calculation

No data download is required:

```python
import numpy as np
from src.models import DHIModel

rng = np.random.default_rng(42)
image = rng.normal(60, 15, (32, 32, 32)).astype(np.float32)
z, y, x = np.indices(image.shape)
mask = ((z - 16)**2 + (y - 16)**2 + (x - 16)**2 <= 9**2)

result = DHIModel().compute(image, mask)
print(result["score"])
print(result["features"])
```

Or run `python examples/quickstart.py`. See [Getting Started](docs/en/getting-started.md) for real images, geometry checks, and CSV export.

## Models

| Model | Perspective | Method |
| --- | --- | --- |
| BIH | Largest axial slice | Signal-weighted box analysis |
| DHI | Intensity | Distribution and local variation |
| HAB | Habitats | GMM with spatial priors |
| PTH | Peritumoral tissue | Ring, SLIC blocks, and block variation |
| SHI | Shape | Geometry, convex hull, fractals, and lacunarity |
| THI | Texture partitions | Voxel features and KMeans |
| ITH-FS | Multi-scale spatial partitions | Spatial spectral clustering |

Scores differ in definition and range; do not average them directly. ITH-FS is a separate model, not an average of the other scores. See [Models and outputs](docs/en/models.md).

## Process a cohort

Prepare identically named `.nii.gz` files in `data/demo/image_process/` and `data/demo/mask_process/`:

```python
from src.core import DataConfig
from src.pipeline import ModelRunner

config = DataConfig(dataset="demo", data_root="./data", output_root="./results")
paths = ModelRunner(config, models=["bih", "dhi"]).run(visualize=False)
```

Or run from the repository root:

```bash
python run_all_scores.py --dataset demo --models bih dhi --no-visualize
```

Inputs must already be aligned. Prepare 1 mm isotropic data before running all models. `compute()` does not register or resample automatically. See [Data and batch processing](docs/en/data-and-batch.md), including current preprocessing limitations.

## Optional components

```bash
python -m pip install -e ".[survival]"
python -m pip install -e ./QuanTAV
```

- **QuanTAV:** vessel morphology and organization features through `quantav_py.analyze_case()`.
- **DualCT:** CT phase analysis script, not yet part of the main public API.
- **analysis / R:** gene-analysis and exploratory workflows.
- **ITHscore:** separate third-party reference repository.

## Documentation and development

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Chinese and English entry points are `docs/_build/html/index.html` and `docs/_build/html/en/index.html`. Every page has a language switch; the site includes navigation and search. Read the Docs build configuration is included. Organization is inspired by [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/); interfaces and content describe this repository.

See [Development](docs/en/contributing.md) for tests. `重构说明.md` records local refactoring work.

## Citation and license

Before a formal package release, the core license, third-party attribution, and software citation need to be finalized. The root currently lacks a formal LICENSE file. The original ITHscore paper and license do not automatically describe or cover every implementation here. These are research descriptors, without a claim of clinical validation.
