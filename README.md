# ITH Analysis System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive multi-dimensional intratumor heterogeneity (ITH) scoring system based on 3D CT imaging and tumor segmentation masks. This framework quantifies tumor internal heterogeneity from fractal, density, habitat, peritumoral, shape, and texture perspectives for prognostic prediction and treatment response assessment.

---

## Overview

### 7 Core Scoring Models

| Model | Full Name | Dimension | Core Principle | Primary Score |
|-------|-----------|-----------|----------------|---------------|
| **BIH** | Box-counting ITH | 2D | Signal-weighted box-counting → fractal dimension | FD (Fractal Dimension) |
| **DHI** | Density Heterogeneity Index | 3D | First-order statistics + local coefficient of variation | CV (Coefficient of Variation) |
| **HAB** | Habitat Analysis Based | 3D | GMM clustering + spatial priors → tumor habitats | H_HAB (Composite Score) |
| **PTH** | Peritumoral Heterogeneity | 3D | Peritumoral ring + SLIC supervoxels + inter-block variation | Mean CV |
| **SHI** | Shape Heterogeneity Index | 3D | Geometry + convex hull + fractal dimension + lacunarity | Shape Composite |
| **THI** | Texture Heterogeneity Index | 3D | Voxel-level radiomic features + KMeans clustering | ITH Score |
| **ITH-FS** | ITH Fusion Score | 3D | Multi-scale features + spatial spectral clustering + confidence map | ITH Fusion Score |

### Auxiliary Analysis Tools

| Module | Function |
|--------|----------|
| **ICC Analysis** | Assess whether heterogeneity features are affected by different observers (ICC 2,1) |
| **Survival Analysis** | Cox regression, Kaplan-Meier curves, risk score construction |
| **Model Comparison** | Multi-survival model (Cox/RSF/GBSA) C-index/Brier comparison |
| **Visualization** | Per-model result visualization + cross-model summary reports |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ITH.git
cd ITH

# Install Python dependencies
pip install numpy pandas scipy scikit-learn scikit-image SimpleITK nibabel matplotlib seaborn tqdm

# Optional: survival analysis
pip install lifelines

# Optional: survival models (RSF, GBSA)
pip install scikit-survival
```

### Requirements

- Python >= 3.9
- NumPy, Pandas, SciPy
- scikit-learn, scikit-image
- SimpleITK, nibabel
- matplotlib, seaborn
- tqdm

---

## Project Structure

```
ITH/
├── run_all_scores.py          # Main entry: batch run all models
├── scripts/
│   └── preprocess_all.py      # Batch preprocessing script
│
├── src/
│   ├── core/                  # Core infrastructure
│   │   ├── base.py            # BaseHeterogeneityModel abstract base class
│   │   ├── config.py          # Configuration management
│   │   ├── io.py              # Unified I/O (load_image, load_seg, ...)
│   │   ├── preprocess.py      # Preprocessing utilities
│   │   ├── preprocess_pipeline.py  # CT/mask preprocessing pipeline
│   │   ├── dataset.py         # DatasetLoader unified data loading
│   │   └── types.py           # Type definitions
│   │
│   ├── models/                # Scoring models (7 total)
│   │   ├── __init__.py        # MODEL_REGISTRY
│   │   ├── bih.py             # BIHModel
│   │   ├── dhi.py             # DHIModel
│   │   ├── hab.py             # HABModel
│   │   ├── pth.py             # PTHModel
│   │   ├── shi.py             # SHIModel
│   │   ├── thi.py             # THIModel
│   │   └── ith_fusion.py      # ITHFusionModel
│   │
│   ├── pipeline/              # Execution pipeline
│   │   └── runner.py          # ModelRunner batch execution
│   │
│   ├── features/              # Feature extraction
│   ├── survival/              # Survival analysis
│   ├── evaluation/            # Evaluation tools (ICC, Cox, KM)
│   └── visualization/         # Visualization
│
├── data/                      # Data directory
│   └── {dataset}/
│       ├── image/             # Raw CT images
│       ├── mask/              # Raw segmentation masks
│       ├── image_process/     # Preprocessed CT (windowing + resample + RAI)
│       ├── mask_process/      # Preprocessed masks
│       ├── multimask/         # Multi-observer masks (optional, for ICC)
│       ├── gene/              # Gene expression data (optional)
│       ├── clinical.xlsx      # Clinical features (time, event, age, stage...)
│       └── rad_features.xlsx  # Radiomics features (optional)
│
└── results/                   # Output results
    └── {dataset}/
        ├── {model}/           # Per-model score CSVs
        ├── {model}/viz/       # Per-model visualizations
        └── summary/           # Cross-model summary reports
```

---

## Data Requirements

### Directory Structure

Each dataset should follow this structure:

```
data/NSCLC-Radiomics/
├── image/              # Raw CT NIfTI files
├── mask/               # Raw mask NIfTI files
├── clinical.xlsx       # Clinical data (must contain: PatientID, time, event)
└── rad_features.xlsx   # Radiomics features (optional)
```

### Clinical Data Format

| Column | Description |
|--------|-------------|
| `PatientID` | Case ID (matches filenames) |
| `time` | Survival time (days) |
| `event` | Event occurrence (0/1) |
| `age` | Age (optional) |
| `gender` | Gender (optional) |
| `Overall.Stage` | Clinical stage (optional) |
| `Histology` | Pathology type (optional) |

### ICC Test Additional Requirements

For inter-observer ICC testing, multi-observer masks are required:

```
data/immunotherapy/
├── image/
├── mask/
├── multimask/          # Multi-observer masks
│   ├── observer1/
│   ├── observer2/
│   └── observer3/
└── clinical.xlsx
```

---

## Quick Start

### 1. Preprocessing

Steps: HU windowing → Resample to 1x1x1mm → Re-orient to RAI

```bash
# Process specific datasets
python scripts/preprocess_all.py --datasets NSCLC-Radiomics NSCLC-Radiogenomics

# Process all datasets
python scripts/preprocess_all.py --all
```

Preprocessing parameters:
- HU window: `[-1200, 800]`
- Target voxel spacing: `(1.0, 1.0, 1.0)` mm
- CT interpolation: Bspline
- Mask interpolation: NearestNeighbor
- Orientation: RAI `(1 0 0 0 1 0 0 0 1)`

Output: `image_process/` and `mask_process/`.

### 2. Run ITH Scoring Models

```bash
# Run all models
python run_all_scores.py --dataset NSCLC-Radiomics

# Run selected models + visualization
python run_all_scores.py --dataset NSCLC-Radiomics --models bih dhi hab --visualize

# Batch process all datasets
python run_all_scores.py --all-datasets --visualize

# ICC inter-observer test
python run_all_scores.py --dataset immunotherapy --icc-test
```

### 3. Python API

```python
from src.core import DataConfig, DatasetLoader
from src.models import get_model

# Load data
loader = DatasetLoader("NSCLC-Radiomics")
vol, msk = loader.load_case("LUNG1-001")

# Run a single model
model = get_model("bih")()
result = model.compute(vol, msk)
print(f"BIH FD: {result['score']}")

# Batch run
from src.pipeline import ModelRunner
cfg = DataConfig(dataset="NSCLC-Radiomics")
runner = ModelRunner(cfg, models=["bih", "dhi", "hab"])
results = runner.run(visualize=True)

# Access clinical and radiomics data
clinical = loader.clinical              # DataFrame
radiomics = loader.radiomics            # DataFrame
survival = loader.get_survival_data()   # DataFrame: case_id, time, event

# Build merged table
merged = loader.build_merged_table(
    include_clinical=True,
    include_radiomics=True,
)
```

### 4. ICC Inter-Observer Test

```bash
# Command line
python -m src.evaluation.inter_observer_test \
    --dataset immunotherapy \
    --observers observer1 observer2 observer3 \
    --models bih dhi hab pth shi thi \
    --output ./results/ICC

# Python API
from src.evaluation import run_icc_workflow
import pandas as pd

data = pd.read_csv("features_all_observers.csv")
results, summary = run_icc_workflow(
    data=data,
    feature_cols=["bih_fd", "dhi_CV", "hab_H_HAB"],
    output_dir="./results/ICC",
    icc_type="ICC(2,1)",
    icc_threshold=0.75,
)
print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
```

---

## Preprocessing Pipeline

```
Raw Data (image/, mask/)
    │
    ├── Raw CT: spacing (0.98, 0.98, 3.0), HU: [-1024, 3000]
    │
    ↓  Step 1: HU Windowing
    ├── CT: clamp(-1200, 800)
    │
    ↓  Step 2: Resample to 1x1x1 mm
    ├── CT: Bspline interpolation
    ├── Mask: NearestNeighbor interpolation
    │
    ↓  Step 3: Re-orient to RAI
    ├── Direction matrix: (1,0,0, 0,1,0, 0,0,1)
    │
    ↓
Output (image_process/, mask_process/)
    ├── CT: spacing (1,1,1), HU: [-1200, 800], RAI orientation
    └── Mask: spacing (1,1,1), binary, RAI orientation
```

---

## Model Details

### BIH (Box-counting ITH)

- **Principle**: Signal-weighted box-counting fractal analysis on the maximum axial slice
- **Input**: 3D CT + mask
- **Output**: `fd` (fractal dimension), `logA` (intercept), `r2` (goodness of fit)

### DHI (Density Heterogeneity Index)

- **Principle**: 3D local coefficient of variation (CV) + first-order histogram statistics
- **Input**: 3D CT + mask
- **Output**: 16 density features, primary score is `CV`

### HAB (Habitat Analysis Based)

- **Principle**: GMM clustering + spatial priors → tumor habitat subregions
- **Input**: 3D CT + mask
- **Output**: Habitat volume ratios, JS divergence, composite score `H_HAB`

### PTH (Peritumoral Heterogeneity)

- **Principle**: 5mm peritumoral ring + SLIC supervoxels + inter-block texture variation
- **Input**: 3D CT + mask
- **Output**: Supervoxel count, per-feature coefficient of variation

### SHI (Shape Heterogeneity Index)

- **Principle**: Geometric features + convex hull analysis + fractal dimension + lacunarity
- **Input**: mask only (no CT required)
- **Output**: Sphericity, solidity, fractal dimension, lacunarity

### THI (Texture Heterogeneity Index)

- **Principle**: Voxel-level 3x3x3 local radiomic features + KMeans clustering + spatial fragmentation score
- **Input**: 3D CT + mask
- **Output**: `ith_score` [0,1], with cluster label map

### ITH-FS (ITH Fusion Score)

- **Principle**: Multi-scale 3D voxel features + spatial spectral clustering + heterogeneity confidence spatial map
- **Innovations**:
  - 3D whole-tumor analysis (vs ITHscore's 2D slice approach)
  - Spatial spectral clustering ensures spatial continuity (vs KMeans)
  - Outputs heterogeneity confidence spatial map (low confidence = high heterogeneity transition zone)
- **Input**: 3D CT + mask
- **Output**: ITH Fusion Score + confidence map + cluster label map

---

## ICC Reliability Standards

Used to assess whether heterogeneity features are affected by different observer segmentation variations:

| ICC Range | Grade | Recommendation |
|-----------|-------|----------------|
| >= 0.90 | Excellent | Highly reliable, safe to use |
| 0.75-0.90 | Good | Reliable, acceptable for use |
| 0.50-0.75 | Moderate | Moderate consistency, use with caution |
| < 0.50 | Poor | Strongly affected by observer differences, not recommended |

**Recommendation**: Features with ICC >= 0.75 are considered "reliable".

---

## Datasets

| Dataset | Cases | Gene Data | Multi-Observer |
|---------|-------|-----------|----------------|
| NSCLC-Radiomics | 422 | No | No |
| NSCLC-Radiogenomics | 211 | Yes | No |
| LungCT-Diagnosis | 61 | No | No |
| CPTAC-TCGA | 103 | Yes | No |
| TongJi | 189 | No | No |
| immunotherapy | 237 | No | Yes |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{li2022ithscore,
  title={ITHscore: comprehensive quantification of intra-tumor heterogeneity in NSCLC by multi-scale radiomic features},
  author={Li, Y. and others},
  journal={European Radiology},
  year={2022}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Multi-center tumor heterogeneity research project
- Based on the ITHscore methodology (Li et al., European Radiology, 2022)
