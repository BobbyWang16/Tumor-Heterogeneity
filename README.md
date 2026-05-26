# Tumor Heterogeneity Analysis Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains three interconnected modules for comprehensive tumor heterogeneity analysis based on CT imaging:

| Module | Purpose | Location |
|--------|---------|----------|
| **ITH** | Multi-dimensional intratumor heterogeneity (ITH) scoring system | [`src/`](src/) |
| **DualCT** | Dual-energy CT-based ITH analysis with clustering | [`DualCT/`](DualCT/) |
| **QuanTAV** | Quantitative Tumor Analysis & Visualization toolkit | [`QuanTAV/`](QuanTAV/) |

---

## 1. ITH Scoring System (`src/`)

A comprehensive multi-dimensional intratumor heterogeneity scoring framework based on 3D CT imaging and tumor segmentation masks. This system quantifies tumor internal heterogeneity from seven perspectives for prognostic prediction.

### 7 Core Scoring Models

| Model | Full Name | Dimension | Core Principle |
|-------|-----------|-----------|----------------|
| **BIH** | Box-counting ITH | 2D | Signal-weighted box-counting fractal analysis |
| **DHI** | Density Heterogeneity Index | 3D | First-order statistics + local coefficient of variation |
| **HAB** | Habitat Analysis Based | 3D | GMM clustering with spatial priors |
| **PTH** | Peritumoral Heterogeneity | 3D | Peritumoral ring + SLIC supervoxels + inter-block variation |
| **SHI** | Shape Heterogeneity Index | 3D | Geometry + convex hull + fractal dimension + lacunarity |
| **THI** | Texture Heterogeneity Index | 3D | Voxel-level radiomic features + KMeans clustering |
| **ITH-FS** | ITH Fusion Score | 3D | Multi-scale features + spatial spectral clustering |

### Quick Start

```bash
# Preprocess CT images
python scripts/preprocess_all.py --datasets NSCLC-Radiomics

# Run all models
python run_all_scores.py --dataset NSCLC-Radiomics --visualize
```

### Project Structure

```
src/
├── core/           # Base model, config, I/O, preprocessing
├── models/         # 7 scoring model implementations
├── pipeline/       # Batch execution runner
├── features/       # Feature extraction utilities
├── survival/       # Survival analysis tools
├── evaluation/     # ICC, Cox, Kaplan-Meier evaluation
└── visualization/  # Result visualization
```

---

## 2. DualCT Analysis (`DualCT/`)

Dual-energy CT-based intratumor heterogeneity analysis module. Performs clustering-based analysis on dual-energy CT images to identify distinct tumor subregions and compute heterogeneity metrics.

### Key Files

| File | Description |
|------|-------------|
| `ith_analysis.py` | Main analysis script for dual-energy CT clustering |
| `ith_analysis.md` | Methodology documentation |
| `cluster/` | Clustering results and visualizations |

### Features

- Dual-energy CT data processing
- Tumor subregion clustering
- Heterogeneity score computation from clustered regions

---

## 3. QuanTAV (`QuanTAV/`)

Quantitative Tumor Analysis & Visualization toolkit. A dedicated module for quantitative analysis of tumor CT features with comprehensive visualization capabilities.

### Key Files

| File | Description |
|------|-------------|
| `quantav_py/api.py` | Main API interface |
| `quantav_py/features.py` | Feature extraction algorithms |
| `quantav_py/visualization.py` | Visualization tools |
| `quantav_py/roi.py` | Region-of-interest processing |
| `quantav_py/skeleton.py` | Vessel skeleton analysis |
| `example_api.py` | Usage examples |
| `算法.md` | Algorithm documentation (Chinese) |

### Installation

```bash
cd QuanTAV
pip install -e .
```

### Usage

```python
from quantav_py.api import analyze_tumor

# Analyze tumor CT
results = analyze_tumor(ct_image, mask)
```

---

## Global Installation

```bash
# Clone repository
git clone https://github.com/BobbyWang16/Tumor-Heterogeneity.git
cd Tumor-Heterogeneity

# Install ITH dependencies
pip install numpy pandas scipy scikit-learn scikit-image SimpleITK nibabel matplotlib seaborn tqdm

# Optional: survival analysis
pip install lifelines scikit-survival
```

### Requirements

- Python >= 3.9
- NumPy, Pandas, SciPy
- scikit-learn, scikit-image
- SimpleITK, nibabel
- matplotlib, seaborn

---

## Data Requirements

Each dataset should follow this structure:

```
data/{dataset}/
├── image/              # Raw CT NIfTI files
├── mask/               # Raw mask NIfTI files
├── image_process/      # Preprocessed CT
├── mask_process/       # Preprocessed masks
├── clinical.xlsx       # Clinical data (PatientID, time, event)
└── rad_features.xlsx   # Radiomics features (optional)
```

---

## Module Comparison

| Aspect | ITH (`src/`) | DualCT (`DualCT/`) | QuanTAV (`QuanTAV/`) |
|--------|-------------|-------------------|---------------------|
| **Input** | Single-energy CT | Dual-energy CT | Single-energy CT |
| **Models** | 7 scoring models | Clustering-based | Feature-based API |
| **Output** | Heterogeneity scores | Cluster labels + scores | Feature vectors + viz |
| **Focus** | Multi-model comparison | Energy-specific analysis | Quantitative features |
| **Batch** | Yes | Case-by-case | Yes |
| **Viz** | Per-model + summary | Clustering plots | Comprehensive charts |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{li2022ithscore,
  title={ITHscore: comprehensive quantification of intra-tumor heterogeneity in NSCLC},
  author={Li, Y. and others},
  journal={European Radiology},
  year={2022}
}
```

## License

MIT License
