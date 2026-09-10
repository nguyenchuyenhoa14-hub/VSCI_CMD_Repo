# Volume--Surface Cascaded Integration (VSCI) Framework
## Official Code Repository for Computer Physics Communications

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21925226.svg)](https://doi.org/10.5281/zenodo.21925226)

This repository contains the official implementation of the **Volume--Surface Cascaded Integration (VSCI)** framework described in:

> **"Physics-Guided Machine Learning for Near-Field Metrology of High-Aspect-Ratio Through-Silicon Vias Using Cylindrical Modal Decomposition"**  
> Nguyen Vo, Song-En Chen, Chao-Ching Ho, Jia-Han Li  
> *Submitted to Computer Physics Communications (Elsevier, 2026).*

---

## Overview

High-aspect-ratio Through-Silicon Via (TSV) interconnects in 3D-IC packaging require non-destructive, sub-nanometer geometric metrology (Top Critical Dimension [TCD], Bottom Critical Dimension [BCD], and Depth) from optical near-field scattering signatures. 

Conventional end-to-end deep learning baselines (such as SVD-ResNet) treat the inverse scattering process as an unconstrained black-box regression, suffering severe accuracy degradation under out-of-distribution (OOD) shifts. 

The **VSCI** framework deterministically factorizes the inverse scattering problem:
1. **Stage 1 (TCD)**: Multi-spectral modal power and Cross-Wavelength Phase Coherence (CWPC) features with $C_{\infty v}$ spatial parity projection.
2. **Stage 2 (BCD)**: Partial Least Squares (PLS, 150 components) latent subspace with Cross-Modal Ratios (CMR) and Polynomial Kernel Ridge Regression (KRR).
3. **Stage 3 (Depth)**: Deep Physical Integration (DPI) phase inversion solving the 1D cylindrical Fabry-Pérot modal phase integral via Legendre-Gauss quadrature with zero empirical calibration parameters.

---

## Directory Structure

```
VSCI_CMD_Repo/
├── LICENSE                        # GNU General Public License v3.0
├── PROGRAM_SUMMARY.txt            # Elsevier CPC Program Summary
├── requirements.txt               # Python package dependencies
├── environment.yml                # Conda environment specification
├── README.md                      # This documentation
├── src/                           # Core implementation pipeline
│   ├── run_pipeline.py            # Main 4-stage VSCI pipeline (Fixed split + 5-fold CV)
│   ├── generate_multimode_lut.py  # 1D FDM PML cylindrical modal eigensolver (LUT builder)
│   ├── generate_figures.py        # Reproduction script for manuscript Figures 1-7
│   ├── paper_constants.py         # Ground-truth constants and benchmark registries
│   ├── run_baselines.py           # SVD-ResNet, Raw-w1/w3/w5 benchmark baselines
│   ├── measure_exact_latency.py   # Latency and throughput benchmarking
│   ├── run_learning_curve.py      # Sample efficiency & learning curve analysis
│   ├── generate_real_learning_curve.py
│   └── verify_1d_cnn_5fold.py     # 1D-CNN baseline verification
├── evaluation/                    # Verification and statistical evaluation suites
│   ├── verify_paper_numbers.py    # Automated paper number & table consistency checker
│   ├── run_vsdcmd_ablation.py     # Component ablation study (Table 5)
│   ├── run_groupkfold_vsdcmd.py   # TCD-stratified OOD covariate shift evaluation
│   ├── run_foldlevel_bootstrap.py # Bootstrap 95% CI (B=10000) & permutation test
│   └── run_pls_component_sweep.py # PLS component sensitivity sweep (Table 6)
└── data/
    └── DATASET_DESCRIPTION.md     # Detailed specification of data variables and units
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- Standard scientific stack (NumPy, SciPy, scikit-learn, Matplotlib)

### Setup via pip
```bash
pip install -r requirements.txt
```

### Setup via conda
```bash
conda env create -f environment.yml
conda activate vsci
```

---

## Quick-Start Reproduction Guide

### 1. Build the Multimode Physics LUT
```bash
python src/generate_multimode_lut.py
```
*Computes the finite-difference PML eigensystem for azimuthal modes $m \in \{0, 1, 2, 3, 4\}$ across all target diameters and wavelengths (~2 min).*

### 2. Run the Main VSCI Pipeline
```bash
python src/run_pipeline.py
```
*Executes fixed-split evaluation, 5-fold cross-validation, and outputs out-of-fold metrics matching Table 3 and Table 4 of the manuscript.*

### 3. Run Component Ablation & Statistical Tests
```bash
python evaluation/run_vsdcmd_ablation.py
python evaluation/run_foldlevel_bootstrap.py
```

### 4. Reproduce Publication Figures
```bash
python src/generate_figures.py
```
*Generates publication-ready vector and high-resolution figures into `docs/figs/`.*

---

## Metrology Benchmark Summary

| Metric / Method | Raw-w1 (Chen 2025) | SVD-ResNet (Chen 2025) | **VSCI (Proposed)** |
|:---|:---:|:---:|:---:|
| **TCD MAPE (%)** | 0.110% | 0.260% | **0.042%** |
| **BCD MAPE (%)** | 3.130% | 1.380% | **0.886%** |
| **Depth MAPE (%)** | 3.650% | 0.790% | **0.195%** |
| **Sum MAPE (%)** | 6.890% | 2.430% | **1.124%** |
| **Depth MAE** | 108.5 nm | 32.7 nm | **5.89 nm** |

---

## Dataset Availability

The synthetic 3D FDTD near-field dataset (1,768 simulated TSV configurations across 55 spectral bands) is permanently archived on **Zenodo**:
- **DOI**: [10.5281/zenodo.21925226](https://doi.org/10.5281/zenodo.21925226)

For further inquiries or access to raw multi-gigabyte field snapshots, contact:
- `jiahan@ntu.edu.tw` (Prof. Jia-Han Li, National Taiwan University)
- `hochao@ntut.edu.tw` (Prof. Chao-Ching Ho, National Taipei University of Technology)

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
