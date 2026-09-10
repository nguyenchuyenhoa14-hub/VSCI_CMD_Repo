# Volume--Surface Cascaded Integration (VSCI) Framework
## Official Code Repository for Computer Physics Communications

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21925226.svg)](https://doi.org/10.5281/zenodo.21925226)

This repository contains the official, minimal Python implementation of the **Volume--Surface Cascaded Integration (VSCI)** framework described in:

> **"Physics-Guided Machine Learning for Near-Field Metrology of High-Aspect-Ratio Through-Silicon Vias Using Cylindrical Modal Decomposition"**  
> Nguyen Vo, Song-En Chen, Chao-Ching Ho, Jia-Han Li  
> *Submitted to Computer Physics Communications (Elsevier, 2026).*

---

## Overview

High-aspect-ratio Through-Silicon Via (TSV) interconnects in 3D-IC packaging require non-destructive, sub-nanometer geometric metrology:
- **Top Critical Dimension (TCD)**
- **Bottom Critical Dimension (BCD)**
- **Depth (Height $h$)**

from optical near-field scattering signatures. 

The **VSCI** framework deterministically factorizes the inverse scattering problem into:
1. **Stage 1 (TCD)**: Multi-spectral modal power and Cross-Wavelength Phase Coherence (CWPC) features with $C_{\infty v}$ spatial parity projection that algebraically eliminates Cartesian Yee-grid discretization noise.
2. **Stage 2 (BCD)**: Partial Least Squares (PLS, 150 components) latent subspace with Cross-Modal Ratios (CMR) and Polynomial Kernel Ridge Regression (KRR).
3. **Stage 3 (Depth)**: Deep Physical Integration (DPI) phase inversion, solving the 1D cylindrical Fabry-Pérot modal phase integral via Legendre-Gauss quadrature with zero empirical calibration parameters.

---

## Directory Structure

```
VSCI_CMD_Repo/
├── LICENSE                        # GNU General Public License v3.0
├── README.md                      # Documentation & reproduction guide
└── src/
    ├── run_pipeline.py            # Main VSCI pipeline (Fixed-split & 5-fold CV evaluation)
    ├── generate_multimode_lut.py  # 1D FDM PML cylindrical modal eigensolver (LUT builder)
    └── generate_figures.py        # Reproduction script for manuscript Figures 1-7
```

---

## Requirements & Installation

The framework requires a standard scientific Python environment (Python 3.9+).

```bash
pip install numpy scipy scikit-learn matplotlib
```

No custom C/C++ compilation or specialized GPU hardware is required.

---

## Step-by-Step Reproduction Guide

### Step 1 — Solve the Cylindrical Waveguide Physics Eigensystem (LUT)
```bash
python src/generate_multimode_lut.py
```
*Computes the finite-difference PML eigensystem for azimuthal modes $m \in \{0, 1, 2, 3, 4\}$ across all target diameters and wavelengths (~2 min on standard CPU).*

### Step 2 — Run the Core VSCI Metrology Pipeline
```bash
python src/run_pipeline.py
```
*Executes the full 4-stage VSCI reconstruction, performs fixed-split evaluation (1629 train / 139 test) and 5-fold cross-validation, and outputs metrics matching Table 3 and Table 4 of the manuscript.*

### Step 3 — Reproduce Publication Figures
```bash
python src/generate_figures.py
```
*Generates publication-quality figures (Figures 1-7) in high-resolution vector/bitmap format.*

---

## Benchmark Summary

| Metric / Method | Raw-w1 (Chen 2025) | SVD-ResNet (Chen 2025) | **VSCI (Proposed)** |
|:---|:---:|:---:|:---:|
| **TCD MAPE (%)** | 0.110% | 0.260% | **0.042%** |
| **BCD MAPE (%)** | 3.130% | 1.380% | **0.886%** |
| **Depth MAPE (%)** | 3.650% | 0.790% | **0.195%** |
| **Sum MAPE (%)** | 6.890% | 2.430% | **1.124%** |
| **Depth MAE** | 108.5 nm | 32.7 nm | **5.89 nm** |

---

## Dataset Specification & Availability

The full synthetic near-field dataset comprises 1,768 3D FDTD simulations across 55 spectral bands (wavelengths 240 nm to 432 nm). Each sample consists of:
- Transverse modal coefficients ($m = 0, 1, 2, 3, 4$) extracted along 16 radial observation rings ($R = 90\text{ nm}$ to $330\text{ nm}$).
- Ground-truth geometry labels: `[TCD, BCD, Depth]` in nanometers.

The permanent digital archive for datasets is hosted on **Zenodo**:
- **DOI**: [10.5281/zenodo.21925226](https://doi.org/10.5281/zenodo.21925226)

For inquiries regarding raw FDTD electromagnetic fields or data access, contact:
- `jiahan@ntu.edu.tw` (Prof. Jia-Han Li, National Taiwan University)
- `hochao@ntut.edu.tw` (Prof. Chao-Ching Ho, National Taipei University of Technology)

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
