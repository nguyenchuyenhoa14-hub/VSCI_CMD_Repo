# Volume--Surface Cascaded Integration (VSCI)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21925226.svg)](https://doi.org/10.5281/zenodo.21925226)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](https://github.com/nguyenchuyenhoa14-hub/VSCI_CMD_Repo)

Official open-source implementation of the **Volume--Surface Cascaded Integration (VSCI)** framework, a hybrid physics-guided machine learning framework for sub-nanometer optical scatterometry of high-aspect-ratio Through-Silicon Vias (TSVs).

> **Paper Title:**  
> *Physics-Guided Machine Learning for Near-Field Metrology of High-Aspect-Ratio Through-Silicon Vias Using Cylindrical Modal Decomposition*  
> **Authors:** Nguyen Vo (1), Song-En Chen (1), Chao-Ching Ho (2), Jia-Han Li (1)*  
> *(1) National Taiwan University, Taipei, Taiwan*  
> *(2) National Taipei University of Technology, Taipei, Taiwan*  
> ***Submitted to:*** *Computer Physics Communications (Elsevier, 2026)*

---

## Key Mathematical & Physical Architecture

Near-field optical metrology of high-aspect-ratio (HAR) TSVs is a severely ill-posed electromagnetic inverse scattering problem. Purely data-driven black-box neural networks (e.g., SVD-ResNet) frequently suffer catastrophic failure under slight geometric variations due to the absence of wave-propagation constraints.

VSCI overcomes this by factorizing the inverse scattering problem into three cascaded physical operators:

`
[Near-Field CMD Spectrum] 
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 1: Top Critical Dimension (TCD)                           │
  │ • Radial modal power + Cross-Wavelength Phase Coherence (CWPC) │
  │ • C_∞v parity projector: deterministically annihilates Yee noise│
  │ • RidgeCV estimator → Sub-grid TCD reconstruction              │
  └────────────────────────────────┬────────────────────────────────┘
                                   │  TCD estimate
                                   ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 2: Bottom Critical Dimension (BCD)                        │
  │ • Cross-Modal Ratio (CMR) log-power features                    │
  │ • 150-component PLS latent space + Polynomial Kernel Ridge      │
  │ • Decouples entry-aperture loss from deep sidewall taper        │
  └────────────────────────────────┬────────────────────────────────┘
                                   │  TCD & BCD estimates
                                   ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 3: Depth Inversion (Deep Physical Integration - DPI)      │
  │ • Inverts Fabry-Perot phase condition: ΔΦ = 2∫β(z)dz + 2Γ = 2πp │
  │ • Legendre-Gauss numerical quadrature with 1D FDM PML eigensystem│
  │ • Zero empirical free parameters: out-of-fold Depth MAE = 5.89nm │
  └─────────────────────────────────────────────────────────────────┘
`

---

## Physical Domain of Validity

The numerical eigensolvers and calibrated models operate within validated semiconductor packaging ranges:

| Parameter | Symbol | Range / Value | Description |
| :--- | :---: | :---: | :--- |
| **Top Critical Dimension** | $\mathrm{TCD}$ |  - 330\text{ nm}$ | Entry aperture diameter |
| **Bottom Critical Dimension** | $\mathrm{BCD}$ |  - 300\text{ nm}$ | Via floor diameter |
| **Via Depth (Height)** | $ |  - 3300\text{ nm}$ | Nominal HAR aspect ratio $\sim 10:1$ |
| **Wavelength Band** | $\lambda$ |  - 432\text{ nm}$ | 55 discrete spectral bands |
| **Core Material** | - | Air ( = 1.00$) | Air-filled cylindrical cavity |
| **Substrate Material** | - | Silicon ($\mathrm{Si}$) | Dispersive {\mathrm{Si}}(\lambda)$, {\mathrm{Si}}(\lambda)$ |
| **FDTD Grid Pitch** | $\Delta$ | .64\text{ nm}$ | Uniform cubic Yee discretization |

---

## Repository Structure

`
VSCI_CMD_Repo/
├── LICENSE                        # GNU General Public License v3.0
├── README.md                      # Documentation & reproducibility guide
└── src/
    ├── run_pipeline.py            # Main VSCI pipeline (Fixed-split & 5-fold CV evaluation)
    ├── generate_multimode_lut.py  # 1D FDM PML cylindrical modal eigensolver (LUT builder)
    └── generate_figures.py        # Reproduction script for manuscript Figures 1-7
`

---

## System Requirements & Installation

- **Operating System:** Linux (Ubuntu 20.04+), Windows 10/11, or macOS 12+
- **Hardware:** Standard 64-bit x86 or ARM CPU (Intel Core i5/i7/i9, AMD Ryzen, Apple M-series). GPU is **not** required.
- **RAM:** 8 GB minimum; 16 GB recommended for full 5-fold cross-validation.
- **Python Version:** Python 3.9+

### Installation
`ash
pip install numpy scipy scikit-learn matplotlib
`

---

## Step-by-Step Reproduction Guide

### 1. Generate the Cylindrical Waveguide Physics LUT (~2 min)
`ash
python src/generate_multimode_lut.py
`
Solves the 1D radial finite-difference with perfectly matched layers (FDM-PML) eigensystem:
\left[ \frac{d^2}{dr^2} + \frac{1}{r}\frac{d}{dr} + \left( k_0^2 n^2(r) - \frac{m^2}{r^2} \right) \right] R_m(r) = \beta_m^2 R_m(r)
for azimuthal orders  \in \{0, 1, 2, 3, 4\}$ across all grid diameters and wavelengths.

### 2. Execute the VSCI Reconstruction Pipeline (~15 min on 8-core CPU)
`ash
python src/run_pipeline.py
`
Runs fixed-split evaluation (1629 train / 139 test) and complete 5-fold cross-validation (1768 samples). Expected output:
`
======================================================================
  VSCI RECONSTRUCTION EVALUATION SUMMARY (Table 3 & Table 4)
======================================================================
  Target       MAPE (%)      MAE (nm)      Std (nm)      Baseline Gain
  --------------------------------------------------------------------
  TCD           0.042%        0.126 nm      0.161 nm     6.19x vs SVD-ResNet
  BCD           0.886%        2.215 nm      2.834 nm     1.56x vs SVD-ResNet
  Depth         0.195%        5.890 nm      7.521 nm     5.55x vs SVD-ResNet
  --------------------------------------------------------------------
  Sum MAPE:     1.124%        Total Runtime: ~15 min (Standard CPU)
======================================================================
`

### 3. Generate Publication Figures
`ash
python src/generate_figures.py
`
Outputs high-resolution vector PDFs and 600-DPI images for Figures 1 through 7 corresponding to the manuscript.

---

## Metrology Benchmark Comparison

Evaluated on the benchmark dataset against previous state-of-the-art methods:

| Metrology Method | TCD MAPE (%) | BCD MAPE (%) | Depth MAPE (%) | Sum MAPE (%) | Depth MAE (nm) | Calibration Free? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw-w1** (Chen et al. 2025) | 0.110% | 3.130% | 3.650% | 6.890% | 108.5 nm | No |
| **Raw-w3** (Chen et al. 2025) | 0.120% | 2.940% | 3.210% | 6.270% | 95.4 nm | No |
| **Raw-w5** (Chen et al. 2025) | 0.100% | 2.810% | 3.050% | 5.960% | 90.8 nm | No |
| **SVD-ResNet** (Chen et al. 2025) | 0.260% | 1.380% | 0.790% | 2.430% | 32.7 nm | No |
| **VSCI (This Work)** | **0.042%** | **0.886%** | **0.195%** | **1.124%** | **5.89 nm** | **Yes (DPI)** |

---

## Data Availability & Archiving

The complete synthetic FDTD near-field dataset (1,768 simulated TSV configurations across 55 spectral bands) is permanently archived on **Zenodo**:
- **DOI:** [10.5281/zenodo.21925226](https://doi.org/10.5281/zenodo.21925226)
- **Concept DOI:** [10.5281/zenodo.21925225](https://doi.org/10.5281/zenodo.21925225)

For access inquiries or raw 3D electromagnetic snapshots, contact:
- Prof. Jia-Han Li (jiahan@ntu.edu.tw), National Taiwan University
- Prof. Chao-Ching Ho (hochao@ntut.edu.tw), National Taipei University of Technology

---

## Citation

If you use this codebase or find our framework useful in your research, please cite:

`ibtex
@article{vo2026vsci,
  title   = {Physics-Guided Machine Learning for Near-Field Metrology of High-Aspect-Ratio Through-Silicon Vias Using Cylindrical Modal Decomposition},
  author  = {Vo, Nguyen and Chen, Song-En and Ho, Chao-Ching and Li, Jia-Han},
  journal = {Computer Physics Communications},
  year    = {2026},
  doi     = {10.5281/zenodo.21925226},
  note    = {Submitted}
}
`

---

## License

This software is released under the **GNU General Public License v3.0 (GPLv3)**. See [LICENSE](LICENSE) for details.
