# VSCI-CMD: Volume–Surface Cascaded Integration Cylindrical Modal Decomposition



---

## Overview

VSCI-CMD is a hybrid physics–machine learning framework for non-destructive
geometric metrology of high-aspect-ratio Through-Silicon Vias (TSVs) from
near-field optical measurements. The framework factorizes the electromagnetic
inverse problem into:

1. A macroscopic 1D cylindrical modal phase integral (physics stage)
2. A localized boundary surface correction via Kernel Ridge Regression (ML stage)

This achieves a 5-fold cross-validated Depth MAPE of **0.224%** (6.50 nm MAE),
a **5.02×** improvement over the SVD-ResNet baseline (Chen et al., CPC 2025).

---

## Repository Structure

```
VSCI_CMD_package/
├── README.md                        ← This file
├── LICENSE                          ← GPL-3.0 full text
├── requirements.txt                 ← Python dependencies
├── PROGRAM_SUMMARY.txt              ← CPC Program Summary (upload separately to EM)
│
├── src/
│   ├── vsci_cmd_main.py             ← MAIN: full VSCI-CMD pipeline (Tables 3–5)
│   ├── paper_constants.py           ← Single source of truth for all paper numbers
│   └── generate_multimode_lut.py    ← Step 0: build FDM physics LUT (~2 min)
│
├── evaluation/
│   ├── verify_paper_numbers.py      ← Verify all paper numbers for consistency
│   ├── run_vsdcmd_ablation.py       ← Reproduce Table 5 ablation study
│   ├── run_foldlevel_bootstrap.py   ← Bootstrap CI + permutation test
│   └── run_pls_component_sweep.py   ← PLS component sweep (hyperparameter study)
│
└── data/
    └── README_data.txt              ← Data format, variable names, units
                                       (actual .npz files available on request)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy ≥ 1.24, scipy ≥ 1.10, scikit-learn ≥ 1.3, matplotlib ≥ 3.7  
No compilation required. Python 3.9+ recommended.

---

## Data Requirements

The pipeline requires three pre-computed data files placed in a `data/` directory:

| File | Description |
|------|-------------|
| `fdtd_cache_highres_training_1629s.npz` | Training split (1629 FDTD samples, ~300 MB) |
| `fdtd_cache_highres_validation_139s.npz` | Validation split (139 samples, ~26 MB) |
| `tsv_physics_lut_multimode.npz` | FDM eigenmode LUT (~14 MB, or auto-generate) |

The FDTD simulation data is available upon request to the corresponding author:
**jiahan@ntu.edu.tw**

See `data/README_data.txt` for the full data format specification.

---

## Step-by-Step Reproduction

### Step 0 — Build the physics LUT (skip if pre-built LUT is available)
```bash
python src/generate_multimode_lut.py
```
Generates `tsv_physics_lut_multimode.npz` via 1D FDM eigensolving.  
Runtime: ~2 min on CPU (Intel i7).

### Step 1 — Run the main VSCI-CMD pipeline
```bash
python src/vsci_cmd_main.py
```
Reproduces:
- **Table 3** — Fixed-split benchmark vs. SVD-ResNet (Sum MAPE = 1.072%)
- **Table 4** — 5-fold cross-validation (Depth MAPE = 0.224% ± 0.012%)
- **Table 5** — Ablation study (via inline ablation mode)

Runtime: ~5 min (fixed split) or ~15 min (with 5-fold CV) on CPU (Intel i7).

### Step 2 — Ablation study (Table 5)
```bash
python evaluation/run_vsdcmd_ablation.py
```

### Step 3 — Statistical validation (bootstrap CI + permutation test)
```bash
python evaluation/run_foldlevel_bootstrap.py
```
Expected output: 95% CI [0.88%, 1.11%], permutation p < 0.0001.

### Step 4 — Verify all paper numbers are internally consistent
```bash
python evaluation/verify_paper_numbers.py
```
Expected output: `[PASS] All internal consistency checks passed.`

---

## Key Results

| Method | TCD MAPE | BCD MAPE | Depth MAPE | Sum MAPE |
|--------|----------|----------|------------|----------|
| Raw-w1 (baseline) | 0.110% | 3.130% | 3.650% | 6.890% |
| SVD-ResNet (Chen et al. 2025) | 0.260% | 1.380% | 0.790% | 2.430% |
| **VSCI-CMD (this work)** | **0.041%** | **0.806%** | **0.225%** | **1.072%** |

*Fixed validation split (1629 train / 139 test). See Table 3 of the paper.*

---

## Citation

If you use this code, please cite:

```bibtex
@article{vsci_cmd_2026,
  title   = {VSCI-CMD: Volume--Surface Cascaded Integration Cylindrical Modal
             Decomposition for Near-Field TSV Metrology via Physics-Guided
             Kernel Ridge Regression},
  author  = {Vo, Nguyen and Chen, Song-En and Li, Jia-Han},
  journal = {Computer Physics Communications},
  year    = {2026},
  doi     = {[TBD upon acceptance]}
}
```

---

## Contact

For questions about the code or data, contact the corresponding author:  
**Jia-Han Li** — jiahan@ntu.edu.tw  
Department of Mechanical Engineering, National Taiwan University, Taipei, Taiwan
