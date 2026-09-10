# Volume--Surface Cascaded Integration (VSCI)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21925226.svg)](https://doi.org/10.5281/zenodo.21925226)

Official Python implementation of the **Volume--Surface Cascaded Integration (VSCI)** framework for high-aspect-ratio Through-Silicon Via (TSV) near-field metrology.

Accompanying manuscript submitted to *Computer Physics Communications* (Elsevier, 2026).

---

## Structure

```
VSCI_CMD_Repo/
├── LICENSE
├── README.md
└── src/
    ├── run_pipeline.py            # Main VSCI pipeline (Fixed-split & 5-fold CV)
    ├── generate_multimode_lut.py  # 1D FDM PML cylindrical modal eigensolver
    └── generate_figures.py        # Figure reproduction script (Figs. 1-7)
```

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## Quick Start

### 1. Build Physics LUT (~2 min)
```bash
python src/generate_multimode_lut.py
```

### 2. Run Pipeline (~15 min)
```bash
python src/run_pipeline.py
```

### 3. Generate Figures
```bash
python src/generate_figures.py
```

---

## Results Summary

| Metric | Raw-w1 (Chen 2025) | SVD-ResNet (Chen 2025) | **VSCI (Proposed)** |
| :--- | :---: | :---: | :---: |
| **TCD MAPE** | 0.110% | 0.260% | **0.042%** |
| **BCD MAPE** | 3.130% | 1.380% | **0.886%** |
| **Depth MAPE** | 3.650% | 0.790% | **0.195%** |
| **Sum MAPE** | 6.890% | 2.430% | **1.124%** |
| **Depth MAE** | 108.5 nm | 32.7 nm | **5.89 nm** |

---

## Citation

```bibtex
@article{vo2026vsci,
  title   = {Physics-Guided Machine Learning for Near-Field Metrology of High-Aspect-Ratio Through-Silicon Vias Using Cylindrical Modal Decomposition},
  author  = {Vo, Nguyen and Chen, Song-En and Ho, Chao-Ching and Li, Jia-Han},
  journal = {Computer Physics Communications},
  year    = {2026},
  doi     = {10.5281/zenodo.21925226}
}
```

---

## License

GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE) for details.
