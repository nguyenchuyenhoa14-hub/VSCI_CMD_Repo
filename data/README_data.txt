# Dataset Description — VSCI-CMD Near-Field FDTD Simulation Data

## Overview

This directory contains pre-computed near-field electromagnetic simulation data
for Through-Silicon Via (TSV) geometry metrology, generated using Lumerical FDTD.

## Files

### fdtd_cache_highres_training_1629s.npz
- **Type:** NumPy compressed archive (.npz)
- **Samples:** 1629 training samples
- **Arrays:**
  - `cmd`: Complex near-field CMD coefficients, shape (1629, 100, 16, 10)
             Dimensions: (samples, wavelengths, radial_rings, [Re/Im for modes 0-4])
  - `labels`: TSV geometry parameters, shape (1629, 3)
               Columns: [TCD (nm), BCD (nm), Depth (nm)]

### fdtd_cache_highres_validation_139s.npz
- **Type:** NumPy compressed archive (.npz)
- **Samples:** 139 validation samples (fixed held-out test set)
- **Arrays:** Same structure as training file.

### tsv_physics_lut_multimode.npz
- **Type:** NumPy compressed archive (.npz)
- **Description:** Lookup table (LUT) of FDM eigenmode solutions
- **Arrays:**
  - `tcd_grid`: TCD values for LUT, shape (13,), range [270, 330] nm
  - `wavelengths`: Wavelength grid, shape (100,), range ~230–500 nm
  - `r`: Radial coordinate grid (nm)
  - `beta_m0` ... `beta_m4`: Complex propagation constants, shape (13, 100)
  - `profile_m0` ... `profile_m4`: Modal field profiles, shape (13, 100, N_r)

## Parameter Ranges
| Parameter | Min  | Max  | Unit |
|-----------|------|------|------|
| TCD       | 270  | 330  | nm   |
| BCD       | 200  | 300  | nm   |
| Depth     | 2700 | 3300 | nm   |
| Wavelength| ~230 | ~500 | nm   |

## Data Availability
The FDTD simulation data (.npz files) are available upon request
to the corresponding author: jiahan@ntu.edu.tw

Raw FDTD generation scripts (Lumerical) are proprietary to the
originating lab and provided under separate data-sharing agreements.

## Loading Example
```python
import numpy as np
tr = np.load("fdtd_cache_highres_training_1629s.npz")
cmd_train   = tr['cmd']    # shape: (1629, 100, 16, 10)
label_train = tr['labels'] # shape: (1629, 3) → [TCD, BCD, Depth]
```
