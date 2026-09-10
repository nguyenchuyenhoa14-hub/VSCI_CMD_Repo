"""
run_groupkfold_vsdcmd.py
========================
Runs VSD-CMD 5-fold GroupKFold (stratified by TCD value) to measure
TCD-stratified covariate shift sensitivity, comparing against the
previously-run SVD-ResNet GroupKFold results from the server.

SVD-ResNet GroupKFold (from server log, task-3522):
  TCD   = 0.042 +/- 0.007%
  BCD   = 1.414 +/- 0.236%
  Depth = 1.649 +/- 0.060%   <-- key comparison number

SVD-ResNet random KFold (Table 4):
  Depth = 1.140 +/- 0.065%   --> degradation when stratifying = +44.6%

VSD-CMD random KFold (Table 4):
  Depth = 0.175 +/- 0.055%

This script measures VSD-CMD GroupKFold Depth to complete the comparison.
"""
import numpy as np
import time
import os
import sys

from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_TR = os.path.join(DIR, "data", "fdtd_cache_highres_training_1629s.npz")
CACHE_VL = os.path.join(DIR, "data", "fdtd_cache_highres_validation_139s.npz")
LUT_PATH  = os.path.join(DIR, "data", "tsv_physics_lut_multimode.npz")

mape = lambda a, b: float(np.mean(np.abs((a - b) / (a + 1e-8))) * 100)

# ── Import the full pipeline from run_final_dual_pipeline.py ─────────────────
sys.path.insert(0, os.path.join(DIR, "src"))
from vsd_cmd_pipeline import run_unified_pipeline


def main():
    print("=" * 70)
    print("  VSD-CMD GroupKFold (TCD-stratified) Covariate Shift Test")
    print("=" * 70)

    tr  = np.load(CACHE_TR)
    vl  = np.load(CACHE_VL)
    lut = np.load(LUT_PATH)

    cmd_all = np.vstack([tr["cmd"], vl["cmd"]])
    y_all   = np.vstack([tr["labels"], vl["labels"]])

    # GroupKFold stratified by TCD value (continuous → used as group)
    # sklearn GroupKFold requires integer groups, so bin TCD into 5 quantile bins
    tcd_vals = y_all[:, 0]
    tcd_bins = np.digitize(tcd_vals, np.percentile(tcd_vals, [20, 40, 60, 80]))
    print(f"  TCD range: {tcd_vals.min():.1f} – {tcd_vals.max():.1f} nm")
    print(f"  Group bin counts: {np.bincount(tcd_bins)}")

    gkf = GroupKFold(n_splits=5)

    tcds, bcds, depths = [], [], []

    print()
    for fold_idx, (idx_tr, idx_vl) in enumerate(
        gkf.split(cmd_all, y_all[:, 1], groups=tcd_bins)
    ):
        t0 = time.time()
        cmd_tr_cv, y_tr_cv = cmd_all[idx_tr], y_all[idx_tr]
        cmd_vl_cv, y_vl_cv = cmd_all[idx_vl], y_all[idx_vl]

        tcd_held = y_vl_cv[:, 0]
        print(f"  Fold {fold_idx+1}/5: held-out TCD range = "
              f"{tcd_held.min():.1f}–{tcd_held.max():.1f} nm "
              f"(N_val={len(y_vl_cv)})...")

        pt_cv, pb_cv, pd_cv = run_unified_pipeline(
            cmd_tr_cv, y_tr_cv, cmd_vl_cv, y_vl_cv, lut, run_cv=True
        )

        t_mape = mape(y_vl_cv[:, 0], pt_cv)
        b_mape = mape(y_vl_cv[:, 1], pb_cv)
        d_mape = mape(y_vl_cv[:, 2], pd_cv)

        tcds.append(t_mape)
        bcds.append(b_mape)
        depths.append(d_mape)
        print(f"    TCD={t_mape:.3f}%  BCD={b_mape:.3f}%  Depth={d_mape:.3f}%  [{time.time()-t0:.0f}s]")

    print()
    print("=" * 70)
    print("  VSD-CMD GroupKFold (TCD-stratified) Results:")
    print(f"    TCD   = {np.mean(tcds):.3f} +/- {np.std(tcds):.3f}%")
    print(f"    BCD   = {np.mean(bcds):.3f} +/- {np.std(bcds):.3f}%")
    print(f"    Depth = {np.mean(depths):.3f} +/- {np.std(depths):.3f}%")
    print()

    # ── Covariate shift comparison ──────────────────────────────────────────
    svd_rand_depth  = 1.140   # SVD-ResNet random KFold (Table 4)
    svd_group_depth = 1.649   # SVD-ResNet GroupKFold (server log task-3522)
    svd_degradation = (svd_group_depth - svd_rand_depth) / svd_rand_depth * 100

    vsd_rand_depth  = 0.175   # VSD-CMD random KFold (Table 4)
    vsd_group_depth = np.mean(depths)
    vsd_degradation = (vsd_group_depth - vsd_rand_depth) / vsd_rand_depth * 100

    print("  COVARIATE SHIFT COMPARISON (Depth MAPE):")
    print(f"  {'Method':<18} {'Random KFold':>14} {'GroupKFold':>12} {'Degradation':>13}")
    print("  " + "-" * 60)
    print(f"  {'SVD-ResNet':<18} {svd_rand_depth:>14.3f}% {svd_group_depth:>12.3f}% {svd_degradation:>+12.1f}%")
    print(f"  {'VSD-CMD':<18} {vsd_rand_depth:>14.3f}% {vsd_group_depth:>12.3f}% {vsd_degradation:>+12.1f}%")
    print()
    print(f"  SVD-ResNet GroupKFold / random KFold ratio: {svd_group_depth/svd_rand_depth:.2f}x")
    print(f"  VSD-CMD    GroupKFold / random KFold ratio: {vsd_group_depth/vsd_rand_depth:.2f}x")
    print()

    print("  COPY-PASTE FOR PAPER:")
    print(f"  VSD-CMD GroupKFold Depth: {np.mean(depths):.3f}% +/- {np.std(depths):.3f}%")
    print(f"  VSD-CMD degradation:      {vsd_degradation:+.1f}%")
    print(f"  SVD-ResNet degradation:   {svd_degradation:+.1f}%")
    print(f"  Resilience advantage:     VSD-CMD degrades {abs(vsd_degradation):.1f}% vs SVD-ResNet {abs(svd_degradation):.1f}%")
    print("=" * 70)

    np.save(os.path.join(DIR, "groupkfold_vsdcmd_results.npy"), {
        "tcds": tcds, "bcds": bcds, "depths": depths,
        "vsd_group_depth_mean": np.mean(depths),
        "vsd_group_depth_std":  np.std(depths),
        "vsd_degradation_pct":  vsd_degradation,
        "svd_degradation_pct":  svd_degradation,
    })
    print(f"  Saved to groupkfold_vsdcmd_results.npy")


if __name__ == "__main__":
    main()
