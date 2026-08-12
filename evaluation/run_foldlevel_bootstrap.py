"""
run_foldlevel_bootstrap.py
==========================
Paired fold-level bootstrap CI using already-computed fold MAPE values.
No additional model training required.

This avoids the sample-level approach's GPU non-determinism issue while 
still providing a valid, non-parametric alternative to t-test df=4.

Fold data from run_final_dual_pipeline.py and run_svd_resnet_torch.py (Table 4):
  VSD-CMD   5-fold Depth (seed=42): [0.154, 0.169, 0.130, 0.281, 0.141]%
  SVD-ResNet 5-fold Depth (GPU):    [1.108, 1.211, 1.065, 1.139, 1.108]%

Bootstrap procedure:
  Resample the 5 paired (vsd_k, svd_k) fold results with replacement,
  compute mean(svd) - mean(vsd) for each resample.
  95% CI = [2.5th, 97.5th] percentile of B=10,000 bootstrap means.
"""
import numpy as np

# ── Fold-level results (from PAPER_NUMBERS.py / Table 4) ─────────────────────
# Per-fold Depth MAPE — these are the EXACT values that produced the Table 4 means
VSD_FOLDS = np.array([0.154, 0.169, 0.130, 0.281, 0.141])  # VSD-CMD 5-fold (random KFold seed=42)
SVD_FOLDS = np.array([1.108, 1.211, 1.065, 1.139, 1.108])  # SVD-ResNet GPU server (latest run)

# Sanity-check means vs Table 4
print("Fold means (sanity check vs Table 4):")
print(f"  VSD-CMD  mean: {VSD_FOLDS.mean():.3f}% ± {VSD_FOLDS.std():.3f}%  (Table 4: 0.175% ± 0.055%)")
print(f"  SVD-ResNet mean: {SVD_FOLDS.mean():.3f}% ± {SVD_FOLDS.std():.3f}%  (Table 4: 1.126% ± 0.049%)")

B = 10_000
rng = np.random.default_rng(42)

# Paired resample — always resample (vsd_k, svd_k) jointly to preserve pairing
boot_mean_diffs = np.empty(B)
n = len(VSD_FOLDS)
for b in range(B):
    idx = rng.integers(0, n, size=n)
    boot_mean_diffs[b] = SVD_FOLDS[idx].mean() - VSD_FOLDS[idx].mean()

# Point estimate and CI
observed_diff = SVD_FOLDS.mean() - VSD_FOLDS.mean()
ci_lo = float(np.percentile(boot_mean_diffs, 2.5))
ci_hi = float(np.percentile(boot_mean_diffs, 97.5))
p_one_sided = float(np.mean(boot_mean_diffs <= 0))

print()
print("="*65)
print("  FOLD-LEVEL PAIRED BOOTSTRAP CI (B=10,000, seed=42)")
print("="*65)
print(f"  Observed mean advantage (SVD - VSD): {observed_diff:.4f}%")
print(f"  Bootstrap mean:                      {boot_mean_diffs.mean():.4f}%")
print(f"  95% Bootstrap CI:                    [{ci_lo:.4f}%, {ci_hi:.4f}%]")
print(f"  99% Bootstrap CI:                    [{np.percentile(boot_mean_diffs,0.5):.4f}%, {np.percentile(boot_mean_diffs,99.5):.4f}%]")
print(f"  P(advantage <= 0) empirical:         {p_one_sided:.5f}")
print()

# Null-distribution permutation test (exact, since n=5)
from itertools import product
signs = list(product([1,-1], repeat=n))
perm_diffs = np.array([
    np.mean(SVD_FOLDS - s * (SVD_FOLDS - VSD_FOLDS)) - np.mean(VSD_FOLDS + s * (SVD_FOLDS - VSD_FOLDS))
    for s in signs
])
# Each sign vector flips whether the advantage belongs to method A or B
diffs_perm = []
for s in signs:
    sv = np.where(np.array(s) == 1, SVD_FOLDS, VSD_FOLDS)
    vv = np.where(np.array(s) == 1, VSD_FOLDS, SVD_FOLDS)
    diffs_perm.append(sv.mean() - vv.mean())
diffs_perm = np.array(diffs_perm)
p_perm = float(np.mean(diffs_perm >= observed_diff))

print(f"  Permutation test p-value (exact, 2^5={2**n} arrangements): {p_perm:.4f}")
print()
print("  COPY-PASTE FOR PAPER (LaTeX):")
print(f"  95\\% bootstrap CI $[{ci_lo:.3f}\\%, {ci_hi:.3f}\\%]$ (B=10,000)")
print(f"  Exact permutation p-value: $p = {p_perm:.4f}$")
print()
print("  NOTE: 95% CI entirely above zero confirms VSD-CMD advantage")
print(f"        is robust across all {B:,} fold resamples.")
print("="*65)
