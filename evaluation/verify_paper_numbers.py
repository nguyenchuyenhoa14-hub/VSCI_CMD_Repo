"""
verify_paper_numbers.py — Cross-Check LaTeX Paper Against PAPER_NUMBERS.py
===========================================================================
Run this before EVERY submission to catch text/figure/table inconsistencies.

Usage:
    python verify_paper_numbers.py

Output:
    PASS / FAIL for each numeric claim in the paper.
    Any FAIL indicates a mismatch between LaTeX and ground truth.
"""
import re, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from PAPER_NUMBERS import (
    DATASET, TABLE3, TABLE3_DERIVED,
    TABLE4, TABLE4_DERIVED,
    TABLE5, TABLE5_DERIVED,
    FIG6, FIG7, PHYSICS, TEXT_CLAIMS,
    GROUP_KFOLD, GROUP_KFOLD_DERIVED
)

TEX_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "VSD_CMD_Paper.tex")

# ============================================================================
# Helper: read the tex file
# ============================================================================
def read_tex():
    with open(TEX_FILE, encoding="utf-8") as f:
        return f.read()

def find_all(text, pattern, flags=0):
    return re.findall(pattern, text, flags)

# ============================================================================
# Registry of checks
# Each check: (description, expected_str, search_pattern)
# If expected_str appears literally in tex → PASS
# ============================================================================
def build_checks(tex):
    checks = []
    PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

    def add(desc, expected, must_exist=True):
        val = str(expected)
        found = val in tex
        status = PASS if found else (FAIL if must_exist else WARN)
        checks.append((status, desc, val))

    def add_float(desc, expected, tol=0.001, must_exist=True):
        """Check that a float value appears somewhere in the tex with given tolerance."""
        # Look for exact match first
        val_str = str(expected)
        if val_str in tex:
            checks.append((PASS, desc, val_str))
            return
        # Try rounding variants
        for decimals in [1, 2, 3, 4]:
            rounded = f"{float(expected):.{decimals}f}"
            if rounded in tex:
                checks.append((PASS, desc, rounded))
                return
        status = FAIL if must_exist else WARN
        checks.append((status, desc, f"{expected} (not found in any rounding)"))

    # -----------------------------------------------------------------------
    # TABLE 3: Main results
    # -----------------------------------------------------------------------
    t3 = TABLE3["vsd_cmd_fixed"]
    add_float("Table 3: VSD-CMD TCD",   t3["tcd"])    # 0.042
    add_float("Table 3: VSD-CMD BCD",   t3["bcd"])    # 0.886
    add_float("Table 3: VSD-CMD Depth", t3["depth"])  # 0.195
    add_float("Table 3: VSD-CMD Sum",   t3["sum"])    # 1.124

    t3b = TABLE3["svd_resnet_chen"]
    add_float("Table 3: SVD-ResNet Sum", t3b["sum"])  # 2.430
    add_float("Table 3: SVD-ResNet TCD", t3b["tcd"])  # 0.260
    add_float("Table 3: SVD-ResNet BCD", t3b["bcd"])  # 1.380
    add_float("Table 3: SVD-ResNet Depth", t3b["depth"])  # 0.790

    # Chen et al. baseline
    t3c = TABLE3["raw_w1"]
    add_float("Table 3: Raw-w1 Sum", t3c["sum"])  # 6.890

    # Derived text claims
    add_float("Text: Sum abs improvement",    TABLE3_DERIVED["sum_abs_improvement_vs_svdresnet"])  # 1.306
    add_float("Text: Sum rel improvement %",  TABLE3_DERIVED["sum_rel_improvement_vs_svdresnet_pct"])  # 53.8

    # TCD in nm
    add("Text: TCD in nm (0.13)",  "0.13")

    # -----------------------------------------------------------------------
    # TABLE 4: Cross-validation
    # -----------------------------------------------------------------------
    t4v = TABLE4["vsd_cmd_cv"]
    add_float("Table 4: VSD-CMD CV Depth mean",  t4v["depth_mean"])  # 0.175
    add_float("Table 4: VSD-CMD CV Depth std",   t4v["depth_std"])   # 0.055
    add_float("Table 4: VSD-CMD CV Sum mean",    t4v["sum_mean"])    # 1.113
    add_float("Table 4: VSD-CMD CV BCD mean",    t4v["bcd_mean"])    # 0.860

    t4s = TABLE4["svd_resnet_cv"]
    add_float("Table 4: SVD-ResNet CV Depth mean", t4s["depth_mean"])  # 1.140
    add_float("Table 4: SVD-ResNet CV Sum mean",   t4s["sum_mean"])    # 2.912

    add_float("Text: Depth CV improvement factor", TABLE4_DERIVED["depth_rel_improvement_vs_svdresnet"])  # 6.51

    # --- SIGNIFICANCE CHECKS (bootstrap CI + exact permutation, not t-test df=4) ---
    add_float("Text: Bootstrap CI lower", TABLE4_DERIVED["boot_ci_lo"])  # 0.896
    add_float("Text: Bootstrap CI upper", TABLE4_DERIVED["boot_ci_hi"])  # 1.005
    add_float("Text: Permutation p-value", TABLE4_DERIVED["perm_p_value"])  # 0.0313
    add("Text: Depth Cohen's d", f"Cohen's $d > {int(TABLE4_DERIVED['depth_cohens_d'])}$")

    # -----------------------------------------------------------------------
    # TABLE 5: Ablation
    # -----------------------------------------------------------------------
    add_float("Table 5: DPI only MAPE",    TABLE5["dpi_only"]["depth_mape"])    # 4.903
    add_float("Table 5: DPI only MAE",     TABLE5["dpi_only"]["depth_mae"])     # 145.4
    add_float("Table 5: Ridge corrector",    TABLE5["dpi_ridge"]["depth_mape"])   # 1.084
    add_float("Table 5: CWPC-only MAPE",     TABLE5["dpi_rbf_cwpc"]["depth_mape"])  # 0.361
    add_float("Table 5: CWPC-only MAE",      TABLE5["dpi_rbf_cwpc"]["depth_mae"])   # 10.7
    add_float("Table 5: geo-only MAPE",      TABLE5["dpi_rbf_geo"]["depth_mape"])   # 1.326
    add_float("Table 5: geo-only MAE",       TABLE5["dpi_rbf_geo"]["depth_mae"])    # 39.4

    # Derived table 5 text
    add_float("Text: CWPC vs geo improvement %", TABLE5_DERIVED["cwpc_vs_geo_improvement_pct"])  # 72.7
    add_float("Text: Ridge vs DPI improvement %", TABLE5_DERIVED["ridge_vs_dpi_improvement_pct"])  # 77.6
    add_float("Text: RBF vs Ridge improvement %", TABLE5_DERIVED["rbf_vs_ridge_improvement_pct"])  # 66.7

    # -----------------------------------------------------------------------
    # FIGURE 6: Wavelength sweep
    # -----------------------------------------------------------------------
    add_float("Fig6: Optimal WL count",  FIG6["optimal_wl"])      # 65
    add_float("Fig6: Optimal Sum MAPE",  FIG6["optimal_mape"])    # 1.124 (same as Table3)
    add_float("Fig6: SVD baseline",      FIG6["svd_baseline"])    # 2.430 (same as Table3)

    # -----------------------------------------------------------------------
    # FIGURE 7: Jitter
    # -----------------------------------------------------------------------
    add("Fig7: Jitter offset range 0.82", "0.82")
    add_float("Fig7: Max physical mode variance", FIG7["physical_modes_max_var_pct"])  # 3e-4 → 0.0003
    add("Fig7: CI lower bound 0.0001", "0.0001")
    add("Fig7: CI upper bound 0.0003", "0.0003")

    # -----------------------------------------------------------------------
    # COVARIATE SHIFT (GroupKFold)
    # -----------------------------------------------------------------------
    add_float("OOD: SVD-ResNet GroupKFold Depth", GROUP_KFOLD["svd_resnet"]["depth_mean"])
    add_float("OOD: VSD-CMD GroupKFold Depth", GROUP_KFOLD["vsd_cmd"]["depth_mean"])
    add_float("OOD: VSD-CMD Degradation Abs", GROUP_KFOLD_DERIVED["vsd_cmd_degradation_abs"])

    # -----------------------------------------------------------------------
    # PHYSICS constants in text
    # -----------------------------------------------------------------------
    add_float("Physics: gamma value",    PHYSICS["gamma"])   # -0.193
    add_float("Physics: n_Cu real",      PHYSICS["n_cu_real"])  # 1.22
    add_float("Physics: n_Cu imag",      PHYSICS["n_cu_imag"])  # 1.88
    add("Physics: n_rings = 16",  "N_r = 16")

    # -----------------------------------------------------------------------
    # DATASET stats in text
    # -----------------------------------------------------------------------
    add(f"Dataset: n_train={DATASET['n_train']}", str(DATASET["n_train"]))  # 1629
    add(f"Dataset: n_val={DATASET['n_val']}",    str(DATASET["n_val"]))    # 139
    add(f"Dataset: n_total={DATASET['n_total']}", str(DATASET["n_total"])) # 1768
    add(f"Dataset: n_folds={DATASET['n_folds']}", str(DATASET["n_folds"])) # 5
    add(f"Dataset: Yee-grid pitch {DATASET['yee_grid_pitch']}nm",
        str(DATASET["yee_grid_pitch"]))  # 1.64

    return checks


# ============================================================================
# Run and report
# ============================================================================
def main():
    tex = read_tex()
    checks = build_checks(tex)

    n_pass = sum(1 for s,_,_ in checks if s == "PASS")
    n_fail = sum(1 for s,_,_ in checks if s == "FAIL")
    n_warn = sum(1 for s,_,_ in checks if s == "WARN")

    print("=" * 70)
    print("  VSD-CMD Paper Cross-Verification Report")
    print("=" * 70)
    print(f"  File: {TEX_FILE}")
    print(f"  Checks: {len(checks)} total  |  {n_pass} PASS  |  {n_fail} FAIL  |  {n_warn} WARN")
    print("=" * 70)

    # Print fails first
    if n_fail > 0:
        print("\n  [FAILURES - must fix before submission]")
        for status, desc, val in checks:
            if status == "FAIL":
                print(f"    FAIL  {desc}")
                print(f"          Expected value: '{val}' not found in paper")

    if n_warn > 0:
        print("\n  [WARNINGS - review recommended]")
        for status, desc, val in checks:
            if status == "WARN":
                print(f"    WARN  {desc}")
                print(f"          Value '{val}' not found (optional check)")

    print("\n  [FULL RESULTS]")
    for status, desc, val in checks:
        mark = "[PASS]" if status == "PASS" else ("[FAIL]" if status == "FAIL" else "[WARN]")
        print(f"  {mark}  {desc}  ('{val}')")

    print("=" * 70)
    if n_fail == 0:
        print("  ALL REQUIRED CHECKS PASSED. Paper is consistent.")
    else:
        print(f"  {n_fail} FAILURE(S) FOUND. Fix before submission.")
    print("=" * 70)

    return n_fail


if __name__ == "__main__":
    n_fail = main()
    sys.exit(n_fail)
