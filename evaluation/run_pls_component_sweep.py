"""
run_pls_component_sweep.py
==========================
Sweeps PLS n_components for the BCD regression stage
to justify the choice of n_components=150 in VSD-CMD.

Outputs:
  - BCD MAPE vs n_components (5-fold CV on training set)
  - Cumulative PLS variance explained (X and Y) at n=150
  - A scree-like table for the paper supplementary

This script uses the same feature pipeline as run_final_dual_pipeline.py,
but only the BCD PLS stage, for speed.
"""
import numpy as np
import time
import os
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_TR = os.path.join(DIR, "data", "fdtd_cache_highres_training_1629s.npz")
LUT_PATH  = os.path.join(DIR, "data", "tsv_physics_lut_multimode.npz")

mape = lambda a, b: float(np.mean(np.abs((a - b) / (a + 1e-8))) * 100)

# ── Feature builder (matches run_final_dual_pipeline.py) ──────────────────────
def compute_dynamic_cwpc(cmd_data, radii, rw, MODES):
    n_wl = cmd_data.shape[1]
    N = len(cmd_data)
    c_complex = np.zeros((N, n_wl, len(radii), len(MODES)), dtype=np.complex64)
    for i, m in enumerate(MODES):
        c_complex[:, :, :, i] = cmd_data[:, :, :, m] + 1j * cmd_data[:, :, :, 8 + m]
    weights = np.sqrt(radii * rw)
    c_weighted = c_complex * weights[None, None, :, None]
    c_flat = c_weighted.reshape(N, n_wl, -1)
    coh_mat = np.einsum("nvi,nwi->nvw", c_flat, np.conj(c_flat))
    phase_mat = np.angle(coh_mat)
    idx = np.triu_indices(n_wl, k=1)
    return phase_mat[:, idx[0], idx[1]]


def build_bcd_features(cmd, lut, n_wl=65):
    """Builds the full BCD feature matrix (power + CMR + CWPC)."""
    wl_mask = np.arange(n_wl)
    cmd = cmd[:, wl_mask]
    tcd_grid = lut["tcd_grid"]
    r_fdm = lut["r"]
    MODES = [0, 1, 2, 3, 4]
    MODES_BCD = [0, 1, 2]

    beta_luts = {m: lut[f"beta_m{m}"][:, wl_mask] for m in MODES}
    profile_luts = {m: lut[f"profile_m{m}"][:, wl_mask, :] for m in MODES}

    N_R = 16; R0, R1 = 90., 330.
    radii = np.linspace(R0, R1, N_R)
    dr = (R1 - R0) / 15.
    rw = np.ones(N_R) * dr; rw[0] = rw[-1] = dr / 2.

    phase = compute_dynamic_cwpc(cmd, radii, rw, MODES)
    cv_v = np.var(phase, axis=0) > 1e-10
    pf = phase[:, cv_v]

    ii = []; iw = []
    for rv in radii:
        k = np.clip(np.searchsorted(r_fdm, rv) - 1, 0, len(r_fdm) - 2)
        ii.append(k); iw.append((rv - r_fdm[k]) / (r_fdm[k + 1] - r_fdm[k]))
    ii = np.array(ii); iw = np.array(iw)

    ref_idx = 4; pnr = {}
    for m in MODES:
        p = profile_luts[m][ref_idx]
        pi = (1 - iw) * p[:, ii] + iw * p[:, ii + 1]
        nf = np.sqrt(2 * np.pi * np.sum(np.abs(pi) ** 2 * radii * rw, axis=1)) + 1e-12
        pnr[m] = pi / nf[:, None]

    def fixed_power(cmd_in, MODES_list):
        N = len(cmd_in); F = np.zeros((N, n_wl * len(MODES_list)), np.float32)
        for i in range(N):
            for mi, m in enumerate(MODES_list):
                c = cmd_in[i, :, :, m] + 1j * cmd_in[i, :, :, 8 + m]
                S = 2 * np.pi * np.sum(c * np.conj(pnr[m]) * radii * rw, axis=1)
                F[i, mi * n_wl:(mi + 1) * n_wl] = np.abs(S) ** 2
        return F

    Xf = fixed_power(cmd, MODES)

    # Depth-compensated power for BCD
    h_init = 3000.0  # nominal initial depth
    log_P_corr_list = []
    for m in MODES_BCD:
        beta_im = np.imag(beta_luts[m][ref_idx])
        correction = 2 * beta_im * h_init / np.log(10)
        Xm = fixed_power(cmd, [m]).reshape(len(cmd), n_wl)
        log_P_corr = np.log10(Xm + 1e-30) + correction[None, :]
        log_P_corr_list.append(log_P_corr)

    feats = list(log_P_corr_list)
    for i in range(len(MODES_BCD)):
        for j in range(i + 1, len(MODES_BCD)):
            cmr_ij = log_P_corr_list[j] - log_P_corr_list[i]
            d_cmr = np.diff(cmr_ij, axis=1)
            feats.extend([cmr_ij, d_cmr])

    Xb = np.hstack(feats)
    X_full = np.hstack([Xb, pf])
    return X_full


def main():
    print("=" * 65)
    print("  PLS Component Sweep — BCD Regression")
    print("=" * 65)
    t_total = time.time()

    tr = np.load(CACHE_TR)
    lut = np.load(LUT_PATH)

    cmd_all = tr["cmd"]
    y_all = tr["labels"]

    print(f"\n  Building BCD features for {len(cmd_all)} training samples...")
    t0 = time.time()
    X_all = build_bcd_features(cmd_all, lut, n_wl=65)
    print(f"  Feature matrix shape: {X_all.shape}  ({time.time()-t0:.1f}s)")
    print(f"  Max PLS components possible: min({X_all.shape[1]}, {len(cmd_all)}) = {min(X_all.shape[1], len(cmd_all))}")

    # ── Variance Explained at n_components=150 ────────────────────────────────
    print("\n  Computing PLS variance explained (n=150)...")
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_all)
    y_log = np.log10(y_all[:, 1])

    pls_full = PLSRegression(n_components=150, max_iter=500)
    pls_full.fit(X_sc, y_log)

    # X-variance explained per component (x_scores_ x_weights_)
    x_var = np.var(pls_full.x_scores_, axis=0)
    x_var_frac = x_var / np.sum(np.var(X_sc, axis=0))

    print(f"\n  PLS X-variance explained:")
    cumvar = 0
    for k in [1, 5, 10, 20, 50, 75, 100, 125, 150]:
        cumvar_k = float(np.sum(x_var_frac[:k]))
        print(f"    k={k:>3}: cumulative X-var = {cumvar_k*100:.1f}%")

    # ── 5-fold CV BCD MAPE sweep ──────────────────────────────────────────────
    component_grid = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n  5-fold CV BCD MAPE vs n_components:")
    print(f"  {'n_comp':>6}  {'mean MAPE':>10}  {'std MAPE':>9}  {'min fold':>9}  {'max fold':>9}")
    print("  " + "-" * 52)

    sweep_results = {}
    for nc in component_grid:
        fold_mapes = []
        for idx_tr, idx_vl in kf.split(X_sc):
            Xtr, Xvl = X_sc[idx_tr], X_sc[idx_vl]
            ytr, yvl = y_log[idx_tr], y_log[idx_vl]
            bcd_tr = y_all[idx_tr, 1]
            bcd_vl = y_all[idx_vl, 1]

            pls = PLSRegression(n_components=nc, max_iter=500)
            T_tr = pls.fit_transform(Xtr, ytr)[0]
            T_vl_raw = pls.transform(Xvl)
            T_vl = T_vl_raw[0] if isinstance(T_vl_raw, tuple) else T_vl_raw

            kr = make_pipeline(StandardScaler(), KernelRidge(kernel="polynomial", degree=2, coef0=1, alpha=0.1))
            kr.fit(T_tr, ytr)
            pred = 10 ** np.clip(kr.predict(T_vl), np.log10(200), np.log10(400))
            fold_mapes.append(mape(bcd_vl, pred))

        m = np.mean(fold_mapes)
        s = np.std(fold_mapes)
        best_marker = " ◀" if nc == 150 else ""
        print(f"  {nc:>6}  {m:>10.4f}%  {s:>9.4f}%  {min(fold_mapes):>9.4f}%  {max(fold_mapes):>9.4f}%{best_marker}")
        sweep_results[nc] = (m, s, fold_mapes)

    # ── Summary ───────────────────────────────────────────────────────────────
    best_nc = min(sweep_results, key=lambda k: sweep_results[k][0])
    print(f"\n  Best n_components (by mean CV BCD MAPE): {best_nc}")
    print(f"  Selected in paper:  150")
    print(f"  Paper BCD (n=150):  {sweep_results[150][0]:.4f}% ± {sweep_results[150][1]:.4f}%")
    if best_nc != 150:
        diff = sweep_results[150][0] - sweep_results[best_nc][0]
        print(f"  Δ vs best (n={best_nc}):  +{diff:.4f}% (within rounding noise if <0.05%)")

    print(f"\n  COPY-PASTE FOR PAPER (Table or footnote):")
    print(f"  BCD MAPE by n_components (5-fold CV, KFold seed=42):")
    for nc, (m, s, _) in sweep_results.items():
        marker = " ← selected" if nc == 150 else ""
        print(f"    {nc:>3}: {m:.4f}% ± {s:.4f}%{marker}")

    print(f"\n  Total: {(time.time()-t_total)/60:.1f} min")
    print("=" * 65)

    np.save(os.path.join(DIR, "pls_sweep_results.npy"), sweep_results)
    print(f"  Results saved to pls_sweep_results.npy")


if __name__ == "__main__":
    main()
