"""
run_baselines.py — Standard "Off-the-shelf" ML Baselines for VSCI-CMD Comparison

Baselines implemented (end-to-end, zero physics knowledge):
  1. MLP (Multilayer Perceptron)       — standard first-line baseline
  2. 1D CNN (Spectral CNN)              — best-practice for spectral data
  3. GPR (Gaussian Process Regression)  — canonical small-data baseline

Input features (raw, physics-free):
  - Flatten |cmd|^2 power spectrum: (N, 55 WL, 5 modes) -> (N, 275) after mean over radii

Targets: TCD, BCD, Depth (in nm) -> MAPE reported

Usage:
  python src/run_baselines.py                 # CPU
  python src/run_baselines.py --gpu           # GPU (auto-detect)
  python src/run_baselines.py --skip-gpr      # skip slow GPR
  python src/run_baselines.py --mlp-only      # only MLP
"""

import sys, os, time, argparse
import numpy as np
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Constants ─────────────────────────────────────────────────────────────
DIR = r"./data"
CACHE_TR = os.path.join(DIR, "fdtd_cache_highres_training_1629s.npz")
CACHE_VL = os.path.join(DIR, "fdtd_cache_highres_validation_139s.npz")
N_WL    = 55
N_MODES = 5
N_FEAT  = N_WL * N_MODES  # 275

mape = lambda a, b: float(np.mean(np.abs((a - b) / (a + 1e-8))) * 100)


# ── Feature Extraction ────────────────────────────────────────────────────
def build_raw_features(cmd_full, n_wl=N_WL, n_modes=N_MODES):
    """
    Raw mode-power features: zero physics, zero modal decomposition.
    Takes |cmd|^2 averaged over radii -> (N, n_wl * n_modes)
    cmd_full shape: (N, 100 WL, 16 radii, 24 components)
    """
    cmd = cmd_full[:, :n_wl, :, :]           # (N, 55, 16, 24)
    pwr = np.zeros((len(cmd), n_wl, n_modes), dtype=np.float32)
    for m in range(n_modes):
        re = cmd[:, :, :, m]
        im = cmd[:, :, :, 8 + m]
        pwr[:, :, m] = np.mean(re**2 + im**2, axis=2)   # average over radii
    X = pwr.reshape(len(cmd), -1)    # (N, 275)
    return X


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE 1: MLP
# ═══════════════════════════════════════════════════════════════════════════
def run_mlp_baseline(X_tr, y_tr, X_vl, y_vl, device="cpu", epochs=600, batch=128):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "-" * 60)
    print("  BASELINE 1: End-to-End MLP")
    print("-" * 60)
    print(f"  Feature dim: {X_tr.shape[1]}  |  Train N: {len(X_tr)}  |  Val N: {len(X_vl)}")

    sc_x = StandardScaler(); sc_y = StandardScaler()
    Xs_tr = sc_x.fit_transform(X_tr).astype(np.float32)
    ys_tr = sc_y.fit_transform(y_tr).astype(np.float32)
    Xs_vl = sc_x.transform(X_vl).astype(np.float32)

    in_dim = X_tr.shape[1]
    class MLPRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.GELU(),
                nn.Linear(128, 3)
            )
        def forward(self, x): return self.net(x)

    use_gpu = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_gpu else "cpu")
    model = MLPRegressor().to(dev)
    print(f"  Running on: {dev}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss(delta=1.0)

    ds = TensorDataset(torch.tensor(Xs_tr), torch.tensor(ys_tr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        scheduler.step()
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                pv = sc_y.inverse_transform(
                    model(torch.tensor(Xs_vl).to(dev)).cpu().numpy())
                print(f"  Ep {ep+1:4d} | TCD={mape(y_vl[:,0],pv[:,0]):.3f}% "
                      f"BCD={mape(y_vl[:,1],pv[:,1]):.3f}% Dep={mape(y_vl[:,2],pv[:,2]):.3f}%")
            model.train()

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pv_final = sc_y.inverse_transform(
            model(torch.tensor(Xs_vl).to(dev)).cpu().numpy())

    results = {
        "TCD_MAPE":   mape(y_vl[:, 0], pv_final[:, 0]),
        "BCD_MAPE":   mape(y_vl[:, 1], pv_final[:, 1]),
        "Depth_MAPE": mape(y_vl[:, 2], pv_final[:, 2]),
        "time_min":   elapsed / 60,
    }
    results["Sum_MAPE"] = results["TCD_MAPE"] + results["BCD_MAPE"] + results["Depth_MAPE"]
    print(f"\n  MLP done in {elapsed/60:.1f} min")
    print(f"  TCD={results['TCD_MAPE']:.3f}%  BCD={results['BCD_MAPE']:.3f}%  "
          f"Depth={results['Depth_MAPE']:.3f}%  Sum={results['Sum_MAPE']:.3f}%")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE 2: 1D SPECTRAL CNN
# ═══════════════════════════════════════════════════════════════════════════
def run_cnn_baseline(X_tr, y_tr, X_vl, y_vl, device="cpu", epochs=800, batch=64):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "-" * 60)
    print("  BASELINE 2: 1D Spectral CNN")
    print("-" * 60)

    # Reshape to (N, modes=5, wl=55)
    X_tr_c = X_tr.reshape(-1, N_WL, N_MODES).transpose(0, 2, 1).astype(np.float32)
    X_vl_c = X_vl.reshape(-1, N_WL, N_MODES).transpose(0, 2, 1).astype(np.float32)

    ch_mean = X_tr_c.mean(axis=(0, 2), keepdims=True)
    ch_std  = X_tr_c.std(axis=(0, 2),  keepdims=True) + 1e-12
    X_tr_c = (X_tr_c - ch_mean) / ch_std
    X_vl_c = (X_vl_c - ch_mean) / ch_std

    sc_y = StandardScaler()
    ys_tr = sc_y.fit_transform(y_tr).astype(np.float32)

    class SpectralCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(N_MODES, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),      nn.BatchNorm1d(64), nn.GELU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),     nn.BatchNorm1d(128), nn.GELU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),    nn.BatchNorm1d(256), nn.GELU(),
                nn.MaxPool1d(2),
            )
            # 55 -> 27 -> 13 -> 6  (after 3x MaxPool1d(2))
            flat_dim = 256 * 6
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, 64),       nn.GELU(),
                nn.Linear(64, 3)
            )
        def forward(self, x): return self.head(self.encoder(x))

    use_gpu = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_gpu else "cpu")
    model = SpectralCNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Running on: {dev}  |  Parameters: {n_params:,}")

    ds = TensorDataset(torch.tensor(X_tr_c), torch.tensor(ys_tr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    steps_per_epoch = len(dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=5e-4, steps_per_epoch=steps_per_epoch, epochs=epochs)
    crit = nn.HuberLoss(delta=1.0)

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                pv = sc_y.inverse_transform(
                    model(torch.tensor(X_vl_c).to(dev)).cpu().numpy())
                print(f"  Ep {ep+1:4d} | TCD={mape(y_vl[:,0],pv[:,0]):.3f}% "
                      f"BCD={mape(y_vl[:,1],pv[:,1]):.3f}% Dep={mape(y_vl[:,2],pv[:,2]):.3f}%")
            model.train()

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pv_final = sc_y.inverse_transform(
            model(torch.tensor(X_vl_c).to(dev)).cpu().numpy())

    results = {
        "TCD_MAPE":   mape(y_vl[:, 0], pv_final[:, 0]),
        "BCD_MAPE":   mape(y_vl[:, 1], pv_final[:, 1]),
        "Depth_MAPE": mape(y_vl[:, 2], pv_final[:, 2]),
        "time_min":   elapsed / 60,
    }
    results["Sum_MAPE"] = results["TCD_MAPE"] + results["BCD_MAPE"] + results["Depth_MAPE"]
    print(f"\n  CNN done in {elapsed/60:.1f} min")
    print(f"  TCD={results['TCD_MAPE']:.3f}%  BCD={results['BCD_MAPE']:.3f}%  "
          f"Depth={results['Depth_MAPE']:.3f}%  Sum={results['Sum_MAPE']:.3f}%")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE 3: GPR
# ═══════════════════════════════════════════════════════════════════════════
def run_gpr_baseline(X_tr, y_tr, X_vl, y_vl, n_pca=60):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("\n" + "-" * 60)
    print("  BASELINE 3: Gaussian Process Regression (GPR)")
    print("-" * 60)
    print(f"  PCA({n_pca}) preprocessing  |  Train N: {len(X_tr)}")

    sc = StandardScaler()
    pca = PCA(n_components=n_pca, random_state=42)
    X_tr_s = pca.fit_transform(sc.fit_transform(X_tr))
    X_vl_s = pca.transform(sc.transform(X_vl))
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    target_names = ["TCD", "BCD", "Depth"]
    preds = np.zeros((len(X_vl), 3))
    t0 = time.time()

    for i, tname in enumerate(target_names):
        print(f"  Fitting GPR for {tname}...", end=" ", flush=True)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3,
            alpha=1e-6, normalize_y=True)
        gpr.fit(X_tr_s, y_tr[:, i])
        preds[:, i] = gpr.predict(X_vl_s)
        print(f"done ({time.time()-t0:.0f}s total)")

    elapsed = time.time() - t0
    results = {
        "TCD_MAPE":   mape(y_vl[:, 0], preds[:, 0]),
        "BCD_MAPE":   mape(y_vl[:, 1], preds[:, 1]),
        "Depth_MAPE": mape(y_vl[:, 2], preds[:, 2]),
        "time_min":   elapsed / 60,
    }
    results["Sum_MAPE"] = results["TCD_MAPE"] + results["BCD_MAPE"] + results["Depth_MAPE"]
    print(f"\n  GPR done in {elapsed/60:.1f} min")
    print(f"  TCD={results['TCD_MAPE']:.3f}%  BCD={results['BCD_MAPE']:.3f}%  "
          f"Depth={results['Depth_MAPE']:.3f}%  Sum={results['Sum_MAPE']:.3f}%")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# TABLE + MAIN
# ═══════════════════════════════════════════════════════════════════════════
def print_table(all_results):
    print("\n" + "=" * 78)
    print("  FINAL COMPARISON TABLE — VSCI-CMD vs Standard ML Baselines")
    print("=" * 78)
    print(f"  {'Method':<28} | {'TCD%':>7} | {'BCD%':>7} | {'Depth%':>8} | {'Sum%':>7}")
    sep = "  " + "-" * 28 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 10 + "+" + "-" * 9
    print(sep)
    rows = [
        ("GPR",              all_results.get("gpr")),
        ("MLP (End-to-End)", all_results.get("mlp")),
        ("1D Spectral CNN",  all_results.get("cnn")),
        ("SVD-ResNet (ref.)", {"TCD_MAPE": 0.155, "BCD_MAPE": 2.090,
                               "Depth_MAPE": 0.790, "Sum_MAPE": 3.035}),
        ("VSCI-CMD (ours)",  {"TCD_MAPE": 0.041, "BCD_MAPE": 0.806,
                               "Depth_MAPE": 0.225, "Sum_MAPE": 1.073}),
    ]
    for name, res in rows:
        if res is None:
            print(f"  {name:<28} | {'--':>7} | {'--':>7} | {'--':>8} | {'--':>7}")
        else:
            print(f"  {name:<28} | {res['TCD_MAPE']:>7.3f} | {res['BCD_MAPE']:>7.3f} "
                  f"| {res['Depth_MAPE']:>8.3f} | {res['Sum_MAPE']:>7.3f}")
    print(sep)
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Run ML baselines for VSCI-CMD paper")
    parser.add_argument("--gpu",       action="store_true", help="Use GPU (CUDA)")
    parser.add_argument("--skip-gpr",  action="store_true", help="Skip GPR (slow)")
    parser.add_argument("--mlp-only",  action="store_true", help="Only run MLP")
    parser.add_argument("--epochs-mlp", type=int, default=600)
    parser.add_argument("--epochs-cnn", type=int, default=800)
    args = parser.parse_args()

    device = "cuda" if args.gpu else "cpu"

    print("=" * 78)
    print("  STANDARD ML BASELINES for VSCI-CMD Paper")
    print("  Raw input: |cmd|^2 power spectrum -> 275 features (zero physics)")
    print("=" * 78)

    print("\nLoading data...")
    tr = np.load(CACHE_TR)
    vl = np.load(CACHE_VL)
    y_tr = tr["labels"].astype(np.float64)
    y_vl = vl["labels"].astype(np.float64)

    print("Building raw features...")
    X_tr = build_raw_features(tr["cmd"]).astype(np.float64)
    X_vl = build_raw_features(vl["cmd"]).astype(np.float64)
    print(f"  X_tr: {X_tr.shape}  X_vl: {X_vl.shape}")

    all_results = {}
    all_results["mlp"] = run_mlp_baseline(
        X_tr, y_tr, X_vl, y_vl, device=device, epochs=args.epochs_mlp)

    if not args.mlp_only:
        all_results["cnn"] = run_cnn_baseline(
            X_tr, y_tr, X_vl, y_vl, device=device, epochs=args.epochs_cnn)
        if not args.skip_gpr:
            all_results["gpr"] = run_gpr_baseline(X_tr, y_tr, X_vl, y_vl)
        else:
            print("\n  [SKIP] GPR (--skip-gpr)")

    print_table(all_results)
    np.save("data/baseline_results.npy", all_results)
    print("  Results saved -> data/baseline_results.npy")


if __name__ == "__main__":
    main()

