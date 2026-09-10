import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

os.chdir(r"C:\Users\Jack\VSCI_Baselines")
data = np.load("baseline_features_dataset.npz")
X_all = np.vstack([data["X_tr"], data["X_vl"]])
y_all = np.vstack([data["y_tr"], data["y_vl"]])
print("Loaded dataset shape:", X_all.shape, y_all.shape)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", dev, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

class SpectralCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.head(self.encoder(x))

tcd_list, bcd_list, dep_list, dep_mae_list = [], [], [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr_f, y_tr_f = X_all[tr_idx], y_all[tr_idx]
    X_vl_f, y_vl_f = X_all[val_idx], y_all[val_idx]
    
    X_tr_c = X_tr_f.reshape(-1, 55, 5).transpose(0, 2, 1).astype(np.float32)
    X_vl_c = X_vl_f.reshape(-1, 55, 5).transpose(0, 2, 1).astype(np.float32)
    
    ch_mean = X_tr_c.mean(axis=(0, 2), keepdims=True)
    ch_std  = X_tr_c.std(axis=(0, 2), keepdims=True) + 1e-12
    X_tr_c = (X_tr_c - ch_mean) / ch_std
    X_vl_c = (X_vl_c - ch_mean) / ch_std
    
    sc_y = StandardScaler()
    ys_tr = sc_y.fit_transform(y_tr_f).astype(np.float32)
    
    model = SpectralCNN().to(dev)
    ds = TensorDataset(torch.tensor(X_tr_c), torch.tensor(ys_tr))
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, steps_per_epoch=len(dl), epochs=600)
    crit = nn.HuberLoss(delta=1.0)
    
    model.train()
    for ep in range(600):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
    model.eval()
    with torch.no_grad():
        preds = sc_y.inverse_transform(model(torch.tensor(X_vl_c).to(dev)).cpu().numpy())
        
    tcd_m = np.mean(np.abs((y_vl_f[:, 0] - preds[:, 0]) / y_vl_f[:, 0])) * 100
    bcd_m = np.mean(np.abs((y_vl_f[:, 1] - preds[:, 1]) / y_vl_f[:, 1])) * 100
    dep_m = np.mean(np.abs((y_vl_f[:, 2] - preds[:, 2]) / y_vl_f[:, 2])) * 100
    dep_mae = np.mean(np.abs(y_vl_f[:, 2] - preds[:, 2]))
    
    tcd_list.append(tcd_m); bcd_list.append(bcd_m); dep_list.append(dep_m); dep_mae_list.append(dep_mae)
    print(f"Fold {fold+1} | TCD: {tcd_m:.3f}%, BCD: {bcd_m:.3f}%, Dep: {dep_m:.3f}% (MAE: {dep_mae:.2f} nm)")

print("\n--- 5-FOLD CV SUMMARY (MEAN +/- STD) ---")
print(f"TCD MAPE: {np.mean(tcd_list):.3f} +/- {np.std(tcd_list):.3f}%")
print(f"BCD MAPE: {np.mean(bcd_list):.3f} +/- {np.std(bcd_list):.3f}%")
print(f"Depth MAPE: {np.mean(dep_list):.3f} +/- {np.std(dep_list):.3f}%")
print(f"Depth MAE: {np.mean(dep_mae_list):.2f} +/- {np.std(dep_mae_list):.2f} nm")
