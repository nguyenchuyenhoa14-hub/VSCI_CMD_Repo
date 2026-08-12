import numpy as np
import os
import scipy.linalg as la

DIR = r"C:\Users\Nguyen\Documents\JHL_SVD\Paper_nearfield\CPC_submission\data"
LUT_PATH = os.path.join(DIR, "tsv_physics_lut_multimode.npz")
OUT_LUT_PATH = os.path.join(DIR, "tsv_physics_lut_multimode.npz")

print("Loading original single-mode LUT...")
lut = np.load(LUT_PATH)
r = lut['r']
tcd_grid = lut['tcd_grid']
wls = lut['wavelengths']
beta_m0_orig = lut['beta_m0']
profile_m0_orig = lut['profile_m0']

N_tcd = len(tcd_grid)
N_wl = len(wls)
N_r = len(r)
dr = r[1] - r[0]

# Allocate arrays for all 5 modes
betas = np.zeros((5, N_tcd, N_wl), dtype=np.complex128)
profiles = np.zeros((5, N_tcd, N_wl, N_r), dtype=np.complex128)

print("Solving multi-mode FDM eigensystem for m = 0, 1, 2, 3, 4...")
for itcd in range(N_tcd):
    print(f"  TCD {tcd_grid[itcd]} nm ({itcd+1}/{N_tcd})...")
    for iwl in range(N_wl):
        wl = wls[iwl]
        k0 = 2 * np.pi / wl
        
        # 1. Reconstruct effective permittivity eps_eff using original m=0 mode
        psi0 = profile_m0_orig[itcd, iwl]
        beta0 = beta_m0_orig[itcd, iwl]
        
        d1_psi = np.zeros(N_r, dtype=np.complex128)
        d2_psi = np.zeros(N_r, dtype=np.complex128)
        for i in range(1, N_r - 1):
            d1_psi[i] = (psi0[i+1] - psi0[i-1]) / (2 * dr)
            d2_psi[i] = (psi0[i+1] - 2*psi0[i] + psi0[i-1]) / (dr**2)
        d1_psi[0] = 0.0
        d2_psi[0] = 2 * (psi0[1] - psi0[0]) / (dr**2)
        d1_psi[-1] = (3*psi0[-1] - 4*psi0[-2] + psi0[-3]) / (2*dr)
        d2_psi[-1] = (psi0[-1] - 2*psi0[-2] + psi0[-3]) / (dr**2)
        
        term_deriv = np.zeros(N_r, dtype=np.complex128)
        for i in range(N_r):
            if i == 0:
                term_deriv[i] = 2 * d2_psi[i]
            else:
                term_deriv[i] = d2_psi[i] + (1.0 / r[i]) * d1_psi[i]
                
        eps_eff = (beta0**2 * psi0 - term_deriv) / (psi0 + 1e-30) / (k0**2)
        eps_eff[0] = eps_eff[1] # fix singularity at r=0
        
        # 2. Solve for m = 0
        H0 = np.zeros((N_r, N_r), dtype=np.complex128)
        H0[0, 0] = -4.0 / (dr**2) + k0**2 * eps_eff[0]
        H0[0, 1] = 4.0 / (dr**2)
        for i in range(1, N_r - 1):
            H0[i, i-1] = 1.0 / (dr**2) - 1.0 / (2 * r[i] * dr)
            H0[i, i] = -2.0 / (dr**2) + k0**2 * eps_eff[i]
            H0[i, i+1] = 1.0 / (dr**2) + 1.0 / (2 * r[i] * dr)
        H0[-1, -2] = 1.0 / (dr**2) - 1.0 / (2 * r[-1] * dr)
        H0[-1, -1] = -2.0 / (dr**2) + k0**2 * eps_eff[-1]
        
        eigenvals0, eigenvecs0 = la.eig(H0)
        betas0 = np.sqrt(eigenvals0)
        idx_closest0 = np.argmin(np.abs(betas0 - beta0))
        
        # Store solved m=0
        betas[0, itcd, iwl] = betas0[idx_closest0]
        prof0 = eigenvecs0[:, idx_closest0]
        # Align phase
        idx_max = np.argmax(np.abs(prof0))
        phase_factor = prof0[idx_max] / np.abs(prof0[idx_max])
        profiles[0, itcd, iwl] = prof0 / phase_factor
        
        # 3. Solve for m = 1, 2, 3, 4
        for m in [1, 2, 3, 4]:
            Hm = np.copy(H0)
            Hm[0, 0] = -1e20 # Enforce psi(0) = 0 for m >= 1
            for i in range(1, N_r):
                Hm[i, i] -= (m**2) / (r[i]**2)
            
            eigenvals_m, eigenvecs_m = la.eig(Hm)
            betas_m = np.sqrt(eigenvals_m + 0j)
            
            # Use overlap integral to track the correct guided mode
            overlaps = np.abs(np.dot(eigenvecs_m.conj().T, prof0))
            idx_best_m = np.argmax(overlaps)
            
            betas[m, itcd, iwl] = betas_m[idx_best_m]
            
            prof_m = eigenvecs_m[:, idx_best_m]
            idx_max_m = np.argmax(np.abs(prof_m))
            phase_factor_m = prof_m[idx_max_m] / (np.abs(prof_m[idx_max_m]) + 1e-30)
            profiles[m, itcd, iwl] = prof_m / phase_factor_m

print("Saving multi-mode LUT to", OUT_LUT_PATH)
np.savez(OUT_LUT_PATH,
         r=r,
         tcd_grid=tcd_grid,
         wavelengths=wls,
         beta_m0=betas[0], beta_m1=betas[1], beta_m2=betas[2], beta_m3=betas[3], beta_m4=betas[4],
         profile_m0=profiles[0], profile_m1=profiles[1], profile_m2=profiles[2], profile_m3=profiles[3], profile_m4=profiles[4])
print("Done!")