import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# CPC Journal Style — Elsevier Q1 standards
DPI = 600
W_SINGLE = 88 / 25.4       # 3.46 inch (single column width)
W_DOUBLE = 180 / 25.4      # 7.09 inch (double column width)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': DPI,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Academic Palette
DEEP_TEAL = '#005F73'
CRIMSON  = '#8C2131'
AMBER    = '#D48C29'
STEEL    = '#4B5C6B'
MGRAY    = '#A0AAB2'
COL_PROPOSED = '#1C4587'

# Resolve root directory dynamically so scripts execute seamlessly on any system
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_project_root(start_dir):
    cur = os.path.abspath(start_dir)
    for _ in range(5):
        if os.path.exists(os.path.join(cur, "data")) and os.path.exists(os.path.join(cur, "elsarticle_template")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(start_dir, ".."))

DIR = find_project_root(SCRIPT_DIR)
DATA_DIR = os.path.join(DIR, "data")
LUT_PATH = os.path.join(DATA_DIR, "tsv_physics_lut_multimode.npz")
FIGS_DIR = os.path.join(DIR, "docs", "figs")
TEMPLATE_FIGS_DIR = os.path.join(DIR, "elsarticle_template", "elsarticle", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(TEMPLATE_FIGS_DIR, exist_ok=True)

def savefig(name):
    for fdir in [FIGS_DIR, TEMPLATE_FIGS_DIR]:
        path = os.path.join(fdir, name)
        plt.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved {name} to both directories")


# -------------------------------------------------------------------------
# Figure 2: Multimode Intensity Profiles
# -------------------------------------------------------------------------
def fig2_multimode_profiles():
    if not os.path.exists(LUT_PATH):
        print(f"LUT not found at {LUT_PATH}")
        return
        
    lut = np.load(LUT_PATH)
    wls = lut['wavelengths']
    r_fdm = lut['r']
    tcd_idx = 4
    wl_targets = [300.0, 400.0]
    wl_indices = [np.argmin(np.abs(wls - wt)) for wt in wl_targets]
    modes = [0, 1, 2, 3, 4]
    
    fig, axes = plt.subplots(1, 5, figsize=(W_DOUBLE, 1.9), sharey=True)
    axes_flat = axes.flatten()
    
    for ax, m in zip(axes_flat, modes):
        profile = lut[f'profile_m{m}'][tcd_idx]
        for wi, (wl_idx, wt) in enumerate(zip(wl_indices, wl_targets)):
            psi = np.abs(profile[wl_idx])**2
            psi = psi / (np.max(psi) + 1e-12)
            col = DEEP_TEAL if wi == 0 else AMBER
            ls  = '-' if wi == 0 else '--'
            lbl = f'$\\lambda = {int(wt)}\\,$nm'
            if m == 0:
                ax.plot(r_fdm, psi, color=col, ls=ls, lw=1.8, label=lbl)
            else:
                ax.plot(r_fdm, psi, color=col, ls=ls, lw=1.8)
                
        ax.axvline(150, color='gray', linestyle=':', lw=0.9)
        ax.axvline(175, color='black', linestyle=':', lw=0.9)
        if m == 0:
            ax.plot([], [], color='gray', linestyle=':', lw=0.9, label='Ref. radius ($r=150\\,$nm)')
            ax.plot([], [], color='black', linestyle=':', lw=0.9, label='$\\mathrm{SiO}_2$/Si interface ($r=175\\,$nm)')
            
        ax.set_title(f'Mode $m={m}$', fontsize=10.5, fontweight='bold', pad=3)
        ax.set_xlim(0, 300)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlabel('$r$ (nm)', fontsize=10)
        if m == 0:
            ax.set_ylabel('Normalized $|\\Psi_m|^2$ (log)', fontsize=10)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0, 150, 300])
        ax.tick_params(axis='both', labelsize=9)
        
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=4, frameon=False, fontsize=8.5, columnspacing=1.4, handlelength=1.5)
    
    plt.subplots_adjust(top=0.83, bottom=0.17, left=0.08, right=0.98, wspace=0.18)
    
    savefig('fig2_multimode_profiles.pdf')
    plt.close()


# -------------------------------------------------------------------------
# Figure 4: Cross-Wavelength Phase Coherence (CWPC) Matrix (Real FDTD Data)
# -------------------------------------------------------------------------
def fig4_cwpc_heatmap():
    data_path = os.path.join(DATA_DIR, 'real_cwpc_matrix.npz')
    data = np.load(data_path)
    Z = data['cwpc_nominal'] # Real CWPC phase matrix computed directly from FDTD near-field dataset
    wls = data['wavelengths']
    
    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.25, W_SINGLE * 1.15), constrained_layout=True)
    im = ax.imshow(Z, extent=[wls.min(), wls.max(), wls.max(), wls.min()], 
                   cmap='viridis', aspect='equal', vmin=0, vmax=np.pi, rasterized=True)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_ticks([0, np.pi/2, np.pi])
    cbar.set_ticklabels(['0', r'$\pi/2$', r'$\pi$'])
    cbar.set_label(r'$|\Delta\phi|$ (rad)', fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    
    ax.invert_yaxis()
    ax.set_xlabel(r'$\lambda_b$ (nm)', fontsize=10)
    ax.set_ylabel(r'$\lambda_a$ (nm)', fontsize=10)
    ax.set_title('Cross-Wavelength Phase Coherence (CWPC)', fontsize=10, pad=8)
    
    ax.tick_params(axis='both', which='both', labelsize=9)
    ax.set_xticks([250, 300, 350, 400, 450])
    ax.set_yticks([250, 300, 350, 400, 450])
    
    savefig('fig4_cwpc_heatmap.pdf')
    plt.close()


# -------------------------------------------------------------------------
# Figure 5: Learning Curve
# -------------------------------------------------------------------------
def fig5_learning_curve():
    data_path = os.path.join(DATA_DIR, 'real_learning_curve.npz')
    data = np.load(data_path)
    N_sizes = data['N_sizes']
    train_mae = data['train_mae']
    val_mae = data['val_mae']

    fig, ax = plt.subplots(figsize=(6.5, 2.7), constrained_layout=True)

    ax.plot(N_sizes, train_mae, '-o', lw=1.8, markersize=5.0, color='#D55E00',
            label='Train MAE ($\\Delta h_{\\mathrm{residual}}$)')

    ax.plot(N_sizes, val_mae, '--s', lw=1.8, markersize=5.0, color='#56B4E9',
            label='Validation MAE ($\\Delta h_{\\mathrm{residual}}$)')

    final_tr = train_mae[-1]
    final_vl = val_mae[-1]
    
    # Position annotations cleanly without white background box directly near final points
    ax.annotate(f'Val $\\approx {final_vl:.1f}$ nm', xy=(1629, final_vl), xytext=(1480, 22.0),
                arrowprops=dict(arrowstyle='->', lw=1.0, color='#1A5276'),
                fontsize=9.5, color='#1A5276', fontweight='semibold')
    ax.annotate(f'Train $\\approx {final_tr:.1f}$ nm', xy=(1629, final_tr), xytext=(1280, 12.0),
                arrowprops=dict(arrowstyle='->', lw=1.0, color='#9C3D00'),
                fontsize=9.5, color='#9C3D00', fontweight='semibold')

    ax.set_xlabel('Training sample size $N$', fontsize=10.5)
    ax.set_ylabel('MAE (nm)', fontsize=10.5)
    ax.set_xlim(50, 1750)
    ax.set_ylim(0, 68)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    
    # Low-opacity horizontal shaded region across 6.1-6.5 nm
    ax.axhspan(6.1, 6.5, color='lightgray', alpha=0.35, lw=0)
    
    ax.tick_params(axis='both', which='major', labelsize=9.5)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color='#E0E4E8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, fontsize=9.5, framealpha=0.9, edgecolor='0.85', borderpad=0.4)

    savefig('fig5_learning_curve.pdf')
    plt.close()


# -------------------------------------------------------------------------
# Figure 6: Wavelength Subsampling Sweep (Spectral Efficiency)
# -------------------------------------------------------------------------
def fig6_wavelength_sweep():
    # Spectral efficiency sweep data
    wl = list(range(5, 101, 5))
    sum_mape = [3.598, 2.541, 2.409, 2.059, 1.599, 1.583, 1.903, 1.259, 1.094, 1.150, 1.041, 1.911, 2.034, 1.740, 1.628, 1.652, 1.772, 1.880, 1.928, 2.066]
                
    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.5, 2.6))
    
    ax.plot(wl, sum_mape, 'D-', color='black', mfc=AMBER, lw=1.2, markersize=4, label='VSCI')
    
    opt_idx = wl.index(55)
    opt_val = sum_mape[opt_idx]
    ax.plot(55, opt_val, '*', color='#2C5F2D', ms=10, mec='#1F4520', mew=0.8, zorder=10)
    
    ax.annotate(f'Optimal: 55 WL\n({opt_val:.3f}%)', xy=(55, opt_val), xytext=(55, 2.2),
                arrowprops=dict(arrowstyle='-', color='#2C5F2D', lw=0.8),
                fontsize=7, color='#2C5F2D', ha='center', fontweight='bold')
                
    ax.set_xlabel('Number of sampled wavelengths $N_\\lambda$')
    ax.set_ylabel('Sum MAPE (%)')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 4.0)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 1, 2, 3, 4])
    
    ax.legend(loc='upper right', framealpha=0.92, frameon=False)
    ax.grid(axis='y', color='#E0E4E8', lw=0.3, alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    savefig('fig6_wavelength_sweep.pdf')
    plt.close()


# -------------------------------------------------------------------------
# Figure 7: Sub-Pixel Jitter Robustness
# -------------------------------------------------------------------------
def fig7_jitter_stress():
    modes = [0, 1, 2, 3, 4]
    p_mean = [41235.1, 0.45, 7.82, 0.22, 10.45]
    v_rel = [0.00000, 25.342, 0.00025, 30.940, 0.00002]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_SINGLE, 4.5), constrained_layout=False)
    fig.subplots_adjust(hspace=0.5, left=0.15, right=0.95, bottom=0.1, top=0.95)
    
    bars1 = ax1.bar(modes, p_mean, color=[DEEP_TEAL, AMBER, DEEP_TEAL, AMBER, DEEP_TEAL], width=0.6, edgecolor='white')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-2, 1e5)
    ax1.set_xticks(modes)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax1.set_xlabel('Azimuthal harmonic $m$', fontsize=9)
    ax1.set_ylabel('Mean modal power (a.u., log)', fontsize=9)
    ax1.set_title('(a) Modal power $|P_m|$ (log scale)', fontsize=10, fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    
    import matplotlib.patches as mpatches
    p_patch = mpatches.Patch(facecolor=DEEP_TEAL, edgecolor='white', label='Physical mode')
    w_patch = mpatches.Patch(facecolor=AMBER, edgecolor='white', label='Weak mode')
    ax1.legend(handles=[p_patch, w_patch], loc='upper right', fontsize=8, frameon=False)
    
    bars2 = ax2.bar(modes, v_rel, color=[DEEP_TEAL, AMBER, DEEP_TEAL, AMBER, DEEP_TEAL], width=0.5, edgecolor='white')
    ax2.set_ylim(0, 35)
    ax2.set_xticks(modes)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    ax2.set_xlabel('Azimuthal harmonic $m$', fontsize=9)
    ax2.set_ylabel('Relative variance (%)', fontsize=9)
    ax2.set_title('(b) Jitter-induced relative variance (%)', fontsize=10, fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)
    
    ax2.axhline(0.05, color=CRIMSON, linestyle='--', lw=2, zorder=0)
    
    import matplotlib.lines as mlines
    line = mlines.Line2D([], [], color=CRIMSON, linestyle='--', lw=2, label='Noise floor: 0.05%')
    ax2.legend(handles=[line], loc='upper left', fontsize=8, frameon=False)
    
    for i, v in enumerate(v_rel):
        if v == 0:
            label = "0%"
        elif v < 1:
            exp = int(np.floor(np.log10(v)))
            coeff = v / 10**exp
            label = f"${coeff:.1f} \\times 10^{{{exp}}}\\%$"
        else:
            label = f"{v:.1f}%"
        ax2.text(i, v + 0.5, label, ha='center', va='bottom', fontsize=8, color='#2C3E50')
        
    savefig('fig7_jitter_stress.pdf')
    plt.close()


# -------------------------------------------------------------------------
# Main Execution: Generate all figures in sequential order
# -------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating CPC Q1 Style Figures (8pt, B&W-safe)...')
    fig2_multimode_profiles()
    fig4_cwpc_heatmap()
    fig5_learning_curve()
    fig6_wavelength_sweep()
    fig7_jitter_stress()
    print('Done!')