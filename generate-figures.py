#!/usr/bin/env python3
"""
generate_figures.py — Generate all paper figures

Produces Figures 1-6 for:
    Parwez (2026). Microtubule Sub-Harmonics as Broadband Neural Source:
    A Cascaded Tissue Filter Model of the EEG Aperiodic Spectrum.

Usage:
    python generate_figures.py          # Figures 1-4 only (no data needed)
    python generate_figures.py --lemon  # All 6 figures (requires LEMON dataset)

Output:
    figures/figure1_source_spectrum.png
    figures/figure2_subband_gradient_source.png
    figures/figure3_ssd_bandwidth.png
    figures/figure4_consciousness_states.png
    figures/figure5_lemon_gradient.png      (--lemon only)
    figures/figure6_ec_eo_comparison.png    (--lemon only)

Author: Dr. Syed Mohsin Parwez
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Local imports
from mt_model import (
    healthy, ssd_bandwidth, light_sedation, moderate_sedation,
    general_anaesthesia, deep_anaesthesia,
    compute_harmonic_density, generate_synthetic_eeg,
    fit_specparam, fit_specparam_subbands, BandConfig, Condition,
)

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
})


# ═════════════════════════════════════════════════════════════════════
# Figure 1: Source spectrum — harmonic density + PSD
# ═════════════════════════════════════════════════════════════════════

def figure1():
    """
    Figure 1. Left (A): Harmonic density contributions from each MT band.
    Right (B): PSD of synthetic EEG with specparam fit.
    """
    print("  Figure 1: Source spectrum... ", end="", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Per-band harmonic density
    band_configs = {
        'kHz (k=1.0)': {'color': '#e74c3c', 'f_lo': 10e3, 'f_hi': 300e3, 'k': 1.0},
        'MHz (k=1.5)': {'color': '#3498db', 'f_lo': 10e6, 'f_hi': 230e6, 'k': 1.5},
        'GHz (k=2.0)': {'color': '#2ecc71', 'f_lo': 1e9,  'f_hi': 20e9,  'k': 2.0},
    }

    for label, cfg in band_configs.items():
        cond = Condition(
            name=label,
            bands=[BandConfig(cfg['f_lo'], cfg['f_hi'], cfg['k'], 5)]
        )
        bc, dens = compute_harmonic_density(cond)
        valid = dens > 0
        ax1.loglog(bc[valid], dens[valid], 'o-', color=cfg['color'],
                   label=label, markersize=3, alpha=0.7)

    # Combined
    h = healthy()
    bc, dens = compute_harmonic_density(h)
    valid = dens > 0
    ax1.loglog(bc[valid], dens[valid], 'k-', linewidth=2, label='Combined')

    # Fit line for reference
    log_f = np.log10(bc[valid])
    log_d = np.log10(dens[valid])
    slope, intercept = np.polyfit(log_f, log_d, 1)
    fit_line = 10**(intercept + slope * log_f)
    ax1.loglog(bc[valid], fit_line, 'k--', alpha=0.5,
               label=f'Fit: slope = {slope:.2f}')

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Harmonic density (count/Hz)')
    ax1.set_title('A. Harmonic Density by MT Band')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: PSD with specparam fit
    sig = generate_synthetic_eeg(h, fs=250, duration=180.0, seed=42)
    from scipy.signal import welch
    f_psd, psd = welch(sig, fs=250, nperseg=2048, noverlap=1024)

    result = fit_specparam(sig, fs=250)
    beta_val = result['exponent']

    mask = (f_psd >= 1) & (f_psd <= 50)
    ax2.semilogy(f_psd[mask], psd[mask], 'k-', alpha=0.7, label='Synthetic PSD')

    # Aperiodic fit line
    offset = result['offset']
    fit_freqs = f_psd[mask]
    fit_psd = 10**(offset - beta_val * np.log10(fit_freqs))
    ax2.semilogy(fit_freqs, fit_psd, 'r--', linewidth=2,
                 label=f'Aperiodic fit: $\\beta$ = {beta_val:.2f}')

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power ($\\mu V^2$/Hz)')
    ax2.set_title('B. Source PSD (specparam)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, 'figure1_source_spectrum.png')
    plt.savefig(outpath)
    plt.close()
    print(f"saved ({outpath})")


# ═════════════════════════════════════════════════════════════════════
# Figure 2: Sub-band gradient from source model
# ═════════════════════════════════════════════════════════════════════

def figure2():
    """
    Figure 2. Sub-band slope variation from synthetic MT source.
    """
    print("  Figure 2: Sub-band gradient (source)... ", end="", flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    h = healthy()
    sig = generate_synthetic_eeg(h, fs=250, duration=180.0, seed=42)
    sb = fit_specparam_subbands(sig, fs=250)

    # Also get global beta
    result = fit_specparam(sig, fs=250)
    global_beta = result['exponent']

    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_centers = [2.5, 6, 10.5, 21.5, 40]
    band_labels = ['$\\delta$\n(1.5-4)', '$\\theta$\n(4-8)', '$\\alpha$\n(8-13)',
                   '$\\beta$\n(13-30)', '$\\gamma$\n(30-50)']

    values = [sb.get(b) for b in band_names]
    valid_vals = [(c, v) for c, v in zip(band_centers, values) if v is not None]

    if valid_vals:
        x_vals, y_vals = zip(*valid_vals)
        ax.plot(x_vals, y_vals, 'bo-', markersize=10, linewidth=2, label='Sub-band $\\beta$')

        # Global beta reference
        ax.axhline(global_beta, color='red', linestyle='--', linewidth=1.5,
                   label=f'Global $\\beta$ = {global_beta:.2f}')

        for c, v, lbl in zip(band_centers, values, band_labels):
            if v is not None:
                ax.annotate(f'{v:.2f}', (c, v), textcoords="offset points",
                           xytext=(0, 12), ha='center', fontsize=10)

    ax.set_xticks(band_centers)
    ax.set_xticklabels(band_labels)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Aperiodic Exponent ($\\beta$)')
    ax.set_title('Sub-band Slope Variation — MT Source Model (specparam)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, 'figure2_subband_gradient_source.png')
    plt.savefig(outpath)
    plt.close()
    print(f"saved ({outpath})")


# ═════════════════════════════════════════════════════════════════════
# Figure 3: SSD bandwidth narrowing
# ═════════════════════════════════════════════════════════════════════

def figure3():
    """
    Figure 3. SSD Delta_beta as a function of bandwidth narrowing.
    """
    print("  Figure 3: SSD bandwidth... ", end="", flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get healthy baseline
    h = healthy()
    sig_h = generate_synthetic_eeg(h, fs=250, duration=180.0, seed=42)
    r_h = fit_specparam(sig_h, fs=250)
    beta_healthy = r_h['exponent']

    # Sweep bandwidth fractions
    bw_fractions = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    delta_betas = []

    for bw in bw_fractions:
        cond = ssd_bandwidth(bw)
        sig_s = generate_synthetic_eeg(cond, fs=250, duration=180.0, seed=42)
        r_s = fit_specparam(sig_s, fs=250)
        db = r_s['exponent'] - beta_healthy
        delta_betas.append(db)

    ax.plot([bw * 100 for bw in bw_fractions], delta_betas,
            'bo-', markersize=8, linewidth=2, label='Source model $\\Delta\\beta$')

    # Hasanaj reference
    hasanaj_val = 0.168
    hasanaj_ci = (0.056, 0.281)
    ax.axhline(hasanaj_val, color='red', linestyle='--', linewidth=1.5,
               label=f'Hasanaj et al. (2025): +{hasanaj_val:.3f}')
    ax.axhspan(hasanaj_ci[0], hasanaj_ci[1], color='red', alpha=0.1,
               label=f'95% CI [{hasanaj_ci[0]:.3f}, {hasanaj_ci[1]:.3f}]')

    # Mark 60% point
    idx_60 = bw_fractions.index(0.60)
    ax.plot(60, delta_betas[idx_60], 'r*', markersize=15, zorder=5)
    ax.annotate(f'60%: $\\Delta\\beta$ = {delta_betas[idx_60]:+.3f}',
               (60, delta_betas[idx_60]), textcoords="offset points",
               xytext=(-80, 15), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('MT Bandwidth (% of healthy)')
    ax.set_ylabel('$\\Delta\\beta$ (relative to healthy)')
    ax.set_title('SSD Source Mechanism: Bandwidth Narrowing')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, 'figure3_ssd_bandwidth.png')
    plt.savefig(outpath)
    plt.close()
    print(f"saved ({outpath})")


# ═════════════════════════════════════════════════════════════════════
# Figure 4: Consciousness state mapping
# ═════════════════════════════════════════════════════════════════════

def figure4():
    """
    Figure 4. Source-level consciousness state mapping.
    """
    print("  Figure 4: Consciousness states... ", end="", flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    states = [
        ("Healthy", healthy),
        ("Light\nSedation", light_sedation),
        ("Moderate\nSedation", moderate_sedation),
        ("General\nAnaesthesia", general_anaesthesia),
        ("Deep\nAnaesthesia", deep_anaesthesia),
    ]

    betas = []
    labels = []
    for name, cond_fn in states:
        cond = cond_fn()
        sig = generate_synthetic_eeg(cond, fs=250, duration=180.0, seed=42)
        r = fit_specparam(sig, fs=250)
        betas.append(r['exponent'])
        labels.append(name)

    x = np.arange(len(labels))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.bar(x, betas, color=colors, edgecolor='black', linewidth=0.5)

    for i, (xi, b) in enumerate(zip(x, betas)):
        ax.text(xi, b + 0.05, f'{b:.2f}', ha='center', va='bottom', fontsize=10)

    # Reference thresholds
    ax.axhline(1.0, color='green', linestyle=':', alpha=0.7, label='Awake threshold ($\\beta$ ~ 1.0)')
    ax.axhline(2.0, color='red', linestyle=':', alpha=0.7, label='Unconscious threshold ($\\beta$ ~ 2.0)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Aperiodic Exponent ($\\beta$)')
    ax.set_title('Source-Level Consciousness State Mapping')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, 'figure4_consciousness_states.png')
    plt.savefig(outpath)
    plt.close()
    print(f"saved ({outpath})")


# ═════════════════════════════════════════════════════════════════════
# Figure 5: LEMON sub-band gradient (requires data)
# ═════════════════════════════════════════════════════════════════════

def figure5(lemon_csv='data/lemon_results.csv'):
    """
    Figure 5. Sub-band gradient: cascade model vs LEMON observed.
    Requires pre-computed LEMON results CSV from analysis_lemon_full.py.
    """
    import pandas as pd

    print("  Figure 5: LEMON gradient... ", end="", flush=True)

    if not os.path.exists(lemon_csv):
        print(f"SKIPPED (no {lemon_csv} — run analysis_lemon_full.py first)")
        return

    df = pd.read_csv(lemon_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Sub-band profiles — observed vs cascade predictions
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_centers = [2.5, 6, 10.5, 21.5, 40]

    # Observed means
    obs_means = []
    for b in band_names:
        col = f'irasa_{b}'
        if col in df.columns:
            obs_means.append(df[col].mean())
        else:
            obs_means.append(np.nan)

    ax1.plot(band_centers, obs_means, 'rs-', markersize=10, linewidth=2,
             label=f'LEMON observed (N={len(df)})', zorder=5)

    # Cascade predictions from cascade_transition_analysis.py
    # These are the Table 2 values for selected parameterisations
    cascade_scenarios = {
        'Standard ($\\tau_m$=20ms)': [1.88, 2.58, 3.51, 4.63, 5.29],
        'Mild ($\\tau_m$=10ms)':     [1.74, 1.96, 2.44, 3.47, 4.56],
        'Weak ($\\tau_m$=8ms)':      [1.72, 1.86, 2.18, 3.05, 4.16],
    }
    cascade_colors = ['#2ecc71', '#3498db', '#e74c3c']

    for (label, vals), color in zip(cascade_scenarios.items(), cascade_colors):
        ax1.plot(band_centers, vals, '--o', color=color, markersize=5,
                 alpha=0.7, label=label)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Aperiodic Exponent ($\\beta$)')
    ax1.set_title('A. Sub-band Exponents: Model vs Observed')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel B: Gradient distribution
    if 'irasa_delta' in df.columns and 'irasa_gamma' in df.columns:
        gradients = df['irasa_gamma'] - df['irasa_delta']
        ax2.hist(gradients, bins=30, color='steelblue', edgecolor='black',
                 alpha=0.7, density=True)
        ax2.axvline(gradients.mean(), color='red', linewidth=2,
                    label=f'Mean = {gradients.mean():.2f} $\\pm$ {gradients.sem():.2f} (SE)')
        ax2.axvline(0, color='black', linestyle=':', alpha=0.5)

        # Cascade prediction lines
        ax2.axvline(2.82, color='#2ecc71', linestyle='--',
                    label='Mild cascade: +2.82')
        ax2.axvline(2.43, color='#e74c3c', linestyle='--',
                    label='Weak cascade: +2.43')

        ax2.set_xlabel('Gradient ($\\beta_\\gamma - \\beta_\\delta$)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'B. Per-Recording Gradients (N={len(gradients)})')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, 'figure5_lemon_gradient.png')
    plt.savefig(outpath)
    plt.close()
    print(f"saved ({outpath})")


# ═════════════════════════════════════════════════════════════════════
# Figure 6: EC vs EO comparison (requires data)
# ═════════════════════════════════════════════════════════════════════

def figure6(lemon_csv='data/lemon_results.csv'):
    """
    Figure 6. Eyes-closed vs eyes-open comparison.
    """
    import pandas as pd

    print("  Figure 6: EC vs EO... ", end="", flush=True)

    if not os.path.exists(lemon_csv):
        print(f"SKIPPED (no {lemon_csv} — run analysis_lemon_full.py first)")
        return

    df = pd.read_csv(lemon_csv)

    if 'condition' not in df.columns:
        print("SKIPPED (no 'condition' column in CSV)")
        return

    ec = df[df['condition'] == 'EC']
    eo = df[df['condition'] == 'EO']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Sub-band profiles EC vs EO
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_centers = [2.5, 6, 10.5, 21.5, 40]

    for subset, label, color in [(ec, f'EC (N={len(ec)})', 'blue'),
                                  (eo, f'EO (N={len(eo)})', 'green')]:
        means = []
        sems = []
        for b in band_names:
            col = f'irasa_{b}'
            if col in subset.columns:
                means.append(subset[col].mean())
                sems.append(subset[col].sem())
            else:
                means.append(np.nan)
                sems.append(np.nan)
        ax1.errorbar(band_centers, means, yerr=sems, fmt='o-', color=color,
                     markersize=8, linewidth=2, capsize=4, label=label)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Aperiodic Exponent ($\\beta$)')
    ax1.set_title('A. Sub-band Profiles: EC vs EO')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel B: Global beta distributions
    if 'irasa_global' in df.columns:
        ec_beta = ec['irasa_global'].dropna()
        eo_beta = eo['irasa_global'].dropna()

        bins = np.linspace(
            min(ec_beta.min(), eo_beta.min()) - 0.1,
            max(ec_beta.max(), eo_beta.max()) + 0.1,
            30
        )

        ax2.hist(ec_beta, bins=bins, alpha=0.5, color='blue', label=f'EC: {ec_beta.mean():.2f} $\\pm$ {ec_beta.std():.2f}')
        ax2.hist(eo_beta, bins=bins, alpha=0.5, color='green', label=f'EO: {eo_beta.mean():.2f} $\\pm$ {eo_beta.std():.2f}')

        from scipy.stats import ttest_ind
        t, p = ttest_ind(ec_beta, eo_beta)
        ax2.set_title(f'B. Global $\\beta$ (t={t:.2f}, p={p:.4f})')
        ax2.set_xlabel('Global $\\beta$ (IRASA)')
        ax2.set_ylabel('Count')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, 'figure6_ec_eo_comparison.png')
    plt.savefig(outpath)
    plt.close()
    print(f"saved ({outpath})")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING PAPER FIGURES")
    print("=" * 70)
    print(f"Output directory: {os.path.abspath(FIGDIR)}/\n")

    run_lemon = '--lemon' in sys.argv

    # Figures 1-4: No data required
    print("Figures 1-4 (model-only, no data required):")
    figure1()
    figure2()
    figure3()
    figure4()

    # Figures 5-6: Require LEMON results
    if run_lemon:
        print("\nFigures 5-6 (LEMON data required):")
        lemon_csv = 'data/lemon_results.csv'
        # Check common alternative paths
        for alt in ['data/lemon_results.csv', 'lemon_results.csv',
                    '../data/lemon_results.csv']:
            if os.path.exists(alt):
                lemon_csv = alt
                break
        figure5(lemon_csv)
        figure6(lemon_csv)
    else:
        print("\nFigures 5-6: SKIPPED (pass --lemon to generate; requires LEMON data)")

    print("\n" + "=" * 70)
    print("DONE.")
    generated = [f for f in os.listdir(FIGDIR) if f.endswith('.png')]
    print(f"Generated {len(generated)} figures in {FIGDIR}/:")
    for f in sorted(generated):
        print(f"  {f}")
    print("=" * 70)
