#!/usr/bin/env python3
"""
analysis_lemon_full.py — Full LEMON Cohort Validation (N=228)
==============================================================

Runs IRASA-based aperiodic analysis on the complete LEMON preprocessed
dataset. Compares sub-band gradients against the cascade transition-regime
predictions.

Key outputs:
  1. Global β distribution (EC and EO separately)
  2. Sub-band gradient (δ→γ) with confidence intervals
  3. Cascade model overlay (fc sweep predictions vs observed)
  4. EC vs EO comparison
  5. Publication-grade figures and CSV results

Prerequisites:
  pip install mne numpy scipy matplotlib yasa pandas

LEMON data:
  Preprocessed .set files from MPI Leipzig LEMON dataset.
  Expected location: ~/Downloads/lemon_full/ or data/lemon/
  Can also be .vhdr BrainVision files from OpenNeuro.

Usage:
  python analysis_lemon_full.py [--data-dir /path/to/lemon] [--max-subjects N]
"""

import os
import sys
import glob
import argparse
import time
import gc
import warnings
import traceback

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import welch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    sys.exit("ERROR: pip install mne")

try:
    import yasa
except ImportError:
    sys.exit("ERROR: pip install yasa")

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pip install pandas")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

SUB_BANDS = {
    'delta': (1.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),  # Conservative upper bound for IRASA
}

GLOBAL_RANGE = (1.0, 50.0)

# IRASA config — match the TDBRAIN analysis that produced clean results
IRASA_HSET = np.arange(1.1, 1.96, 0.05)  # 18 resampling factors
MAX_DURATION_SEC = 120  # Use first 120s of each recording
TARGET_SFREQ = 250.0   # Downsample if higher (speeds IRASA ~10x)


# ═══════════════════════════════════════════════════════════════════════
# CASCADE MODEL (from cascade_transition_analysis.py)
# ═══════════════════════════════════════════════════════════════════════

def cascade_model_subband_betas(fc_mem, fc_skull, source_alpha=1.633):
    """
    Compute predicted sub-band betas from the cascade model.
    
    P_EEG(f) = f^(-source_alpha) * 1/(1+(f/fc_mem)^2) * 1/(1+(f/fc_skull)^2)
    
    Returns dict of {band_name: predicted_beta}.
    """
    results = {}
    for band_name, (flo, fhi) in SUB_BANDS.items():
        freqs = np.linspace(flo, fhi, 500)
        source = freqs ** (-source_alpha)
        h_mem = 1.0 / (1.0 + (freqs / fc_mem) ** 2)
        h_skull = 1.0 / (1.0 + (freqs / fc_skull) ** 2)
        psd = source * h_mem * h_skull
        
        log_f = np.log10(freqs)
        log_p = np.log10(psd)
        slope, _, _, _, _ = sp_stats.linregress(log_f, log_p)
        results[band_name] = -slope
    return results


# ═══════════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════

def find_lemon_files(data_dir=None):
    """
    Search for LEMON EEG files. Supports .set (EEGLAB) and .vhdr (BrainVision).
    """
    search_paths = []
    
    if data_dir:
        search_paths.append(data_dir)
    
    # Common locations from previous sessions
    home = os.path.expanduser('~')
    search_paths.extend([
        os.path.join(home, 'Downloads', 'lemon_full'),
        os.path.join(home, 'Downloads', 'lemon_preprocessed'),
        'data/lemon/extracted',
        'data/lemon',
        'data_lemon',
    ])
    
    files = []
    for path in search_paths:
        if os.path.isdir(path):
            # .set files (EEGLAB format — LEMON preprocessed)
            sets = sorted(glob.glob(os.path.join(path, '**', '*.set'), recursive=True))
            if sets:
                files.extend(sets)
                break
            # .vhdr files (BrainVision — OpenNeuro format)
            vhdrs = sorted(glob.glob(os.path.join(path, '**', '*.vhdr'), recursive=True))
            if vhdrs:
                files.extend(vhdrs)
                break
    
    return files


def parse_condition(filepath):
    """Extract EC (eyes-closed) or EO (eyes-open) from filename."""
    fname = os.path.basename(filepath).upper()
    if '_EC' in fname or 'RESTEC' in fname or 'EC.' in fname:
        return 'EC'
    elif '_EO' in fname or 'RESTEO' in fname or 'EO.' in fname:
        return 'EO'
    return 'unknown'


def parse_subject_id(filepath):
    """Extract subject ID from filename."""
    fname = os.path.basename(filepath)
    # Try sub-XXXXXX pattern
    import re
    match = re.search(r'sub-(\d+)', fname)
    if match:
        return match.group(1)
    # Fallback: use filename stem
    return os.path.splitext(fname)[0]


# ═══════════════════════════════════════════════════════════════════════
# SINGLE-RECORDING PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def process_recording(filepath):
    """
    Load one EEG recording, run IRASA, return results dict.
    
    Returns None if processing fails.
    """
    try:
        fname = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        
        # Load
        if ext == '.set':
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        elif ext == '.vhdr':
            raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
        else:
            return None
        
        # Pick EEG channels only
        raw.pick_types(eeg=True, exclude='bads')
        n_channels = len(raw.ch_names)
        sfreq = raw.info['sfreq']
        
        # Truncate to MAX_DURATION_SEC
        max_samples = int(MAX_DURATION_SEC * sfreq)
        if raw.n_times > max_samples:
            raw.crop(tmax=MAX_DURATION_SEC)
        
        duration = raw.n_times / sfreq
        
        # Downsample if needed (IRASA is O(n²) in sample count)
        if sfreq > TARGET_SFREQ * 1.5:
            raw.resample(TARGET_SFREQ, verbose=False)
            sfreq = TARGET_SFREQ
        
        # Get data as (n_channels, n_samples)
        data = raw.get_data()
        
        # ─── Global IRASA ───
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            freqs, psd_aperiodic, psd_oscillatory = yasa.irasa(
                data, sf=sfreq, band=GLOBAL_RANGE,
                hset=IRASA_HSET, return_fit=False
            )
        
        # Average across channels
        if psd_aperiodic.ndim > 1:
            psd_aperiodic_mean = psd_aperiodic.mean(axis=0)
        else:
            psd_aperiodic_mean = psd_aperiodic
        
        # Fit global β via log-log regression
        mask = (freqs >= GLOBAL_RANGE[0]) & (freqs <= GLOBAL_RANGE[1])
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd_aperiodic_mean[mask])
        
        # Remove any inf/nan
        valid = np.isfinite(log_f) & np.isfinite(log_p)
        if valid.sum() < 5:
            return None
        
        slope, intercept, r, p, se = sp_stats.linregress(log_f[valid], log_p[valid])
        global_beta = -slope
        global_r2 = r ** 2
        
        # ─── Sub-band IRASA ───
        subband_betas = {}
        for band_name, (flo, fhi) in SUB_BANDS.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sb_freqs, sb_aperiodic, sb_osc = yasa.irasa(
                        data, sf=sfreq, band=(flo, fhi),
                        hset=IRASA_HSET, return_fit=False
                    )
                
                if sb_aperiodic.ndim > 1:
                    sb_ap_mean = sb_aperiodic.mean(axis=0)
                else:
                    sb_ap_mean = sb_aperiodic
                
                sb_mask = (sb_freqs >= flo) & (sb_freqs <= fhi)
                sb_log_f = np.log10(sb_freqs[sb_mask])
                sb_log_p = np.log10(sb_ap_mean[sb_mask])
                
                sb_valid = np.isfinite(sb_log_f) & np.isfinite(sb_log_p)
                if sb_valid.sum() >= 3:
                    sb_slope, _, _, _, _ = sp_stats.linregress(
                        sb_log_f[sb_valid], sb_log_p[sb_valid]
                    )
                    subband_betas[band_name] = -sb_slope
                else:
                    subband_betas[band_name] = np.nan
            except Exception:
                subband_betas[band_name] = np.nan
        
        condition = parse_condition(filepath)
        subject_id = parse_subject_id(filepath)
        
        result = {
            'file': fname,
            'subject_id': subject_id,
            'condition': condition,
            'n_channels': n_channels,
            'duration_s': round(duration, 1),
            'sfreq': sfreq,
            'global_beta': round(global_beta, 4),
            'global_r2': round(global_r2, 4),
        }
        for band_name in SUB_BANDS:
            result[f'beta_{band_name}'] = round(subband_betas.get(band_name, np.nan), 4)
        
        # Gradient
        if not np.isnan(subband_betas.get('gamma', np.nan)) and not np.isnan(subband_betas.get('delta', np.nan)):
            result['gradient'] = round(subband_betas['gamma'] - subband_betas['delta'], 4)
        else:
            result['gradient'] = np.nan
        
        # Cleanup
        del raw, data, psd_aperiodic, psd_oscillatory
        gc.collect()
        
        return result
    
    except Exception as e:
        print(f"    ERROR processing {os.path.basename(filepath)}: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════

def create_gradient_robustness_figure(df, output_dir='.'):
    """
    THE key figure: cascade sweep vs observed gradient.
    Shows that the positive gradient is a structural consequence of
    physiological filtering, not a tuned model prediction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ─── Panel A: Sub-band β across cascade scenarios ───
    ax = axes[0]
    
    band_names = list(SUB_BANDS.keys())
    band_centers = [np.mean(v) for v in SUB_BANDS.values()]
    
    # Cascade scenarios
    scenarios = [
        (5, 10, 'Strong ($\\tau_m$=32ms)', 'C0', 0.3),
        (10, 20, 'Moderate ($\\tau_m$=16ms)', 'C1', 0.5),
        (16, 30, 'Standard ($\\tau_m$=10ms)', 'C2', 1.0),
        (20, 40, 'Weak ($\\tau_m$=8ms)', 'C3', 0.5),
        (30, 50, 'Very weak', 'C4', 0.3),
    ]
    
    for fc_m, fc_s, label, color, alpha in scenarios:
        model_betas = cascade_model_subband_betas(fc_m, fc_s)
        vals = [model_betas[b] for b in band_names]
        ax.plot(band_centers, vals, 'o--', color=color, alpha=alpha,
                label=f'{label}', markersize=4)
    
    # Observed LEMON data
    observed_means = []
    observed_sems = []
    for band_name in band_names:
        col = f'beta_{band_name}'
        if col in df.columns:
            vals = df[col].dropna()
            observed_means.append(vals.mean())
            observed_sems.append(vals.sem())
        else:
            observed_means.append(np.nan)
            observed_sems.append(np.nan)
    
    ax.errorbar(band_centers, observed_means, yerr=observed_sems,
                fmt='s-', color='red', linewidth=2, markersize=8,
                capsize=4, label=f'LEMON (N={len(df)})', zorder=10)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Aperiodic exponent $\\hat{\\beta}$')
    ax.set_title('A. Sub-band gradient: cascade model vs observed')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xscale('log')
    ax.set_xticks([2, 6, 10, 20, 38])
    ax.set_xticklabels(['δ(2)', 'θ(6)', 'α(10)', 'β(20)', 'γ(38)'])
    ax.grid(True, alpha=0.3)
    
    # ─── Panel B: Gradient distribution ───
    ax = axes[1]
    
    gradients = df['gradient'].dropna()
    
    ax.hist(gradients, bins=25, alpha=0.7, color='steelblue',
            edgecolor='black', density=True, label='LEMON observed')
    
    mean_g = gradients.mean()
    sem_g = gradients.sem()
    ci95 = 1.96 * sem_g
    
    ax.axvline(mean_g, color='red', linewidth=2,
               label=f'Mean = {mean_g:.2f} ± {sem_g:.2f}')
    ax.axvspan(mean_g - ci95, mean_g + ci95, alpha=0.2, color='red',
               label='95% CI')
    ax.axvline(0, color='black', linestyle=':', linewidth=1,
               label='Zero (no gradient)')
    
    # Cascade predictions for standard scenarios
    for fc_m, fc_s, label, color, _ in [(16, 30, 'Standard', 'C2', 1.0),
                                         (20, 40, 'Weak', 'C3', 0.5)]:
        model_betas = cascade_model_subband_betas(fc_m, fc_s)
        pred_gradient = model_betas['gamma'] - model_betas['delta']
        ax.axvline(pred_gradient, color=color, linestyle='--', linewidth=1.5,
                   label=f'Cascade {label}: {pred_gradient:.2f}')
    
    ax.set_xlabel('Sub-band gradient ($\\hat{\\beta}_{\\gamma} - \\hat{\\beta}_{\\delta}$)')
    ax.set_ylabel('Density')
    ax.set_title('B. Gradient distribution')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'fig_gradient_robustness.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath


def create_ec_eo_figure(df, output_dir='.'):
    """EC vs EO comparison across sub-bands."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    band_names = list(SUB_BANDS.keys())
    band_centers = [np.mean(v) for v in SUB_BANDS.values()]
    
    for condition, color, marker in [('EC', 'blue', 's'), ('EO', 'green', 'o')]:
        sub = df[df['condition'] == condition]
        if len(sub) == 0:
            continue
        
        means = []
        sems = []
        for band_name in band_names:
            col = f'beta_{band_name}'
            vals = sub[col].dropna()
            means.append(vals.mean())
            sems.append(vals.sem())
        
        axes[0].errorbar(band_centers, means, yerr=sems,
                         fmt=f'{marker}-', color=color, linewidth=1.5,
                         markersize=6, capsize=3,
                         label=f'{condition} (N={len(sub)})')
    
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('$\\hat{\\beta}$')
    axes[0].set_title('A. Sub-band profile: EC vs EO')
    axes[0].legend()
    axes[0].set_xscale('log')
    axes[0].set_xticks([2, 6, 10, 20, 38])
    axes[0].set_xticklabels(['δ', 'θ', 'α', 'β', 'γ'])
    axes[0].grid(True, alpha=0.3)
    
    # Global β comparison
    ec_beta = df[df['condition'] == 'EC']['global_beta'].dropna()
    eo_beta = df[df['condition'] == 'EO']['global_beta'].dropna()
    
    data_to_plot = []
    labels = []
    if len(ec_beta) > 0:
        data_to_plot.append(ec_beta.values)
        labels.append(f'EC (N={len(ec_beta)})')
    if len(eo_beta) > 0:
        data_to_plot.append(eo_beta.values)
        labels.append(f'EO (N={len(eo_beta)})')
    
    if data_to_plot:
        bp = axes[1].boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen']
        for patch, col in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(col)
        
        if len(ec_beta) > 0 and len(eo_beta) > 0:
            t_stat, p_val = sp_stats.ttest_ind(ec_beta, eo_beta)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            axes[1].set_title(f'B. Global β: EC vs EO (p={p_val:.4f} {sig})')
        else:
            axes[1].set_title('B. Global β by condition')
    
    axes[1].set_ylabel('Global aperiodic exponent $\\beta$')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'fig_ec_eo_comparison.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Full LEMON IRASA validation')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to LEMON .set or .vhdr files')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Limit number of files to process (for testing)')
    parser.add_argument('--output-dir', type=str, default='lemon_full_results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ─── Find files ───
    files = find_lemon_files(args.data_dir)
    
    if not files:
        print("ERROR: No LEMON files found.")
        print("Searched: ~/Downloads/lemon_full/, data/lemon/, etc.")
        print("Use --data-dir /path/to/lemon to specify location.")
        sys.exit(1)
    
    print("=" * 70)
    print("LEMON FULL COHORT VALIDATION")
    print("=" * 70)
    print(f"\n  Found {len(files)} files")
    print(f"  File type: {os.path.splitext(files[0])[1]}")
    print(f"  First: {os.path.basename(files[0])}")
    print(f"  Last:  {os.path.basename(files[-1])}")
    
    if args.max_subjects:
        files = files[:args.max_subjects]
        print(f"  Limited to {len(files)} files (--max-subjects)")
    
    # Count EC/EO
    conditions = [parse_condition(f) for f in files]
    n_ec = conditions.count('EC')
    n_eo = conditions.count('EO')
    n_unk = conditions.count('unknown')
    print(f"  EC: {n_ec}, EO: {n_eo}, unknown: {n_unk}")
    print(f"\n  IRASA config: hset={len(IRASA_HSET)} factors, max {MAX_DURATION_SEC}s, target {TARGET_SFREQ} Hz")
    print()
    
    # ─── Process all files ───
    results = []
    t_start = time.time()
    
    for i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        cond = parse_condition(filepath)
        
        t0 = time.time()
        sys.stdout.write(f"  [{i+1}/{len(files)}] {fname[:50]}... ")
        sys.stdout.flush()
        
        result = process_recording(filepath)
        
        elapsed = time.time() - t0
        
        if result:
            results.append(result)
            print(f"β={result['global_beta']:.3f}, {result['n_channels']}ch, "
                  f"{result['duration_s']:.0f}s, {cond} ({elapsed:.1f}s)")
        else:
            print(f"FAILED ({elapsed:.1f}s)")
        
        # Progress estimate
        if (i + 1) % 10 == 0:
            elapsed_total = time.time() - t_start
            rate = (i + 1) / elapsed_total
            remaining = (len(files) - i - 1) / rate
            print(f"  --- {i+1}/{len(files)} done, "
                  f"~{remaining/60:.1f} min remaining ---")
    
    total_time = time.time() - t_start
    
    if not results:
        print("\nERROR: No recordings processed successfully.")
        sys.exit(1)
    
    # ─── Create DataFrame ───
    df = pd.DataFrame(results)
    
    # Save raw results
    csv_path = os.path.join(args.output_dir, 'lemon_full_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STATISTICAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    
    n_subjects = df['subject_id'].nunique()
    n_recordings = len(df)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Recordings processed: {n_recordings}/{len(files)} "
          f"({n_recordings/len(files)*100:.0f}% success)")
    print(f"  Unique subjects: {n_subjects}")
    print(f"  Total processing time: {total_time/60:.1f} min "
          f"({total_time/n_recordings:.1f}s per recording)")
    
    # ─── Global β ───
    print(f"\n{'─'*70}")
    print("GLOBAL APERIODIC EXPONENT (IRASA, 1-50 Hz)")
    print(f"{'─'*70}")
    
    global_beta = df['global_beta'].dropna()
    print(f"\n  All recordings:  β = {global_beta.mean():.3f} ± {global_beta.std():.3f} "
          f"(SEM={global_beta.sem():.3f}, N={len(global_beta)})")
    
    for cond in ['EC', 'EO']:
        sub = df[df['condition'] == cond]['global_beta'].dropna()
        if len(sub) > 0:
            print(f"  {cond}:              β = {sub.mean():.3f} ± {sub.std():.3f} "
                  f"(SEM={sub.sem():.3f}, N={len(sub)})")
    
    # EC vs EO test
    ec_betas = df[df['condition'] == 'EC']['global_beta'].dropna()
    eo_betas = df[df['condition'] == 'EO']['global_beta'].dropna()
    if len(ec_betas) > 5 and len(eo_betas) > 5:
        t_stat, p_val = sp_stats.ttest_ind(ec_betas, eo_betas)
        print(f"\n  EC vs EO: t={t_stat:.3f}, p={p_val:.4f} "
              f"(Δ={ec_betas.mean()-eo_betas.mean():+.3f})")
        print(f"  Expected: EC > EO (literature consensus)")
        if ec_betas.mean() > eo_betas.mean():
            print(f"  ✓ Direction matches literature")
        else:
            print(f"  ✗ Direction reversed — investigate")
    
    # ─── Sub-band gradient ───
    print(f"\n{'─'*70}")
    print("SUB-BAND GRADIENT (IRASA)")
    print(f"{'─'*70}")
    
    print(f"\n  {'Band':<10} {'Mean β':>10} {'SD':>10} {'SEM':>10} {'N':>6}")
    print(f"  {'─'*46}")
    
    band_means = {}
    for band_name in SUB_BANDS:
        col = f'beta_{band_name}'
        vals = df[col].dropna()
        band_means[band_name] = vals.mean()
        print(f"  {band_name:<10} {vals.mean():>10.3f} {vals.std():>10.3f} "
              f"{vals.sem():>10.3f} {len(vals):>6}")
    
    gradient = df['gradient'].dropna()
    print(f"\n  Gradient (γ−δ): {gradient.mean():+.3f} ± {gradient.std():.3f} "
          f"(SEM={gradient.sem():.3f}, N={len(gradient)})")
    
    # One-sample t-test: gradient > 0?
    if len(gradient) > 5:
        t_stat, p_val = sp_stats.ttest_1samp(gradient, 0)
        p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        print(f"  One-sided t-test (gradient > 0): t={t_stat:.3f}, p={p_one_sided:.2e}")
        
        if p_one_sided < 0.001 and gradient.mean() > 0:
            print(f"  ✓✓✓ GRADIENT IS ROBUSTLY POSITIVE (p < 0.001)")
        elif p_one_sided < 0.05 and gradient.mean() > 0:
            print(f"  ✓ Gradient is positive (p < 0.05)")
        else:
            print(f"  ✗ Gradient is NOT significantly positive")
    
    # ─── Cascade model comparison ───
    print(f"\n{'─'*70}")
    print("CASCADE MODEL COMPARISON")
    print(f"{'─'*70}")
    
    scenarios = [
        (10, 20, 'Moderate'),
        (16, 30, 'Standard'),
        (20, 40, 'Weak'),
    ]
    
    print(f"\n  {'Scenario':<25} {'Pred gradient':>15} {'Obs gradient':>15} {'Match':>8}")
    print(f"  {'─'*63}")
    
    for fc_m, fc_s, label in scenarios:
        pred_betas = cascade_model_subband_betas(fc_m, fc_s)
        pred_grad = pred_betas['gamma'] - pred_betas['delta']
        obs_grad = gradient.mean()
        match = "✓" if abs(pred_grad - obs_grad) < 1.5 else "~"
        print(f"  fc_mem={fc_m}, fc_skull={fc_s} ({label:<10}) "
              f"{pred_grad:>+15.3f} {obs_grad:>+15.3f} {match:>8}")
    
    # ─── Comparison with N=10 results ───
    print(f"\n{'─'*70}")
    print("COMPARISON WITH PREVIOUS N=10 RESULTS")
    print(f"{'─'*70}")
    
    print(f"\n  {'Metric':<30} {'N=10':>12} {'N={0}'.format(n_recordings):>12} {'Δ':>10}")
    print(f"  {'─'*64}")
    
    prev = {
        'Global β': (1.705, global_beta.mean()),
        'Delta β': (0.55, band_means.get('delta', np.nan)),
        'Gamma β': (4.21, band_means.get('gamma', np.nan)),
        'Gradient': (3.66, gradient.mean()),
    }
    
    for metric, (old, new) in prev.items():
        if not np.isnan(new):
            print(f"  {metric:<30} {old:>12.3f} {new:>12.3f} {new-old:>+10.3f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    
    grad_positive = gradient.mean() > 0 and (len(gradient) < 6 or 
                     sp_stats.ttest_1samp(gradient, 0)[1]/2 < 0.05)
    ec_gt_eo = (len(ec_betas) > 5 and len(eo_betas) > 5 and 
                ec_betas.mean() > eo_betas.mean())
    
    verdicts = {
        'Sub-band gradient positive': grad_positive,
        'EC > EO direction': ec_gt_eo if len(ec_betas) > 5 else None,
        'Monotonic δ→γ increase': (
            band_means.get('delta', 0) < band_means.get('theta', 0) <
            band_means.get('alpha', 0) < band_means.get('beta', 0) <
            band_means.get('gamma', 0)
        ) if all(b in band_means for b in SUB_BANDS) else None,
    }
    
    for test, passed in verdicts.items():
        if passed is None:
            symbol = '—'
        elif passed:
            symbol = '✓'
        else:
            symbol = '✗'
        print(f"  {symbol} {test}")
    
    print(f"\n  N={n_recordings} recordings from {n_subjects} subjects")
    print(f"  This is {'PUBLICATION-GRADE' if n_subjects >= 50 else 'PRELIMINARY'} "
          f"statistical power.")
    
    # ═══════════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n{'─'*70}")
    print("GENERATING FIGURES")
    print(f"{'─'*70}\n")
    
    create_gradient_robustness_figure(df, args.output_dir)
    create_ec_eo_figure(df, args.output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"  Output directory: {args.output_dir}/")
    print(f"  CSV results:      lemon_full_results.csv")
    print(f"  Gradient figure:  fig_gradient_robustness.png")
    print(f"  EC/EO figure:     fig_ec_eo_comparison.png")


if __name__ == '__main__':
    main()
