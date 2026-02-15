#!/usr/bin/env python3
"""
analysis_amplitude_decay.py — Test amplitude decay models for mt_model.py

The current mt_model.py (line 271) assigns uniform amplitude=1.0 to all
harmonics. This is physically unrealistic — real MT resonances should
decay with harmonic number. This script monkey-patches generate_synthetic_eeg
to apply decay weights, then checks how β changes.

Decay models tested:
    1. Uniform (current):     A_n = 1.0
    2. 1/n^α for α ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
    3. Exponential:           A_n = exp(-n/τ) for τ ∈ {10, 20, 50}

For each decay model, we generate synthetic EEG, fit specparam, and compare
to the LEMON-observed β ≈ 1.95 (full cohort, N=383).

Usage:
    cd mt_eeg_toolkit
    python analysis_amplitude_decay.py
"""

import sys
import numpy as np
from scipy.signal import welch
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
try:
    from mt_model import Condition, BandConfig, healthy
    from specparam import SpectralModel
except ImportError as e:
    print(f"ERROR: {e}")
    print("Run this script from the mt_eeg_toolkit directory.")
    sys.exit(1)


# ─── Modified EEG generator with amplitude decay ────────────────────────

def generate_eeg_with_decay(condition, decay_func, fs=250, duration=180.0,
                            max_harmonics_per_source=800, seed=42):
    """
    Identical to mt_model.generate_synthetic_eeg (lines 220-276) but with
    amplitude weighting applied at line 271.

    decay_func(n) → amplitude weight, where n is the sub-harmonic index
    (i.e. the same 'ns' array from mt_model.py line 259).
    """
    rng = np.random.RandomState(seed)
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signal = np.zeros(n_samples)

    for band in condition.bands:
        if band.n_sources == 0:
            continue
        sources = np.logspace(
            np.log10(band.f_lo), np.log10(band.f_hi), max(1, band.n_sources)
        )
        for src in sources:
            n_max = min(50000, int(src / 0.1))
            if n_max < 1:
                continue
            ns = np.arange(1, n_max + 1)
            freqs = src / (ns ** band.k)
            mask = (freqs >= 0.5) & (freqs <= fs / 2 - 1)
            valid_freqs = freqs[mask]
            valid_ns = ns[mask]  # Keep track of harmonic numbers

            # Subsample for computational tractability
            if len(valid_freqs) > max_harmonics_per_source:
                idx = rng.choice(len(valid_freqs), max_harmonics_per_source, replace=False)
                valid_freqs = valid_freqs[idx]
                valid_ns = valid_ns[idx]

            # Apply amplitude decay based on sub-harmonic number
            for i, f in enumerate(valid_freqs):
                phase = rng.uniform(0, 2 * np.pi)
                n = valid_ns[i]
                amp = decay_func(n)
                signal += amp * np.sin(2 * np.pi * f * t + phase)

    # Add small noise floor (same as mt_model.py line 274)
    if np.std(signal) > 0:
        signal += rng.normal(0, 0.01 * np.std(signal), n_samples)

    return signal


def fit_beta(signal, fs=250, freq_range=(1, 50)):
    """Fit specparam and return global β."""
    f, psd = welch(signal, fs=fs, nperseg=2048, noverlap=1024)
    sm = SpectralModel(aperiodic_mode='fixed', min_peak_height=0.3,
                       max_n_peaks=6, peak_width_limits=[1, 12])
    sm.fit(f, psd, list(freq_range))
    return float(sm.get_params('aperiodic')[1])


def fit_subbands(signal, fs=250):
    """Fit specparam per sub-band."""
    f, psd = welch(signal, fs=fs, nperseg=2048, noverlap=1024)
    sub_bands = {
        'delta': (1.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'beta': (13, 30), 'gamma': (30, 50),
    }
    results = {}
    for name, (flo, fhi) in sub_bands.items():
        sm = SpectralModel(aperiodic_mode='fixed', min_peak_height=0.5, max_n_peaks=2)
        try:
            sm.fit(f, psd, [flo, fhi])
            results[name] = float(sm.get_params('aperiodic')[1])
        except Exception:
            results[name] = np.nan
    return results


# ─── Decay models ────────────────────────────────────────────────────────

DECAY_MODELS = OrderedDict([
    ('uniform',     {'func': lambda n: 1.0,               'label': 'Uniform (current)'}),
    ('1/n^0.25',    {'func': lambda n: 1.0 / (n ** 0.25), 'label': '1/n^0.25'}),
    ('1/√n',        {'func': lambda n: 1.0 / np.sqrt(n),  'label': '1/√n'}),
    ('1/n^0.75',    {'func': lambda n: 1.0 / (n ** 0.75), 'label': '1/n^0.75'}),
    ('1/n',         {'func': lambda n: 1.0 / n,           'label': '1/n'}),
    ('1/n^1.5',     {'func': lambda n: 1.0 / (n ** 1.5),  'label': '1/n^1.5'}),
    ('1/n²',        {'func': lambda n: 1.0 / (n ** 2),    'label': '1/n²'}),
    ('exp(-n/50)',   {'func': lambda n: np.exp(-n / 50),   'label': 'exp(-n/50)'}),
    ('exp(-n/20)',   {'func': lambda n: np.exp(-n / 20),   'label': 'exp(-n/20)'}),
    ('exp(-n/10)',   {'func': lambda n: np.exp(-n / 10),   'label': 'exp(-n/10)'}),
])


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("AMPLITUDE DECAY SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()
    print("Testing whether amplitude decay in MT harmonics shifts the")
    print("predicted β from ~1.15 (uniform) toward the LEMON-observed ~1.7.")
    print()
    print("This script replicates mt_model's harmonic generation (lines")
    print("249-271) but applies decay_func(n) as an amplitude weight,")
    print("where n is the sub-harmonic index from the source frequency.")
    print()
    print("CRITICAL NOTE: For a 1 GHz source reaching 10 Hz via f=src/n^k")
    print("with k=2.0, the sub-harmonic index n ≈ 10,000. Even mild decay")
    print("like 1/√n gives amplitude ≈ 0.01 at these frequencies. This")
    print("preferentially suppresses low-frequency contributions from the")
    print("GHz band, which could either steepen or flatten the overall")
    print("spectrum depending on which bands dominate.")
    print()

    SEEDS = [42, 123, 456]
    LEMON_BETA = 1.95  # Full cohort IRASA, N=383
    cond = healthy()

    results = OrderedDict()

    for model_name, model_info in DECAY_MODELS.items():
        print(f"--- {model_info['label']} ---")

        betas = []
        subbands = None

        for seed in SEEDS:
            sig = generate_eeg_with_decay(cond, model_info['func'], seed=seed)
            beta = fit_beta(sig)
            betas.append(beta)
            print(f"  seed={seed}: β = {beta:.3f}")

            if seed == SEEDS[0]:
                subbands = fit_subbands(sig)

        mean_beta = np.nanmean(betas)
        sd_beta = np.nanstd(betas)
        print(f"  Mean β = {mean_beta:.3f} ± {sd_beta:.3f}")
        print()

        results[model_name] = {
            'label': model_info['label'],
            'mean_beta': mean_beta,
            'sd_beta': sd_beta,
            'betas': betas,
            'subbands': subbands or {},
            'dist_from_lemon': abs(mean_beta - LEMON_BETA),
        }

    # ─── Summary table ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Decay Model':<20} {'Mean β':>8} {'± SD':>8} "
          f"{'Δ from LEMON':>14} {'In [0.8,1.2]':>14}")
    print(f"  {'-' * 64}")

    for name, r in sorted(results.items(), key=lambda x: x[1]['mean_beta']):
        in_range = "YES" if 0.8 <= r['mean_beta'] <= 1.2 else "NO"
        delta_lemon = r['mean_beta'] - LEMON_BETA
        print(f"  {r['label']:<20} {r['mean_beta']:>8.3f} {r['sd_beta']:>8.3f} "
              f"{delta_lemon:>+14.3f} {in_range:>14}")

    # ─── Sub-band comparison ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUB-BAND SLOPES (seed=42)")
    print("=" * 70)
    print()

    mt_pred = {'delta': 0.85, 'theta': 0.82, 'alpha': 0.79, 'beta': 1.42, 'gamma': 3.47}
    lemon_obs = {'delta': 0.548, 'theta': -0.900, 'alpha': 3.603, 'beta': 2.635, 'gamma': 4.209}
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    header = f"  {'Model':<20}"
    for band in bands:
        header += f" {band:>8}"
    header += "  gradient?"
    print(header)
    print(f"  {'-' * 75}")

    for ref_name, ref_vals in [('LEMON Observed', lemon_obs), ('MT Uniform', mt_pred)]:
        line = f"  {ref_name:<20}"
        for band in bands:
            line += f" {ref_vals[band]:>8.3f}"
        line += f"  {'YES' if ref_vals['gamma'] > ref_vals['delta'] else 'NO'}"
        print(line)
    print(f"  {'-' * 75}")

    for name, r in results.items():
        sb = r['subbands']
        if not sb:
            continue
        line = f"  {r['label']:<20}"
        for band in bands:
            val = sb.get(band, np.nan)
            line += f" {val:>8.3f}" if not np.isnan(val) else f" {'N/A':>8}"
        d = sb.get('delta', np.nan)
        g = sb.get('gamma', np.nan)
        if not np.isnan(d) and not np.isnan(g):
            line += f"  {'YES' if g > d else 'NO'}"
        else:
            line += "  ???"
        print(line)

    # ─── Best match ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("BEST MATCH TO LEMON OBSERVED β")
    print("=" * 70)
    print()

    sorted_results = sorted(results.items(), key=lambda x: x[1]['dist_from_lemon'])
    best_name, best = sorted_results[0]

    print(f"  Target: LEMON β = {LEMON_BETA:.3f}")
    print()
    print(f"  Top 3 closest:")
    for i, (name, r) in enumerate(sorted_results[:3]):
        print(f"    {i+1}. {r['label']:<20} β = {r['mean_beta']:.3f}  "
              f"(distance = {r['dist_from_lemon']:.3f})")

    print()
    if best['dist_from_lemon'] < 0.1:
        print(f"  ✓ {best['label']} matches LEMON within 0.1!")
    elif best['dist_from_lemon'] < 0.3:
        print(f"  ~ {best['label']} is within 0.3 of LEMON.")
    else:
        print(f"  ✗ No tested decay model brings β close to LEMON.")

    # ─── Direction of effect ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("DIRECTION OF EFFECT")
    print("=" * 70)
    print()

    uniform_beta = results['uniform']['mean_beta']
    steeper_count = 0
    flatter_count = 0
    for name, r in results.items():
        if name == 'uniform':
            continue
        direction = "↑ steeper" if r['mean_beta'] > uniform_beta else "↓ flatter"
        delta = r['mean_beta'] - uniform_beta
        print(f"  {r['label']:<20} → β {direction} by {abs(delta):.3f}")
        if r['mean_beta'] > uniform_beta:
            steeper_count += 1
        else:
            flatter_count += 1

    print()
    if steeper_count > flatter_count:
        print("  RESULT: Most decay models produce STEEPER slopes (higher β).")
        print("  Amplitude decay shifts β TOWARD the LEMON value.")
        print("  The uniform-amplitude assumption was underestimating β.")
    elif flatter_count > steeper_count:
        print("  RESULT: Most decay models produce FLATTER slopes (lower β).")
        print("  Amplitude decay shifts β AWAY from LEMON.")
        print("  The global β offset is NOT caused by uniform amplitudes.")
        print("  It must come from biological/methodological factors in real EEG.")
    else:
        print("  RESULT: Mixed — direction depends on decay strength.")

    # ─── Physical justification ──────────────────────────────────────
    print()
    print("=" * 70)
    print("PHYSICAL JUSTIFICATION NOTES")
    print("=" * 70)
    print()
    print("  • 1/n: Natural for resonant systems (overtone decay).")
    print("    Well-established in acoustics and vibrating membranes.")
    print()
    print("  • 1/√n: Diffusive wave propagation through lossy media.")
    print()
    print("  • 1/n²: Radiation from oscillating dipoles (multipole).")
    print()
    print("  • exp(-n/τ): Systems with quality factor Q.")
    print("    Bandyopadhyay reports Q ~ 10-100 for MT modes.")
    print()
    print("  KEY INSIGHT: The sub-harmonic index n is VERY large for")
    print("  EEG-range frequencies. For a 1 GHz source reaching 10 Hz")
    print("  via f = src/n^k with k=2.0: n = √(1e9/10) ≈ 10,000.")
    print("  Even 1/√n gives amplitude ≈ 0.01 at n=10,000.")
    print("  For the kHz band (k=1.0): n = 10e3/10 = 1,000.")
    print("  So decay DIFFERENTIALLY suppresses GHz contributions")
    print("  (large n) relative to kHz (smaller n).")
    print()
    print("  This means decay effectively changes the RELATIVE weighting")
    print("  of bands — the kHz band (k=1.0, flatter) becomes dominant")
    print("  while the GHz band (k=2.0, steeper) is suppressed.")
    print("  The net effect on β depends on this rebalancing.")

    # ─── Figure ──────────────────────────────────────────────────────
    print()
    print("Generating figure...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: β vs decay model
    ax = axes[0]
    sorted_for_plot = sorted(results.items(), key=lambda x: x[1]['mean_beta'])
    names = [r['label'].replace(' (current)', '\n(current)') for _, r in sorted_for_plot]
    means = [r['mean_beta'] for _, r in sorted_for_plot]
    sds = [r['sd_beta'] for _, r in sorted_for_plot]

    colors = ['#228833' if abs(m - LEMON_BETA) < 0.15 else
              '#CCBB44' if abs(m - LEMON_BETA) < 0.3 else
              '#4477AA' for m in means]

    x = np.arange(len(names))
    ax.bar(x, means, yerr=sds, color=colors, alpha=0.8,
           capsize=3, edgecolor='black', linewidth=0.5)
    ax.axhline(LEMON_BETA, color='red', linestyle='--', linewidth=1.5,
               label=f'LEMON observed ({LEMON_BETA})')
    ax.axhline(uniform_beta, color='blue', linestyle=':', linewidth=1.5,
               label=f'Uniform model ({uniform_beta:.3f})')
    ax.axhspan(0.8, 1.2, alpha=0.1, color='green', label='Published range')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Aperiodic exponent (β)')
    ax.set_title('A. Effect of amplitude decay on predicted β')
    ax.legend(fontsize=7, loc='upper left')

    # Panel B: Sub-band profiles for key models
    ax = axes[1]
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    x_b = np.arange(len(band_names))

    lemon_vals = [lemon_obs[b] for b in band_names]
    ax.plot(x_b, lemon_vals, 'rs-', markersize=8, linewidth=2, label='LEMON observed')

    mt_vals = [mt_pred[b] for b in band_names]
    ax.plot(x_b, mt_vals, 'k--', markersize=5, linewidth=1, label='MT uniform', alpha=0.5)

    markers = ['-o', '-^', '-D']
    colors_top = ['#228833', '#CCBB44', '#4477AA']
    for i, (name, r) in enumerate(sorted_results[:3]):
        if not r['subbands']:
            continue
        vals = [r['subbands'].get(b, np.nan) for b in band_names]
        ax.plot(x_b, vals, markers[i], color=colors_top[i],
                markersize=6, linewidth=1.5, label=r['label'], alpha=0.8)

    ax.set_xticks(x_b)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('Aperiodic exponent (β)')
    ax.set_title('B. Sub-band profiles: best decay models vs LEMON')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('amplitude_decay_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('amplitude_decay_analysis.pdf', bbox_inches='tight')
    print("  Saved: amplitude_decay_analysis.png / .pdf")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
