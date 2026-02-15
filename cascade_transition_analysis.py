#!/usr/bin/env python3
"""
CASCADE TRANSITION-REGIME ANALYSIS
===================================
The critical test: does the membrane + skull filter cascade reverse
the sub-band gradient direction from what bare MT source density predicts?

Bare source density (analytic derivation):
  kHz band (k=1.0): β = 1 + 1/k = 2.000 (steepest)
  GHz band (k=2.0): β = 1 + 1/k = 1.500 (flattest)
  → Gradient DECREASES with frequency

But real EEG passes through two filters:
  P_EEG(f) = P_source(f) × |H_mem(f)|² × |H_skull(f)|²

Where:
  |H_mem(f)|² = 1 / (1 + (f/fc_mem)²)     [membrane time constant]
  |H_skull(f)|² = 1 / (1 + (f/fc_skull)²)  [effective skull spatial→temporal]

Question: In the transition regime (1-50 Hz), does the cascade
produce an INCREASING effective β across sub-bands?

If yes → the LEMON/TDBRAIN gradient is explained by filters, not source.
If no  → the gradient claim must be dropped.
"""

import numpy as np
from scipy import stats
import json

# ============================================================
# PART 1: Define the physics
# ============================================================

def mt_source_psd(freqs, bands=None):
    """
    Combined MT source spectrum from all three bands.
    Each band contributes harmonic density ∝ f^(-1/k - 1) = f^(-β_band).
    
    Using Bandyopadhyay's measured bands:
      kHz: f0 ~ 10 kHz, k = 1.0  → density ∝ f^(-2)
      MHz: f0 ~ 10 MHz, k = 1.5  → density ∝ f^(-5/3)
      GHz: f0 ~ 10 GHz, k = 2.0  → density ∝ f^(-3/2)
    """
    if bands is None:
        bands = [
            {'f0': 10e3, 'k': 1.0, 'label': 'kHz'},
            {'f0': 10e6, 'k': 1.5, 'label': 'MHz'},
            {'f0': 10e9, 'k': 2.0, 'label': 'GHz'},
        ]
    
    psd = np.zeros_like(freqs, dtype=float)
    for band in bands:
        k = band['k']
        # Harmonic density contribution: f^(-1/k - 1)
        # This is the analytic result from the density derivation
        psd += freqs ** (-1.0/k - 1.0)
    
    return psd


def membrane_filter(freqs, fc_mem):
    """Single-pole low-pass: |H(f)|² = 1/(1 + (f/fc)²)"""
    return 1.0 / (1.0 + (freqs / fc_mem)**2)


def skull_filter(freqs, fc_skull):
    """Effective spatial→temporal low-pass: |H(f)|² = 1/(1 + (f/fc)²)"""
    return 1.0 / (1.0 + (freqs / fc_skull)**2)


def full_cascade(freqs, fc_mem, fc_skull, bands=None):
    """P_EEG(f) = P_source(f) × |H_mem|² × |H_skull|²"""
    source = mt_source_psd(freqs, bands)
    h_mem = membrane_filter(freqs, fc_mem)
    h_skull = skull_filter(freqs, fc_skull)
    return source * h_mem * h_skull


def fit_subband_beta(freqs, psd, band_edges):
    """Fit log-log slope in a sub-band. Returns β (positive = steeper)."""
    mask = (freqs >= band_edges[0]) & (freqs <= band_edges[1])
    if mask.sum() < 3:
        return np.nan
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    slope, intercept, r, p, se = stats.linregress(log_f, log_p)
    return -slope  # Convention: β positive for 1/f^β


# ============================================================
# PART 2: Sub-band definitions
# ============================================================

SUBBANDS = {
    'delta':  (1.0, 4.0),
    'theta':  (4.0, 8.0),
    'alpha':  (8.0, 13.0),
    'beta':   (13.0, 30.0),
    'gamma':  (30.0, 50.0),
}

# ============================================================
# PART 3: Sweep fc values and check gradient direction
# ============================================================

def analyse_cascade(fc_mem, fc_skull, freqs=None, label=""):
    """Full analysis for one (fc_mem, fc_skull) pair."""
    if freqs is None:
        freqs = np.linspace(1.0, 50.0, 5000)
    
    # Bare source (no filters)
    source_psd = mt_source_psd(freqs)
    
    # Filtered cascade
    cascade_psd = full_cascade(freqs, fc_mem, fc_skull)
    
    results = {
        'fc_mem': fc_mem,
        'fc_skull': fc_skull,
        'label': label,
        'source_betas': {},
        'cascade_betas': {},
    }
    
    for band_name, edges in SUBBANDS.items():
        results['source_betas'][band_name] = fit_subband_beta(freqs, source_psd, edges)
        results['cascade_betas'][band_name] = fit_subband_beta(freqs, cascade_psd, edges)
    
    # Global β (1-50 Hz)
    results['source_global'] = fit_subband_beta(freqs, source_psd, (1.0, 50.0))
    results['cascade_global'] = fit_subband_beta(freqs, cascade_psd, (1.0, 50.0))
    
    # Gradient: gamma_β - delta_β
    results['source_gradient'] = results['source_betas']['gamma'] - results['source_betas']['delta']
    results['cascade_gradient'] = results['cascade_betas']['gamma'] - results['cascade_betas']['delta']
    
    return results


def print_results(r):
    """Pretty-print one analysis result."""
    print(f"\n{'='*70}")
    print(f"  {r['label']}")
    print(f"  fc_mem = {r['fc_mem']:.1f} Hz, fc_skull = {r['fc_skull']:.1f} Hz")
    print(f"{'='*70}")
    print(f"  {'Band':<10} {'Source β':>12} {'Cascade β':>12} {'Filter Δ':>12}")
    print(f"  {'-'*46}")
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        sb = r['source_betas'][band]
        cb = r['cascade_betas'][band]
        print(f"  {band:<10} {sb:>12.3f} {cb:>12.3f} {cb-sb:>+12.3f}")
    print(f"  {'-'*46}")
    print(f"  {'GLOBAL':<10} {r['source_global']:>12.3f} {r['cascade_global']:>12.3f} {r['cascade_global']-r['source_global']:>+12.3f}")
    print(f"  {'-'*46}")
    print(f"  Gradient (γ-δ):  Source = {r['source_gradient']:+.3f}  |  Cascade = {r['cascade_gradient']:+.3f}")
    
    if r['cascade_gradient'] > 0:
        print(f"  >>> GRADIENT IS POSITIVE (increasing with freq) — MATCHES LEMON <<<")
    else:
        print(f"  >>> GRADIENT IS NEGATIVE (decreasing with freq) — STILL INVERTED <<<")


# ============================================================
# PART 4: Run the analysis
# ============================================================

if __name__ == '__main__':
    freqs = np.linspace(1.0, 50.0, 5000)
    
    print("=" * 70)
    print("CASCADE TRANSITION-REGIME ANALYSIS")
    print("=" * 70)
    print()
    print("QUESTION: Does the membrane + skull filter cascade reverse")
    print("the sub-band gradient from decreasing to increasing?")
    print()
    print("LEMON observed:  delta β ≈ 0.55, gamma β ≈ 4.21  (gradient = +3.66)")
    print("TDBRAIN observed: gradient ≈ +1.15 (δ→β direction)")
    print("Bare source:     gradient should be NEGATIVE (analytic derivation)")
    print()
    
    # --------------------------------------------------------
    # Test 1: Bare source only (sanity check)
    # --------------------------------------------------------
    r0 = analyse_cascade(1e6, 1e6, freqs, "NO FILTERS (fc → ∞)")
    print_results(r0)
    
    # --------------------------------------------------------
    # Test 2: Physiological fc values from literature
    # Membrane τ_m ≈ 10-30 ms → fc = 1/(2πτ) ≈ 5-16 Hz
    # Skull effective fc ≈ 10-40 Hz (from spatial smoothing models)
    # --------------------------------------------------------
    
    physiological_scenarios = [
        # (fc_mem, fc_skull, label)
        (5.0,  10.0, "STRONG FILTERING: τ_m=32ms, skull tight"),
        (8.0,  15.0, "MODERATE-STRONG: τ_m=20ms, skull moderate"),
        (10.0, 20.0, "MODERATE: τ_m=16ms, skull moderate"),
        (12.0, 25.0, "MODERATE-WEAK: τ_m=13ms, skull loose"),
        (16.0, 30.0, "STANDARD: τ_m=10ms, skull typical"),
        (16.0, 40.0, "STANDARD MEM + WEAK SKULL"),
        (20.0, 40.0, "WEAK FILTERING: τ_m=8ms, skull loose"),
        (30.0, 50.0, "VERY WEAK: both fc near upper EEG edge"),
    ]
    
    all_results = [r0]
    
    for fc_m, fc_s, label in physiological_scenarios:
        r = analyse_cascade(fc_m, fc_s, freqs, label)
        print_results(r)
        all_results.append(r)
    
    # --------------------------------------------------------
    # Test 3: Individual band contributions through cascade
    # Which MT band's gradient dominates after filtering?
    # --------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("INDIVIDUAL BAND CONTRIBUTIONS THROUGH CASCADE")
    print("fc_mem=16 Hz, fc_skull=30 Hz (standard values)")
    print("=" * 70)
    
    fc_m, fc_s = 16.0, 30.0
    
    for band_info in [
        {'f0': 10e3, 'k': 1.0, 'label': 'kHz only'},
        {'f0': 10e6, 'k': 1.5, 'label': 'MHz only'},
        {'f0': 10e9, 'k': 2.0, 'label': 'GHz only'},
    ]:
        r = analyse_cascade(fc_m, fc_s, freqs, f"{band_info['label']} (k={band_info['k']})")
        # Override with single-band source
        single_psd = freqs ** (-1.0/band_info['k'] - 1.0)
        cascade_single = single_psd * membrane_filter(freqs, fc_m) * skull_filter(freqs, fc_s)
        
        for band_name, edges in SUBBANDS.items():
            r['source_betas'][band_name] = fit_subband_beta(freqs, single_psd, edges)
            r['cascade_betas'][band_name] = fit_subband_beta(freqs, cascade_single, edges)
        
        r['source_global'] = fit_subband_beta(freqs, single_psd, (1.0, 50.0))
        r['cascade_global'] = fit_subband_beta(freqs, cascade_single, (1.0, 50.0))
        r['source_gradient'] = r['source_betas']['gamma'] - r['source_betas']['delta']
        r['cascade_gradient'] = r['cascade_betas']['gamma'] - r['cascade_betas']['delta']
        
        print_results(r)
    
    # --------------------------------------------------------
    # Test 4: Asymptotic regime verification
    # --------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("ASYMPTOTIC VERIFICATION")
    print("=" * 70)
    
    # If f >> both fc, slope should approach β_source + 2 + 2 = β_source + 4
    # For combined source (effective β ≈ 1.7), asymptotic should be ≈ 5.7
    
    # Check at very high frequencies
    high_freqs = np.linspace(200, 1000, 1000)
    cascade_high = full_cascade(high_freqs, 16.0, 30.0)
    beta_asymp = fit_subband_beta(high_freqs, cascade_high, (200, 1000))
    source_high = mt_source_psd(high_freqs)
    beta_source_high = fit_subband_beta(high_freqs, source_high, (200, 1000))
    
    print(f"\n  High-f regime (200-1000 Hz):")
    print(f"  Source β = {beta_source_high:.3f}")
    print(f"  Cascade β = {beta_asymp:.3f}")
    print(f"  Filter contribution = {beta_asymp - beta_source_high:.3f} (expect ≈ 4.0)")
    
    # Low frequency check
    low_freqs = np.linspace(0.1, 1.0, 1000)
    cascade_low = full_cascade(low_freqs, 16.0, 30.0)
    beta_low = fit_subband_beta(low_freqs, cascade_low, (0.1, 1.0))
    source_low = mt_source_psd(low_freqs)
    beta_source_low = fit_subband_beta(low_freqs, source_low, (0.1, 1.0))
    
    print(f"\n  Low-f regime (0.1-1 Hz):")
    print(f"  Source β = {beta_source_low:.3f}")
    print(f"  Cascade β = {beta_low:.3f}")
    print(f"  Filter contribution = {beta_low - beta_source_low:.3f} (expect ≈ 0.0)")
    
    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("SUMMARY: GRADIENT DIRECTION ACROSS ALL SCENARIOS")
    print("=" * 70)
    print(f"\n  {'Scenario':<45} {'Cascade Gradient':>16} {'Direction':>12}")
    print(f"  {'-'*73}")
    
    for r in all_results:
        direction = "POSITIVE ✓" if r['cascade_gradient'] > 0 else "NEGATIVE ✗"
        label = r['label'][:44]
        print(f"  {label:<45} {r['cascade_gradient']:>+16.3f} {direction:>12}")
    
    print(f"\n  LEMON observed gradient: +3.66")
    print(f"  TDBRAIN observed gradient: +1.15")
    
    # Count how many scenarios produce positive gradient
    positive = sum(1 for r in all_results if r['cascade_gradient'] > 0)
    total = len(all_results)
    print(f"\n  Scenarios with positive gradient: {positive}/{total}")
    
    if positive > 0:
        print("\n  ✓ FILTER CASCADE CAN REVERSE THE GRADIENT DIRECTION")
        print("    The transition-regime effect of membrane + skull filters")
        print("    steepens high-frequency sub-bands more than low-frequency,")
        print("    producing the increasing gradient seen in real EEG.")
    else:
        print("\n  ✗ FILTER CASCADE DOES NOT REVERSE THE GRADIENT")
        print("    The gradient claim cannot be rescued by filters alone.")
    
    # --------------------------------------------------------
    # Save numerical results for paper
    # --------------------------------------------------------
    output = {
        'scenarios': [],
        'lemon_gradient': 3.66,
        'tdbrain_gradient': 1.15,
    }
    for r in all_results:
        output['scenarios'].append({
            'label': r['label'],
            'fc_mem': r['fc_mem'],
            'fc_skull': r['fc_skull'],
            'cascade_global_beta': round(r['cascade_global'], 3),
            'cascade_gradient': round(r['cascade_gradient'], 3),
            'cascade_betas': {k: round(v, 3) for k, v in r['cascade_betas'].items()},
        })
    
    with open('cascade_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n  Results saved to cascade_results.json")
