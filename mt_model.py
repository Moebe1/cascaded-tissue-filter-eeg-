"""
mt_model.py — Core Microtubule Oscillator Diversity Model

This module contains the mathematical framework for generating
EEG-range frequencies from Bandyopadhyay's measured MT resonance bands.

When run directly, reproduces key paper claims:
    - Source beta_density ~ 1.0
    - Source beta_PSD (specparam) ~ 1.15
    - SSD source-mechanism Delta_beta = +0.161 (60% bandwidth)
    - Consciousness state mapping (Table 3)
    - Sub-band gradient from source model
    - Sensitivity analyses (n_sources, random seed)

Reference:
    Saxena, K., et al. (2020). Fractal, Scale Free Electromagnetic
    Resonance of a Single Brain Extracted Microtubule.
    Fractal and Fractional, 4(2), 11.

Author: Dr. Syed Mohsin Parwez
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────────────────────────────
# Bandyopadhyay's measured MT resonance bands
# ─────────────────────────────────────────────────────────────────────

BANDS = {
    'kHz': {'f_lo': 10e3, 'f_hi': 300e3, 'k': 1.0},
    'MHz': {'f_lo': 10e6, 'f_hi': 230e6, 'k': 1.5},
    'GHz': {'f_lo': 1e9,  'f_hi': 20e9,  'k': 2.0},
}


# ─────────────────────────────────────────────────────────────────────
# Condition configurations
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BandConfig:
    """Configuration for a single MT resonance band."""
    f_lo: float
    f_hi: float
    k: float
    n_sources: int


@dataclass
class Condition:
    """A clinical/pharmacological condition modelled as MT disruption."""
    name: str
    bands: List[BandConfig]
    description: str = ""


def healthy() -> Condition:
    """Healthy waking adult — full MT resonance ensemble."""
    return Condition(
        name="Healthy",
        bands=[
            BandConfig(10e3, 300e3, 1.0, 5),
            BandConfig(10e6, 230e6, 1.5, 5),
            BandConfig(1e9, 20e9, 2.0, 5),
        ],
        description="Full MT oscillator diversity across all Bandyopadhyay bands"
    )


def ssd_bandwidth(bw_fraction: float = 0.70) -> Condition:
    """Schizophrenia spectrum disorder — narrowed resonance bands."""
    return Condition(
        name=f"SSD ({int(bw_fraction*100)}% bandwidth)",
        bands=[
            BandConfig(10e3, 10e3 + (300e3 - 10e3) * bw_fraction, 1.0, 5),
            BandConfig(10e6, 10e6 + (230e6 - 10e6) * bw_fraction, 1.5, 5),
            BandConfig(1e9, 1e9 + (20e9 - 1e9) * bw_fraction, 2.0, 5),
        ],
        description=f"MT resonance bands narrowed to {int(bw_fraction*100)}% of healthy range"
    )


def ssd_kshift(dk: float = 0.05) -> Condition:
    """SSD modelled as altered fractal exponents."""
    return Condition(
        name=f"SSD (k+{dk})",
        bands=[
            BandConfig(10e3, 300e3, 1.0 + dk, 5),
            BandConfig(10e6, 230e6, 1.5 + dk, 5),
            BandConfig(1e9, 20e9, 2.0 + dk, 5),
        ],
        description=f"Fractal exponents shifted by +{dk}"
    )


def ptsd_dissociation() -> Condition:
    """Dissociative PTSD — restricted oscillator diversity."""
    return Condition(
        name="PTSD-Dissociation",
        bands=[
            BandConfig(10e3, 150e3, 1.0, 4),
            BandConfig(10e6, 230e6, 1.5, 4),
            BandConfig(1e9, 20e9, 2.0, 3),
        ],
        description="Reduced source count and narrowed kHz band"
    )


def light_sedation() -> Condition:
    return Condition(
        name="Light Sedation",
        bands=[
            BandConfig(10e3, 300e3, 1.0, 5),
            BandConfig(10e6, 230e6, 1.5, 4),
            BandConfig(1e9, 20e9, 2.0, 3),
        ],
        description="Early GHz band suppression"
    )


def moderate_sedation() -> Condition:
    return Condition(
        name="Moderate Sedation",
        bands=[
            BandConfig(10e3, 300e3, 1.0, 5),
            BandConfig(10e6, 150e6, 1.5, 3),
            BandConfig(1e9, 10e9, 2.0, 2),
        ],
        description="MHz and GHz suppression"
    )


def general_anaesthesia() -> Condition:
    return Condition(
        name="General Anaesthesia",
        bands=[
            BandConfig(10e3, 300e3, 1.0, 4),
            BandConfig(10e6, 100e6, 1.5, 2),
        ],
        description="GHz silenced, MHz severely reduced"
    )


def deep_anaesthesia() -> Condition:
    return Condition(
        name="Deep Anaesthesia",
        bands=[
            BandConfig(10e3, 200e3, 1.0, 3),
        ],
        description="Only residual kHz activity"
    )


ALL_CONDITIONS = [
    healthy, ssd_bandwidth, ptsd_dissociation,
    light_sedation, moderate_sedation, general_anaesthesia, deep_anaesthesia,
]


# ─────────────────────────────────────────────────────────────────────
# Core computation: harmonic density
# ─────────────────────────────────────────────────────────────────────

def compute_harmonic_density(
    condition: Condition,
    f_min: float = 0.5,
    f_max: float = 80.0,
    n_bins: int = 150,
    n_max_cap: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the harmonic frequency density in the EEG range.

    Returns:
        bin_centers: array of frequency bin centers (Hz)
        density: harmonic count per Hz at each bin center
    """
    freq_bins = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)
    bin_centers = np.sqrt(freq_bins[:-1] * freq_bins[1:])
    bin_widths = freq_bins[1:] - freq_bins[:-1]
    density = np.zeros(len(bin_centers))

    for band in condition.bands:
        if band.n_sources == 0:
            continue
        sources = np.logspace(
            np.log10(band.f_lo), np.log10(band.f_hi), max(1, band.n_sources)
        )
        for src in sources:
            n_max = min(n_max_cap, int(src / 0.1))
            if n_max < 1:
                continue
            ns = np.arange(1, n_max + 1)
            freqs = src / (ns ** band.k)
            mask = (freqs >= f_min) & (freqs <= f_max)
            valid = freqs[mask]
            if len(valid) > 0:
                counts, _ = np.histogram(valid, bins=freq_bins)
                density += counts / bin_widths

    return bin_centers, density


def compute_density_beta(
    condition: Condition,
    f_min: float = 0.5,
    f_max: float = 80.0,
) -> Optional[float]:
    """
    Compute the spectral exponent beta from harmonic density.
    beta is defined such that density ~ f^(-beta).
    """
    bin_centers, density = compute_harmonic_density(condition, f_min, f_max)
    valid = density > 0
    if np.sum(valid) < 5:
        return None
    log_f = np.log10(bin_centers[valid])
    log_d = np.log10(density[valid])
    slope, _ = np.polyfit(log_f, log_d, 1)
    return -slope


# ─────────────────────────────────────────────────────────────────────
# Synthetic EEG generation
# ─────────────────────────────────────────────────────────────────────

def generate_synthetic_eeg(
    condition: Condition,
    fs: int = 250,
    duration: float = 180.0,
    max_harmonics_per_source: int = 800,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic EEG time series from MT harmonic frequencies.

    Each MT source contributes sinusoids at its harmonic frequencies
    with UNIFORM amplitude (consistent with resonance amplification).
    A small white noise floor is added.

    Args:
        condition: MT configuration to generate
        fs: sampling frequency (Hz)
        duration: signal duration (seconds)
        max_harmonics_per_source: cap on harmonics per source (for speed)
        seed: random seed for reproducibility

    Returns:
        signal: 1D numpy array of synthetic EEG
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

            # Subsample for computational tractability
            if len(valid_freqs) > max_harmonics_per_source:
                idx = rng.choice(len(valid_freqs), max_harmonics_per_source, replace=False)
                valid_freqs = valid_freqs[idx]

            for f in valid_freqs:
                phase = rng.uniform(0, 2 * np.pi)
                signal += np.sin(2 * np.pi * f * t + phase)

    # Add small noise floor
    signal += rng.normal(0, 0.01 * np.std(signal), n_samples)

    return signal


# ─────────────────────────────────────────────────────────────────────
# Specparam analysis
# ─────────────────────────────────────────────────────────────────────

def fit_specparam(
    signal: np.ndarray,
    fs: int = 250,
    freq_range: Tuple[float, float] = (1, 50),
    nperseg: int = 2048,
) -> Dict:
    """
    Run specparam on a signal and return aperiodic parameters.

    Returns dict with:
        'exponent': aperiodic exponent (beta)
        'offset': aperiodic offset
        'freqs': PSD frequencies
        'powers': PSD power values
    """
    from scipy.signal import welch
    from specparam import SpectralModel

    f_psd, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    sm = SpectralModel(
        aperiodic_mode='fixed',
        min_peak_height=0.3,
        max_n_peaks=6,
        peak_width_limits=[1, 12],
    )
    sm.fit(f_psd, psd, freq_range)

    ap = sm.get_params('aperiodic')

    return {
        'exponent': ap[1],
        'offset': ap[0],
        'freqs': f_psd,
        'powers': psd,
    }


def fit_specparam_subbands(
    signal: np.ndarray,
    fs: int = 250,
    nperseg: int = 4096,
) -> Dict[str, Optional[float]]:
    """
    Fit specparam separately to each EEG sub-band.

    Returns dict mapping band name to aperiodic exponent.
    """
    from scipy.signal import welch
    from specparam import SpectralModel

    f_psd, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    sub_bands = {
        'delta': (1.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50),
    }

    results = {}
    for name, (flo, fhi) in sub_bands.items():
        sm = SpectralModel(
            aperiodic_mode='fixed',
            min_peak_height=0.5,
            max_n_peaks=2,
        )
        try:
            sm.fit(f_psd, psd, [flo, fhi])
            ap = sm.get_params('aperiodic')
            results[name] = ap[1]
        except Exception:
            results[name] = None

    return results


# ═════════════════════════════════════════════════════════════════════
# MAIN — Reproduce key paper claims when run directly
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    import time

    print("=" * 70)
    print("MT SOURCE MODEL — Reproducing Paper Claims")
    print("=" * 70)

    # ── 1. Harmonic density exponent (Paper: ~1.0) ──────────────────
    print("\n[1] Harmonic density exponent (Paper Section 2.1: ~1.0)")
    h = healthy()
    beta_d = compute_density_beta(h)
    print(f"    Computed: {beta_d:.3f}")
    print(f"    Paper:    ~1.0")
    print(f"    {'PASS' if beta_d is not None and 0.8 <= beta_d <= 1.2 else 'CHECK'}")

    # ── 2. Source PSD via specparam (Paper: ~1.15) ──────────────────
    print("\n[2] Source PSD beta via specparam (Paper Figure 1B: ~1.15)")
    print("    Generating synthetic EEG (fs=250, 180s)... ", end="", flush=True)
    t0 = time.time()
    sig = generate_synthetic_eeg(h, fs=250, duration=180.0, seed=42)
    print(f"done ({time.time()-t0:.1f}s)")

    print("    Fitting specparam... ", end="", flush=True)
    try:
        result = fit_specparam(sig, fs=250)
        beta_sp = result['exponent']
        print(f"done")
        print(f"    Computed: {beta_sp:.3f}")
        print(f"    Paper:    ~1.15")
        print(f"    {'PASS' if 0.9 <= beta_sp <= 1.4 else 'CHECK'}")
    except Exception as e:
        print(f"FAILED: {e}")
        beta_sp = None

    # ── 3. Sub-band gradient from source (Paper Figure 2) ──────────
    print("\n[3] Sub-band gradient from source model (Paper Figure 2)")
    print("    Fitting sub-bands... ", end="", flush=True)
    try:
        sb = fit_specparam_subbands(sig, fs=250)
        print("done")
        for band, val in sb.items():
            v = f"{val:.3f}" if val is not None else "NaN"
            print(f"    {band:>8s}: {v}")
        if sb.get('gamma') is not None and sb.get('delta') is not None:
            gradient = sb['gamma'] - sb['delta']
            print(f"    Gradient (gamma-delta): {gradient:+.3f}")
            print(f"    Paper Figure 2 gradient: +2.62")
    except Exception as e:
        print(f"FAILED: {e}")

    # ── 4. SSD bandwidth narrowing (Paper Section 3.2) ─────────────
    print("\n[4] SSD source-mechanism Delta_beta (Paper Section 3.2: +0.161)")
    print("    Testing 60% bandwidth... ", end="", flush=True)
    try:
        ssd60 = ssd_bandwidth(0.60)
        sig_ssd = generate_synthetic_eeg(ssd60, fs=250, duration=180.0, seed=42)
        r_ssd = fit_specparam(sig_ssd, fs=250)
        delta_beta = r_ssd['exponent'] - beta_sp if beta_sp else None
        print("done")
        if delta_beta is not None:
            print(f"    Healthy beta:  {beta_sp:.3f}")
            print(f"    SSD 60% beta:  {r_ssd['exponent']:.3f}")
            print(f"    Delta_beta:    {delta_beta:+.3f}")
            print(f"    Paper:         +0.161")
            print(f"    {'PASS' if 0.05 <= delta_beta <= 0.30 else 'CHECK'}")
    except Exception as e:
        print(f"FAILED: {e}")

    # ── 5. Consciousness state mapping (Paper Table 3) ─────────────
    print("\n[5] Consciousness state mapping (Paper Table 3)")
    states = [
        ("Healthy",           healthy),
        ("Light Sedation",    light_sedation),
        ("Moderate Sedation", moderate_sedation),
        ("General Anaesth.",  general_anaesthesia),
        ("Deep Anaesthesia",  deep_anaesthesia),
    ]
    betas = []
    for name, cond_fn in states:
        print(f"    {name:.<25s} ", end="", flush=True)
        try:
            cond = cond_fn()
            sig_c = generate_synthetic_eeg(cond, fs=250, duration=180.0, seed=42)
            r_c = fit_specparam(sig_c, fs=250)
            betas.append(r_c['exponent'])
            print(f"beta = {r_c['exponent']:.3f}")
        except Exception as e:
            betas.append(None)
            print(f"FAILED: {e}")

    valid_betas = [b for b in betas if b is not None]
    if len(valid_betas) > 1:
        monotonic = all(valid_betas[i] <= valid_betas[i+1] for i in range(len(valid_betas)-1))
        print(f"    Monotonically increasing: {'PASS' if monotonic else 'CHECK'}")
        print(f"    Paper predicts: monotonic increase with anaesthetic depth")

    # ── 6. Sensitivity: n_sources (Paper Section 4.4) ──────────────
    print("\n[6] Source count sensitivity (Paper Section 4.4: spread 0.060)")
    spreads = []
    for ns in [3, 5, 10, 20]:
        seed_betas = []
        for seed in [42, 123, 999]:
            cond = Condition(
                name=f"n={ns}",
                bands=[
                    BandConfig(10e3, 300e3, 1.0, ns),
                    BandConfig(10e6, 230e6, 1.5, ns),
                    BandConfig(1e9, 20e9, 2.0, ns),
                ],
            )
            try:
                sig_s = generate_synthetic_eeg(cond, fs=250, duration=180.0, seed=seed)
                r_s = fit_specparam(sig_s, fs=250)
                seed_betas.append(r_s['exponent'])
            except Exception:
                pass
        if seed_betas:
            avg = np.mean(seed_betas)
            spreads.append(avg)
            print(f"    n_sources={ns:>2d}: mean beta = {avg:.3f}")
    if len(spreads) > 1:
        spread = max(spreads) - min(spreads)
        print(f"    Total spread: {spread:.3f}")
        print(f"    Paper:        0.060")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE. All key claims from Sections 2.1, 3.2, 3.3, 4.4 tested.")
    print("=" * 70)
