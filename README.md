# Cascaded Tissue Filter Model of the EEG Aperiodic Spectrum

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)

**Microtubule Sub-Harmonics as Broadband Neural Source: A Cascaded Tissue Filter Model of the EEG Aperiodic Spectrum with Literature-Constrained Biophysical Parameters**

Dr. Syed Mohsin Parwez  
Independent Researcher  
ORCID: [0009-0004-6318-0309](https://orcid.org/0009-0004-6318-0309)

---

## Overview

This repository contains the computational code and analysis pipelines supporting the preprint:

> Parwez, S. M. (2026). Microtubule Sub-Harmonics as Broadband Neural Source: A Cascaded Tissue Filter Model of the EEG Aperiodic Spectrum with Literature-Constrained Biophysical Parameters. *bioRxiv*. https://doi.org/PLACEHOLDER

The model derives the observed EEG aperiodic spectral slope (1/f^β) from three independently parameterised components:

1. **Microtubule (MT) broadband source** — Sub-harmonic oscillations from experimentally measured kHz/MHz/GHz resonance bands (Saxena et al., 2020) produce a composite source spectrum with β_source ≈ 1.6 (PSD slope).
2. **Neuronal membrane RC filter** — A first-order low-pass filter with cutoff frequency determined by the membrane time constant τ_m = 20 ms (McCormick et al., 1985; guinea pig neocortex).
3. **Skull spatial low-pass filter** — Volume conduction through the skull imposes an effective temporal low-pass at ~15 Hz for dipolar cortical sources (Nunez & Srinivasan, 2006).

The cascaded filter produces a sub-band gradient (δ→γ) that is positive across all physiologically plausible parameterisations (6/6 tested scenarios), validated on N = 195 subjects (383 recordings) from the LEMON dataset.

## Repository Structure

```
cascade-filter-model/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── CITATION.cff                     # Citation metadata
├── requirements.txt                 # Python dependencies
├── mt_model.py                      # Core MT source spectrum model
├── cascade_transition_analysis.py   # Analytical cascade filter (Table 2)
├── analysis_lemon_full.py           # LEMON dataset IRASA pipeline (Section 4)
├── analysis_amplitude_decay.py      # Amplitude decay comparison (Appendix B)
└── figures/                         # Generated figures (after running scripts)
```

## Scripts and Paper Claims

| Script | Paper Section | Key Output |
|--------|--------------|------------|
| `mt_model.py` | §2.1, §3.2, §3.3, §4.4, Figs 1–4 | Source spectrum β ≈ 1.6, SSD Δβ = +0.161, consciousness state mapping, sensitivity analyses |
| `cascade_transition_analysis.py` | §2.4, §3.1, Table 2 | Sub-band gradient range +1.91 to +3.40, SSD filter Δβ = +0.151, asymptotic verification |
| `analysis_lemon_full.py` | §4.1, §4.2, Figs 5–6 | LEMON N=383 validation: gradient +4.82 ± 0.76, EC/EO comparison, sub-band statistics |
| `analysis_amplitude_decay.py` | Appendix B | 10 amplitude decay model comparison |

## Quick Start

### Requirements

```bash
# Create environment
conda create -n cascade_eeg python=3.10
conda activate cascade_eeg
pip install -r requirements.txt
```

### Run core model (no data needed)

```bash
# Reproduce Table 2 and cascade predictions
python cascade_transition_analysis.py

# Reproduce MT source spectrum and clinical predictions
python mt_model.py

# Reproduce Appendix B amplitude decay analysis
python analysis_amplitude_decay.py
```

### Run LEMON validation (requires dataset)

The LEMON dataset (Babayan et al., 2019) must be downloaded separately:

1. Download preprocessed EEG (.set/.fdt files) from the [LEMON dataset](https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/)
2. Place files in `data/lemon_preprocessed/`
3. Run:

```bash
python analysis_lemon_full.py
```

This processes 406 recordings from 203 subjects (383 recordings from 195 subjects pass quality control). Runtime: approximately 2–4 hours depending on hardware.

## Key Parameters

All model parameters are drawn from the independent experimental literature:

| Parameter | Value | Source |
|-----------|-------|--------|
| MT kHz band | 10–300 kHz, k = 1.0 | Saxena et al. (2020) |
| MT MHz band | 10–230 MHz, k = 1.5 | Saxena et al. (2020) |
| MT GHz band | 1–20 GHz, k = 2.0 | Saxena et al. (2020) |
| Membrane τ_m | 20 ms (f_c = 8.0 Hz) | McCormick et al. (1985) |
| Skull f_c | ~15 Hz | Nunez & Srinivasan (2006) |
| Amplitude decay α | 0.5 | Assumed (sole free parameter) |

## Data Availability

- **LEMON dataset**: Babayan, A., et al. (2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology. *Scientific Data*, 6, 180308.
- **Hasanaj et al. (2025)**: SSD aperiodic exponent data from Munich Clinical Deep Phenotyping study. *European Journal of Neuroscience*, 62, e70263.

## Known Limitations

- The sub-band gradient magnitude (+4.82 observed vs +1.91–3.40 predicted) exceeds the two-pole cascade prediction, suggesting higher-order or distributed filter models are needed for quantitative magnitude matching.
- The MT amplitude decay exponent (α = 0.5) has not been directly measured.
- MT resonance data derive exclusively from the Bandyopadhyay laboratory; independent replication has not been achieved.
- The model treats the sub-harmonic cascade as a mathematical framework conditional on the validity of the MHz/GHz-to-Hz coupling mechanism.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{parwez2026cascaded,
  title={Microtubule Sub-Harmonics as Broadband Neural Source: A Cascaded Tissue Filter Model of the EEG Aperiodic Spectrum with Literature-Constrained Biophysical Parameters},
  author={Parwez, Syed Mohsin},
  journal={bioRxiv},
  year={2026},
  doi={PLACEHOLDER}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

The author acknowledges the use of Claude (Anthropic) for writing assistance, data analysis support, and code generation throughout this research. All scientific hypotheses, theoretical framework design, model architecture, interpretation of results, and final conclusions are the sole intellectual contribution and responsibility of the author.
