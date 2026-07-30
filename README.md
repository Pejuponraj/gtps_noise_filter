# gtps_noise_filter
Generalized Topological Physiological Spaces: hereditary-class noise filter for persistent homology
# GTPS-TDA: A Hereditary-Class Noise Filter for Persistent Homology

Code and analysis scripts for the paper:

**"Generalized Topological Physiological Spaces: A Hereditary-Class Noise Filter for Persistent Homology of Physiological Signals"**
Ponraj A. P., Muthu Ramachandran, Hari Friedrich Schuth.

This repository reproduces every quantitative result reported in the paper on the
real BIDMC PhysioNet dataset.

## Data

The analysis uses the **BIDMC PPG and Respiration Dataset** (53 ICU recordings),
publicly available from PhysioNet:

> https://physionet.org/content/bidmc/1.0.0/

The main script downloads the records automatically via the WFDB package; no manual
download is required.

## Requirements

    pip install -r requirements.txt

Python 3.10+ recommended. Core packages: numpy, scipy, ripser, persim, wfdb.

## Scripts

| Script | What it produces |
|--------|------------------|
| `bidmc_full53.py` | Self-contained loader; builds the three-channel delay embedding for all 53 recordings and reports H1/H2 detection counts. |
| `bidmc_stats_fixed.py` | Main result: bottleneck distance of GTPS vs standard TDA to the true topology under additive contamination (Wilcoxon signed-rank, effect sizes). |
| `bidmc_gtps_vs_dtm_fair.py` | Fair three-way comparison of standard TDA, GTPS and DTM in an identical removal-plus-Rips framework (ground-truth metric, non-degeneracy check). |
| `bidmc_structured_contam.py` | Robustness across corruption models (additive, deletion, dropped frames). |
| `bidmc_h2_surrogate_v2.py` | IAAFT surrogate test for the degree-two features. |

## How to reproduce

Run from a directory where `bidmc_full53.py` and the downloaded `bidmc_data`
folder both live:

    python bidmc_full53.py            # loads data, detection counts
    python bidmc_stats_fixed.py       # main GTPS vs standard result
    python bidmc_gtps_vs_dtm_fair.py  # DTM comparison
    python bidmc_structured_contam.py # robustness by corruption model
    python bidmc_h2_surrogate_v2.py   # surrogate test

All scripts use a fixed random seed; the numbers they print are the numbers in the
paper's tables.

## Note on scope

The filter is evaluated on **additive** contamination (spurious points added to the
cloud). Homology degrees H0, H1, H2 are treated as mathematical descriptors of the
recovered structure; no physiological claim is attached to any individual degree.

## License

MIT License (see LICENSE).
