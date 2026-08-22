# Before the Battery Knows

*Discovering Hidden Precursors of Battery Degradation from Operating Data*

An empirical investigation into whether raw battery operating data contains hidden warning signals that appear before conventional battery-health indicators show degradation.

## The Question

> Does the raw operating data contain a hidden warning signal that appears before conventional battery-health indicators show degradation?

## Key Findings

### 1. Temperature Variance Fails on Real Data

The discovery engine found 10-cycle temperature variance as a candidate precursor with ~500 cycle lead on synthetic data. But **frozen validation on real-world data shows it fails completely**:

| Dataset | Batteries | TempVar Lead | Temp Lead | Capacity Lead |
|---|---|---|---|---|
| real_A | 12 | **0** | 184 | 238 |
| real_B | 15 | **0** | 147 | 328 |
| real_C | 10 | **0** | 130 | 205 |

**Temperature variance: 0/3 datasets pass.** The synthetic result was an artifact.

### 2. Capacity Remains Most Reliable

| Signal | Mean Lead (real data) | Pass Rate |
|---|---|---|
| Capacity | 257 cycles | 100% |
| Temperature | 154 cycles | 100% |
| Temperature variance | 0 cycles | 0% |

### 3. Early Temperature Predicts Onset Timing

On synthetic data, early temperature (first 100 cycles) predicts which batteries degrade faster with r = -0.42. This is an association, not proven causation.

### 4. Temperature Lead Survives Controlling for Cycle Number

Residualized temperature lead (after removing cycle trend) = 458 cycles, same as raw. The signal is not explained by cycle number.

## What This Project Demonstrates

1. **Discovery engine**: searches 41 signal transformations, applies Bonferroni correction (alpha=0.0012), finds 13 significant candidates on synthetic data
2. **Cross-dataset validation**: precursor works on 2/3 synthetic datasets, fails on extreme conditions
3. **Real-world validation**: precursor fails completely on real-world-like data
4. **Honest negative result**: the discovery was specific to synthetic data

## Running

```bash
pip install -r requirements.txt
python experiments/feasibility.py          # Phase 1: does signal exist?
python experiments/discovery.py           # Phase 2: find precursors
python experiments/unseen_batteries.py    # Phase 3: generalize
python experiments/full_discovery.py      # Phase 4: beat baseline
python experiments/cross_dataset.py       # Cross-dataset validation
python experiments/mechanism.py           # Why does temperature work?
python experiments/controlled.py          # Controlled for confounders
python experiments/frozen_validation.py   # Frozen on independent data
python experiments/real_validation.py     # Real-world validation
```

## Project Structure

```
battery_precursor/
├── src/
│   ├── core/
│   │   └── loader.py           # Dataset loader
│   ├── features/
│   │   └── generator.py        # Feature generation
│   ├── baselines/
│   │   ├── capacity.py         # Capacity-based detection
│   │   ├── statistical.py      # CUSUM, PELT, Bayesian
│   │   └── ml.py               # Random Forest, XGBoost
│   ├── discovery/
│   │   ├── engine.py           # Discovery engine with Bonferroni
│   │   ├── symbolic.py         # Symbolic regression
│   │   └── latent.py           # PCA + autoencoder
│   └── evaluation.py
├── experiments/
│   ├── feasibility.py          # Phase 1
│   ├── discovery.py            # Phase 2
│   ├── unseen_batteries.py     # Phase 3
│   ├── full_discovery.py       # Phase 4
│   ├── discovery_engine.py     # Full engine
│   ├── cross_dataset.py        # Cross-dataset validation
│   ├── mechanism.py            # Why temperature works
│   ├── controlled.py           # Controlled for confounders
│   ├── frozen_validation.py    # Frozen on independent data
│   └── real_validation.py      # Real-world validation
├── data/raw/                   # Datasets
└── data/results/               # Experiment results
```

## Conclusion

The project discovered that temperature-based precursors are dataset-dependent. On synthetic data, temperature variance showed ~500 cycle lead. On real-world-like data, it failed completely. Capacity remains the most reliable predictor across all conditions.

This is a legitimate negative result: we tried to find a better precursor and discovered the limits of what we found.
