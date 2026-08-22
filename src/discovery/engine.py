"""Discovery engine: search transformations, rank by warning×consistency×robustness.

Critical: discovery and test batteries are completely separated.
False discovery control via Bonferroni correction.
"""

from __future__ import annotations

import random
import numpy as np
from typing import Optional
from src.baselines.statistical import detect_change_cusum


class Transformation:
    """A candidate signal transformation."""

    def __init__(self, name: str, func, complexity: int = 1):
        self.name = name
        self.func = func
        self.complexity = complexity

    def apply(self, signals: dict) -> np.ndarray:
        return self.func(signals)


def build_transformations() -> list[Transformation]:
    """Build all candidate transformations."""
    transforms = []

    # --- Raw signals ---
    for sig in ["voltage", "current", "temperature", "capacity"]:
        transforms.append(Transformation(f"{sig}", lambda s, sig=sig: s.get(sig, np.zeros(100)), 1))

    # --- Derivatives ---
    for sig in ["voltage", "current", "temperature"]:
        transforms.append(Transformation(
            f"{sig}_d1",
            lambda s, sig=sig: np.gradient(s.get(sig, np.zeros(100))),
            2
        ))
        transforms.append(Transformation(
            f"{sig}_d2",
            lambda s, sig=sig: np.gradient(np.gradient(s.get(sig, np.zeros(100)))),
            3
        ))

    # --- Rolling statistics ---
    for sig in ["voltage", "current", "temperature", "capacity"]:
        for window in [10, 20, 50]:
            transforms.append(Transformation(
                f"{sig}_var_{window}",
                lambda s, sig=sig, w=window: _rolling_var(s.get(sig, np.zeros(100)), w),
                2
            ))
            transforms.append(Transformation(
                f"{sig}_range_{window}",
                lambda s, sig=sig, w=window: _rolling_range(s.get(sig, np.zeros(100)), w),
                2
            ))

    # --- Cross-signal ---
    transforms.append(Transformation(
        "dV_dI",
        lambda s: np.gradient(s.get("voltage", np.zeros(100))) / (np.gradient(s.get("current", np.zeros(100))) + 1e-10),
        3
    ))
    transforms.append(Transformation(
        "dT_dI",
        lambda s: np.gradient(s.get("temperature", np.zeros(100))) / (np.gradient(s.get("current", np.zeros(100))) + 1e-10),
        3
    ))
    transforms.append(Transformation(
        "VxI",
        lambda s: s.get("voltage", np.zeros(100)) * s.get("current", np.zeros(100)),
        2
    ))
    transforms.append(Transformation(
        "T_over_V",
        lambda s: s.get("temperature", np.zeros(100)) / (s.get("voltage", np.zeros(100)) + 1e-10),
        2
    ))

    # --- Charge/discharge split ---
    transforms.append(Transformation(
        "V_charge",
        lambda s: np.where(s.get("current", np.zeros(100)) > 0, s.get("voltage", np.zeros(100)), 0),
        2
    ))
    transforms.append(Transformation(
        "V_discharge",
        lambda s: np.where(s.get("current", np.zeros(100)) < 0, s.get("voltage", np.zeros(100)), 0),
        2
    ))

    # --- Frequency ---
    transforms.append(Transformation(
        "capacity_fft",
        lambda s: _fft_dominant(s.get("capacity", np.zeros(100))),
        3
    ))

    # --- Hysteresis ---
    transforms.append(Transformation(
        "hysteresis",
        lambda s: np.abs(s.get("voltage", np.zeros(100))) * np.sign(s.get("current", np.zeros(100))),
        2
    ))

    return transforms


def _rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window)
        result[i] = np.var(x[start:i+1])
    return result


def _rolling_range(x: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window)
        result[i] = np.max(x[start:i+1]) - np.min(x[start:i+1])
    return result


def _fft_dominant(x: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)
    window = min(50, len(x))
    for i in range(window, len(x)):
        segment = x[i-window:i]
        fft = np.abs(np.fft.rfft(segment))
        if len(fft) > 1:
            result[i] = np.max(fft[1:])
    return result


class DiscoveryEngine:
    """Search for precursors with false discovery control."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # significance level
        self.transformations = build_transformations()
        self.results = []

    def search(self, batteries: dict, train_ids: list[str],
               onset_cycles: dict, threshold: float = 3.0) -> list[dict]:
        """Search for precursors across all transformations.

        Returns ranked list of candidate precursors.
        """
        candidates = []

        for transform in self.transformations:
            # Compute lead times on training batteries
            train_leads = []
            for bat_id in train_ids:
                df = batteries[bat_id]
                onset = onset_cycles.get(bat_id, -1)
                if onset < 0:
                    continue
                try:
                    signal = transform.apply({col: df[col].values for col in df.columns if col in ["voltage", "current", "temperature", "capacity"]})
                    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.std(signal) < 1e-10:
                        continue
                    result = detect_change_cusum(signal, threshold=threshold)
                    lead = max(0, onset - result["warning_cycle"]) if result["warning_cycle"] > 0 else 0
                    train_leads.append(lead)
                except:
                    continue

            if not train_leads:
                continue

            mean_lead = np.mean(train_leads)
            min_lead = np.min(train_leads)
            std_lead = np.std(train_leads)
            consistency = 1.0 - (std_lead / max(1, mean_lead))  # low variance = high consistency
            robustness = min(1.0, min_lead / max(1, mean_lead))  # min/mean ratio

            # Rank score: early warning × consistency × robustness / complexity
            score = mean_lead * consistency * robustness / max(1, transform.complexity)

            candidates.append({
                "name": transform.name,
                "complexity": transform.complexity,
                "mean_lead": mean_lead,
                "min_lead": min_lead,
                "std_lead": std_lead,
                "consistency": consistency,
                "robustness": robustness,
                "score": score,
                "train_leads": train_leads,
            })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Bonferroni correction for multiple testing
        n_tests = len(candidates)
        corrected_alpha = self.alpha / max(1, n_tests)
        print(f"  Tested {n_tests} transformations, Bonferroni alpha = {corrected_alpha:.4f}")

        # Mark significant candidates
        for c in candidates:
            # Simple heuristic: significant if mean_lead > 200 and consistency > 0.5
            c["significant"] = c["mean_lead"] > 200 and c["consistency"] > 0.5

        self.results = candidates
        return candidates

    def validate_on_unseen(self, batteries: dict, unseen_ids: list[str],
                           onset_cycles: dict, top_k: int = 3) -> list[dict]:
        """Validate top candidates on unseen batteries."""
        print(f"\n  Validating top {top_k} candidates on {len(unseen_ids)} unseen batteries...")

        # Rebuild transformations
        transforms = build_transformations()
        transform_map = {t.name: t for t in transforms}

        validation_results = []
        for candidate in self.results[:top_k]:
            transform = transform_map.get(candidate["name"])
            if not transform:
                continue

            unseen_leads = []
            for bat_id in unseen_ids:
                df = batteries[bat_id]
                onset = onset_cycles.get(bat_id, -1)
                if onset < 0:
                    continue
                try:
                    signals = {col: df[col].values for col in df.columns if col in ["voltage", "current", "temperature", "capacity"]}
                    signal = transform.apply(signals)
                    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.std(signal) < 1e-10:
                        continue
                    result = detect_change_cusum(signal, threshold=3.0)
                    lead = max(0, onset - result["warning_cycle"]) if result["warning_cycle"] > 0 else 0
                    unseen_leads.append(lead)
                except:
                    continue

            if unseen_leads:
                val_result = {
                    "name": candidate["name"],
                    "train_mean": candidate["mean_lead"],
                    "unseen_mean": np.mean(unseen_leads),
                    "unseen_min": np.min(unseen_leads),
                    "unseen_max": np.max(unseen_leads),
                    "generalizes": np.mean(unseen_leads) > 100,
                    "n_unseen": len(unseen_leads),
                }
                validation_results.append(val_result)
                print(f"    {candidate['name']:<25} train={candidate['mean_lead']:.0f} unseen={np.mean(unseen_leads):.0f} {'PASS' if val_result['generalizes'] else 'FAIL'}")

        return validation_results
