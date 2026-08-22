"""Baseline B: Statistical change-point detection."""

from __future__ import annotations

import numpy as np
from typing import Optional


def detect_change_cusum(signal: np.ndarray, threshold: float = 5.0) -> dict:
    """CUSUM change-point detection."""
    mean = np.mean(signal)
    cusum_pos = 0.0
    cusum_neg = 0.0
    change_points = []

    for i in range(len(signal)):
        diff = signal[i] - mean
        cusum_pos = max(0, cusum_pos + diff)
        cusum_neg = max(0, cusum_neg - diff)
        if cusum_pos > threshold or cusum_neg > threshold:
            change_points.append(i)
            cusum_pos = 0
            cusum_neg = 0

    if change_points:
        return {
            "onset_cycle": change_points[0],
            "warning_cycle": change_points[0],
            "n_changes": len(change_points),
            "change_points": change_points[:10],
        }
    return {"onset_cycle": -1, "warning_cycle": -1, "n_changes": 0}


def detect_change_pelt(signal: np.ndarray, penalty: float = 1.0) -> dict:
    """Simplified PELT change-point detection."""
    n = len(signal)
    if n < 10:
        return {"onset_cycle": -1, "warning_cycle": -1}

    # Sliding window comparison
    window = min(50, n // 4)
    changes = []
    for i in range(window, n - window):
        before = signal[i - window:i]
        after = signal[i:i + window]
        # Two-sample t-test statistic
        if np.std(before) > 0 and np.std(after) > 0:
            t_stat = abs(np.mean(before) - np.mean(after)) / np.sqrt(
                np.var(before) / window + np.var(after) / window
            )
            if t_stat > 2.0:
                changes.append(i)

    if changes:
        return {
            "onset_cycle": changes[0],
            "warning_cycle": changes[0],
            "n_changes": len(changes),
        }
    return {"onset_cycle": -1, "warning_cycle": -1, "n_changes": 0}


def detect_change_bayesian(signal: np.ndarray, prior_mean: float = None) -> dict:
    """Bayesian change-point detection (simplified)."""
    n = len(signal)
    if prior_mean is None:
        prior_mean = np.mean(signal[:min(100, n)])

    best_cp = -1
    best_score = 0

    for i in range(10, n - 10):
        before = signal[:i]
        after = signal[i:]
        # Log-likelihood ratio
        mu_before = np.mean(before)
        mu_after = np.mean(after)
        var_before = np.var(before) + 1e-10
        var_after = np.var(after) + 1e-10

        ll_before = -0.5 * np.sum(((before - mu_before) ** 2) / var_before)
        ll_after = -0.5 * np.sum(((after - mu_after) ** 2) / var_after)
        score = abs(ll_after - ll_before) / n

        if score > best_score:
            best_score = score
            best_cp = i

    if best_cp > 0:
        return {"onset_cycle": best_cp, "warning_cycle": best_cp, "score": best_score}
    return {"onset_cycle": -1, "warning_cycle": -1}
