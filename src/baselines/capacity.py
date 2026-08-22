"""Baseline A: Capacity-based degradation detection."""

from __future__ import annotations

import numpy as np
from typing import Optional


def detect_degradation_capacity(capacity: np.ndarray, threshold: float = 0.8) -> dict:
    """Detect degradation when capacity drops below threshold of initial capacity."""
    initial = capacity[0]
    degraded = np.where(capacity < initial * threshold)[0]

    if len(degraded) > 0:
        onset_cycle = degraded[0]
        warning_cycle = degraded[0]
    else:
        onset_cycle = -1
        warning_cycle = -1

    return {
        "onset_cycle": onset_cycle,
        "warning_cycle": warning_cycle,
        "lead_time": 0,
        "threshold": threshold,
    }


def detect_degradation_capacity_rate(capacity: np.ndarray, window: int = 50) -> dict:
    """Detect degradation by rate of capacity decline."""
    if len(capacity) < window:
        return {"onset_cycle": -1, "warning_cycle": -1}

    rates = []
    for i in range(window, len(capacity)):
        rate = (capacity[i] - capacity[i - window]) / window
        rates.append(rate)

    # Find where rate becomes significantly negative
    rates = np.array(rates)
    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    threshold = mean_rate - 2 * std_rate

    decline_start = np.where(rates < threshold)[0]
    if len(decline_start) > 0:
        onset_cycle = decline_start[0] + window
    else:
        onset_cycle = -1

    return {
        "onset_cycle": onset_cycle,
        "warning_cycle": onset_cycle,
        "lead_time": 0,
        "rates": rates.tolist(),
    }
