"""Feature generation: temporal, electrical, cross-signal transformations."""

from __future__ import annotations

import numpy as np
from typing import Optional


def generate_features(df: dict, window: int = 20) -> dict:
    """Generate all feature categories from raw signals.

    Input: dict of signal_name -> numpy array
    Output: dict of feature_name -> numpy array
    """
    features = {}

    # Temporal features
    for sig_name, sig in df.items():
        if not isinstance(sig, np.ndarray) or len(sig) < window:
            continue

        # Derivatives
        features[f"{sig_name}_d1"] = np.gradient(sig)
        features[f"{sig_name}_d2"] = np.gradient(np.gradient(sig))

        # Moving averages
        kernel = np.ones(window) / window
        features[f"{sig_name}_ma"] = np.convolve(sig, kernel, mode="same")

        # Variance (rolling)
        features[f"{sig_name}_var"] = rolling_var(sig, window)

        # Autocorrelation at lag 1
        features[f"{sig_name}_acf1"] = rolling_autocorr(sig, window, lag=1)

        # Frequency components (FFT magnitude at dominant frequency)
        features[f"{sig_name}_fft_dom"] = rolling_fft_dominant(sig, window)

        # Change-point statistics
        features[f"{sig_name}_cpd"] = rolling_cusum(sig, window)

    # Cross-signal features
    if "voltage" in df and "current" in df:
        V = df["voltage"]
        I = df["current"]
        features["dV_dI"] = np.gradient(V) / (np.gradient(I) + 1e-10)
        features["V_times_I"] = V * I
        features["power"] = V * I

    if "voltage" in df and "temperature" in df:
        V = df["voltage"]
        T = df["temperature"]
        features["dT_dV"] = np.gradient(T) / (np.gradient(V) + 1e-10)

    if "current" in df and "temperature" in df:
        I = df["current"]
        T = df["temperature"]
        features["dT_dI"] = np.gradient(T) / (np.gradient(I) + 1e-10)

    if "voltage" in df and "current" in df:
        V = df["voltage"]
        I = df["current"]
        features["impedance_proxy"] = V / (I + 1e-10)

    # Charge/discharge asymmetry
    if "current" in df:
        I = df["current"]
        features["charge_ratio"] = np.where(I > 0, I, 0)
        features["discharge_ratio"] = np.where(I < 0, -I, 0)

    # Hysteresis proxy
    if "voltage" in df and "current" in df:
        V = df["voltage"]
        I = df["current"]
        features["hysteresis"] = np.abs(V) * np.sign(I)

    return features


def rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling variance."""
    result = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window)
        result[i] = np.var(x[start:i+1])
    return result


def rolling_autocorr(x: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    """Rolling autocorrelation at given lag."""
    result = np.zeros_like(x, dtype=float)
    for i in range(lag, len(x)):
        start = max(0, i - window)
        segment = x[start:i+1]
        if len(segment) > lag:
            result[i] = np.corrcoef(segment[:-lag], segment[lag:])[0, 1]
    return result


def rolling_fft_dominant(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling FFT dominant frequency magnitude."""
    result = np.zeros_like(x, dtype=float)
    for i in range(window, len(x)):
        segment = x[i-window:i]
        fft = np.abs(np.fft.rfft(segment))
        if len(fft) > 1:
            result[i] = np.max(fft[1:])  # skip DC
    return result


def rolling_cusum(x: np.ndarray, window: int) -> np.ndarray:
    """Cumulative sum change-point detection."""
    result = np.zeros_like(x, dtype=float)
    mean = np.mean(x[:window]) if len(x) >= window else np.mean(x)
    cusum_pos = 0.0
    cusum_neg = 0.0
    for i in range(len(x)):
        diff = x[i] - mean
        cusum_pos = max(0, cusum_pos + diff)
        cusum_neg = max(0, cusum_neg - diff)
        result[i] = max(cusum_pos, cusum_neg)
    return result


def extract_cycle_features(cycles_df: dict) -> dict:
    """Extract features from cycle-level data."""
    features = {}
    for sig_name, sig in cycles_df.items():
        if not isinstance(sig, np.ndarray) or len(sig) < 2:
            continue
        features[f"{sig_name}_mean"] = np.full(len(sig), np.mean(sig))
        features[f"{sig_name}_std"] = np.full(len(sig), np.std(sig))
    return features
