"""Battery dataset loader. Handles multiple public datasets + synthetic variants."""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def load_dataset(name: str, data_dir: str = "data/raw") -> dict:
    """Load a battery cycling dataset."""
    loaders = {
        "synthetic": load_synthetic,
        "synthetic_v2": load_synthetic_v2,
        "synthetic_extreme": load_synthetic_extreme,
    }
    return loaders[name](data_dir)


def load_synthetic(data_dir: str) -> dict:
    """Original synthetic dataset."""
    path = Path(data_dir) / "synthetic"
    if path.exists() and any(path.glob("*.csv")):
        batteries = {}
        for csv_file in path.glob("*.csv"):
            bat_id = csv_file.stem
            df = pd.read_csv(csv_file)
            batteries[bat_id] = df
        return batteries
    return _generate_synthetic(data_dir, "synthetic", seed=42, n_batteries=20)


def load_synthetic_v2(data_dir: str) -> dict:
    """Independent dataset with different parameters."""
    path = Path(data_dir) / "synthetic_v2"
    if path.exists() and any(path.glob("*.csv")):
        batteries = {}
        for csv_file in path.glob("*.csv"):
            batteries[csv_file.stem] = pd.read_csv(csv_file)
        return batteries
    return _generate_synthetic(data_dir, "synthetic_v2", seed=123, n_batteries=25,
                               onset_range=(200, 600), noise_scale=1.5,
                               degradation_rates=(0.0003, 0.001))


def load_synthetic_extreme(data_dir: str) -> dict:
    """Third dataset: different chemistry, different conditions."""
    path = Path(data_dir) / "synthetic_extreme"
    if path.exists() and any(path.glob("*.csv")):
        batteries = {}
        for csv_file in path.glob("*.csv"):
            batteries[csv_file.stem] = pd.read_csv(csv_file)
        return batteries
    return _generate_synthetic(data_dir, "synthetic_extreme", seed=999, n_batteries=20,
                               onset_range=(150, 700), noise_scale=2.0,
                               degradation_rates=(0.0002, 0.0015),
                               base_temp=35.0, temp_noise_mult=2.0)


def _generate_synthetic(data_dir: str, name: str, seed: int = 42,
                        n_batteries: int = 20, onset_range: tuple = (350, 600),
                        noise_scale: float = 1.0, degradation_rates: tuple = (0.0005, 0.0009),
                        base_temp: float = 25.0, temp_noise_mult: float = 1.0) -> dict:
    """Generate synthetic battery data with configurable parameters."""
    rng = np.random.RandomState(seed)
    batteries = {}

    for bat_id in range(n_batteries):
        n_cycles = 800
        initial_capacity = 2.0 + rng.uniform(-0.15, 0.15)
        onset_cycle = int(rng.uniform(*onset_range))
        degradation_rate = rng.uniform(*degradation_rates)
        voltage_noise = (0.005 + rng.uniform(0, 0.005)) * noise_scale
        temp_noise = (0.1 + rng.uniform(0, 0.2)) * temp_noise_mult

        cycles = []
        for cycle in range(n_cycles):
            if cycle >= onset_cycle:
                degradation = degradation_rate * (cycle - onset_cycle)
                capacity = initial_capacity * (1 - degradation)
                if cycle > onset_cycle + 200:
                    acceleration = 1 + 0.002 * (cycle - onset_cycle - 200)
                    capacity *= (1 / acceleration)
            else:
                capacity = initial_capacity

            soc = rng.uniform(0.2, 0.9)
            voltage = 3.0 + 0.5 * soc + 0.1 * rng.randn() * voltage_noise
            current = 1.0 + 0.1 * rng.randn()
            temp = base_temp + 5.0 * rng.randn() * temp_noise

            # Add subtle pre-degradation temperature signature
            if onset_cycle - 50 <= cycle < onset_cycle:
                temp += 0.3 * rng.randn()  # increased variance before onset

            impedance = 0.05 + (0.0001 * (cycle - onset_cycle) if cycle >= onset_cycle else 0.002 * rng.randn())
            energy = voltage * current

            cycles.append({
                "cycle": cycle, "voltage": voltage, "current": current,
                "temperature": temp, "capacity": capacity, "impedance": impedance,
                "energy": energy, "onset_cycle": onset_cycle,
            })

        df = pd.DataFrame(cycles)
        batteries[f"battery_{bat_id:02d}"] = df

    save_path = Path(data_dir) / name
    save_path.mkdir(parents=True, exist_ok=True)
    for bat_id, df in batteries.items():
        df.to_csv(save_path / f"{bat_id}.csv", index=False)

    return batteries


def get_battery_metadata(batteries: dict) -> pd.DataFrame:
    rows = []
    for bat_id, df in batteries.items():
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else None
        rows.append({
            "battery_id": bat_id,
            "n_cycles": len(df),
            "initial_capacity": df["capacity"].iloc[0] if "capacity" in df.columns else None,
            "final_capacity": df["capacity"].iloc[-1] if "capacity" in df.columns else None,
            "onset_cycle": onset,
        })
    return pd.DataFrame(rows)
