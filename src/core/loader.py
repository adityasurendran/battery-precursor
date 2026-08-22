"""Battery dataset loader. Handles multiple public datasets."""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def load_dataset(name: str, data_dir: str = "data/raw") -> dict:
    """Load a battery cycling dataset. Returns dict of battery_id -> DataFrame."""
    loaders = {
        "nasa": load_nasa,
        "calce": load_calce,
        "oxford": load_oxford,
        "synthetic": load_synthetic,
    }
    return loaders[name](data_dir)


def load_nasa(data_dir: str) -> dict:
    """NASA battery dataset. Download from:
    https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
    """
    path = Path(data_dir) / "nasa"
    if not path.exists():
        print(f"  NASA dataset not found at {path}")
        print("  Download from: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository")
        print("  Or use synthetic dataset")
        return load_synthetic(data_dir)

    batteries = {}
    for csv_file in path.glob("*.csv"):
        bat_id = csv_file.stem
        df = pd.read_csv(csv_file)
        batteries[bat_id] = df
    return batteries


def load_calce(data_dir: str) -> dict:
    """CALCE battery dataset from University of Maryland."""
    path = Path(data_dir) / "calce"
    if not path.exists():
        print(f"  CALCE dataset not found at {path}")
        return load_synthetic(data_dir)
    batteries = {}
    for csv_file in path.glob("*.csv"):
        bat_id = csv_file.stem
        df = pd.read_csv(csv_file)
        batteries[bat_id] = df
    return batteries


def load_oxford(data_dir: str) -> dict:
    """Oxford battery degradation dataset."""
    path = Path(data_dir) / "oxford"
    if not path.exists():
        print(f"  Oxford dataset not found at {path}")
        return load_synthetic(data_dir)
    batteries = {}
    for csv_file in path.glob("*.csv"):
        bat_id = csv_file.stem
        df = pd.read_csv(csv_file)
        batteries[bat_id] = df
    return batteries


def load_synthetic(data_dir: str) -> dict:
    """Generate synthetic battery cycling data for feasibility testing.

    Creates 20 batteries with realistic degradation patterns.
    Each battery has voltage, current, temperature, capacity per cycle.
    """
    print("  Generating synthetic battery data (20 batteries, 800 cycles each)")
    rng = np.random.RandomState(42)
    batteries = {}

    for bat_id in range(20):
        n_cycles = 800
        initial_capacity = 2.0 + rng.uniform(-0.1, 0.1)  # Ah

        # Degradation onset: varies per battery (the "hidden transition")
        onset_cycle = int(400 + rng.uniform(-100, 200))

        # Degradation rate varies per battery
        degradation_rate = 0.0005 + rng.uniform(-0.0002, 0.0004)

        # Noise levels
        voltage_noise = 0.005 + rng.uniform(0, 0.005)
        temp_noise = 0.1 + rng.uniform(0, 0.2)

        cycles = []
        for cycle in range(n_cycles):
            # Capacity degrades after onset
            if cycle >= onset_cycle:
                degradation = degradation_rate * (cycle - onset_cycle)
                capacity = initial_capacity * (1 - degradation)
                # Accelerating degradation near end
                if cycle > onset_cycle + 200:
                    acceleration = 1 + 0.002 * (cycle - onset_cycle - 200)
                    capacity *= (1 / acceleration)
            else:
                capacity = initial_capacity

            # Voltage curve (simplified)
            soc = rng.uniform(0.2, 0.9)
            voltage = 3.0 + 0.5 * soc + 0.1 * rng.randn() * voltage_noise

            # Current (charge/discharge)
            current = 1.0 + 0.1 * rng.randn()

            # Temperature
            temp = 25.0 + 5.0 * rng.randn() * temp_noise

            # Impedance (increases with degradation)
            if cycle >= onset_cycle:
                impedance = 0.05 + 0.0001 * (cycle - onset_cycle)
            else:
                impedance = 0.05 + 0.002 * rng.randn()

            # Energy
            energy = voltage * current

            cycles.append({
                "cycle": cycle,
                "voltage": voltage,
                "current": current,
                "temperature": temp,
                "capacity": capacity,
                "impedance": impedance,
                "energy": energy,
                "onset_cycle": onset_cycle,
            })

        df = pd.DataFrame(cycles)
        batteries[f"battery_{bat_id:02d}"] = df

    # Save to disk
    save_path = Path(data_dir) / "synthetic"
    save_path.mkdir(parents=True, exist_ok=True)
    for bat_id, df in batteries.items():
        df.to_csv(save_path / f"{bat_id}.csv", index=False)

    print(f"  Saved {len(batteries)} batteries to {save_path}")
    return batteries


def get_battery_metadata(batteries: dict) -> pd.DataFrame:
    """Extract metadata from battery datasets."""
    rows = []
    for bat_id, df in batteries.items():
        if "onset_cycle" in df.columns:
            onset = df["onset_cycle"].iloc[0]
        else:
            onset = None
        rows.append({
            "battery_id": bat_id,
            "n_cycles": len(df),
            "initial_capacity": df["capacity"].iloc[0] if "capacity" in df.columns else None,
            "final_capacity": df["capacity"].iloc[-1] if "capacity" in df.columns else None,
            "onset_cycle": onset,
        })
    return pd.DataFrame(rows)
