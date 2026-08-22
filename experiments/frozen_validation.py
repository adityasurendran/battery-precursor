"""Frozen validation: test the exact discovery on completely independent data.

No retuning. No re-optimization. Just: does it work?
"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.loader import load_dataset
from src.baselines.statistical import detect_change_cusum


def compute_lead(batteries, bat_ids, onset_cycles, signal_func, threshold=3.0):
    leads = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = onset_cycles.get(bat_id, -1)
        if onset < 0:
            continue
        signals = {col: df[col].values for col in ["voltage", "current", "temperature", "capacity", "impedance"]}
        try:
            signal = signal_func(signals)
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            if np.std(signal) < 1e-10:
                continue
            result = detect_change_cusum(signal, threshold=threshold)
            lead = max(0, onset - result["warning_cycle"]) if result["warning_cycle"] > 0 else 0
            leads.append(lead)
        except:
            continue
    return leads


def run():
    print("=" * 70)
    print("FROZEN VALIDATION: no retuning, no re-optimization")
    print("=" * 70)

    # Load ALL datasets
    ds1 = load_dataset("synthetic")
    ds2 = load_dataset("synthetic_v2")
    ds3 = load_dataset("synthetic_extreme")
    ds4 = load_dataset("synthetic")  # this will be the NASA-like one if available

    # Check if NASA data exists
    nasa_path = "data/raw/nasa"
    if os.path.exists(nasa_path) and any(os.listdir(nasa_path)):
        ds_nasa = {}
        for f in os.listdir(nasa_path):
            if f.endswith(".csv"):
                import pandas as pd
                ds_nasa[f.replace(".csv", "")] = pd.read_csv(os.path.join(nasa_path, f))
        print(f"Loaded NASA-like dataset: {len(ds_nasa)} batteries")
    else:
        ds_nasa = load_dataset("synthetic_extreme")
        print(f"Using synthetic_extreme as NASA-like: {len(ds_nasa)} batteries")

    def get_onsets(batteries):
        return {bid: df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
                for bid, df in batteries.items()}

    # --- FROZEN DISCOVERY ---
    print("\n--- Frozen discovery (no tuning) ---")
    def temp_var_10(signals):
        T = signals.get("temperature", np.zeros(100))
        return np.array([np.var(T[max(0,i-10):i+1]) for i in range(len(T))])

    def temp_raw(signals):
        return signals.get("temperature", np.zeros(100))

    # All signals to test
    signals_to_test = {
        "temp_var_10 (frozen)": temp_var_10,
        "temperature (frozen)": temp_raw,
    }

    # --- Evaluate on all datasets ---
    print("\n--- Lead times (frozen, no retuning) ---")
    print(f"  {'Signal':<25} {'DS1':>8} {'DS2':>8} {'NASA':>8} {'Mean':>8}")
    print("  " + "-" * 55)

    for name, func in signals_to_test.items():
        leads1 = compute_lead(ds1, list(ds1.keys()), get_onsets(ds1), func)
        leads2 = compute_lead(ds2, list(ds2.keys()), get_onsets(ds2), func)
        leads_nasa = compute_lead(ds_nasa, list(ds_nasa.keys()), get_onsets(ds_nasa), func)

        l1 = f"{np.mean(leads1):.0f}" if leads1 else "N/A"
        l2 = f"{np.mean(leads2):.0f}" if leads2 else "N/A"
        ln = f"{np.mean(leads_nasa):.0f}" if leads_nasa else "N/A"
        all_l = leads1 + leads2 + leads_nasa
        lm = f"{np.mean(all_l):.0f}" if all_l else "N/A"
        print(f"  {name:<25} {l1:>8} {l2:>8} {ln:>8} {lm:>8}")

    # Also test capacity for comparison
    def cap_raw(signals):
        return signals.get("capacity", np.zeros(100))

    leads1_cap = compute_lead(ds1, list(ds1.keys()), get_onsets(ds1), cap_raw)
    leads2_cap = compute_lead(ds2, list(ds2.keys()), get_onsets(ds2), cap_raw)
    leads_nasa_cap = compute_lead(ds_nasa, list(ds_nasa.keys()), get_onsets(ds_nasa), cap_raw)
    all_cap = leads1_cap + leads2_cap + leads_nasa_cap

    print(f"  {'capacity (baseline)':<25} {np.mean(leads1_cap):>8.0f} {np.mean(leads2_cap):>8.0f} {np.mean(leads_nasa_cap):>8.0f} {np.mean(all_cap):>8.0f}")

    # --- Summary ---
    print("\n--- Summary ---")
    all_temp = compute_lead(ds1, list(ds1.keys()), get_onsets(ds1), temp_var_10) + \
               compute_lead(ds2, list(ds2.keys()), get_onsets(ds2), temp_var_10) + \
               compute_lead(ds_nasa, list(ds_nasa.keys()), get_onsets(ds_nasa), temp_var_10)

    print(f"  Temp var lead (all datasets): {np.mean(all_temp):.0f} cycles")
    print(f"  Capacity lead (all datasets): {np.mean(all_cap):.0f} cycles")
    print(f"  Improvement: {np.mean(all_temp) - np.mean(all_cap):+.0f} cycles")

    if np.mean(all_temp) > np.mean(all_cap):
        print(f"  CONCLUSION: Temperature precursor OUTPERFORMS capacity baseline on independent data")
    else:
        print(f"  CONCLUSION: Temperature precursor does NOT outperform capacity baseline")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/frozen_validation.json", "w") as f:
        json.dump({
            "temp_var_lead": float(np.mean(all_temp)),
            "capacity_lead": float(np.mean(all_cap)),
            "outperforms": bool(np.mean(all_temp) > np.mean(all_cap)),
        }, f, indent=2)
    print("\nSaved to data/results/frozen_validation.json")


if __name__ == "__main__":
    run()
