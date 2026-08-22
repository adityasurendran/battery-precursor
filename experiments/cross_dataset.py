"""Cross-dataset validation: freeze discovery, test on independent dataset."""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.loader import load_dataset, get_battery_metadata
from src.baselines.statistical import detect_change_cusum
from src.features.generator import generate_features


def compute_lead(batteries, bat_ids, onset_cycles, signal_func, threshold=3.0):
    """Compute lead times for a signal function across batteries."""
    leads = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = onset_cycles.get(bat_id, -1)
        if onset < 0:
            continue
        signals = {col: df[col].values for col in ["voltage", "current", "temperature", "capacity"]}
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
    print("CROSS-DATASET VALIDATION: frozen discovery on independent data")
    print("=" * 70)

    # --- Load datasets ---
    print("\n--- Loading datasets ---")
    ds1 = load_dataset("synthetic")
    ds2 = load_dataset("synthetic_v2")
    ds3 = load_dataset("synthetic_extreme")

    print(f"  Dataset 1 (original): {len(ds1)} batteries")
    print(f"  Dataset 2 (independent): {len(ds2)} batteries")
    print(f"  Dataset 3 (extreme): {len(ds3)} batteries")

    # Get onset cycles
    def get_onsets(batteries):
        onsets = {}
        for bat_id, df in batteries.items():
            onsets[bat_id] = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        return onsets

    onsets1 = get_onsets(ds1)
    onsets2 = get_onsets(ds2)
    onsets3 = get_onsets(ds3)

    # --- Define the frozen precursor ---
    print("\n--- Frozen precursor: 10-cycle temperature variance ---")
    def temp_var_10(signals):
        T = signals.get("temperature", np.zeros(100))
        result = np.zeros_like(T, dtype=float)
        for i in range(10, len(T)):
            result[i] = np.var(T[i-10:i])
        return result

    # Also test raw temperature and capacity for comparison
    def temp_raw(signals):
        return signals.get("temperature", np.zeros(100))

    def capacity_raw(signals):
        return signals.get("capacity", np.zeros(100))

    # --- Evaluate on all 3 datasets ---
    print("\n--- Lead times on all datasets ---")
    print(f"  {'Signal':<25} {'DS1 (n=20)':<15} {'DS2 (n=25)':<15} {'DS3 (n=20)':<15}")
    print("  " + "-" * 70)

    for name, func in [("temp_var_10 (FROZEN)", temp_var_10),
                       ("temperature", temp_raw),
                       ("capacity", capacity_raw)]:
        leads1 = compute_lead(ds1, list(ds1.keys()), onsets1, func)
        leads2 = compute_lead(ds2, list(ds2.keys()), onsets2, func)
        leads3 = compute_lead(ds3, list(ds3.keys()), onsets3, func)

        l1 = f"{np.mean(leads1):.0f}" if leads1 else "N/A"
        l2 = f"{np.mean(leads2):.0f}" if leads2 else "N/A"
        l3 = f"{np.mean(leads3):.0f}" if leads3 else "N/A"
        print(f"  {name:<25} {l1:>12}  {l2:>12}  {l3:>12}")

    # --- Detailed breakdown ---
    print("\n--- Detailed breakdown per dataset ---")
    for ds_name, ds, ons in [("DS1", ds1, onsets1), ("DS2", ds2, onsets2), ("DS3", ds3, onsets3)]:
        leads = compute_lead(ds, list(ds.keys()), ons, temp_var_10)
        if leads:
            print(f"\n  {ds_name} (n={len(ds)}):")
            print(f"    Mean: {np.mean(leads):.0f}, Min: {np.min(leads):.0f}, Max: {np.max(leads):.0f}")
            print(f"    Std: {np.std(leads):.0f}")
            # Per-battery breakdown
            for bat_id in list(ds.keys())[:5]:
                df = ds[bat_id]
                onset = ons[bat_id]
                signals = {col: df[col].values for col in ["voltage", "current", "temperature", "capacity"]}
                signal = temp_var_10(signals)
                signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
                result = detect_change_cusum(signal, threshold=3.0)
                lead = max(0, onset - result["warning_cycle"]) if result["warning_cycle"] > 0 else 0
                print(f"    {bat_id}: onset={onset}, detected={result['warning_cycle']}, lead={lead}")

    # --- Summary ---
    print("\n--- Summary ---")
    leads1 = compute_lead(ds1, list(ds1.keys()), onsets1, temp_var_10)
    leads2 = compute_lead(ds2, list(ds2.keys()), onsets2, temp_var_10)
    leads3 = compute_lead(ds3, list(ds3.keys()), onsets3, temp_var_10)
    all_leads = leads1 + leads2 + leads3

    print(f"  Cross-dataset mean lead: {np.mean(all_leads):.0f} cycles")
    print(f"  DS1: {np.mean(leads1):.0f}, DS2: {np.mean(leads2):.0f}, DS3: {np.mean(leads3):.0f}")
    print(f"  Generalizes: {'YES' if np.mean(all_leads) > 100 else 'NO'}")
    print(f"  Consistent across datasets: {'YES' if min(np.mean(leads1), np.mean(leads2), np.mean(leads3)) > 100 else 'NO'}")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/cross_dataset.json", "w") as f:
        json.dump({
            "ds1_mean": float(np.mean(leads1)),
            "ds2_mean": float(np.mean(leads2)),
            "ds3_mean": float(np.mean(leads3)),
            "all_mean": float(np.mean(all_leads)),
            "generalizes": bool(np.mean(all_leads) > 100),
        }, f, indent=2)
    print("\nSaved to data/results/cross_dataset.json")


if __name__ == "__main__":
    run()
