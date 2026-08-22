"""Real validation: frozen discovery on independent datasets.

Protocol:
1. Freeze temperature-variance-10 precursor
2. Freeze detection threshold (3.0)
3. Don't retune per dataset
4. Test on 3 independent datasets
5. Report every result
"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from src.baselines.statistical import detect_change_cusum


def load_real_datasets():
    """Load all real datasets."""
    datasets = {}
    for ds_name in ["real_A", "real_B", "real_C"]:
        path = f"data/raw/real"
        if not os.path.exists(path):
            continue
        bats = {}
        for f in sorted(os.listdir(path)):
            if f.endswith(".csv") and ds_name in f:
                df = pd.read_csv(os.path.join(path, f))
                bats[f.replace(".csv", "")] = df
        if bats:
            datasets[ds_name] = bats
    return datasets


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
    print("REAL VALIDATION: frozen discovery, no retuning")
    print("=" * 70)

    datasets = load_real_datasets()
    print(f"Loaded {len(datasets)} datasets")

    # --- FROZEN DISCOVERY ---
    print("\n--- Frozen discovery (NO TUNING) ---")
    print("  Threshold: 3.0 (frozen from original)")
    print("  Window: 10 cycles (frozen)")
    print("  Signal: temperature variance (frozen)")

    def temp_var_10(signals):
        T = signals.get("temperature", np.zeros(100))
        return np.array([np.var(T[max(0,i-10):i+1]) for i in range(len(T))])

    def temp_raw(signals):
        return signals.get("temperature", np.zeros(100))

    def cap_raw(signals):
        return signals.get("capacity", np.zeros(100))

    # --- Evaluate on all datasets ---
    print("\n--- Lead times (frozen, no retuning) ---")
    print(f"  {'Dataset':<12} {'Batts':>5} {'TempVar':>10} {'Temp':>10} {'Capacity':>10}")
    print("  " + "-" * 55)

    all_results = {}
    for ds_name, bats in datasets.items():
        onsets = {bid: df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
                  for bid, df in bats.items()}
        bat_ids = list(bats.keys())

        leads_tv = compute_lead(bats, bat_ids, onsets, temp_var_10)
        leads_t = compute_lead(bats, bat_ids, onsets, temp_raw)
        leads_c = compute_lead(bats, bat_ids, onsets, cap_raw)

        tv_str = f"{np.mean(leads_tv):.0f}" if leads_tv else "FAIL"
        t_str = f"{np.mean(leads_t):.0f}" if leads_t else "FAIL"
        c_str = f"{np.mean(leads_c):.0f}" if leads_c else "FAIL"

        print(f"  {ds_name:<12} {len(bats):>5} {tv_str:>10} {t_str:>10} {c_str:>10}")

        all_results[ds_name] = {
            "n_batteries": len(bats),
            "temp_var_lead": float(np.mean(leads_tv)) if leads_tv else 0,
            "temp_lead": float(np.mean(leads_t)) if leads_t else 0,
            "capacity_lead": float(np.mean(leads_c)) if leads_c else 0,
            "temp_var_pass": bool(leads_tv and np.mean(leads_tv) > 100),
            "temp_pass": bool(leads_t and np.mean(leads_t) > 100),
            "capacity_pass": bool(leads_c and np.mean(leads_c) > 100),
        }

    # --- Summary ---
    print("\n--- Summary ---")
    tv_pass = sum(1 for r in all_results.values() if r["temp_var_pass"])
    t_pass = sum(1 for r in all_results.values() if r["temp_pass"])
    c_pass = sum(1 for r in all_results.values() if r["capacity_pass"])
    n_ds = len(all_results)

    print(f"  Temperature variance: {tv_pass}/{n_ds} datasets pass")
    print(f"  Temperature:          {t_pass}/{n_ds} datasets pass")
    print(f"  Capacity:             {c_pass}/{n_ds} datasets pass")

    all_tv = [r["temp_var_lead"] for r in all_results.values() if r["temp_var_lead"] > 0]
    all_t = [r["temp_lead"] for r in all_results.values() if r["temp_lead"] > 0]
    all_c = [r["capacity_lead"] for r in all_results.values() if r["capacity_lead"] > 0]

    if all_tv: print(f"  Temp var mean (where detected): {np.mean(all_tv):.0f} cycles")
    if all_t: print(f"  Temp mean (where detected): {np.mean(all_t):.0f} cycles")
    if all_c: print(f"  Capacity mean (where detected): {np.mean(all_c):.0f} cycles")

    print("\n--- Verdict ---")
    if tv_pass >= n_ds * 0.6:
        print("  Temperature precursor SURVIVES real-world validation")
    elif tv_pass >= n_ds * 0.3:
        print("  Temperature precursor PARTIALLY survives (needs more data)")
    else:
        print("  Temperature precursor FAILS on real-world data")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/real_validation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to data/results/real_validation.json")


if __name__ == "__main__":
    run()
