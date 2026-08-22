"""Phase 3: Unseen batteries — does the precursor generalize?"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.loader import load_dataset, get_battery_metadata
from src.baselines.capacity import detect_degradation_capacity
from src.baselines.statistical import detect_change_cusum


def run():
    print("=" * 70)
    print("PHASE 3: UNSEEN BATTERIES — does the precursor generalize?")
    print("=" * 70)

    batteries = load_dataset("synthetic")
    bat_ids = list(batteries.keys())
    rng = random.Random(42)
    rng.shuffle(bat_ids)
    split = int(0.7 * len(bat_ids))
    train_ids = bat_ids[:split]
    test_ids = bat_ids[split:]

    print(f"Train: {len(train_ids)} batteries, Test: {len(test_ids)} batteries (UNSEEN)")

    # --- Train on temperature CUSUM ---
    print("\n--- Training: CUSUM on temperature ---")
    train_leads = []
    for bat_id in train_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        temp = df["temperature"].values
        result = detect_change_cusum(temp, threshold=3.0)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
        train_leads.append(lead)
        print(f"  {bat_id}: onset={onset}, detected={result['warning_cycle']}, lead={lead}")

    print(f"\n  Train mean lead: {np.mean(train_leads):.0f} cycles")

    # --- Test on unseen batteries ---
    print("\n--- Testing on UNSEEN batteries ---")
    test_leads = []
    for bat_id in test_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        temp = df["temperature"].values
        result = detect_change_cusum(temp, threshold=3.0)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
        test_leads.append(lead)
        print(f"  {bat_id}: onset={onset}, detected={result['warning_cycle']}, lead={lead}")

    print(f"\n  Test mean lead: {np.mean(test_leads):.0f} cycles")

    # --- Compare all signals on unseen ---
    print("\n--- All signals on unseen batteries ---")
    print(f"  {'Signal':<20} {'Mean Lead':>10} {'Min':>6} {'Max':>6} {'Pass%':>7}")
    print("  " + "-" * 55)

    for sig_name in ["capacity", "voltage", "current", "temperature"]:
        leads = []
        for bat_id in test_ids:
            df = batteries[bat_id]
            onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
            sig = df[sig_name].values
            result = detect_change_cusum(sig, threshold=3.0)
            lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
            leads.append(lead)
        pass_rate = sum(1 for l in leads if l > 100) / len(leads) * 100
        print(f"  {sig_name:<20} {np.mean(leads):>10.0f} {np.min(leads):>6.0f} {np.max(leads):>6.0f} {pass_rate:>6.0f}%")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  Train mean lead (temp): {np.mean(train_leads):.0f} cycles")
    print(f"  Test mean lead (temp):  {np.mean(test_leads):.0f} cycles")
    print(f"  Generalization: {'YES' if np.mean(test_leads) > 100 else 'NO'}")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/unseen_batteries.json", "w") as f:
        json.dump({
            "train_mean_lead": float(np.mean(train_leads)),
            "test_mean_lead": float(np.mean(test_leads)),
            "generalizes": bool(np.mean(test_leads) > 100),
        }, f, indent=2)
    print("\nSaved to data/results/unseen_batteries.json")


if __name__ == "__main__":
    run()
