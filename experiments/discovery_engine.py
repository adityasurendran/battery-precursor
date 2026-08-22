"""Run discovery engine: search, rank, validate on unseen batteries."""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.loader import load_dataset
from src.discovery.engine import DiscoveryEngine
from src.baselines.statistical import detect_change_cusum


def run():
    print("=" * 70)
    print("DISCOVERY ENGINE: search + rank + validate on unseen")
    print("=" * 70)

    batteries = load_dataset("synthetic")
    bat_ids = list(batteries.keys())
    rng = random.Random(42)
    rng.shuffle(bat_ids)
    split = int(0.7 * len(bat_ids))
    train_ids = bat_ids[:split]
    test_ids = bat_ids[split:]

    onset_cycles = {}
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset_cycles[bat_id] = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1

    print(f"Train: {len(train_ids)}, Test: {len(test_ids)} (UNSEEN)")

    # --- Phase 1: Discovery on training batteries ---
    print("\n--- Phase 1: Discovery on training batteries ---")
    engine = DiscoveryEngine(alpha=0.05)
    candidates = engine.search(batteries, train_ids, onset_cycles, threshold=3.0)

    print(f"\n  Top 5 candidates:")
    for i, c in enumerate(candidates[:5]):
        sig = "*" if c["significant"] else " "
        print(f"  {i+1}. {sig} {c['name']:<25} lead={c['mean_lead']:.0f} "
              f"consistency={c['consistency']:.2f} robustness={c['robustness']:.2f} "
              f"score={c['score']:.1f}")

    # --- Phase 2: Validate on unseen batteries ---
    print("\n--- Phase 2: Validate on UNSEEN batteries ---")
    validation = engine.validate_on_unseen(batteries, test_ids, onset_cycles, top_k=5)

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  Discovery: tested {len(candidates)} transformations")
    significant = [c for c in candidates if c["significant"]]
    print(f"  Significant candidates: {len(significant)}")

    if validation:
        best = max(validation, key=lambda x: x["unseen_mean"])
        print(f"\n  Best validated precursor: {best['name']}")
        print(f"    Train lead: {best['train_mean']:.0f} cycles")
        print(f"    Unseen lead: {best['unseen_mean']:.0f} cycles")
        print(f"    Generalizes: {best['generalizes']}")

    # Compare to temperature CUSUM baseline
    print("\n--- vs Temperature CUSUM baseline ---")
    temp_leads = []
    for bat_id in test_ids:
        df = batteries[bat_id]
        onset = onset_cycles[bat_id]
        temp = df["temperature"].values
        result = detect_change_cusum(temp, threshold=3.0)
        lead = max(0, onset - result["warning_cycle"]) if result["warning_cycle"] > 0 else 0
        temp_leads.append(lead)
    print(f"  Temperature CUSUM: {np.mean(temp_leads):.0f} cycles (test)")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/discovery_engine.json", "w") as f:
        json.dump({
            "n_candidates": len(candidates),
            "n_significant": len(significant),
            "validation": validation,
            "temp_cusum_test": float(np.mean(temp_leads)),
        }, f, indent=2, default=str)
    print("\nSaved to data/results/discovery_engine.json")


if __name__ == "__main__":
    run()
