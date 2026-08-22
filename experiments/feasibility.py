"""Phase 1: Feasibility — is there any reproducible pre-degradation signal?"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.loader import load_dataset, get_battery_metadata
from src.features.generator import generate_features
from src.baselines.capacity import detect_degradation_capacity, detect_degradation_capacity_rate
from src.baselines.statistical import detect_change_cusum, detect_change_pelt
from src.discovery.latent import LatentDiscovery


def run():
    print("=" * 70)
    print("PHASE 1: FEASIBILITY — Is there a pre-degradation signal?")
    print("=" * 70)

    # Load data
    batteries = load_dataset("synthetic")
    meta = get_battery_metadata(batteries)
    print(f"\nLoaded {len(batteries)} batteries")
    print(meta.to_string(index=False))

    # Split: 70% train, 30% test
    bat_ids = list(batteries.keys())
    rng = random.Random(42)
    rng.shuffle(bat_ids)
    split = int(0.7 * len(bat_ids))
    train_ids = bat_ids[:split]
    test_ids = bat_ids[split:]
    print(f"\nTrain: {len(train_ids)} batteries, Test: {len(test_ids)} batteries")

    # --- Baseline A: Capacity-based ---
    print("\n--- Baseline A: Capacity-based ---")
    for bat_id in train_ids[:3]:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        capacity = df["capacity"].values
        result = detect_degradation_capacity(capacity, threshold=0.8)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 else 0
        print(f"  {bat_id}: actual_onset={onset}, detected={result['warning_cycle']}, lead={lead} cycles")

    # --- Baseline B: CUSUM ---
    print("\n--- Baseline B: CUSUM change-point ---")
    for bat_id in train_ids[:3]:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        capacity = df["capacity"].values
        result = detect_change_cusum(capacity, threshold=3.0)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
        print(f"  {bat_id}: actual_onset={onset}, detected={result['warning_cycle']}, lead={lead} cycles")

    # --- Latent discovery ---
    print("\n--- Latent State Discovery ---")
    # Combine all battery features
    all_features = []
    for bat_id in train_ids[:5]:
        df = batteries[bat_id]
        signals = {col: df[col].values for col in ["voltage", "current", "temperature", "capacity"]}
        features = generate_features(signals, window=20)
        # Stack features
        feat_matrix = np.column_stack([v for v in features.values() if isinstance(v, np.ndarray) and len(v) == len(signals["voltage"])])
        all_features.append(feat_matrix)

    X = np.vstack(all_features) if all_features else np.zeros((100, 5))
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Feature matrix: {X.shape}")

    # PCA
    ld = LatentDiscovery(n_components=5)
    Z, variance = ld.fit_pca(X)
    print(f"  PCA explained variance: {[f'{v:.3f}' for v in variance]}")

    # Detect transitions
    transition_result = ld.detect_transition(Z)
    print(f"  Latent transitions detected: {transition_result['n_transitions']}")
    if transition_result["transitions"]:
        print(f"  First transition at index: {transition_result['transitions'][0]}")

    # Test on unseen batteries
    print("\n--- Test on unseen batteries ---")
    for bat_id in test_ids[:3]:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        print(f"  {bat_id}: actual_onset={onset}")

        # Capacity baseline
        cap_result = detect_degradation_capacity(df["capacity"].values, threshold=0.8)
        cap_lead = max(0, onset - cap_result["warning_cycle"]) if onset > 0 and cap_result["warning_cycle"] > 0 else 0

        # CUSUM
        cusum_result = detect_change_cusum(df["capacity"].values)
        cusum_lead = max(0, onset - cusum_result["warning_cycle"]) if onset > 0 and cusum_result["warning_cycle"] > 0 else 0

        print(f"    Capacity: detected={cap_result['warning_cycle']}, lead={cap_lead}")
        print(f"    CUSUM:    detected={cusum_result['warning_cycle']}, lead={cusum_lead}")

    print("\n--- Summary ---")
    print("Feasibility check: are there pre-degradation signals?")
    print("If CUSUM detects changes before capacity threshold, answer is YES.")
    print("If not, answer is NO for this dataset.")

    # Save results
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/feasibility.json", "w") as f:
        json.dump({
            "n_batteries": len(batteries),
            "train": len(train_ids),
            "test": len(test_ids),
            "latent_transitions": transition_result["n_transitions"],
        }, f, indent=2)
    print("\nSaved to data/results/feasibility.json")


if __name__ == "__main__":
    import random
    run()
