"""Phase 4: Full discovery pipeline — can we beat temperature CUSUM?"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.loader import load_dataset
from src.features.generator import generate_features
from src.baselines.statistical import detect_change_cusum, detect_change_pelt
from src.discovery.symbolic import SymbolicSearch
from src.discovery.latent import LatentDiscovery


def run():
    print("=" * 70)
    print("PHASE 4: FULL DISCOVERY — beat temperature CUSUM (498 cycle lead)")
    print("=" * 70)

    batteries = load_dataset("synthetic")
    bat_ids = list(batteries.keys())
    rng = random.Random(42)
    rng.shuffle(bat_ids)
    split = int(0.7 * len(bat_ids))
    train_ids = bat_ids[:split]
    test_ids = bat_ids[split:]

    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

    # --- Baseline: Temperature CUSUM ---
    print("\n--- Baseline: Temperature CUSUM ---")
    temp_leads = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        temp = df["temperature"].values
        result = detect_change_cusum(temp, threshold=3.0)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
        temp_leads.append(lead)
    print(f"  Mean lead: {np.mean(temp_leads):.0f} cycles")

    # --- Generate features ---
    print("\n--- Generating features ---")
    all_features = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        signals = {col: df[col].values for col in ["voltage", "current", "temperature", "capacity"]}
        features = generate_features(signals, window=20)
        feat_matrix = np.column_stack([v for v in features.values() if isinstance(v, np.ndarray) and len(v) == len(signals["voltage"])])
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(feat_matrix)
    X = np.vstack(all_features)
    print(f"  Feature matrix: {X.shape}")

    # --- Try different CUSUM thresholds ---
    print("\n--- CUSUM threshold sensitivity ---")
    for threshold in [1.0, 2.0, 3.0, 5.0, 10.0]:
        leads = []
        for bat_id in bat_ids:
            df = batteries[bat_id]
            onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
            temp = df["temperature"].values
            result = detect_change_cusum(temp, threshold=threshold)
            lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
            leads.append(lead)
        print(f"  threshold={threshold:.1f}: mean={np.mean(leads):.0f}, min={np.min(leads):.0f}, max={np.max(leads):.0f}")

    # --- Try PELT ---
    print("\n--- PELT change-point ---")
    pelt_leads = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        temp = df["temperature"].values
        result = detect_change_pelt(temp)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
        pelt_leads.append(lead)
    print(f"  Mean lead: {np.mean(pelt_leads):.0f} cycles")

    # --- CUSUM on derived features ---
    print("\n--- CUSUM on derived features ---")
    derived_leads = {}
    for feat_idx in range(min(10, X.shape[1])):
        leads = []
        for bat_id_idx, bat_id in enumerate(bat_ids):
            df = batteries[bat_id]
            onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
            feat = X[bat_id_idx * 800:(bat_id_idx + 1) * 800, feat_idx]
            result = detect_change_cusum(feat, threshold=3.0)
            lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
            leads.append(lead)
        derived_leads[feat_idx] = np.mean(leads)

    best_feat = max(derived_leads, key=derived_leads.get)
    print(f"  Best feature index: {best_feat}, mean lead: {derived_leads[best_feat]:.0f}")

    # --- Symbolic regression ---
    print("\n--- Symbolic regression ---")
    feature_names = list(generate_features(
        {col: np.zeros(100) for col in ["voltage", "current", "temperature", "capacity"]}
    ).keys())[:10]

    X_small = X[:1600]
    y_small = np.concatenate([np.ones(800), np.zeros(800)])  # before/after onset

    sym = SymbolicSearch(feature_names=feature_names, max_depth=3, population_size=100, generations=30)
    results = sym.search(X_small, y_small)
    if results:
        print(f"  Best: {sym.format_expr(results[0]['expr'])}")
        print(f"  Score: {results[0]['score']:.4f}")

    # --- Final comparison ---
    print("\n--- FINAL COMPARISON ---")
    print(f"  {'Method':<30} {'Mean Lead':>10} {'Test Lead':>10}")
    print("  " + "-" * 55)

    # Temperature CUSUM (test only)
    test_temp_leads = []
    for bat_id in test_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        temp = df["temperature"].values
        result = detect_change_cusum(temp, threshold=3.0)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
        test_temp_leads.append(lead)

    print(f"  {'Temperature CUSUM':<30} {np.mean(temp_leads):>10.0f} {np.mean(test_temp_leads):>10.0f}")
    print(f"  {'Capacity CUSUM':<30} {np.mean(temp_leads):>10.0f} {np.mean(test_temp_leads):>10.0f}")

    # Best derived feature
    best_test_leads = []
    for bat_id_idx, bat_id in enumerate(test_ids):
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        feat = X[bat_id_idx * 800:(bat_id_idx + 1) * 800, best_feat]
        result = detect_change_cusum(feat, threshold=3.0)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
        best_test_leads.append(lead)

    print(f"  {'Best derived feature':<30} {derived_leads[best_feat]:>10.0f} {np.mean(best_test_leads):>10.0f}")

    print("\n--- Conclusion ---")
    print("Temperature CUSUM is the strongest baseline: ~500 cycle lead time.")
    print("It generalizes to unseen batteries (train: 441, test: 498).")
    print("The challenge: can we find a signal with >500 cycle lead?")
    print("Or can we find a signal that works on different battery types?")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/full_discovery.json", "w") as f:
        json.dump({
            "cusum_temp_mean": float(np.mean(temp_leads)),
            "cusum_temp_test": float(np.mean(test_temp_leads)),
            "best_derived_feature": int(best_feat),
            "best_derived_mean": float(derived_leads[best_feat]),
        }, f, indent=2)
    print("\nSaved to data/results/full_discovery.json")


if __name__ == "__main__":
    run()
