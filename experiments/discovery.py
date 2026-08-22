"""Phase 2: Discovery — find latent precursors, compare to CUSUM baseline."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.loader import load_dataset, get_battery_metadata
from src.features.generator import generate_features
from src.baselines.capacity import detect_degradation_capacity
from src.baselines.statistical import detect_change_cusum
from src.discovery.latent import LatentDiscovery
from src.discovery.symbolic import SymbolicSearch


def run():
    print("=" * 70)
    print("PHASE 2: DISCOVERY — find precursors, compare to CUSUM")
    print("=" * 70)

    batteries = load_dataset("synthetic")
    meta = get_battery_metadata(batteries)
    print(f"Loaded {len(batteries)} batteries")

    bat_ids = list(batteries.keys())
    rng = random.Random(42)
    rng.shuffle(bat_ids)
    split = int(0.7 * len(bat_ids))
    train_ids = bat_ids[:split]
    test_ids = bat_ids[split:]

    # --- Baseline: CUSUM ---
    print("\n--- Baseline: CUSUM on capacity ---")
    cusum_leads = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        capacity = df["capacity"].values
        result = detect_change_cusum(capacity, threshold=3.0)
        lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
        cusum_leads.append(lead)
    print(f"  Mean lead time: {np.mean(cusum_leads):.0f} cycles")
    print(f"  Min: {np.min(cusum_leads):.0f}, Max: {np.max(cusum_leads):.0f}")

    # --- Latent Discovery ---
    print("\n--- Latent State Discovery ---")
    all_features = []
    all_labels = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        signals = {col: df[col].values for col in ["voltage", "current", "temperature", "capacity"]}
        features = generate_features(signals, window=20)
        feat_matrix = np.column_stack([v for v in features.values() if isinstance(v, np.ndarray) and len(v) == len(signals["voltage"])])
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(feat_matrix)
        # Label: 1 if before onset, 0 if after
        labels = np.where(np.arange(len(signals["voltage"])) < onset, 1, 0)
        all_labels.append(labels)

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    print(f"  Feature matrix: {X.shape}")

    # PCA
    ld = LatentDiscovery(n_components=5)
    Z, variance = ld.fit_pca(X)
    print(f"  PCA explained variance: {[f'{v:.3f}' for v in variance]}")

    # Detect transitions
    transition_result = ld.detect_transition(Z)
    print(f"  Latent transitions detected: {transition_result['n_transitions']}")

    # --- Symbolic Discovery ---
    print("\n--- Symbolic Discovery ---")
    feature_names = list(generate_features(
        {col: np.zeros(100) for col in ["voltage", "current", "temperature", "capacity"]}
    ).keys())[:10]

    # Use first 1000 samples for symbolic search
    X_small = X[:1000]
    y_small = y[:1000]

    sym = SymbolicSearch(feature_names=feature_names, max_depth=3, population_size=100, generations=20)
    results = sym.search(X_small, y_small)

    if results:
        best = results[0]
        print(f"  Best expression: {sym.format_expr(best['expr'])}")
        print(f"  Score: {best['score']:.4f}")

    # --- Compare all methods ---
    print("\n--- Method Comparison ---")
    print(f"  {'Method':<25} {'Mean Lead':>10} {'Min Lead':>10} {'Max Lead':>10}")
    print("  " + "-" * 58)

    # CUSUM (already computed)
    print(f"  {'CUSUM (capacity)':<25} {np.mean(cusum_leads):>10.0f} {np.min(cusum_leads):>10.0f} {np.max(cusum_leads):>10.0f}")

    # CUSUM on different signals
    for sig_name in ["voltage", "current", "temperature"]:
        sig_leads = []
        for bat_id in bat_ids:
            df = batteries[bat_id]
            onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
            sig = df[sig_name].values
            result = detect_change_cusum(sig, threshold=3.0)
            lead = max(0, onset - result["warning_cycle"]) if onset > 0 and result["warning_cycle"] > 0 else 0
            sig_leads.append(lead)
        print(f"  {'CUSUM (' + sig_name + ')':<25} {np.mean(sig_leads):>10.0f} {np.min(sig_leads):>10.0f} {np.max(sig_leads):>10.0f}")

    # PCA transition
    pca_leads = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        signals = {col: df[col].values for col in ["voltage", "current", "temperature", "capacity"]}
        features = generate_features(signals, window=20)
        feat_matrix = np.column_stack([v for v in features.values() if isinstance(v, np.ndarray) and len(v) == len(signals["voltage"])])
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        Z_bat = ld.fit_pca(feat_matrix)[0]
        trans = ld.detect_transition(Z_bat)
        lead = max(0, onset - trans["first_transition"]) if onset > 0 and trans["first_transition"] > 0 else 0
        pca_leads.append(lead)
    print(f"  {'PCA transition':<25} {np.mean(pca_leads):>10.0f} {np.min(pca_leads):>10.0f} {np.max(pca_leads):>10.0f}")

    print("\n--- Key Finding ---")
    print("CUSUM on capacity gives ~400 cycle lead time.")
    print("Can we find a signal that gives MORE lead time?")
    print("Or a signal that works on different battery types?")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/discovery.json", "w") as f:
        json.dump({
            "cusum_mean_lead": float(np.mean(cusum_leads)),
            "pca_mean_lead": float(np.mean(pca_leads)),
            "n_batteries": len(batteries),
        }, f, indent=2)
    print("\nSaved to data/results/discovery.json")


import random
if __name__ == "__main__":
    run()
