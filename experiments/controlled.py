"""Controlled experiments: does temperature predict AFTER controlling for confounders?

Test whether temperature signal is a genuine precursor or merely an indirect correlate
of factors like cycle number, charging regime, ambient conditions, etc.
"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.loader import load_dataset
from src.baselines.statistical import detect_change_cusum


def partial_correlation(x, y, z):
    """Partial correlation between x and y, controlling for z."""
    if len(x) < 10:
        return 0.0
    # Regress x on z, y on z, correlate residuals
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    if np.std(z) < 1e-10:
        return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0

    # Simple linear regression residuals
    z_mean = np.mean(z)
    z_var = np.var(z) + 1e-10

    x_res = x - (np.mean(x) + np.cov(x, z)[0, 1] / z_var * (z - z_mean))
    y_res = y - (np.mean(y) + np.cov(x, z)[0, 1] / z_var * (z - z_mean))

    if np.std(x_res) < 1e-10 or np.std(y_res) < 1e-10:
        return 0.0

    return np.corrcoef(x_res, y_res)[0, 1]


def compute_lead_time(batteries, bat_ids, onset_cycles, signal_func, threshold=3.0):
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
    print("CONTROLLED EXPERIMENTS: temperature as causal precursor")
    print("=" * 70)

    batteries = load_dataset("synthetic")
    bat_ids = list(batteries.keys())

    # --- Experiment 1: Partial correlations ---
    print("\n--- Experiment 1: Partial Correlations ---")
    print("  Does temperature correlate with degradation AFTER controlling for each confounder?")

    confounders = {
        "cycle_number": lambda df: np.arange(len(df)),
        "voltage": lambda df: df["voltage"].values,
        "current": lambda df: df["current"].values,
        "impedance": lambda df: df["impedance"].values,
        "capacity": lambda df: df["capacity"].values,
    }

    print(f"\n  {'Confounder':<20} {'Raw r':>8} {'Partial r':>10} {'Delta':>8}")
    print("  " + "-" * 50)

    for conf_name, conf_func in confounders.items():
        raw_rs = []
        partial_rs = []
        for bat_id in bat_ids:
            df = batteries[bat_id]
            onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
            T = df["temperature"].values
            C = df["capacity"].values
            conf = conf_func(df)

            # Before onset
            pre = min(onset, len(T))
            if pre > 50:
                # Raw correlation: temperature vs capacity decline
                cap_decline = C[0] - C[:pre]
                raw_r = np.corrcoef(T[:pre], cap_decline)[0, 1] if np.std(T[:pre]) > 0 else 0
                raw_rs.append(raw_r)

                # Partial correlation: temperature vs capacity decline, controlling for confounder
                partial_r = partial_correlation(T[:pre], cap_decline, conf[:pre])
                partial_rs.append(partial_r)

        raw_mean = np.mean(raw_rs) if raw_rs else 0
        partial_mean = np.mean(partial_rs) if partial_rs else 0
        delta = partial_mean - raw_mean
        print(f"  {conf_name:<20} {raw_mean:>8.3f} {partial_mean:>10.3f} {delta:>+8.3f}")

    # --- Experiment 2: Controlling for multiple confounders simultaneously ---
    print("\n--- Experiment 2: Multiple regression ---")
    print("  Does temperature predict degradation after controlling for ALL confounders?")

    all_features = []
    all_targets = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
        if onset < 0:
            continue
        T = df["temperature"].values
        C = df["capacity"].values
        V = df["voltage"].values
        I = df["current"].values
        Z = df["impedance"].values
        cycle = np.arange(len(T))

        pre = min(onset, len(T))
        if pre < 50:
            continue

        cap_decline = C[0] - C[:pre]

        # Features: cycle, voltage, current, impedance, temperature
        X = np.column_stack([cycle[:pre], V[:pre], I[:pre], Z[:pre], T[:pre]])
        X = np.nan_to_num(X, nan=0.0)
        all_features.append(X)
        all_targets.append(cap_decline)

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_targets)

    # Full model (all features)
    from sklearn.linear_model import LinearRegression
    lr_full = LinearRegression()
    lr_full.fit(X_all, y_all)
    r2_full = lr_full.score(X_all, y_all)

    # Without temperature
    X_no_temp = X_all[:, :-1]  # remove last column (temperature)
    lr_no_temp = LinearRegression()
    lr_no_temp.fit(X_no_temp, y_all)
    r2_no_temp = lr_no_temp.score(X_no_temp, y_all)

    # Without cycle number
    X_no_cycle = np.delete(X_all, 0, axis=1)
    lr_no_cycle = LinearRegression()
    lr_no_cycle.fit(X_no_cycle, y_all)
    r2_no_cycle = lr_no_cycle.score(X_no_cycle, y_all)

    print(f"  Full model (all features):        R² = {r2_full:.4f}")
    print(f"  Without temperature:              R² = {r2_no_temp:.4f}  (delta = {r2_full - r2_no_temp:+.4f})")
    print(f"  Without cycle number:             R² = {r2_no_cycle:.4f}  (delta = {r2_full - r2_no_cycle:+.4f})")

    if r2_full - r2_no_temp > 0.01:
        print(f"\n  Temperature adds {r2_full - r2_no_temp:.4f} R² — it has INDEPENDENT predictive power")
    else:
        print(f"\n  Temperature adds only {r2_full - r2_no_temp:.4f} R² — mostly correlated with other factors")

    # --- Experiment 3: Lead time after controlling ---
    print("\n--- Experiment 3: Lead time with residualized signals ---")
    print("  Remove cycle trend, then measure lead time")

    for sig_name in ["temperature", "capacity", "impedance"]:
        leads_raw = []
        leads_residual = []
        for bat_id in bat_ids:
            df = batteries[bat_id]
            onset = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1
            if onset < 0:
                continue
            sig = df[sig_name].values
            cycle = np.arange(len(sig))

            # Raw lead
            result = detect_change_cusum(sig, threshold=3.0)
            lead = max(0, onset - result["warning_cycle"]) if result["warning_cycle"] > 0 else 0
            leads_raw.append(lead)

            # Residualized: remove cycle trend
            if np.std(cycle) > 0 and np.std(sig) > 0:
                slope = np.cov(sig, cycle)[0, 1] / np.var(cycle)
                intercept = np.mean(sig) - slope * np.mean(cycle)
                residual = sig - (slope * cycle + intercept)
                result_r = detect_change_cusum(residual, threshold=3.0)
                lead_r = max(0, onset - result_r["warning_cycle"]) if result_r["warning_cycle"] > 0 else 0
                leads_residual.append(lead_r)

        raw_mean = np.mean(leads_raw) if leads_raw else 0
        res_mean = np.mean(leads_residual) if leads_residual else 0
        print(f"  {sig_name:<15} raw={raw_mean:>6.0f}  residualized={res_mean:>6.0f}  delta={res_mean - raw_mean:>+6.0f}")

    # --- Experiment 4: Per-battery temperature trajectory ---
    print("\n--- Experiment 4: Temperature trajectory analysis ---")
    print("  Does temperature trajectory predict WHICH batteries degrade faster?")

    onset_cycles = {}
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset_cycles[bat_id] = df["onset_cycle"].iloc[0] if "onset_cycle" in df.columns else -1

    # Compute: average temp in first 100 cycles vs onset
    early_temps = []
    onsets = []
    for bat_id in bat_ids:
        df = batteries[bat_id]
        onset = onset_cycles[bat_id]
        early_temp = np.mean(df["temperature"].values[:100])
        early_temps.append(early_temp)
        onsets.append(onset)

    corr = np.corrcoef(early_temps, onsets)[0, 1]
    print(f"  Correlation (early temp vs onset): {corr:.3f}")
    print(f"  If |r| > 0.3: early temperature predicts onset timing")
    print(f"  If |r| < 0.3: early temperature does NOT predict onset timing")

    # --- Summary ---
    print("\n--- Summary ---")
    print("If temperature adds R² > 0.01 after controlling for all confounders,")
    print("it has INDEPENDENT predictive power — not just an indirect correlate.")
    print("If residualized lead time is similar to raw lead time,")
    print("the signal survives controlling for cycle trend.")

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/controlled.json", "w") as f:
        json.dump({
            "r2_full": float(r2_full),
            "r2_no_temp": float(r2_no_temp),
            "r2_no_cycle": float(r2_no_cycle),
            "temp_adds_r2": float(r2_full - r2_no_temp),
            "early_temp_corr": float(corr),
        }, f, indent=2)
    print("\nSaved to data/results/controlled.json")


if __name__ == "__main__":
    run()
