"""
Step 1: try to BREAK the current adapter with realistic heavy-tailed latency
noise (occasional genuine outlier requests even during healthy operation --
GC pauses, a slow DB query, a network hiccup) instead of smooth Gaussian
jitter.

Step 2: if it breaks, try an adapter-level fix (a knee-curve on latency
normalization, same pattern as the theta_c fix) -- NOT touching engine.py.

Step 3: compare before/after on both this noise scenario AND a genuine
incident, to make sure any fix doesn't just mask real problems.
"""
import numpy as np
from engine import JOREngine
from service_adapter import ServiceHealthAdapter


def simulate_heavy_tailed_requests(mean_latency_ms, error_rate, n=200,
                                    outlier_prob=0.03, outlier_mult=6.0):
    """
    Realistic latency distribution: mostly tight around the mean, but with a
    small fraction of genuine outlier requests (much slower) even when the
    service is healthy. This is what real p95 latency actually looks like --
    a pure Gaussian never produces this shape.
    """
    base = np.clip(np.random.normal(mean_latency_ms, mean_latency_ms * 0.12, n), 1.0, None)
    is_outlier = np.random.random(n) < outlier_prob
    outlier_latencies = mean_latency_ms * outlier_mult * np.random.uniform(0.7, 1.3, n)
    latencies = np.where(is_outlier, outlier_latencies, base)
    error_count = np.random.binomial(n, min(max(error_rate, 0.0), 1.0))
    return latencies, error_count, n


def run_noise_scenario(adapter_factory, label, n_steps=80, seed=42,
                        outlier_prob=0.03, outlier_mult=6.0):
    np.random.seed(seed)
    engine = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                        retention=0.70, calibration_steps=20)
    adapter = adapter_factory()

    transitions = 0
    prev_alert = None
    run_lengths = []
    current_len = 0
    max_nhp_during_noise = 0.0

    # Calibration: healthy, but WITH the heavy-tailed noise already present
    # (this is realistic -- you don't get to calibrate against noise-free
    # conditions in a real service).
    for _ in range(20):
        latencies, err, req = simulate_heavy_tailed_requests(
            50, 0.003, outlier_prob=outlier_prob, outlier_mult=outlier_mult)
        obs = adapter.extract_features(latencies, err, req)
        theta_c = adapter.normalize_context(150, 500)
        theta_s = adapter.normalize_system_state(40, 1.0)
        engine.fusion_step(theta_s, theta_c, obs['theta_o'])

    # Ordinary noisy healthy operation -- NO real incident, just realistic
    # heavy-tailed latency noise, for n_steps.
    for i in range(n_steps):
        mean_latency = np.random.normal(50, 5)
        error_rate = max(0.0, np.random.normal(0.004, 0.004))
        latencies, err, req = simulate_heavy_tailed_requests(
            mean_latency, error_rate, outlier_prob=outlier_prob, outlier_mult=outlier_mult)
        obs = adapter.extract_features(latencies, err, req)
        theta_c = adapter.normalize_context(np.random.normal(155, 20), 500)
        theta_s = adapter.normalize_system_state(np.random.normal(41, 3), max(0.2, np.random.normal(1.2, 0.5)))
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs['theta_o'])
        max_nhp_during_noise = max(max_nhp_during_noise, nhp)

        if prev_alert is not None and alert != prev_alert:
            transitions += 1
            run_lengths.append(current_len)
            current_len = 0
        current_len += 1
        prev_alert = alert
    run_lengths.append(current_len)

    short_runs = [n for n in run_lengths if n <= 2]

    print(f"\n=== {label} (outlier_prob={outlier_prob}, outlier_mult={outlier_mult}) ===")
    print(f"Steps: {n_steps} (pure noise, no real incident)")
    print(f"Transitions: {transitions}")
    print(f"Run lengths: {run_lengths}")
    print(f"Peak NHP during noise-only period: {max_nhp_during_noise:.3f}")
    if short_runs:
        print(f"Short runs (<=2, possible rapid alternation): {len(short_runs)}  <-- ISSUE")
    else:
        print("No short runs -- stable under this noise profile.")

    return transitions, max_nhp_during_noise, short_runs


def run_incident_check(adapter_factory, label, seed=42):
    """Confirm a genuine incident still gets caught after any adapter change."""
    np.random.seed(seed)
    engine = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                        retention=0.70, calibration_steps=20)
    adapter = adapter_factory()

    for _ in range(20):
        latencies, err, req = simulate_heavy_tailed_requests(50, 0.003)
        obs = adapter.extract_features(latencies, err, req)
        theta_c = adapter.normalize_context(150, 500)
        theta_s = adapter.normalize_system_state(40, 1.0)
        engine.fusion_step(theta_s, theta_c, obs['theta_o'])

    alert_fired = False
    alert_step = None
    for i in range(10):
        # Genuine incident: real sustained latency degradation (not just
        # outliers) -- e.g. a slow downstream dependency
        mean_latency = 60 + i * 35
        latencies, err, req = simulate_heavy_tailed_requests(mean_latency, 0.01 + i * 0.008)
        obs = adapter.extract_features(latencies, err, req)
        theta_c = adapter.normalize_context(160, 500)
        theta_s = adapter.normalize_system_state(45, 1.5)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs['theta_o'])
        if alert and not alert_fired:
            alert_fired = True
            alert_step = i

    print(f"{label}: genuine incident alert fired={alert_fired}"
          f"{f' at step {alert_step}/10' if alert_fired else ' (MISSED)'}")
    return alert_fired, alert_step


if __name__ == "__main__":
    print("STEP 1: Test current (unmodified) adapter against heavy-tailed noise\n")
    run_noise_scenario(lambda: ServiceHealthAdapter(), "Current adapter -- heavy-tailed noise")
    run_incident_check(lambda: ServiceHealthAdapter(), "Current adapter -- genuine incident check")
