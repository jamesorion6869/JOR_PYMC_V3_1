"""
Calibration/operating-condition mismatch test, applied to the per-instance
path specifically (extract_features_per_instance).

Same pattern as the aggregate-path test that broke p95 and the tail-ratio
attempts: calibrate under one noise regime, then operate under a
DIFFERENT, harsher-but-still-healthy regime. No bad instance is introduced
at any point -- this tests whether instance_ema / the IQR-based flagging
itself produces false alerts purely from a calibration/operating mismatch,
independent of any real degradation.

Mismatch modeled here: calibrate during an unusually LOW-variance period
(all instances very tightly clustered -- e.g. a quiet low-traffic window),
then operate during NORMAL, somewhat higher instance-to-instance jitter
(still completely healthy, just more realistic spread).
"""
import numpy as np
from engine import JOREngine
from service_adapter import ServiceHealthAdapter


def simulate_fleet(n_instances, mean_latency=50, jitter_pct=0.05, n_per=40):
    """jitter_pct controls how much instance-to-instance variance exists --
    low jitter_pct = tightly clustered (quiet calibration window),
    higher jitter_pct = normal realistic spread."""
    instance_latencies = {}
    # Each instance gets its own slightly different mean, controlled by jitter_pct
    instance_means = np.random.normal(mean_latency, mean_latency * jitter_pct, n_instances)
    for i, inst_mean in enumerate(instance_means):
        lat = np.clip(np.random.normal(inst_mean, inst_mean * 0.1, n_per), 1, None)
        instance_latencies[f"inst_{i}"] = lat
    return instance_latencies


def run(n_instances, seed, calib_jitter=0.02, operate_jitter=0.08, operate_steps=80):
    np.random.seed(seed)
    engine = JOREngine(calibration_steps=20)
    adapter = ServiceHealthAdapter()

    # Calibration: unusually LOW fleet-to-fleet variance (quiet window)
    for _ in range(20):
        lats = simulate_fleet(n_instances, jitter_pct=calib_jitter)
        all_lat = np.concatenate(list(lats.values()))
        obs = adapter.extract_features(all_lat, 0, len(all_lat))
        engine.fusion_step(0.2, 0.3, obs['theta_o'])

    # Operation: normal, HIGHER fleet-to-fleet variance -- still healthy,
    # no bad instance, just more realistic ordinary spread than calibration saw
    transitions = 0
    prev_alert = None
    max_nhp = 0.0
    for i in range(operate_steps):
        lats = simulate_fleet(n_instances, jitter_pct=operate_jitter)
        obs = adapter.extract_features_per_instance(
            lats, 0, sum(len(v) for v in lats.values()))
        sop, nhp, alert = engine.fusion_step(0.2, 0.3, obs['theta_o'])
        max_nhp = max(max_nhp, nhp)
        if prev_alert is not None and alert != prev_alert:
            transitions += 1
        prev_alert = alert

    return transitions, max_nhp


if __name__ == "__main__":
    print("=== Per-instance calibration/operating-condition mismatch test ===")
    print("Calibrate on LOW fleet variance (2%), operate on HIGHER but still")
    print("healthy fleet variance (8%). No bad instance introduced at any point.\n")

    total_false = 0
    for n_instances in [4, 5, 6, 8, 10, 20]:
        for seed in [1, 2, 3, 4]:
            transitions, max_nhp = run(n_instances, seed)
            flag = "  <-- FALSE ALERT" if transitions > 0 else ""
            print(f"fleet={n_instances:3d}  seed={seed}  peak_NHP={max_nhp:.3f}  "
                  f"false_alerts={transitions}{flag}")
            total_false += transitions

    print(f"\nTotal false alerts across all runs: {total_false}")
