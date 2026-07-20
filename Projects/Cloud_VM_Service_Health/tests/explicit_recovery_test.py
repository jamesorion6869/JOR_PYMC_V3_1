"""
Explicit Recovery Test Harness (Option A: Bad instance removed from fleet)

Simulates:
  - Healthy baseline
  - Degradation (one bad instance)
  - Sustained degradation (EMA rises)
  - Recovery (bad instance removed from fleet)
  - Full normalization (EMA resets, no sticky alerts)
"""

import numpy as np
from service_adapter import ServiceHealthAdapter


def simulate_instance_latencies(instance_ids, baseline, jitter, bad_ids=None, bad_mult=10):
    if bad_ids is None:
        bad_ids = []

    instance_latencies = {}
    for inst in instance_ids:
        mean = baseline * bad_mult if inst in bad_ids else baseline
        lat = np.clip(np.random.normal(mean, jitter, 50), 1.0, None)
        instance_latencies[inst] = lat

    return instance_latencies


def run_step(label, adapter, instance_ids, baseline, jitter, bad_ids=None):
    print(f"\n=== {label} ===")

    instance_lats = simulate_instance_latencies(
        instance_ids=instance_ids,
        baseline=baseline,
        jitter=jitter,
        bad_ids=bad_ids
    )

    obs = adapter.extract_features_per_instance(
        instance_lats,
        error_count=0,
        request_count=len(instance_ids) * 50
    )

    print(f"fleet_size={len(instance_ids)}")
    print(f"fleet_median={obs['fleet_median_latency']:.2f}ms")
    print(f"outlier_fraction={obs['outlier_fraction']:.3f}")
    print(f"persistent_outliers={obs['persistent_outliers']}")
    print(f"worst_instance_ema={obs['worst_instance_ema']:.3f}")
    print(f"severity_boost={obs['severity_boost']:.3f}")
    print(f"theta_o={obs['theta_o']:.3f}")


def main():
    print("\n" + "=" * 80)
    print("### Explicit Recovery Test (Option A: Remove Bad Instance) ###")

    baseline = 50
    jitter = baseline * 0.15
    adapter = ServiceHealthAdapter(latency_baseline_ms=baseline)

    # Fleet
    fleet = [f"inst_{i}" for i in range(10)]
    bad_inst = "inst_3"

    # 1. Healthy baseline
    run_step("Healthy baseline", adapter, fleet, baseline, jitter)

    # 2. Degradation begins
    run_step("Degradation begins", adapter, fleet, baseline, jitter, bad_ids=[bad_inst])

    # 3. Sustained degradation (EMA rises)
    for step in range(3):
        run_step(f"Sustained degradation step {step}", adapter, fleet, baseline, jitter, bad_ids=[bad_inst])

    # 4. Recovery begins — REMOVE bad instance from fleet
    recovered_fleet = [inst for inst in fleet if inst != bad_inst]
    run_step("Recovery begins (bad instance removed)", adapter, recovered_fleet, baseline, jitter)

    # 5. Multi-step recovery — EMA should fully reset
    for step in range(5):
        run_step(f"Recovery step {step}", adapter, recovered_fleet, baseline, jitter)

    # 6. Full normalization
    run_step("Full normalization", adapter, recovered_fleet, baseline, jitter)


if __name__ == "__main__":
    main()
