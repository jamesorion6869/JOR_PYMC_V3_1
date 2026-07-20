"""
Baseline Scaling Test Harness for ServiceHealthAdapter

Purpose:
Validate that the default IQR floor rule:
    min_iqr_ms = max(latency_baseline_ms * 0.1, 1.0)
behaves correctly across very different latency baselines.

Baselines tested:
  - 5 ms
  - 50 ms
  - 500 ms
  - 5000 ms

Each baseline runs:
  - healthy fleet
  - one bad instance (gray failure)
  - recovery (bad instance removed)
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


def run_cycle(label, adapter, instance_ids, baseline, jitter, bad_ids=None):
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

    print(f"baseline={baseline}ms  min_iqr_ms={adapter.min_iqr_ms:.2f}ms")
    print(f"fleet_median={obs['fleet_median_latency']:.2f}ms")
    print(f"outlier_fraction={obs['outlier_fraction']:.3f}")
    print(f"persistent_outliers={obs['persistent_outliers']}")
    print(f"worst_instance_ema={obs['worst_instance_ema']:.3f}")
    print(f"severity_boost={obs['severity_boost']:.3f}")
    print(f"theta_o={obs['theta_o']:.3f}")


def test_baseline(baseline):
    print("\n" + "=" * 80)
    print(f"### Baseline Scaling Test: baseline={baseline}ms ###")

    jitter = baseline * 0.15
    adapter = ServiceHealthAdapter(latency_baseline_ms=baseline)

    fleet = [f"inst_{i}" for i in range(10)]
    bad_inst = "inst_3"

    # Healthy
    run_cycle("Healthy fleet", adapter, fleet, baseline, jitter)

    # Degradation
    run_cycle("Gray failure (inst_3 bad)", adapter, fleet, baseline, jitter, bad_ids=[bad_inst])

    # Sustained degradation
    for step in range(2):
        run_cycle(f"Sustained degradation step {step}", adapter, fleet, baseline, jitter, bad_ids=[bad_inst])

    # Recovery (remove bad instance)
    recovered_fleet = [inst for inst in fleet if inst != bad_inst]
    run_cycle("Recovery (bad instance removed)", adapter, recovered_fleet, baseline, jitter)

    # Final normalization
    run_cycle("Full normalization", adapter, recovered_fleet, baseline, jitter)


def main():
    baselines = [5, 50, 500, 5000]
    for b in baselines:
        test_baseline(b)


if __name__ == "__main__":
    main()
