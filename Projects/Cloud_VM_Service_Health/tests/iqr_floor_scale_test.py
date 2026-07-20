"""
Multi-Scale IQR Floor Test Harness for ServiceHealthAdapter

Purpose:
Validate whether the default IQR floor (10% of latency_baseline_ms)
behaves correctly across radically different latency scales:

  - Ultra-low latency (baseline ~5ms)
  - Medium latency (baseline ~50ms)
  - High latency (baseline ~500ms)

Each scale is tested under:
  - Noise-only
  - One bad instance
  - Two bad instances
  - Rotating bad instance
  - Heterogeneous fleet baselines
  - Autoscaling fleet size changes
"""

import numpy as np
from service_adapter import ServiceHealthAdapter


def simulate_instance_latencies(num_instances, baseline, jitter, bad_instances=None, bad_mult=10):
    """
    Generate per-instance latency arrays.
    bad_instances: list of instance indices that should be degraded.
    """
    if bad_instances is None:
        bad_instances = []

    instance_latencies = {}
    for i in range(num_instances):
        if i in bad_instances:
            mean = baseline * bad_mult
        else:
            mean = baseline

        lat = np.clip(np.random.normal(mean, jitter, 50), 1.0, None)
        instance_latencies[f"inst_{i}"] = lat

    return instance_latencies


def run_test(label, adapter, num_instances, baseline, jitter, bad_instances=None):
    """
    Run a single test scenario and print results.
    """
    print(f"\n=== {label} ===")
    instance_lats = simulate_instance_latencies(
        num_instances=num_instances,
        baseline=baseline,
        jitter=jitter,
        bad_instances=bad_instances
    )

    obs = adapter.extract_features_per_instance(
        instance_lats,
        error_count=0,
        request_count=num_instances * 50
    )

    print(f"baseline={baseline}ms  min_iqr_ms={adapter.min_iqr_ms:.2f}")
    print(f"fleet_median={obs['fleet_median_latency']:.2f}ms")
    print(f"outlier_fraction={obs['outlier_fraction']:.3f}")
    print(f"persistent_outliers={obs['persistent_outliers']}")
    print(f"worst_instance_ema={obs['worst_instance_ema']:.3f}")
    print(f"severity_boost={obs['severity_boost']:.3f}")
    print(f"theta_o={obs['theta_o']:.3f}")


def run_scale_tests(baseline):
    """
    Run the full battery of tests for a given latency baseline.
    """
    print("\n" + "=" * 80)
    print(f"### Testing latency baseline = {baseline}ms ###")

    adapter = ServiceHealthAdapter(latency_baseline_ms=baseline)

    jitter = baseline * 0.15

    # 1. Noise-only
    run_test("Noise-only", adapter, num_instances=10, baseline=baseline, jitter=jitter)

    # 2. One bad instance
    run_test("One bad instance", adapter, num_instances=10, baseline=baseline,
             jitter=jitter, bad_instances=[3])

    # 3. Two bad instances
    run_test("Two bad instances", adapter, num_instances=10, baseline=baseline,
             jitter=jitter, bad_instances=[2, 7])

    # 4. Rotating bad instance (simulate churn)
    for step in range(3):
        bad = [step]
        run_test(f"Rotating bad instance step={step}", adapter, num_instances=10,
                 baseline=baseline, jitter=jitter, bad_instances=bad)

    # 5. Heterogeneous fleet baselines
    print("\n--- Heterogeneous fleet baseline test ---")
    instance_lats = {}
    for i in range(10):
        mean = baseline * (1.0 + (i % 3) * 0.2)  # three clusters
        lat = np.clip(np.random.normal(mean, jitter, 50), 1.0, None)
        instance_lats[f"inst_{i}"] = lat

    obs = adapter.extract_features_per_instance(instance_lats, 0, 500)
    print(f"fleet_median={obs['fleet_median_latency']:.2f}ms")
    print(f"outlier_fraction={obs['outlier_fraction']:.3f}")
    print(f"persistent_outliers={obs['persistent_outliers']}")
    print(f"worst_instance_ema={obs['worst_instance_ema']:.3f}")
    print(f"severity_boost={obs['severity_boost']:.3f}")
    print(f"theta_o={obs['theta_o']:.3f}")

    # 6. Autoscaling fleet size changes
    print("\n--- Autoscaling fleet size test ---")
    for size in [3, 5, 10, 20]:
        run_test(f"Fleet size = {size}", adapter, num_instances=size,
                 baseline=baseline, jitter=jitter, bad_instances=[0])


def main():
    # Ultra-low latency baseline
    run_scale_tests(5)

    # Medium latency baseline
    run_scale_tests(50)

    # High latency baseline
    run_scale_tests(500)


if __name__ == "__main__":
    main()
