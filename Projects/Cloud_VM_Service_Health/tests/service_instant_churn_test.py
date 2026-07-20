"""
Instance Churn Test Harness for ServiceHealthAdapter

Purpose:
Validate that per-instance gray-failure detection behaves correctly when
instances appear/disappear mid-run (autoscaling, rolling deploys, churn).

Tests:
  - Fleet shrink (10 → 5 → 3)
  - Fleet expansion (3 → 8 → 12)
  - Churn during degradation (bad instance appears/disappears)
  - Churn during recovery
  - Heterogeneous baselines + churn
  - Rotating bad instance + churn
"""

import numpy as np
from service_adapter import ServiceHealthAdapter


def simulate_instance_latencies(instance_ids, baseline, jitter, bad_ids=None, bad_mult=10):
    """Generate per-instance latency arrays."""
    if bad_ids is None:
        bad_ids = []

    instance_latencies = {}
    for inst in instance_ids:
        mean = baseline * bad_mult if inst in bad_ids else baseline
        lat = np.clip(np.random.normal(mean, jitter, 50), 1.0, None)
        instance_latencies[inst] = lat

    return instance_latencies


def run_step(label, adapter, instance_ids, baseline, jitter, bad_ids=None):
    """Run one churn step and print results."""
    print(f"\n=== {label} ===")
    instance_lats = simulate_instance_latencies(instance_ids, baseline, jitter, bad_ids)

    obs = adapter.extract_features_per_instance(
        instance_lats,
        error_count=0,
        request_count=len(instance_ids) * 50
    )

    print(f"fleet_size={len(instance_ids)}  baseline={baseline}ms  min_iqr_ms={adapter.min_iqr_ms:.2f}")
    print(f"fleet_median={obs['fleet_median_latency']:.2f}ms")
    print(f"outlier_fraction={obs['outlier_fraction']:.3f}")
    print(f"persistent_outliers={obs['persistent_outliers']}")
    print(f"worst_instance_ema={obs['worst_instance_ema']:.3f}")
    print(f"severity_boost={obs['severity_boost']:.3f}")
    print(f"theta_o={obs['theta_o']:.3f}")


def main():
    print("\n" + "=" * 80)
    print("### Instance Churn Test Harness ###")

    baseline = 50
    jitter = baseline * 0.15

    adapter = ServiceHealthAdapter(latency_baseline_ms=baseline)

    # -------------------------------
    # 1. Fleet shrink (10 → 5 → 3)
    # -------------------------------
    print("\n--- Fleet Shrink Test (10 → 5 → 3) ---")
    run_step("Initial fleet (10)", adapter, [f"inst_{i}" for i in range(10)], baseline, jitter)
    run_step("Shrink to 5", adapter, [f"inst_{i}" for i in range(5)], baseline, jitter)
    run_step("Shrink to 3", adapter, [f"inst_{i}" for i in range(3)], baseline, jitter)

    # -------------------------------
    # 2. Fleet expansion (3 → 8 → 12)
    # -------------------------------
    print("\n--- Fleet Expansion Test (3 → 8 → 12) ---")
    run_step("Start small (3)", adapter, [f"inst_{i}" for i in range(3)], baseline, jitter)
    run_step("Expand to 8", adapter, [f"inst_{i}" for i in range(8)], baseline, jitter)
    run_step("Expand to 12", adapter, [f"inst_{i}" for i in range(12)], baseline, jitter)

    # -------------------------------
    # 3. Churn during degradation
    # -------------------------------
    print("\n--- Churn During Degradation (bad instance appears/disappears) ---")
    run_step("Healthy fleet (10)", adapter, [f"inst_{i}" for i in range(10)], baseline, jitter)
    run_step("Bad instance appears", adapter, [f"inst_{i}" for i in range(10)], baseline, jitter, bad_ids=["inst_3"])
    run_step("Shrink fleet (bad removed)", adapter, [f"inst_{i}" for i in range(5)], baseline, jitter)

    # -------------------------------
    # 4. Churn during recovery
    # -------------------------------
    print("\n--- Churn During Recovery ---")
    run_step("Bad instance present", adapter, [f"inst_{i}" for i in range(10)], baseline, jitter, bad_ids=["inst_7"])
    run_step("Bad instance removed (fleet shrinks)", adapter, [f"inst_{i}" for i in range(6)], baseline, jitter)
    run_step("Fleet expands during recovery", adapter, [f"inst_{i}" for i in range(12)], baseline, jitter)

    # -------------------------------
    # 5. Heterogeneous baselines + churn
    # -------------------------------
    print("\n--- Heterogeneous Baselines + Churn ---")
    instance_ids = [f"inst_{i}" for i in range(10)]
    instance_lats = {}

    for i in range(10):
        mean = baseline * (1.0 + (i % 3) * 0.25)
        lat = np.clip(np.random.normal(mean, jitter, 50), 1.0, None)
        instance_lats[f"inst_{i}"] = lat

    obs = adapter.extract_features_per_instance(instance_lats, 0, 500)
    print(f"fleet_median={obs['fleet_median_latency']:.2f}ms  outlier_fraction={obs['outlier_fraction']:.3f}")

    run_step("Shrink heterogeneous fleet", adapter, [f"inst_{i}" for i in range(5)], baseline, jitter)

    # -------------------------------
    # 6. Rotating bad instance + churn
    # -------------------------------
    print("\n--- Rotating Bad Instance + Churn ---")
    for step in range(4):
        bad = [f"inst_{step}"]
        fleet = [f"inst_{i}" for i in range(10 - step)]  # shrink each step
        run_step(f"Step {step} (bad={bad}, fleet={len(fleet)})", adapter, fleet, baseline, jitter, bad_ids=bad)


if __name__ == "__main__":
    main()
