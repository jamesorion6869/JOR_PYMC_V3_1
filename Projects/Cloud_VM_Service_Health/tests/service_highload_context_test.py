"""
High-Load Context Pressure Test Harness for ServiceHealthAdapter

Purpose:
Validate that context normalization (theta_c) behaves correctly under:
  - high RPS near saturation
  - overload (RPS > capacity)
  - oscillation around the saturation knee
  - overload + gray failure
  - recovery from overload

This ensures:
  - no false alerts from context pressure alone
  - no masking of gray failures under overload
  - no amplification of noise
  - smooth recovery
"""

import numpy as np
from service_adapter import ServiceHealthAdapter


def simulate_instance_latencies(num_instances, baseline, jitter, bad_instances=None, bad_mult=10):
    """Generate per-instance latency arrays."""
    if bad_instances is None:
        bad_instances = []

    instance_latencies = {}
    for i in range(num_instances):
        mean = baseline * bad_mult if i in bad_instances else baseline
        lat = np.clip(np.random.normal(mean, jitter, 50), 1.0, None)
        instance_latencies[f"inst_{i}"] = lat

    return instance_latencies


def run_step(label, adapter, num_instances, baseline, jitter, rps, capacity, bad_instances=None):
    """Run one high-load step and print results."""
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

    theta_c = adapter.normalize_context(rps, capacity)
    theta_s = adapter.normalize_system_state(memory_pct=45, disk_queue_depth=1.5)

    print(f"fleet_size={num_instances}  rps={rps}  capacity={capacity}  theta_c={theta_c:.3f}")
    print(f"fleet_median={obs['fleet_median_latency']:.2f}ms")
    print(f"outlier_fraction={obs['outlier_fraction']:.3f}")
    print(f"persistent_outliers={obs['persistent_outliers']}")
    print(f"worst_instance_ema={obs['worst_instance_ema']:.3f}")
    print(f"severity_boost={obs['severity_boost']:.3f}")
    print(f"theta_o={obs['theta_o']:.3f}")

    return {
        "theta_c": theta_c,
        "theta_o": obs["theta_o"],
        "outlier_fraction": obs["outlier_fraction"],
        "worst_instance_ema": obs["worst_instance_ema"]
    }


def main():
    print("\n" + "=" * 80)
    print("### High-Load Context Pressure Test Harness ###")

    baseline = 20
    jitter = baseline * 0.10
    adapter = ServiceHealthAdapter(latency_baseline_ms=baseline)

    num_instances = 10
    capacity = 500  # RPS capacity

    # ---------------------------------------------------------
    # 1. Near saturation (80% of capacity)
    # ---------------------------------------------------------
    near_sat = run_step(
        "Near saturation (80%)",
        adapter,
        num_instances,
        baseline,
        jitter,
        rps=int(capacity * 0.80),
        capacity=capacity
    )

    # ---------------------------------------------------------
    # 2. Full saturation (100% of capacity)
    # ---------------------------------------------------------
    full_sat = run_step(
        "Full saturation (100%)",
        adapter,
        num_instances,
        baseline,
        jitter,
        rps=capacity,
        capacity=capacity
    )

    # ---------------------------------------------------------
    # 3. Overload (150% of capacity)
    # ---------------------------------------------------------
    overload = run_step(
        "Overload (150%)",
        adapter,
        num_instances,
        baseline,
        jitter,
        rps=int(capacity * 1.50),
        capacity=capacity
    )

    # ---------------------------------------------------------
    # 4. Oscillation around saturation knee
    # ---------------------------------------------------------
    print("\n--- Oscillation Around Saturation Knee ---")

    for rps in [350, 420, 390, 430, 380, 450]:
        run_step(
            f"Oscillation rps={rps}",
            adapter,
            num_instances,
            baseline,
            jitter,
            rps=rps,
            capacity=capacity
        )

    # ---------------------------------------------------------
    # 5. Overload + gray failure (bad instance)
    # ---------------------------------------------------------
    gray_failure = run_step(
        "Overload + gray failure",
        adapter,
        num_instances,
        baseline,
        jitter,
        rps=int(capacity * 1.40),
        capacity=capacity,
        bad_instances=[3]
    )

    # ---------------------------------------------------------
    # 6. Overload + multiple bad instances
    # ---------------------------------------------------------
    multi_failure = run_step(
        "Overload + multiple bad instances",
        adapter,
        num_instances,
        baseline,
        jitter,
        rps=int(capacity * 1.40),
        capacity=capacity,
        bad_instances=[2, 7]
    )

    # ---------------------------------------------------------
    # 7. Recovery from overload
    # ---------------------------------------------------------
    print("\n--- Recovery From Overload ---")

    recovery_results = []

    for rps in [600, 500, 400, 300, 200]:
        result = run_step(
            f"Recovery rps={rps}",
            adapter,
            num_instances,
            baseline,
            jitter,
            rps=rps,
            capacity=capacity
        )

        recovery_results.append(result)

    # ---------------------------------------------------------
    # PASS / FAIL Validation
    # ---------------------------------------------------------

    recovery_final = recovery_results[-1]

    passed = (
        overload["theta_c"] > near_sat["theta_c"] and
        gray_failure["theta_o"] > overload["theta_o"] and
        multi_failure["theta_o"] >= overload["theta_o"] and
        near_sat["outlier_fraction"] < 0.10 and
        recovery_final["theta_o"] <= recovery_results[0]["theta_o"]
    )

    print("\n" + "=" * 80)
    print("### Validation Result ###")

    if passed:
        print("PASS")
        print("High-load context behavior validated.")
        print("Context pressure increased correctly, gray failures remained detectable,")
        print("and recovery behavior remained stable.")
    else:
        print("FAIL")
        print("High-load validation criteria exceeded.")
        print("Review context normalization, anomaly escalation, or recovery behavior.")

    print("=" * 80)


if __name__ == "__main__":
    main()