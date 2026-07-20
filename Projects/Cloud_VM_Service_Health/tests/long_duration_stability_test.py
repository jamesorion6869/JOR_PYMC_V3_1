"""
Long-Duration Stability Test Harness for ServiceHealthAdapter

Purpose:
Simulate thousands of cycles of mostly healthy cloud VM behavior with:
  - normal traffic variation
  - mild latency jitter
  - occasional small noise blips
  - no true failures

Goals:
  - Verify theta_o stays bounded and near zero
  - Verify no alert lock (no sticky high theta_o)
  - Verify no numerical drift in EMA / outlier tracking
  - Verify per-instance logic remains quiet under normal conditions
"""

import numpy as np
from service_adapter import ServiceHealthAdapter


def simulate_instance_latencies(num_instances, baseline, jitter, occasional_spike_prob=0.01, spike_mult=3.0):
    """
    Generate per-instance latency arrays for a single cycle.
    Mostly healthy, with rare small spikes on random instances.
    """
    instance_latencies = {}
    for i in range(num_instances):
        mean = baseline
        lat = np.random.normal(mean, jitter, 50)

        # Rare small spike: one-off blip, not a persistent failure
        if np.random.rand() < occasional_spike_prob:
            spike_idx = np.random.randint(0, len(lat))
            lat[spike_idx] *= spike_mult

        lat = np.clip(lat, 1.0, None)
        instance_latencies[f"inst_{i}"] = lat

    return instance_latencies


def main():
    print("\n" + "=" * 80)
    print("### Long-Duration Stability Test Harness ###")

    baseline = 50
    jitter = baseline * 0.10
    num_instances = 10

    adapter = ServiceHealthAdapter(latency_baseline_ms=baseline)

    num_cycles = 2000  # long-duration run
    theta_o_history = []
    worst_ema_history = []
    outlier_fraction_history = []

    for step in range(num_cycles):
        instance_lats = simulate_instance_latencies(
            num_instances=num_instances,
            baseline=baseline,
            jitter=jitter,
            occasional_spike_prob=0.01,
            spike_mult=3.0
        )

        obs = adapter.extract_features_per_instance(
            instance_lats,
            error_count=0,
            request_count=num_instances * 50
        )

        theta_o = obs['theta_o']
        worst_ema = obs['worst_instance_ema']
        outlier_fraction = obs['outlier_fraction']

        theta_o_history.append(theta_o)
        worst_ema_history.append(worst_ema)
        outlier_fraction_history.append(outlier_fraction)

        # Print occasional checkpoints
        if step % 200 == 0:
            print(f"\n--- Step {step} ---")
            print(f"fleet_median={obs['fleet_median_latency']:.2f}ms")
            print(f"theta_o={theta_o:.3f}")
            print(f"worst_instance_ema={worst_ema:.3f}")
            print(f"outlier_fraction={outlier_fraction:.3f}")

    # Summary statistics
    theta_o_arr = np.array(theta_o_history)
    worst_ema_arr = np.array(worst_ema_history)
    outlier_arr = np.array(outlier_fraction_history)

    print("\n" + "=" * 80)
    print("### Long-Duration Stability Summary ###")
    print(f"cycles={num_cycles}")
    print(f"theta_o: min={theta_o_arr.min():.3f}  max={theta_o_arr.max():.3f}  mean={theta_o_arr.mean():.3f}")
    print(f"worst_ema: min={worst_ema_arr.min():.3f}  max={worst_ema_arr.max():.3f}  mean={worst_ema_arr.mean():.3f}")
    print(f"outlier_fraction: min={outlier_arr.min():.3f}  max={outlier_arr.max():.3f}  mean={outlier_arr.mean():.3f}")

    # PASS / FAIL validation criteria
    passed = (
        theta_o_arr.max() < 0.05 and
        outlier_arr.max() < 0.05 and
        worst_ema_arr.max() < 0.50
    )

    print("\n" + "=" * 80)
    print("### Validation Result ###")

    if passed:
        print("PASS")
        print("Long-duration stability maintained.")
        print("No anomaly drift, alert lock, or excessive outlier generation detected.")
    else:
        print("FAIL")
        print("Stability criteria exceeded.")
        print("Review theta_o drift, EMA behavior, or outlier tracking.")

    print("=" * 80)


if __name__ == "__main__":
    main()