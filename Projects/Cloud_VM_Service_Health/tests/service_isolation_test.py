"""
Partial-degradation isolation test.

Purpose: the fusion weights (0.40*theta_s + 0.30*theta_c + 0.30*theta_o) were
inherited unchanged from the vibration domain. This checks whether those
weights produce sensible behavior here, by degrading each evidence channel
ALONE while holding the other two healthy -- three separate scenarios:

  A) theta_o only  -- errors/latency spike, memory & traffic stay normal
     (e.g. a bad code deploy)
  B) theta_c only   -- traffic surges toward capacity, but latency/errors/
     memory stay fine (e.g. autoscaling is keeping up -- NOT a real problem)
  C) theta_s only   -- memory/disk pressure builds silently, latency/errors/
     traffic stay normal (e.g. a slow leak not yet symptomatic)

A well-tuned system should clearly flag (A), probably NOT flag (B) since
handled traffic growth isn't inherently unhealthy, and the interesting
question is what it does with (C) -- a real, dangerous condition that
produces NO user-facing symptom yet.
"""
import numpy as np
from engine import JOREngine
from service_adapter import ServiceHealthAdapter


def simulate_requests(mean_latency_ms, error_rate, n=200):
    jitter_std = max(6.0, mean_latency_ms * 0.12)
    latencies = np.clip(np.random.normal(mean_latency_ms, jitter_std, n), 1.0, None)
    error_count = np.random.binomial(n, min(max(error_rate, 0.0), 1.0))
    return latencies, error_count, n


def run_scenario(name, degrade_fn, n_steps=20):
    print(f"\n=== Scenario: {name} ===")
    engine = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                        retention=0.70, calibration_steps=20)
    adapter = ServiceHealthAdapter()

    # Calibration: healthy baseline
    for _ in range(20):
        latencies, err, req = simulate_requests(45, 0.002)
        obs = adapter.extract_features(latencies, err, req)
        theta_c = adapter.normalize_context(150, 500)
        theta_s = adapter.normalize_system_state(40, 1.0)
        engine.fusion_step(theta_s, theta_c, obs['theta_o'])
    print(f"  baseline_sop = {engine.baseline_sop:.4f}")

    max_nhp = 0.0
    alert_fired = False
    alert_step = None
    for i in range(n_steps):
        mean_latency, error_rate, rps, memory_pct, disk_queue = degrade_fn(i)
        latencies, err, req = simulate_requests(mean_latency, error_rate)
        obs = adapter.extract_features(latencies, err, req)
        theta_c = adapter.normalize_context(rps, 500)
        theta_s = adapter.normalize_system_state(memory_pct, disk_queue)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs['theta_o'])
        max_nhp = max(max_nhp, nhp)
        if alert and not alert_fired:
            alert_fired = True
            alert_step = i
        status = "ALERT" if alert else "Normal"
        print(f"  step {i:2d}  p95={obs['p95_latency_ms']:6.1f}ms  err={obs['error_rate']:.3f}  "
              f"ts={theta_s:.2f}  tc={theta_c:.2f}  SOP={sop:.3f}  NHP={nhp:.3f}  {status}")

    print(f"  -> Peak NHP: {max_nhp:.3f}  |  Alert fired: {alert_fired}"
          f"{f' at step {alert_step}' if alert_fired else ''}")
    return max_nhp, alert_fired


def scenario_a(i):
    # theta_o degrades: latency and error rate climb steadily; everything
    # else stays exactly at healthy baseline.
    return (45 + i * 25, 0.002 + i * 0.006, 150, 40, 1.0)


def scenario_b(i):
    # theta_c degrades: traffic climbs toward capacity, but latency/errors/
    # memory stay healthy (autoscaling/headroom is coping fine).
    return (46, 0.002, 150 + i * 20, 40, 1.0)


def scenario_c(i):
    # theta_s degrades: memory and disk queue climb steadily (a slow leak),
    # latency/errors/traffic stay healthy -- no user-facing symptom yet.
    return (46, 0.002, 150, 40 + i * 2.8, 1.0 + i * 0.45)


def main():
    np.random.seed(11)
    results = {}
    results['A: theta_o only (errors/latency)'] = run_scenario(
        "A -- theta_o only (errors/latency spike)", scenario_a)
    results['B: theta_c only (traffic, handled)'] = run_scenario(
        "B -- theta_c only (traffic surge, otherwise healthy)", scenario_b)
    results['C: theta_s only (silent resource leak)'] = run_scenario(
        "C -- theta_s only (memory/disk pressure, no symptom)", scenario_c)

    print("\n=== Cross-scenario summary ===")
    for name, (peak_nhp, fired) in results.items():
        print(f"{name:42s} peak NHP={peak_nhp:.3f}  alert={'YES' if fired else 'no'}")


if __name__ == "__main__":
    main()
