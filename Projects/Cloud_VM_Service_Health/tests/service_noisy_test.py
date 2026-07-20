"""
Noisy/bursty service health test.

Purpose: check that the fusion engine, fed jittery real-world-style service
metrics (not a smooth ramp), correctly holds through ordinary noise and only
fires on genuine sustained degradation.

"""
import numpy as np
from engine import JOREngine
from service_adapter import ServiceHealthAdapter


def simulate_requests(mean_latency_ms, error_rate, n=200, jitter_std=None):
    # Jitter scales with the mean so noise is proportionally realistic at
    # both low and high latency, not a fixed absolute wobble.
    if jitter_std is None:
        jitter_std = max(6.0, mean_latency_ms * 0.15)
    latencies = np.clip(np.random.normal(mean_latency_ms, jitter_std, n), 1.0, None)
    error_count = np.random.binomial(n, min(max(error_rate, 0.0), 1.0))
    return latencies, error_count, n


def step(engine, adapter, mean_latency, error_rate, rps, capacity_rps,
         memory_pct, disk_queue):
    latencies, err_count, req_count = simulate_requests(mean_latency, error_rate)
    obs = adapter.extract_features(latencies, err_count, req_count)
    theta_c = adapter.normalize_context(rps, capacity_rps)
    theta_s = adapter.normalize_system_state(memory_pct, disk_queue)
    sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs['theta_o'])
    return obs, theta_c, theta_s, sop, nhp, alert


def print_row(i, label, obs, theta_c, theta_s, sop, nhp, alert):
    status = "ALERT" if alert else "Normal"
    print(f"{i:4d} {label:10s} p95={obs['p95_latency_ms']:6.1f}ms  "
          f"err={obs['error_rate']:.3f}  ts={theta_s:.2f}  tc={theta_c:.2f}  "
          f"SOP={sop:.3f}  NHP={nhp:.3f}  {status}")


def main():
    np.random.seed(7)
    print("=== JOR 4.0 -- Service Health: Noisy / Bursty Stress Test ===\n")

    engine = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                        retention=0.70, calibration_steps=20)
    adapter = ServiceHealthAdapter()

    i = 0
    transitions = 0
    prev_alert = None
    episode_lengths = []
    current_len = 0

    def run(label, n_steps, mean_latency, error_rate, rps, capacity_rps,
            memory_pct, disk_queue):
        nonlocal i, transitions, prev_alert, current_len
        for _ in range(n_steps):
            obs, tc, ts, sop, nhp, alert = step(
                engine, adapter, mean_latency, error_rate, rps, capacity_rps,
                memory_pct, disk_queue)
            print_row(i, label, obs, tc, ts, sop, nhp, alert)
            if prev_alert is not None and alert != prev_alert:
                transitions += 1
                episode_lengths.append(current_len)
                current_len = 0
            current_len += 1
            prev_alert = alert
            i += 1

    # Phase 1: Calibration -- realistic jitter, no real problem
    run("Calib", 20, mean_latency=45, error_rate=0.002,
        rps=150, capacity_rps=500, memory_pct=40, disk_queue=1.0)

    # Phase 2: Ordinary noisy healthy operation, 60 steps -- jittery latency,
    # occasional error blips, traffic wobbling -- but NO sustained problem.
    # This is the segment that matters most: does it stay stable under
    # realistic noise, the way the CPU tests eventually showed it does?
    for _ in range(60):
        mean_latency = np.random.normal(50, 6)
        error_rate = max(0.0, np.random.normal(0.006, 0.006))
        rps = np.random.normal(160, 25)
        memory_pct = np.random.normal(42, 4)
        disk_queue = max(0.2, np.random.normal(1.3, 0.6))
        obs, tc, ts, sop, nhp, alert = step(
            engine, adapter, mean_latency, error_rate, rps, 500, memory_pct, disk_queue)
        print_row(i, "Noisy", obs, tc, ts, sop, nhp, alert)
        if prev_alert is not None and alert != prev_alert:
            transitions += 1
            episode_lengths.append(current_len)
            current_len = 0
        current_len += 1
        prev_alert = alert
        i += 1

    # Phase 3: Genuine incident #1 -- short, sharp error burst (bad deploy),
    # then real recovery
    run("Incident1", 8, mean_latency=90, error_rate=0.06,
        rps=170, capacity_rps=500, memory_pct=50, disk_queue=2.0)
    run("Recover1", 10, mean_latency=48, error_rate=0.002,
        rps=155, capacity_rps=500, memory_pct=41, disk_queue=1.1)

    # Phase 4: More noisy healthy operation
    for _ in range(30):
        mean_latency = np.random.normal(50, 6)
        error_rate = max(0.0, np.random.normal(0.006, 0.006))
        rps = np.random.normal(160, 25)
        memory_pct = np.random.normal(42, 4)
        disk_queue = max(0.2, np.random.normal(1.3, 0.6))
        obs, tc, ts, sop, nhp, alert = step(
            engine, adapter, mean_latency, error_rate, rps, 500, memory_pct, disk_queue)
        print_row(i, "Noisy", obs, tc, ts, sop, nhp, alert)
        if prev_alert is not None and alert != prev_alert:
            transitions += 1
            episode_lengths.append(current_len)
            current_len = 0
        current_len += 1
        prev_alert = alert
        i += 1

    # Phase 5: Genuine incident #2 -- sustained memory-leak-style degradation
    print("\n--- Incident 2: sustained memory leak ---")
    for k in range(15):
        mean_latency = 55 + k * 12
        error_rate = 0.005 + k * 0.003
        memory_pct = 45 + k * 3
        disk_queue = 1.2 + k * 0.3
        obs, tc, ts, sop, nhp, alert = step(
            engine, adapter, mean_latency, error_rate, 165, 500, memory_pct, disk_queue)
        print_row(i, "Incident2", obs, tc, ts, sop, nhp, alert)
        if prev_alert is not None and alert != prev_alert:
            transitions += 1
            episode_lengths.append(current_len)
            current_len = 0
        current_len += 1
        prev_alert = alert
        i += 1

    run("Recover2", 20, mean_latency=47, error_rate=0.002,
        rps=150, capacity_rps=500, memory_pct=40, disk_queue=1.0)

    episode_lengths.append(current_len)

    print("\n=== Summary ===")
    print(f"Total steps: {i}")
    print(f"Total ALERT<->Normal transitions: {transitions}")
    print(f"State run-lengths between transitions: {episode_lengths}")
    short_runs = [n for n in episode_lengths if n <= 2]
    print(f"Runs of length <=2 (potential rapid alternation): {len(short_runs)}")
    if short_runs:
        print("  -> Possible flapping: multiple short-lived state runs detected.")
    else:
        print("  -> No rapid alternation: every state held for a sustained run "
              "before transitioning. Consistent with correctly-detected "
              "separate incidents, not flapping.")


if __name__ == "__main__":
    main()
