"""
Prototype: JOR 4.0 fusion engine applied to cloud VM / service health.

Enhanced with per-instance gray-failure detection.
"""
import numpy as np
import json
import os
from engine import JOREngine
from service_adapter import ServiceHealthAdapter
from logger import FusionLogger

STATE_FILE = "cloud_vm_engine_state.json"


def load_engine_state(engine):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            engine.p_final = state.get("p_final", engine.prior_nh)
            engine.alert_status = state.get("alert_status", False)
            engine.baseline_sop = state.get("baseline_sop", engine.baseline_sop)
            engine.is_calibrated = state.get("is_calibrated", engine.is_calibrated)
            engine.calibrating = not engine.is_calibrated
            print("--- State restored from disk ---")


def save_engine_state(engine):
    state = {
        "p_final": engine.p_final,
        "alert_status": engine.alert_status,
        "baseline_sop": engine.baseline_sop,
        "is_calibrated": engine.is_calibrated,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def simulate_requests(mean_latency_ms, error_rate, n=200, jitter_std=8.0):
    latencies = np.clip(np.random.normal(mean_latency_ms, jitter_std, n), 1.0, None)
    error_count = np.random.binomial(n, min(error_rate, 1.0))
    return latencies, error_count, n


def simulate_per_instance_requests(num_instances=10, bad_instance_idx=None, 
                                  mean_latency=50, bad_mean=450, error_rate=0.001, n_per=50):
    """Simulate latencies across instances, with optional one bad instance."""
    instance_latencies = {}
    total_errors = 0
    total_requests = num_instances * n_per
    
    for i in range(num_instances):
        if i == bad_instance_idx:
            latencies, errs, _ = simulate_requests(bad_mean, error_rate * 0.5, n=n_per)  # Fewer errors on bad
        else:
            latencies, errs, _ = simulate_requests(mean_latency, error_rate, n=n_per)
        instance_latencies[f"inst_{i}"] = latencies
        total_errors += errs
    
    return instance_latencies, total_errors, total_requests


def run_phase(engine, adapter, logger, label, n_steps, mean_latency, error_rate,
              rps, capacity_rps, memory_pct, disk_queue, use_per_instance=False, bad_instance=None):
    print(f"\n--- Phase: {label} {'(Per-Instance)' if use_per_instance else ''} ---")
    for i in range(n_steps):
        if use_per_instance:
            instance_lats, err_count, req_count = simulate_per_instance_requests(
                bad_instance_idx=bad_instance if i >= n_steps//2 else None
            )
            obs = adapter.extract_features_per_instance(instance_lats, err_count, req_count)
            extra = f" outliers={obs.get('persistent_outliers',0)}"
        else:
            latencies, err_count, req_count = simulate_requests(mean_latency, error_rate)
            obs = adapter.extract_features(latencies, err_count, req_count)
            extra = ""
        
        theta_c = adapter.normalize_context(rps, capacity_rps)
        theta_s = adapter.normalize_system_state(memory_pct, disk_queue)

        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs['theta_o'])
        status = "ALERT" if alert else "Normal"
        print(f"p95={obs['p95_latency_ms']:6.1f}ms  err={obs['error_rate']:.3f}  "
              f"theta_o={obs['theta_o']:.3f} {extra}  "
              f"theta_s={theta_s:.2f} theta_c={theta_c:.2f}  "
              f"SOP={sop:.3f} NHP={nhp:.3f} {status}")
        logger.log_state(sop, nhp, alert, metadata={
            "phase": label, "step": i,
            "p95_latency_ms": obs['p95_latency_ms'],
            "error_rate": obs['error_rate'],
        })


def main():
    print("=== JOR 4.0 -- Cloud Service Health Prototype (with Per-Instance) ===")
    engine = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                        retention=0.70, calibration_steps=20)
    load_engine_state(engine)

    adapter = ServiceHealthAdapter()
    logger = FusionLogger("cloud_vm_health_log.json")

    # Standard aggregate phases (unchanged)
    run_phase(engine, adapter, logger, "Calibration", 20, 45, 0.001, 150, 500, 40, 1.0)
    print(f"    -> Calibrated baseline_sop = {engine.baseline_sop:.4f}")

    run_phase(engine, adapter, logger, "Healthy", 10, 48, 0.002, 160, 500, 42, 1.2)

    # Per-instance degrading test: one bad instance
    print("\n--- Per-Instance Gray Failure Test (1 bad instance, zero errors) ---")
    for i in range(15):
        # Simulate bad instance appearing mid-way
        bad_idx = 2 if i >= 5 else None
        instance_lats, err_count, req_count = simulate_per_instance_requests(bad_instance_idx=bad_idx)
        obs = adapter.extract_features_per_instance(instance_lats, err_count, req_count)
        theta_c = adapter.normalize_context(160, 500)
        theta_s = adapter.normalize_system_state(45, 1.5)

        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs['theta_o'])
        status = "ALERT" if alert else "Normal"
        print(f"Step {i:2d}: theta_o={obs['theta_o']:.3f} outliers={obs.get('persistent_outliers',0)} "
              f"NHP={nhp:.3f} {status}")

    # Recovery
    run_phase(engine, adapter, logger, "Recovery", 15, 46, 0.001, 155, 500, 40, 1.0)

    save_engine_state(engine)

    print("\n=== Summary ===")
    print(f"Calibrated baseline_sop: {engine.baseline_sop:.4f}")


if __name__ == "__main__":
    main()
