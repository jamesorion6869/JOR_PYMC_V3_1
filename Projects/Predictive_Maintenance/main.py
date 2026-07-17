import numpy as np
import json
import os
from engine import JOREngine
from adapters import VibrationAdapter
from logger import FusionLogger

STATE_FILE = "engine_state.json"


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


def make_vibration_buffer(rms, freq, t, noise_std=0.5):
    return rms * np.sin(2 * np.pi * freq * t) + np.random.normal(0, noise_std, len(t))


def main():
    print("=== JOR V4.0 Production Engine (Self-Calibrating Baseline + Retention + Hysteresis) ===\n")

    engine = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                        retention=0.70, calibration_steps=20)
    load_engine_state(engine)

    adapter = VibrationAdapter(sample_rate=10000)
    logger = FusionLogger("maintenance_log.json")

    # NOTE: theta_s remains a fixed placeholder here, matching all prior test
    # scripts, so results are directly comparable to earlier runs. This still
    # needs to be wired to a real sensor/structural-health input for an
    # actual production deployment.
    theta_s = 0.72
    load, temp = 60.0, 40.0
    t = np.linspace(0, 1, 10000)

    print(f"{'Step':<5}{'Phase':<12}{'SOP':<8}{'NHP':<8}{'Status'}")
    print("-" * 50)

    step = 0

    # --- Phase 1: Calibration (20 steps of healthy, quiet operation) ---
    for i in range(20):
        raw_buffer = make_vibration_buffer(rms=3.0, freq=180, t=t)
        features = adapter.extract_features(raw_buffer)
        theta_c = adapter.normalize_context(machine_load=load, ambient_temp=temp)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, features['theta_o'])
        phase = "Calibrating"
        print(f"{step:<5}{phase:<12}{sop:<8.4f}{nhp:<8.4f}{'ALERT' if alert else 'Normal'}")
        logger.log_state(sop, nhp, alert, metadata={"phase": phase, "step": step})
        step += 1
    print(f"      -> Calibrated baseline_sop = {engine.baseline_sop:.4f}\n")

    # --- Phase 2: Healthy operation, post-calibration (confirm stability) ---
    for i in range(15):
        raw_buffer = make_vibration_buffer(rms=3.0, freq=180, t=t)
        features = adapter.extract_features(raw_buffer)
        theta_c = adapter.normalize_context(machine_load=load, ambient_temp=temp)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, features['theta_o'])
        print(f"{step:<5}{'Healthy':<12}{sop:<8.4f}{nhp:<8.4f}{'ALERT' if alert else 'Normal'}")
        logger.log_state(sop, nhp, alert, metadata={"phase": "Healthy", "step": step})
        step += 1

    # --- Phase 3: Escalating danger (vibration ramps toward rated limit) ---
    for i in range(20):
        rms = 3.0 + i * 0.5   # ramps up toward/past the rated safe limit
        raw_buffer = make_vibration_buffer(rms=rms, freq=180, t=t)
        features = adapter.extract_features(raw_buffer)
        theta_c = adapter.normalize_context(machine_load=load, ambient_temp=temp)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, features['theta_o'])
        print(f"{step:<5}{'Escalating':<12}{sop:<8.4f}{nhp:<8.4f}{'ALERT' if alert else 'Normal'}")
        logger.log_state(sop, nhp, alert, metadata={"phase": "Escalating", "step": step, "rms": rms})
        step += 1

    # --- Phase 4: Recovery (vibration returns to healthy) ---
    for i in range(25):
        raw_buffer = make_vibration_buffer(rms=3.0, freq=180, t=t)
        features = adapter.extract_features(raw_buffer)
        theta_c = adapter.normalize_context(machine_load=load, ambient_temp=temp)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, features['theta_o'])
        print(f"{step:<5}{'Recovery':<12}{sop:<8.4f}{nhp:<8.4f}{'ALERT' if alert else 'Normal'}")
        logger.log_state(sop, nhp, alert, metadata={"phase": "Recovery", "step": step})
        step += 1

    save_engine_state(engine)

    print("\n=== Self-Validation Summary ===")
    print(f"Calibrated baseline_sop: {engine.baseline_sop:.4f}")
    print("Expected: Phase 1-2 stay Normal, Phase 3 transitions to ALERT, "
          "Phase 4 returns to Normal.")


if __name__ == "__main__":
    main()
