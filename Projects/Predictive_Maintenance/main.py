import numpy as np
import json
import os
from engine import JOREngine
from adapters import VibrationAdapter
from logger import FusionLogger

STATE_FILE = "engine_state.json"


def load_state(engine, adapter):
    """
    Restores BOTH the engine's baseline_sop and the adapter's Criterion II
    baseline_velocity_mm_s from the same state file. These two baselines
    represent "what's normal" for two different halves of the same
    pipeline (fused SOP vs. raw vibration velocity) -- they should always
    be restored together, or not at all, so they can't silently drift out
    of sync with each other across a restart.
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            engine.p_final = state.get("p_final", engine.prior_nh)
            engine.alert_status = state.get("alert_status", False)
            engine.baseline_sop = state.get("baseline_sop", engine.baseline_sop)
            engine.is_calibrated = state.get("is_calibrated", engine.is_calibrated)
            engine.calibrating = not engine.is_calibrated

            # Older state files (saved before Criterion II persistence was
            # added) won't have an "adapter" key -- adapter.load_state(None)
            # is a no-op in that case, so the adapter just re-establishes
            # its own baseline fresh, same as it always did before.
            adapter.load_state(state.get("adapter"))

            print("--- State restored from disk ---")
            if adapter.baseline_established:
                print(f"--- Criterion II baseline restored: {adapter.baseline_velocity_mm_s:.3f} mm/s ---")


def save_state(engine, adapter):
    state = {
        "p_final": engine.p_final,
        "alert_status": engine.alert_status,
        "baseline_sop": engine.baseline_sop,
        "is_calibrated": engine.is_calibrated,
        "adapter": adapter.get_state(),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def make_vibration_buffer(peak_accel_g, freq, t, noise_std=0.5):
    """
    peak_accel_g: the zero-to-peak acceleration amplitude of the simulated
    sinusoid, in g's. NOT the RMS value -- for a pure sinusoid,
    RMS = peak / sqrt(2) (~0.707x peak). The true RMS of the full buffer
    (including the added noise and harmonic components elsewhere in this
    file) is computed downstream by VibrationAdapter.extract_features(),
    which is the only place "RMS" should be read as an actual RMS value.
    """
    return peak_accel_g * np.sin(2 * np.pi * freq * t) + np.random.normal(0, noise_std, len(t))


def print_row(step, phase, sop, nhp, features, alert):
    crit_ii_marker = "FLAG" if features['criterion_ii_flag'] else "-"
    print(f"{step:<5}{phase:<12}{sop:<8.4f}{nhp:<8.4f}{features['iso_zone']:<6}"
          f"{features['velocity_rms_mm_s']:<12.3f}{crit_ii_marker:<6}"
          f"{'ALERT' if alert else 'Normal'}")


def log_row(logger, sop, nhp, alert, phase, step, features, extra_metadata=None):
    metadata = {
        "phase": phase,
        "step": step,
        "iso_zone": features['iso_zone'],
        "velocity_rms_mm_s": features['velocity_rms_mm_s'],
        "baseline_velocity_mm_s": features['baseline_velocity_mm_s'],
        "criterion_ii_delta_mm_s": features['criterion_ii_delta_mm_s'],
        "criterion_ii_threshold_mm_s": features['criterion_ii_threshold_mm_s'],
        "criterion_ii_flag": features['criterion_ii_flag'],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    logger.log_state(sop, nhp, alert, metadata=metadata)


def main():
    print("=== JOR V4.0 Production Engine (Self-Calibrating Baseline + Retention + Hysteresis) ===")
    print("=== theta_o derived from ISO 20816-3 Group 2: Criterion I (zone) + Criterion II (25%-of-B/C rate of change) ===\n")

    engine = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                        retention=0.70, calibration_steps=20)
    adapter = VibrationAdapter(sample_rate=10000)
    load_state(engine, adapter)

    logger = FusionLogger("maintenance_log.json")

    print(f"Criterion II threshold: {adapter.criterion_ii_threshold_mm_s:.3f} mm/s "
          f"(25% of the {adapter.zone_bc} mm/s B/C boundary)")
    print("'CritII' column below reads FLAG when a reading changes by more than that\n"
          "threshold from the machine's established baseline -- even if it stays\n"
          "within the same absolute zone, which Criterion I alone would miss.\n")

    # NOTE: theta_s remains a fixed placeholder here, matching all prior test
    # scripts, so results are directly comparable to earlier runs. This still
    # needs to be wired to a real sensor/structural-health input for an
    # actual production deployment.
    theta_s = 0.72
    load, temp = 60.0, 40.0
    t = np.linspace(0, 1, 10000)

    # Realistic PEAK acceleration amplitudes (g's, zero-to-peak) for a
    # Group 2 machine at 180 Hz. These are peak values, not RMS -- for a
    # pure sinusoid, true RMS = peak / sqrt(2). They were chosen so the
    # RMS *velocity* actually computed downstream by VibrationAdapter
    # (mm/s, after the acceleration-to-velocity conversion) lands in
    # sensible ISO 20816-3 zones for each phase of the demo:
    #   healthy    ~0.12g pk  -> ~0.7  mm/s RMS velocity -> Zone A
    #   escalating  0.12-1.45g pk -> ~0.7-8.9 mm/s RMS velocity -> Zone A through D
    #   recovery   ~0.12g pk  -> back to Zone A
    # (Previous amplitudes of 3.0-12.5g were arbitrary units tuned against a
    # single linear safe_rms_g ceiling; they don't correspond to physically
    # realistic accelerations once mapped through real ISO velocity zones.)
    HEALTHY_PEAK_ACCEL_G = 0.12
    ESCALATION_STEP_PEAK_ACCEL_G = 0.07

    print(f"{'Step':<5}{'Phase':<12}{'SOP':<8}{'NHP':<8}{'Zone':<6}{'v_rms(mm/s)':<12}{'CritII':<6}{'Status'}")
    print("-" * 74)

    step = 0

    # --- Phase 1: Calibration (20 steps of healthy, quiet operation) ---
    # Note: the adapter's Criterion II baseline_window also defaults to 20,
    # so its own baseline finishes establishing around the same time the
    # engine finishes calibrating -- both "what's normal" learners warm up
    # together over this phase.
    for i in range(20):
        raw_buffer = make_vibration_buffer(peak_accel_g=HEALTHY_PEAK_ACCEL_G, freq=180, t=t,
                                            noise_std=HEALTHY_PEAK_ACCEL_G * 0.15)
        features = adapter.extract_features(raw_buffer)
        theta_c = adapter.normalize_context(machine_load=load, ambient_temp=temp)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, features['theta_o'])
        print_row(step, "Calibrating", sop, nhp, features, alert)
        log_row(logger, sop, nhp, alert, "Calibrating", step, features)
        step += 1
    print(f"      -> Calibrated baseline_sop = {engine.baseline_sop:.4f}")
    print(f"      -> Criterion II baseline velocity = {adapter.baseline_velocity_mm_s:.3f} mm/s\n")

    # --- Phase 2: Healthy operation, post-calibration (confirm stability) ---
    for i in range(15):
        raw_buffer = make_vibration_buffer(peak_accel_g=HEALTHY_PEAK_ACCEL_G, freq=180, t=t,
                                            noise_std=HEALTHY_PEAK_ACCEL_G * 0.15)
        features = adapter.extract_features(raw_buffer)
        theta_c = adapter.normalize_context(machine_load=load, ambient_temp=temp)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, features['theta_o'])
        print_row(step, "Healthy", sop, nhp, features, alert)
        log_row(logger, sop, nhp, alert, "Healthy", step, features)
        step += 1

    # --- Phase 3: Escalating danger (vibration ramps toward/through ISO zones) ---
    for i in range(20):
        peak_accel_g = HEALTHY_PEAK_ACCEL_G + i * ESCALATION_STEP_PEAK_ACCEL_G
        raw_buffer = make_vibration_buffer(peak_accel_g=peak_accel_g, freq=180, t=t,
                                            noise_std=peak_accel_g * 0.05)
        features = adapter.extract_features(raw_buffer)
        theta_c = adapter.normalize_context(machine_load=load, ambient_temp=temp)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, features['theta_o'])
        print_row(step, "Escalating", sop, nhp, features, alert)
        log_row(logger, sop, nhp, alert, "Escalating", step, features,
                extra_metadata={"peak_accel_g": peak_accel_g})
        step += 1

    # --- Phase 4: Recovery (vibration returns to healthy) ---
    for i in range(25):
        raw_buffer = make_vibration_buffer(peak_accel_g=HEALTHY_PEAK_ACCEL_G, freq=180, t=t,
                                            noise_std=HEALTHY_PEAK_ACCEL_G * 0.15)
        features = adapter.extract_features(raw_buffer)
        theta_c = adapter.normalize_context(machine_load=load, ambient_temp=temp)
        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, features['theta_o'])
        print_row(step, "Recovery", sop, nhp, features, alert)
        log_row(logger, sop, nhp, alert, "Recovery", step, features)
        step += 1

    save_state(engine, adapter)

    print("\n=== Self-Validation Summary ===")
    print(f"Calibrated baseline_sop: {engine.baseline_sop:.4f}")
    print("Expected: Phase 1-2 stay Normal (Zone A), Phase 3 transitions through "
          "B/C/D to ALERT, Phase 4 returns to Normal (Zone A).")
    print("Watch for 'FLAG' in the CritII column during Phase 3 -- that's the "
          "adapter catching a fast relative change, sometimes before the "
          "absolute zone itself has moved.")


if __name__ == "__main__":
    main()
