import numpy as np
from engine import JOREngine
from adapters import VibrationAdapter


def main():
    print("=== JOR V4.0 Recovery Ramp Validation Test ===\n")

    engine = JOREngine(
        prior_nh=0.05,
        steepness=12.0,     # matches danger-ramp configuration
        baseline_sop=0.55,
        upper_th=0.55,
        lower_th=0.45
    )

    adapter = VibrationAdapter(sample_rate=10000)

    print(f"{'Step':<4} {'SOP':<8} {'NHP':<8} {'Status'}")
    print("-" * 40)

    t = np.linspace(0, 1, 10000)

    # Start in degraded conditions
    start_rms = 85.0
    start_freq = 180 + (39 * 20)

    alert_seen = False
    alert_cleared = False
    nhp_valid = True

    previous_alert = False

    # Recovery ramp: gradually reduce anomaly signatures
    for step in range(40):

        # Reduce vibration severity over time
        rms = max(0.0, start_rms - (step * 5.0))
        freq = start_freq - (step * 5)

        load = 60.0
        temp = 40.0

        harm = 0.3 * np.sin(2 * np.pi * (freq * 2) * t)

        burst = (
            0.2 *
            np.sin(2 * np.pi * 40 * t) *
            (1 - step / 40)
        )

        mod = (
            np.sin(2 * np.pi * freq * t) *
            (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        )

        raw_buffer = (
            rms * mod +
            rms * harm +
            burst +
            np.random.normal(0, 0.5, 10000)
        )

        features = adapter.extract_features(raw_buffer)

        theta_c = adapter.normalize_context(
            machine_load=load,
            ambient_temp=temp
        )

        theta_s = 0.90

        sop, nhp, alert_status = engine.fusion_step(
            theta_s,
            theta_c,
            features['theta_o']
        )

        # Validation tracking
        if alert_status:
            alert_seen = True

        if previous_alert and not alert_status:
            alert_cleared = True

        if not (0.0 <= nhp <= 1.0):
            nhp_valid = False

        previous_alert = alert_status

        status = "ALERT" if alert_status else "Normal"

        print(
            f"{step:<4} {sop:<8.4f} {nhp:<8.4f} {status}"
        )

    print("\n=== Validation Results ===")

    if alert_seen:
        print("PASS: Alert activated during degraded conditions.")
    else:
        print("FAIL: Alert was never activated.")

    if alert_cleared:
        print("PASS: Alert cleared during recovery.")
    else:
        print("FAIL: Alert did not clear during recovery.")

    if nhp_valid:
        print("PASS: NHP remained within valid probability bounds (0-1).")
    else:
        print("FAIL: NHP exceeded valid probability bounds.")

    print()

    if alert_seen and alert_cleared and nhp_valid:
        print("=== RECOVERY RAMP TEST PASSED ===")
    else:
        print("=== RECOVERY RAMP TEST FAILED ===")


if __name__ == "__main__":
    main()