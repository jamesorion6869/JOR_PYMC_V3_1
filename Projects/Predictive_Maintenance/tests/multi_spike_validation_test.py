import numpy as np
from engine import JOREngine
from adapters import VibrationAdapter


def main():

    print("=== JOR V4.0 Multi-Spike Stress Validation Test (Calibrated) ===\n")

    engine = JOREngine(
        prior_nh=0.05,
        steepness=12.0,
        baseline_sop=0.55,
        upper_th=0.55,
        lower_th=0.45
    )

    adapter = VibrationAdapter(sample_rate=10000)

    print(f"{'Step':<4} {'SOP':<8} {'NHP':<8} {'Status'}")
    print("-" * 40)

    t = np.linspace(0, 1, 10000)

    # NOTE: base_peak_accel_g and burst peak_accel_g rescaled from 5.0g/90.0g. Under
    # VibrationAdapter's ISO 20816-3 grounded conversion, BOTH the old
    # "normal" (5.0g) and "burst" (90.0g) values independently computed to
    # Zone D saturation (theta_o pegged at exactly 1.0 for both) -- meaning
    # there was zero vibration-channel signal distinguishing quiet baseline
    # from severe anomaly at all. That's the actual root cause of this
    # test's current FAIL: with no theta_o contrast between phases, the
    # engine had nothing telling it conditions had improved, so NHP never
    # dropped enough to demonstrate recovery between bursts. Rescaled so
    # normal operation sits in Zone A (healthy) and burst events genuinely
    # spike into Zone D (severe), restoring the intended contrast.
    base_peak_accel_g = 0.12
    base_freq = 180

    # Validation tracking
    alert_seen = False
    normal_seen_after_alert = False
    nhp_valid = True
    max_nhp = 0.0

    previous_alert = False

    for step in range(120):

        # Three repeated anomaly bursts:
        # Burst 1: steps 10-34
        # Burst 2: steps 50-74
        # Burst 3: steps 90-114

        if (10 <= step <= 34) or (50 <= step <= 74) or (90 <= step <= 114):

            peak_accel_g = 2.0
            freq = 350.0
            burst_amp = 0.8

            harm = (
                0.4 *
                np.sin(2 * np.pi * (freq * 2) * t)
            )

            mod = (
                np.sin(2 * np.pi * freq * t) *
                (1 + 0.7 * np.sin(2 * np.pi * 5 * t))
            )

            theta_s = 0.90

            theta_c = adapter.normalize_context(
                machine_load=70.0,
                ambient_temp=60.0
            )

        else:

            # Normal operating baseline
            peak_accel_g = base_peak_accel_g + np.random.normal(0, base_peak_accel_g * 0.05)
            freq = base_freq + np.random.normal(0, 2.0)
            burst_amp = 0.0

            harm = (
                0.1 *
                np.sin(2 * np.pi * (freq * 2) * t)
            )

            mod = np.sin(2 * np.pi * freq * t)

            theta_s = 0.75

            theta_c = adapter.normalize_context(
                machine_load=40.0,
                ambient_temp=40.0
            )


        burst = (
            burst_amp *
            np.sin(2 * np.pi * 40 * t)
        )

        noise = np.random.normal(0, peak_accel_g * 0.05, 10000)

        raw_buffer = (
            peak_accel_g * mod +
            peak_accel_g * harm +
            burst +
            noise
        )

        features = adapter.extract_features(raw_buffer)


        sop, nhp, alert_status = engine.fusion_step(
            theta_s,
            theta_c,
            features['theta_o']
        )


        # Validation tracking

        if alert_status:
            alert_seen = True

        if previous_alert and not alert_status:
            normal_seen_after_alert = True

        if not (0.0 <= nhp <= 1.0):
            nhp_valid = False

        max_nhp = max(max_nhp, nhp)

        previous_alert = alert_status


        status = "ALERT" if alert_status else "Normal"

        print(
            f"{step:<4} {sop:<8.4f} "
            f"{nhp:<8.4f} {status}"
        )


    print("\n=== Validation Results ===")

    if alert_seen:
        print("PASS: Alert activated during repeated anomaly spikes.")
    else:
        print("FAIL: No alert activation detected.")

    if normal_seen_after_alert:
        print("PASS: System recovered between anomaly events.")
    else:
        print("FAIL: System did not demonstrate recovery behavior.")

    if nhp_valid:
        print("PASS: NHP remained within valid probability bounds (0-1).")
    else:
        print("FAIL: NHP exceeded valid probability bounds.")

    print(f"INFO: Maximum observed NHP = {max_nhp:.4f}")


    if alert_seen and normal_seen_after_alert and nhp_valid:
        print("\n=== MULTI-SPIKE STRESS TEST PASSED ===")
    else:
        print("\n=== MULTI-SPIKE STRESS TEST FAILED ===")


if __name__ == "__main__":
    main()