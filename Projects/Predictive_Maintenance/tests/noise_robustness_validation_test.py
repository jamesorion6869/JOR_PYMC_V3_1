import numpy as np
from engine import JOREngine
from adapters import VibrationAdapter


def main():
    print("=== JOR V4.0 Noise Robustness Validation Test ===\n")

    engine = JOREngine(
        prior_nh=0.05,
        steepness=0.50,
        baseline_sop=0.55,
        upper_th=0.55,
        lower_th=0.45
    )

    adapter = VibrationAdapter(sample_rate=10000)

    print(f"{'Step':<4} {'SOP':<8} {'NHP':<8} {'Status'}")
    print("-" * 40)

    t = np.linspace(0, 1, 10000)

    # NOTE: base_peak_accel_g rescaled from 5.0g, and noise magnitude rescaled from a
    # fixed absolute 2.0 (which, at the old 5.0g scale, was a deliberately
    # heavy 40% noise-to-signal ratio meant to stress-test robustness).
    # Under VibrationAdapter's ISO 20816-3 grounded conversion, the OLD
    # values computed to Zone D saturation (theta_o pegged at exactly 1.0),
    # which is why the original run reported a perfectly flat SOP range of
    # 0.7365-0.7365 across 100 steps -- despite peak_accel_g/freq/noise all being
    # randomized every step, zero variation was actually reaching the
    # output. That's not "robust to noise," it's "noise never had a chance
    # to matter because the signal was already maxed out." Rescaled to a
    # realistic Zone A/B baseline, preserving the same ~40% heavy-noise
    # ratio relative to the new base_peak_accel_g so this is still a genuine
    # heavy-noise stress test, just at a scale where its result means
    # something.
    base_peak_accel_g = 0.5
    base_freq = 180

    alert_seen = False
    nhp_valid = True

    nhp_values = []
    sop_values = []

    steps = 100

    for step in range(steps):

        # Healthy operating condition
        peak_accel_g = base_peak_accel_g + np.random.normal(0, base_peak_accel_g * 0.10)
        freq = base_freq + np.random.normal(0, 5.0)

        load = 60.0
        temp = 40.0

        mod = np.sin(2 * np.pi * freq * t)

        harm = (
            0.1 *
            np.sin(2 * np.pi * (freq * 2) * t)
        )

        # Increased measurement noise (heavy, ~40% of base_peak_accel_g, same
        # stress-test ratio as the original fixed value relative to the
        # old base_peak_accel_g=5.0)
        noise = np.random.normal(0, base_peak_accel_g * 0.40, 10000)

        raw_buffer = (
            peak_accel_g * mod +
            peak_accel_g * harm +
            noise
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

        sop_values.append(sop)
        nhp_values.append(nhp)

        if alert_status:
            alert_seen = True

        if not (0.0 <= nhp <= 1.0):
            nhp_valid = False

        status = "ALERT" if alert_status else "Normal"

        if step % 10 == 0:
            print(
                f"{step:<4} {sop:<8.4f} {nhp:<8.4f} {status}"
            )

    print("\n=== Validation Results ===")

    if not alert_seen:
        print(
            "PASS: No false alerts triggered under noisy normal operation."
        )
    else:
        print(
            "FAIL: False alert triggered during noise robustness test."
        )

    if nhp_valid:
        print(
            "PASS: NHP remained within valid probability bounds (0-1)."
        )
    else:
        print(
            "FAIL: NHP exceeded valid probability bounds."
        )

    print(
        f"INFO: NHP range = {min(nhp_values):.4f} - {max(nhp_values):.4f}"
    )

    print(
        f"INFO: SOP range = {min(sop_values):.4f} - {max(sop_values):.4f}"
    )

    if not alert_seen and nhp_valid:
        print("\n=== NOISE ROBUSTNESS TEST PASSED ===")
    else:
        print("\n=== NOISE ROBUSTNESS TEST FAILED ===")


if __name__ == "__main__":
    main()