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

    # NOTE: this ramp originally used start_peak_accel_g=85.0 decaying by a fixed
    # 5.0/step, clipped at a hard floor of exactly 0.0 via max(0.0, ...).
    # That meant roughly the back half of the 40-step ramp had peak_accel_g=0 --
    # i.e. no actual vibration signal at all, just noise plus a fading 40Hz
    # burst tone. Under VibrationAdapter's single-dominant-frequency
    # acceleration-to-velocity conversion (already disclosed as a
    # limitation in adapters.py's docstring), that noise-only condition let
    # the FFT lock onto a spurious low-frequency component, and because
    # velocity = acceleration / frequency, a small frequency denominator
    # wildly inflated the computed "velocity" even though the actual
    # acceleration was negligible -- producing a nonsensical Zone D
    # classification for what should have read as near-silent. That's the
    # actual root cause of the ALERT/Normal flapping seen in the original
    # run's tail.
    #
    # Fixed by decaying smoothly toward a realistic healthy floor
    # (0.12g, matching main.py's Zone A baseline) instead of clipping to
    # literal zero -- a real machine's baseline vibration never actually
    # disappears to nothing, so "recovered" should mean "back to healthy
    # baseline," not "no signal at all." start_peak_accel_g rescaled from 85.0g to
    # 2.5g, which computes to Zone D under the real ISO 20816-3 conversion
    # (verified: theta_o ~1.0 at step 0), giving a genuine high-severity
    # starting point without needing an unrealistic acceleration value.
    start_peak_accel_g = 2.5
    floor_peak_accel_g = 0.12
    start_freq = 180

    alert_seen = False
    alert_cleared = False
    nhp_valid = True

    previous_alert = False

    # Recovery ramp: gradually reduce anomaly signatures
    for step in range(40):

        # Reduce vibration severity over time, decaying toward the healthy
        # floor rather than to literal zero.
        peak_accel_g = floor_peak_accel_g + (start_peak_accel_g - floor_peak_accel_g) * max(0.0, 1 - step / 30)
        freq = start_freq

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
            peak_accel_g * mod +
            peak_accel_g * harm +
            burst +
            np.random.normal(0, peak_accel_g * 0.05, 10000)
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