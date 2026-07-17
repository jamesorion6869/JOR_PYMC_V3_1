import numpy as np
from engine import JOREngine
from adapters import VibrationAdapter


def main():
    print("=== JOR V4.0 False-Positive Immunity Validation Test ===\n")

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

    base_rms = 5.0
    base_freq = 180

    alert_triggered = False
    nhp_valid = True
    max_nhp = 0.0

    for step in range(60):

        # Borderline patterns:
        # Individual indicators increase slightly,
        # but conditions never combine into a true anomaly.
        mode = step % 4

        if mode == 0:
            # Slightly elevated RMS
            rms = base_rms + 1.5
            freq = base_freq
            harm_amp = 0.1
            mod_amp = 1.0

        elif mode == 1:
            # Slightly elevated frequency
            rms = base_rms
            freq = base_freq + 20
            harm_amp = 0.1
            mod_amp = 1.0

        elif mode == 2:
            # Slightly elevated harmonics
            rms = base_rms
            freq = base_freq
            harm_amp = 0.25
            mod_amp = 1.0

        else:
            # Slightly elevated modulation depth
            rms = base_rms
            freq = base_freq
            harm_amp = 0.1
            mod_amp = 1.3

        harm = harm_amp * np.sin(
            2 * np.pi * (freq * 2) * t
        )

        mod = (
            np.sin(2 * np.pi * freq * t) *
            mod_amp
        )

        noise = np.random.normal(
            0,
            0.5,
            10000
        )

        raw_buffer = (
            rms * mod +
            rms * harm +
            noise
        )

        features = adapter.extract_features(raw_buffer)

        theta_c = adapter.normalize_context(
            machine_load=60.0,
            ambient_temp=40.0
        )

        theta_s = 0.90

        sop, nhp, alert_status = engine.fusion_step(
            theta_s,
            theta_c,
            features['theta_o']
        )

        if alert_status:
            alert_triggered = True

        if not (0.0 <= nhp <= 1.0):
            nhp_valid = False

        max_nhp = max(max_nhp, nhp)

        status = "ALERT" if alert_status else "Normal"

        print(
            f"{step:<4} {sop:<8.4f} {nhp:<8.4f} {status}"
        )

    print("\n=== Validation Results ===")

    if not alert_triggered:
        print("PASS: No false alerts triggered during normal variation.")
    else:
        print("FAIL: False alert triggered during normal variation.")

    if nhp_valid:
        print("PASS: NHP remained within valid probability bounds (0-1).")
    else:
        print("FAIL: NHP exceeded valid probability bounds.")

    print(f"INFO: Maximum observed NHP = {max_nhp:.4f}")

    print()

    if not alert_triggered and nhp_valid:
        print("=== FALSE-POSITIVE IMMUNITY TEST PASSED ===")
    else:
        print("=== FALSE-POSITIVE IMMUNITY TEST FAILED ===")


if __name__ == "__main__":
    main()