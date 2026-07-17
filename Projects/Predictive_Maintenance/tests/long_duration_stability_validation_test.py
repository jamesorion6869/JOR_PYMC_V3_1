import numpy as np
from engine import JOREngine
from adapters import VibrationAdapter


def main():
    print("=== JOR V4.0 Long-Duration Stability Validation Test ===\n")

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

    alert_seen = False
    nhp_valid = True
    nhp_values = []
    sop_values = []

    steps = 1000

    for step in range(steps):

        # Stable baseline vibration with small random variation
        rms = base_rms + np.random.normal(0, 0.3)
        freq = base_freq + np.random.normal(0, 2.0)

        harm = 0.1 * np.sin(2 * np.pi * (freq * 2) * t)
        mod = np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 0.5, 10000)

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

        sop_values.append(sop)
        nhp_values.append(nhp)

        if alert_status:
            alert_seen = True

        if not (0.0 <= nhp <= 1.0):
            nhp_valid = False

        status = "ALERT" if alert_status else "Normal"

        # Print every 50 steps to keep output manageable
        if step % 50 == 0:
            print(
                f"{step:<4} {sop:<8.4f} {nhp:<8.4f} {status}"
            )

    nhp_min = min(nhp_values)
    nhp_max = max(nhp_values)
    sop_min = min(sop_values)
    sop_max = max(sop_values)

    print("\n=== Validation Results ===")

    if not alert_seen:
        print("PASS: No false alerts occurred during long-duration stable operation.")
    else:
        print("FAIL: False alert detected during stable operation.")

    if nhp_valid:
        print("PASS: NHP remained within valid probability bounds (0-1).")
    else:
        print("FAIL: NHP exceeded valid probability bounds.")

    print(
        f"INFO: NHP range = {nhp_min:.4f} - {nhp_max:.4f}"
    )

    print(
        f"INFO: SOP range = {sop_min:.4f} - {sop_max:.4f}"
    )

    if not alert_seen and nhp_valid:
        print("\n=== LONG-DURATION STABILITY TEST PASSED ===")
    else:
        print("\n=== LONG-DURATION STABILITY TEST FAILED ===")


if __name__ == "__main__":
    main()