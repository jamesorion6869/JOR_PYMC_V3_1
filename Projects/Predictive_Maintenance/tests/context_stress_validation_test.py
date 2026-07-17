import numpy as np
from engine import JOREngine
from adapters import VibrationAdapter


def main():

    print("=== JOR V4.0 Context Stress Validation Test (θc Sweep) ===\n")

    engine = JOREngine(
        prior_nh=0.05,
        steepness=0.50,
        baseline_sop=0.55,
        upper_th=0.55,
        lower_th=0.45
    )

    adapter = VibrationAdapter(sample_rate=10000)

    print(
        f"{'Step':<4} {'Load%':<7} {'Temp':<6} "
        f"{'θc':<8} {'SOP':<8} {'NHP':<8} {'Status'}"
    )
    print("-" * 65)

    t = np.linspace(0, 1, 10000)

    base_rms = 5.0
    base_freq = 180

    steps = 48

    alert_seen = False
    nhp_valid = True
    max_nhp = 0.0

    for step in range(steps):

        # Operational context sweep
        load = np.linspace(0, 100, steps)[step]
        temp = np.linspace(20, 80, steps)[step]

        # Normal vibration conditions
        rms = base_rms + np.random.normal(0, 0.3)
        freq = base_freq + np.random.normal(0, 2.0)

        harm_amp = 0.1
        mod_amp = 1.0

        harm = (
            harm_amp *
            np.sin(2 * np.pi * (freq * 2) * t)
        )

        mod = (
            np.sin(2 * np.pi * freq * t) *
            mod_amp
        )

        noise = np.random.normal(0, 0.5, 10000)

        raw_buffer = (
            rms * mod +
            rms * harm +
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

        # Validation checks

        if alert_status:
            alert_seen = True

        if not (0.0 <= nhp <= 1.0):
            nhp_valid = False

        max_nhp = max(max_nhp, nhp)

        status = "ALERT" if alert_status else "Normal"

        print(
            f"{step:<4} {load:<7.1f} {temp:<6.1f} "
            f"{theta_c:<8.4f} {sop:<8.4f} "
            f"{nhp:<8.4f} {status}"
        )


    print("\n=== Validation Results ===")

    if not alert_seen:
        print("PASS: No false alerts triggered during context stress sweep.")
    else:
        print("FAIL: Alert triggered during normal context variation.")

    if nhp_valid:
        print("PASS: NHP remained within valid probability bounds (0-1).")
    else:
        print("FAIL: NHP exceeded valid probability bounds.")

    print(f"INFO: Maximum observed NHP = {max_nhp:.4f}")

    if not alert_seen and nhp_valid:
        print("\n=== CONTEXT STRESS TEST PASSED ===")
    else:
        print("\n=== CONTEXT STRESS TEST FAILED ===")


if __name__ == "__main__":
    main()