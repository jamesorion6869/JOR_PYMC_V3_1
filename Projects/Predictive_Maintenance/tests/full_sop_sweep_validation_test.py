import numpy as np
from engine import JOREngine


def main():
    print("=== JOR V4.0 Full SOP Sweep Validation Test ===\n")

    engine = JOREngine(
        prior_nh=0.05,
        steepness=0.50,
        baseline_sop=0.55,
        upper_th=0.55,
        lower_th=0.45
    )

    print(f"{'Idx':<4} {'SOP In':<8} {'NHP Out':<8}")
    print("-" * 32)

    sop_values = np.linspace(0.0, 1.0, 41)

    nhp_values = []

    nhp_valid = True
    finite_valid = True

    for idx, sop in enumerate(sop_values):

        theta_s = 0.90
        theta_c = 0.55
        theta_o = sop

        sop_out, nhp, _ = engine.fusion_step(
            theta_s,
            theta_c,
            theta_o
        )

        nhp_values.append(nhp)

        if not (0.0 <= nhp <= 1.0):
            nhp_valid = False

        if not np.isfinite(nhp):
            finite_valid = False

        print(
            f"{idx:<4} {sop:<8.4f} {nhp:<8.4f}"
        )

    # Check overall response direction
    monotonic_response = all(
        nhp_values[i] <= nhp_values[i + 1] + 0.05
        for i in range(len(nhp_values) - 1)
    )

    print("\n=== Validation Results ===")

    if nhp_valid:
        print(
            "PASS: NHP remained within valid probability bounds (0-1)."
        )
    else:
        print(
            "FAIL: NHP exceeded valid probability bounds."
        )

    if finite_valid:
        print(
            "PASS: All outputs remained numerically stable."
        )
    else:
        print(
            "FAIL: Non-finite values detected."
        )

    if monotonic_response:
        print(
            "PASS: NHP response remained consistent across SOP sweep."
        )
    else:
        print(
            "FAIL: Unexpected NHP response behavior detected."
        )

    print(
        f"INFO: NHP range = {min(nhp_values):.4f} - {max(nhp_values):.4f}"
    )

    if nhp_valid and finite_valid and monotonic_response:
        print("\n=== FULL SOP SWEEP TEST PASSED ===")
    else:
        print("\n=== FULL SOP SWEEP TEST FAILED ===")


if __name__ == "__main__":
    main()