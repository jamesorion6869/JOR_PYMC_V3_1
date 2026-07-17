# danger_ramp_validation.py

from engine import JOREngine

jor = JOREngine()

print("=== JOR V4.0 Danger Ramp Validation Test ===\n")
print("Step SOP      NHP      Status")
print("----------------------------------------")

step = 0
alert_triggered = False
nhp_valid = True

# --- HEALTHY PREFIX (20 steps) ---
healthy_sop_values = [0.52 + (0.0005 * i) for i in range(20)]

for sop in healthy_sop_values:
    # SOP fed into all three inputs
    _, nhp, status = jor.fusion_step(sop, sop, sop)

    if not (0.0 <= nhp <= 1.0):
        nhp_valid = False

    print(f"{step:<4} {sop:.4f}   {nhp:.4f}   {status}")
    step += 1


# --- TRUE DANGER RAMP (keeps increasing) ---
# Values above 1.0 intentionally stress-test engine input handling.
danger_sop_values = [
    0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 1.00, 1.05,
    1.10, 1.15, 1.20, 1.25, 1.30
]

for sop in danger_sop_values:
    _, nhp, status = jor.fusion_step(sop, sop, sop)

    if status:
        alert_triggered = True

    if not (0.0 <= nhp <= 1.0):
        nhp_valid = False

    print(f"{step:<4} {sop:.4f}   {nhp:.4f}   {status}")
    step += 1


print("\n=== Validation Results ===")

if alert_triggered:
    print("PASS: Alert triggered during danger ramp.")
else:
    print("FAIL: Alert did not trigger during danger ramp.")

if nhp_valid:
    print("PASS: NHP remained within valid probability bounds (0-1).")
else:
    print("FAIL: NHP exceeded valid probability bounds.")

if alert_triggered and nhp_valid:
    print("\n=== DANGER RAMP TEST PASSED ===")
else:
    print("\n=== DANGER RAMP TEST FAILED ===")