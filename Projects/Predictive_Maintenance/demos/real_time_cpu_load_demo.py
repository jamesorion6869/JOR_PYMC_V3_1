import psutil
import time
from engine import JOREngine

"""
JOR V4.0 Real-Time Telemetry Demonstration
------------------------------------------
Demonstrates live integration of JOR V4.0 using
CPU utilization telemetry as an operational input.

CPU load is normalized into an SOP evidence signal
and processed through the Bayesian fusion engine.

This is an integration demonstration, not an
industrial predictive maintenance benchmark.
"""

# Use your existing JOR V4.0 settings
engine = JOREngine(
    prior_nh=0.05,
    steepness=12.0,
    upper_th=0.55,
    lower_th=0.45,
    retention=0.70
)

print("=== JOR V4.0 Real-Time CPU Load Test ===")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        # Read CPU load (0–100%)
        cpu = psutil.cpu_percent(interval=1)

        # Normalize to SOP (0–1)
        sop = cpu / 100.0

        # Feed into your existing engine
        sop_out, nhp, alert = engine.fusion_step(sop, sop, sop)

        status = "ALERT" if alert else "Normal"
        print(f"CPU Load={cpu:5.1f}% | SOP={sop_out:.3f} | NHP={nhp:.3f} | {status}")

except KeyboardInterrupt:
    print("\n=== Stopped by user. JOR V4.0 test ended. ===")
