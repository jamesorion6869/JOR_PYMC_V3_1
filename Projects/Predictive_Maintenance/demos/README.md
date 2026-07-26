# JOR V4.0 Real-Time Telemetry Demonstration

## Overview

This directory contains a demonstration script showing real-time data ingestion into the JOR V4.0 Bayesian fusion engine.

The example demonstrates how live operational telemetry can be converted into evidence inputs and processed through the JOR recursive fusion architecture.

---

## Real-Time CPU Load Demonstration

**File:**

`real_time_cpu_load_demo.py`

### Purpose

Demonstrates live telemetry integration using system CPU utilization as an operational input signal.

CPU load is sampled in real time via `psutil`, normalized to a 0-1 SOP (Signal Object Processing) value, and fed into the JOR V4.0 fusion engine as all three evidence inputs (`theta_s`, `theta_c`, `theta_o`) simultaneously. This is a simpler engine-reactivity demonstration than the predictive-maintenance vibration pipeline elsewhere in this repository: it doesn't use a domain-specific adapter to derive three independent evidence channels from CPU telemetry, it drives the same fused engine directly from one live signal to show real-time responsiveness with no synthetic data involved.

The demonstration shows:

- Continuous telemetry ingestion (live, not simulated)
- Real-time evidence updates
- Recursive Bayesian state tracking
- Dynamic NHP estimation
- Alert state evaluation, including the engine's 20-step calibration
  window (NHP stays at the prior and no alert can fire until the engine
  has learned a baseline from live conditions first)

### Note on the engine used

This demo imports `JOREngine` directly from the predictive-maintenance
project's `engine.py`, using the same default constructor values
(`steepness=12.0`, `upper_th=0.55`, `lower_th=0.45`, `retention=0.70`).
It is fully compatible with the current engine and its documented
calibration/retention/hysteresis behavior.

---

## Execution

Run from the Predictive Maintenance project root directory:

```bash
python demos/real_time_cpu_load_demo.py
