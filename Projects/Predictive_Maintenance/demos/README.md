# JOR V4.0 Real-Time Telemetry Demonstration

## Overview

This directory contains demonstration scripts showing real-time data ingestion into the JOR V4.0 Bayesian fusion engine.

The examples demonstrate how live operational telemetry can be converted into evidence inputs and processed through the JOR recursive fusion architecture.

---

## Real-Time CPU Load Demonstration

**File:**

`real_time_cpu_load_demo.py`

### Purpose

Demonstrates live telemetry integration using system CPU utilization as an operational input signal.

CPU load is sampled in real time, normalized into an SOP (Solid Object Probability / Signal Observation Probability) input value, and processed through the JOR V4.0 fusion engine.

The demonstration shows:

- Continuous telemetry ingestion
- Real-time evidence updates
- Recursive Bayesian state tracking
- Dynamic NHP estimation
- Alert state evaluation

---

## Execution

Run from the Predictive Maintenance project directory:

```bash
python demos/real_time_cpu_load_demo.py
