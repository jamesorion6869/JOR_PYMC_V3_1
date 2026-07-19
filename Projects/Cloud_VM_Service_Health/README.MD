# JOR 4.0 Cloud VM Service Health Fusion Engine

## Related Project

This demonstration is part of the broader **JOR (James Orion Report) Bayesian Evidence Fusion Framework**.

The JOR framework provides a general recursive evidence-fusion architecture that can be adapted across operational domains through domain-specific evidence adapters.

Related application demonstrations include:

- Predictive Maintenance — industrial sensor health monitoring
- Cloud VM Service Health — cloud telemetry and service reliability monitoring

Main Repository:

https://github.com/jamesorion6869/JOR_PYMC_V3_1

Cite All Versions:

https://doi.org/10.5281/zenodo.18088931

---

## Overview

This project demonstrates a domain adaptation of the James Orion Report (JOR) recursive Bayesian fusion architecture for cloud VM and service health monitoring.

The prototype applies JOR evidence fusion concepts to cloud telemetry by converting operational measurements into structured evidence inputs, recursively updating a non-healthy probability estimate, and generating a hysteresis-controlled health alert state.

This is a research prototype demonstrating architecture transfer. It is not intended to replace production observability platforms or operational monitoring systems.

---

# Architecture

The system follows the JOR evidence-fusion workflow:

Cloud Telemetry Simulation  
→  
ServiceHealthAdapter  
→  
JOR Evidence Inputs (theta_s, theta_c, theta_o)  
→  
JOR Fusion Engine  
→  
SOP + NHP Posterior  
→  
Alert State  
→  
Fusion Logger

---

# JOR Domain Mapping

## Signal Object Processing (SOP)

In this adaptation:

SOP represents the fused evidence state generated from multiple cloud service health signals.

The engine combines:

SOP =  
0.40 * theta_s +  
0.30 * theta_c +  
0.30 * theta_o

Where:

### theta_s — System State Evidence

Represents infrastructure condition indicators.

Examples:

- memory utilization  
- disk queue pressure  
- system resource stress  

---

### theta_c — Operational Context Evidence

Represents workload operating conditions.

Examples:

- request utilization  
- service capacity pressure  
- traffic saturation  

---

### theta_o — Observed Service Health Evidence

Represents observed service behavior.

Examples:

- latency degradation  
- error rates  
- per-instance latency anomalies  
- gray failures  

---

# Non-Healthy Probability (NHP)

NHP represents the recursive posterior probability that the current service state is non-healthy.

The engine updates NHP using accumulated evidence while retaining stability through prior retention.

Features:

- recursive evidence updating  
- configurable prior probability  
- self-calibrating baseline estimation  
- retention-based stabilization  
- hysteresis alert thresholds  

---

# Components

## engine.py

JOR recursive fusion engine.

Responsibilities:

- calculate SOP  
- establish healthy baseline  
- update NHP posterior  
- maintain recursive state  
- control alert transitions  

---

## service_adapter.py

Cloud telemetry adapter.

Responsibilities:

- convert telemetry measurements into JOR evidence inputs  
- calculate observed service health evidence  
- detect per-instance gray failures  

Features:

- aggregate latency analysis  
- instance-level latency tracking  
- IQR-based outlier detection  
- EMA persistence tracking  
- instance state pruning for rotating cloud workloads  

---

## cloud_vm_demo.py

Demonstration workflow.

Includes:

- calibration phase  
- healthy operation phase  
- per-instance gray failure simulation  
- recovery phase  
- state persistence  

---

## logger.py

Fusion state logging.

Records:

- timestamp  
- fused SOP value  
- NHP posterior  
- alert state  
- supporting metadata  

Output format: JSON lines for later analysis.

---

# Validation Tests

The adapter and fusion engine were evaluated with simulated cloud service scenarios including:

- healthy long-duration stability  
- noisy telemetry conditions  
- calibration mismatch conditions  
- instance churn behavior  
- gray failure detection  
- IQR floor protection  
- baseline scaling across latency ranges  

Tested latency baselines:

- 5 ms  
- 50 ms  
- 500 ms  
- 5000 ms  

Validation confirmed:

- bounded theta_o behavior during healthy operation  
- no alert lock under normal noise  
- detection of isolated degraded instances  
- recovery after degraded instance removal  
- stable behavior across different latency scales  

---

# Example Workflow

1. Cloud telemetry is collected or simulated.  
2. The adapter converts telemetry into JOR evidence inputs.  
3. The fusion engine recursively updates SOP and NHP.  
4. Alert state is determined through hysteresis thresholds.  
5. Results are logged for analysis.  

---

# Research Context

This project demonstrates that the JOR recursive fusion architecture can be adapted beyond its original domain by separating:

- domain-specific evidence extraction  
- general Bayesian-style fusion logic  
- state tracking  
- decision thresholds  

The same architecture pattern can be applied to different operational domains by replacing the telemetry adapter while preserving the fusion engine.

---

# Disclaimer

This repository contains a research prototype.

It is intended for experimentation, architecture demonstration, and evaluation.

Production cloud monitoring systems require additional engineering considerations including:

- real telemetry integration  
- operational alert policies  
- distributed state management  
- security controls  
- deployment validation
