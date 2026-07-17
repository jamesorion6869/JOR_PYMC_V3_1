# JOR V4.0 Predictive Maintenance Demonstration

## Overview

This project demonstrates an application of the JOR (James Orion Report) Bayesian Evidence Fusion architecture to a predictive maintenance scenario.

The objective is to demonstrate how the JOR evidence-fusion approach can be adapted from its original evidence-triage applications into an industrial monitoring domain by separating:

- Domain-specific sensor processing
- Evidence extraction
- Bayesian state fusion
- Recursive state tracking
- Operational alerting
- Result logging and analysis

The implementation uses simulated vibration sensor data to demonstrate recursive health-state tracking, anomaly escalation, alert persistence, and recovery behavior.

This project represents a domain adaptation of the JOR evidence-fusion methodology. The sensor-processing layer is separated from the probabilistic reasoning layer, allowing different data sources to be translated into structured evidence inputs.

---

# Architecture

The Predictive Maintenance workflow follows a layered design:

```
Simulated Sensor Data
        |
        v
Vibration Adapter
(adapters.py)
        |
        v
JOR Evidence Representation
(theta_s, theta_c, theta_o)
        |
        v
Predictive Maintenance
Fusion Engine
(engine.py)
        |
        v
Health State Posterior
        |
        v
Fusion Logger
(logger.py)
        |
        v
Log Analysis
(analyze_logs.py)
```

The implementation preserves the JOR evidence-fusion architecture while separating domain-specific sensor processing from probabilistic state estimation.

The adapter layer translates vibration measurements into structured evidence inputs suitable for recursive fusion.

---

# Project Files

```
Predictive_Maintenance/

├── engine.py
│   Predictive maintenance fusion engine.
│   Implements recursive Bayesian-style
│   state tracking, retention, calibration,
│   and hysteresis behavior.
│
├── adapters.py
│   Domain translation layer.
│   Converts vibration signals into
│   structured evidence variables.
│
├── main.py
│   Application execution pipeline.
│   Runs calibration, healthy operation,
│   anomaly escalation, and recovery phases.
│
├── logger.py
│   Persistent JSON-line fusion state logger.
│
└── analyze_logs.py
    Utility for reviewing fusion output history.
```

---

# Evidence Mapping

The predictive maintenance system maps industrial measurements into a JOR-style evidence structure.

| JOR Variable | Predictive Maintenance Interpretation |
|---|---|
| `theta_s` | System or structural condition evidence |
| `theta_c` | Operational context evidence |
| `theta_o` | Sensor observation evidence |

The fusion engine does not directly process raw vibration waveforms.

Instead:

```
Raw Sensor Signal
        |
        v
Feature Extraction
        |
        v
Evidence Variables
        |
        v
Probabilistic State Fusion
```

This separation creates a reusable pattern where domain-specific measurements are translated into evidence inputs before fusion.

---

# Vibration Adapter

`adapters.py` provides the sensor-processing and feature-extraction layer.

The `VibrationAdapter` performs:

- Signal validation
- RMS vibration calculation
- FFT-based dominant frequency extraction
- Normalization into a 0-1 observation scale
- Operational context normalization

Example:

```
Raw vibration waveform
        |
        v
Signal Processing
        |
        v
Normalized Observation Evidence
(theta_o)
```

The normalized observation value represents vibration severity relative to the configured safe operating limit.

---

# Fusion Engine

`engine.py` implements the predictive maintenance adaptation of the JOR recursive evidence-fusion approach.

The engine combines:

```
System Evidence
+
Environmental / Context Evidence
+
Sensor Observation Evidence
        |
        v
Fused Health State Estimate
```

The system maintains state over time:

```
Posterior(t)
      |
      v
Retention / Recursive Update
      |
      v
Posterior(t+1)
```

This allows the model to track gradual changes in equipment condition rather than relying only on instantaneous threshold checks.

---

# Self Calibration

The demonstration begins with an adaptive calibration period.

## Phase 1: Calibration

The system observes healthy operating conditions:

```
20 healthy vibration samples
        |
        v
Baseline health state established
```

The learned baseline is then used for later anomaly evaluation.

---

# Simulation Phases

The demonstration runs four operating phases.

---

## Phase 1 — Calibration

Purpose:

- Establish normal operating baseline
- Initialize adaptive state tracking

Expected result:

```
Normal operation
No alert
```

---

## Phase 2 — Healthy Operation

Purpose:

- Verify stability after calibration

Expected result:

```
Stable posterior state
No alert
```

---

## Phase 3 — Escalating Vibration

Purpose:

- Simulate increasing mechanical stress

The vibration level is gradually increased:

```
Normal vibration
       |
       v
Increasing vibration
       |
       v
Potential fault condition
```

Expected result:

```
Health risk increases
Alert transition occurs
```

---

## Phase 4 — Recovery

Purpose:

- Verify recovery behavior

The vibration returns to healthy conditions.

Expected result:

```
Health risk decreases
Alert clears after hysteresis conditions are satisfied
```

---

# Alert Logic

The engine uses hysteresis-based alerting to reduce unnecessary state switching.

Example:

```
Health Risk > Upper Threshold
          |
          v
       ALERT ON


Health Risk < Lower Threshold
          |
          v
       ALERT OFF
```

The separation between activation and clearing thresholds prevents rapid alert oscillation around the decision boundary.

---

# Logging

`logger.py` records the evolution of the fusion state.

Logged information includes:

- Timestamp
- Fused SOP value
- Posterior health probability
- Alert state
- Simulation metadata

Example output:

```json
{
 "timestamp": "2026-01-01T12:00:00.000",
 "sop_fused": 0.62,
 "nhp_posterior": 0.71,
 "alert_active": true,
 "metadata": {
    "phase": "Escalating",
    "step": 42
 }
}
```

The log provides an audit trail for validating system behavior across operating phases.

---

# Log Analysis

`analyze_logs.py` provides a lightweight review utility.

It extracts:

- Timestamp
- Posterior health state
- Alert status

This supports validation of:

- Calibration stability
- Anomaly escalation
- Recovery behavior
- Alert transitions

---

# Running the Demonstration

From the Predictive Maintenance directory:

```bash
python main.py
```

The simulation will:

1. Initialize the fusion engine
2. Restore previous state if available
3. Perform calibration
4. Simulate healthy operation
5. Introduce escalating vibration conditions
6. Simulate recovery
7. Save final engine state

To inspect logged results:

```bash
python analyze_logs.py
```

---

# Design Philosophy

This project demonstrates a core JOR design principle:

> Evidence fusion should be separated from domain-specific measurement systems.

Domain-specific adapters translate available measurements into structured evidence variables.

The fusion layer operates on those evidence representations rather than requiring direct knowledge of the original sensor domain.

This architecture supports future adaptations where different data sources can provide evidence inputs while maintaining a consistent probabilistic reasoning approach.

---

# Current Status

**JOR V4.0 Predictive Maintenance Demonstration**

Current capabilities:

- Recursive health-state tracking
- Adaptive baseline calibration
- Evidence-based fusion
- Hysteresis-controlled alerting
- Persistent state storage
- Simulation validation workflow
- Log-based output analysis

Future extensions may include:

- Live sensor ingestion
- Additional sensor modalities
- Fleet-level health monitoring
- Historical failure dataset validation
- Extraction of shared fusion components into a reusable core library
