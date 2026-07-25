# JOR V4.0 Predictive Maintenance Demonstration

## Related Project

This demonstration is part of the broader JOR (James Orion Report) Bayesian Evidence Fusion Framework. The Predictive Maintenance demonstration is maintained as an application example within the broader JOR framework repository.

Main Repository:

https://github.com/jamesorion6869/JOR_PYMC_V3_1

Cite All Versions:

https://doi.org/10.5281/zenodo.18088931

## Overview

This project demonstrates an application of the JOR (James Orion Report) Bayesian Evidence Fusion Framework architecture to a predictive maintenance scenario.

The objective is to demonstrate how the JOR evidence-fusion approach can be adapted from its original evidence-triage applications into an industrial monitoring domain by separating:

- Domain-specific sensor processing
- Evidence extraction
- Bayesian state fusion
- Recursive state tracking
- Operational alerting
- Result logging and analysis

The implementation uses simulated vibration sensor data to demonstrate recursive health-state tracking, anomaly escalation, alert persistence, and recovery behavior.

Two of the three evidence inputs are grounded in real, cited industry standards rather than illustrative constants (see **Evidence Mapping** below) — this distinguishes the current version from earlier iterations of this demonstration, which used arbitrary normalization constants throughout.

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
│   structured evidence variables, grounded
│   in ISO 20816-3 (vibration) and
│   NEMA MG-1 (thermal/load).
│
├── main.py
│   Application execution pipeline.
│   Runs calibration, healthy operation,
│   anomaly escalation, and recovery phases.
│
├── logger.py
│   Persistent JSON-line fusion state logger.
│
├── analyze_logs.py
│   Utility for reviewing fusion output history,
│   including ISO zone and Criterion II detail.
│
└── test_jor_predictive_maintenance.py
    Automated test suite (47 tests) covering
    zone boundaries, Criterion II behavior,
    NEMA load/ambient severity, state
    persistence, and full integration arcs.
```

---

# Evidence Mapping

The predictive maintenance system maps industrial measurements into a JOR-style evidence structure.

| JOR Variable | Predictive Maintenance Interpretation | Grounding |
|---|---|---|
| `theta_s` | System or structural condition evidence | Fixed placeholder (see note below) |
| `theta_c` | Operational context evidence (machine load, ambient temperature) | NEMA MG-1, Class F insulation |
| `theta_o` | Sensor observation evidence (vibration) | ISO 20816-3, Group 2 |

**`theta_s` is currently a fixed placeholder (0.72)**, not a validated measurement. It's disclosed this way intentionally: this evidence channel needs a real structural-health input (bearing condition monitoring, motor current signature analysis, oil analysis, or similar) to be meaningful, and no synthetic proxy for that would be honest to call validated. `theta_c` and `theta_o` are grounded in cited standards precisely because doing so didn't require live sensor data — see **Vibration Adapter** below for what that grounding actually means.

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

# JOR V4.0 Evidence Metrics

JOR V4.0 uses a Bayesian evidence-fusion architecture where multiple evidence inputs are combined into fused evidence scores and posterior state estimates.

The fusion engine processes three primary evidence inputs:

| Variable | Description |
|---|---|
| `theta_s` | System or structural condition evidence |
| `theta_c` | Operational context evidence |
| `theta_o` | Sensor observation evidence |

The fusion process produces two primary outputs:

## SOP — Signal Object Processing

SOP is the fused evidence score computed from:

- `theta_s` — system evidence
- `theta_c` — operational context evidence
- `theta_o` — observed sensor evidence

SOP represents the overall strength of the detected operational condition based on the available evidence inputs.

SOP is an evidence score and should not be interpreted as a direct probability.

**A note on the fusion weights themselves:** the relative weighting of theta_s/theta_c/theta_o (currently 0.40/0.30/0.30) is a JOR architectural choice, not a value derived from an industry-standard formula. Real condition-monitoring practice (e.g. ISO 17359) generally does not blend multiple signal types into a single weighted composite the way JOR does — it favors failure-mode-specific technique selection or independent per-channel alarm thresholds (as in API 670) instead. JOR's single fused posterior is a deliberate, different architectural pattern, not an attempt to replicate that practice; the weights are exposed as constructor parameters specifically so they can be revisited if a citable weighting scheme for this kind of multi-signal fusion is ever established.

## NHP — Non-Healthy Probability

NHP is the posterior probability produced by the Bayesian fusion engine that the monitored equipment is in a non-healthy operating state.

Higher NHP values indicate increased probability of:

- equipment degradation
- abnormal operation
- potential fault conditions

NHP is not a direct sensor measurement. It is the result of recursive Bayesian evidence fusion using the configured prior, evidence inputs, retention behavior, and calibration parameters.

Because the engine tracks state recursively, NHP represents an evolving estimate rather than an instantaneous classification. A temporary increase in evidence may therefore persist briefly after the initiating condition has improved, depending on retention settings and recovery behavior. Alert activation and clearing behavior are controlled separately through the hysteresis logic described in the Alert Logic section.

---

# Vibration Adapter

`adapters.py` provides the sensor-processing and feature-extraction layer. Two independent evidence channels are computed here, each grounded in a different, real, cited standard.

## theta_o — ISO 20816-3, Group 2

`theta_o` (vibration severity) is derived from ISO 20816-3's evaluation criteria for a Group 2 machine (medium power, rigid foundation — e.g. a standard industrial motor or pump bolted to a concrete pad). ISO 20816-1 requires both of the following be assessed together:

- **Criterion I — absolute zone boundaries.** RMS velocity (mm/s) is classified into Zone A (new/reconditioned) through Zone D (immediate action needed), using the commonly cited Group 2 boundaries (1.4 / 2.8 / 7.1 mm/s). This is mapped onto a continuous 0-1 severity that preserves each zone's relative jump rather than a single straight line.
- **Criterion II — rate of change from the machine's own baseline.** A change exceeding 25% of the zone B/C boundary is flagged as significant *even if the absolute reading stays in the same zone* — this is the standard's own example of a developing fault that Criterion I alone would miss. A flagged change adds a smooth severity boost proportional to how far past the threshold it is.

The adapter converts the simulated acceleration signal (g's, peak amplitude) to RMS velocity via `v = a / omega`, since ISO 20816-3 zones are defined in velocity, not acceleration. This conversion is exact only for a single dominant frequency component — a reasonable approximation for this synthetic single-tone demo signal, but a real deployment should measure velocity directly or properly integrate a real broadband waveform.

Both the Criterion I zone boundaries and the Criterion II baseline persist across process restarts (see **State Persistence** below).

## theta_c — NEMA MG-1, Class F Insulation

`theta_c` (operational/thermal stress) is derived from NEMA MG-1's published thermal and load limits for a general-purpose Class F insulation motor (the overwhelmingly common insulation class on modern three-phase AC motors):

- **Load severity** is keyed to the motor's nameplate service factor (a common general-purpose value is 1.15): normal operation up to 100% of rated load, a derated-life operating envelope up to 115%, and genuine overload beyond that.
- **Ambient severity** is keyed to NEMA's reference ambient (40°C) and its published 105°C allowable temperature rise for Class F insulation at 1.0 SF, applying NEMA's own rule that allowable rise shrinks 1:1 for every degree ambient exceeds the reference.

Load and ambient severity are combined using the same worst-dominates blend used elsewhere in the JOR adapter family: the more severe of the two drives most of the result, with a smaller contribution from their average.

**Simplification note:** both `theta_o` and `theta_c` are severity proxies grounded in real published limits, not full physical models. Real winding temperature rise scales roughly with load squared, for example, which this adapter does not model directly. Where this demo simplifies, it does so from a cited real-world reference point rather than an arbitrary constant.

```
Raw vibration waveform
        |
        v
Signal Processing (ISO 20816-3 Criterion I + II)
        |
        v
Normalized Observation Evidence
(theta_o)
```

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

Both the engine's own `baseline_sop` and the adapter's Criterion II `baseline_velocity_mm_s` establish over this same 20-step window, so both halves of "what's normal for this machine" are learned together.

---

# State Persistence

Both baselines — the engine's `baseline_sop` and the adapter's Criterion II `baseline_velocity_mm_s` — are saved to and restored from the same state file (`engine_state.json`) across process restarts. They are always restored together, or not at all, so they cannot silently drift out of sync with each other. Older state files saved before Criterion II existed simply cause the adapter to re-establish its baseline fresh, without error.

The adapter's baseline can also be manually cleared via `reset_baseline()` — for example, after real maintenance or a machine rebuild, when comparing new readings against the pre-maintenance baseline would no longer make sense.

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
Criterion II typically flags the developing trend a few steps
before the absolute zone crosses into ALERT territory
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
- Simulation metadata, including ISO zone, RMS velocity, and Criterion II detail

Example output:

```json
{
 "timestamp": "2026-01-01T12:00:00.000",
 "sop_fused": 0.62,
 "nhp_posterior": 0.71,
 "alert_active": true,
 "metadata": {
    "phase": "Escalating",
    "step": 42,
    "iso_zone": "C",
    "velocity_rms_mm_s": 3.68,
    "baseline_velocity_mm_s": 0.74,
    "criterion_ii_delta_mm_s": 2.94,
    "criterion_ii_threshold_mm_s": 0.7,
    "criterion_ii_flag": true
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
- Phase and step (when present)
- ISO zone and RMS velocity (when present)
- Criterion II flag (when present)

Fields are read defensively, so this script runs correctly against log files written before ISO 20816-3 grounding existed — those older lines simply print without the extra detail.

This supports validation of:

- Calibration stability
- Anomaly escalation
- Recovery behavior
- Alert transitions
- Whether Criterion II is flagging developing trends ahead of Criterion I

---

# Testing

`test_jor_predictive_maintenance.py` is an automated pytest suite (47 tests) covering:

- ISO 20816-3 zone boundary correctness, monotonicity, and saturation
- Criterion II threshold math, the same-zone-jump case Criterion I alone would miss, and boost capping
- NEMA MG-1 load and ambient severity curves at their documented breakpoints
- Adapter robustness (empty/zero/NaN/Inf input signals)
- Engine calibration, hysteresis, and retention-blend stability
- State persistence round-tripping for both the engine and adapter baselines
- Full integration arcs (healthy stays Normal, escalation eventually alerts, recovery clears)

Run with:

```bash
pip install pytest
pytest test_jor_predictive_maintenance.py -v
```

---

# Running the Demonstration

From the Predictive Maintenance directory:

```bash
python main.py
```

The simulation will:

1. Initialize the fusion engine and adapter
2. Restore previous state if available (both engine and adapter baselines together)
3. Perform calibration
4. Simulate healthy operation
5. Introduce escalating vibration conditions
6. Simulate recovery
7. Save final state

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

Because each evidence channel is computed independently before fusion, individual channels can be grounded in real standards (as theta_o and theta_c now are) without requiring changes to the fusion engine itself, and the fusion weights themselves remain simple constructor parameters that could be revisited if a citable multi-signal weighting scheme is ever established for this domain.

---

# What This Is, and Isn't

This is a working demonstration of Bayesian sequential fusion applied to a vibration/thermal condition-monitoring problem. Calibration, Criterion I and II evaluation, retention-based smoothing, hysteresis alerting, and cross-restart state persistence are all real, functioning components — not simplified stand-ins.

What it is *not* is a validated operational deployment:

- `theta_s` remains an unvalidated fixed placeholder pending a real structural-health sensor input.
- The acceleration-to-velocity conversion is exact only for a single dominant frequency; real machinery vibration is broadband.
- The demo signal is entirely synthetic (a sine wave plus noise), not real sensor data.
- The fusion weights (0.40/0.30/0.30) are a JOR architectural choice, not a value derived from an industry-standard formula — real condition-monitoring practice generally favors independent per-signal thresholds over a single blended composite.
- No automated fault classification (e.g. bearing defect frequency analysis) or SCADA/historian integration exists; both would be natural, separate next projects built on top of this evidence layer, not gaps in this one.

---

# Current Status

**JOR V4.0 Predictive Maintenance Demonstration**

Current capabilities:

- Recursive health-state tracking
- Adaptive baseline calibration (engine and adapter, together)
- theta_o grounded in ISO 20816-3 Group 2 (Criterion I zone boundaries + Criterion II rate-of-change)
- theta_c grounded in NEMA MG-1 Class F thermal/load limits
- Evidence-based fusion
- Hysteresis-controlled alerting
- Persistent state storage across restarts, for both baselines together
- Simulation validation workflow
- Log-based output analysis, including ISO zone and Criterion II detail
- 47-test automated regression suite

Future extensions may include:

- Live sensor ingestion
- Additional sensor modalities
- Fleet-level health monitoring
- Historical failure dataset validation
- A citable, standards-derived weighting scheme for theta_s/theta_c/theta_o, if one is established
- Extraction of shared fusion components into a reusable core library

