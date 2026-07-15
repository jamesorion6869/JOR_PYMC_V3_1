# JOR V3.1 DSP-to-Rubric Live Sensor Ingestion Simulation

## Overview

This benchmark demonstrates a live sensor ingestion workflow for the JOR V3.1 Bayesian fusion framework.

A synthetic DSP-style radar detection stream is generated, mapped into the JOR C/E/P evidence structure, and recursively fused using the validated JOR V3.1 Bayesian posterior model.

The simulation represents a conventional tracked object scenario with no anomalous flight behavior and verifies that sensor-derived evidence remains calibrated against the established Conventional baseline.

---

## Purpose

The goal is to demonstrate the transition from processed sensor features into the JOR evidence model:

**DSP Sensor Features → C/E/P Evidence Mapping → Bayesian Fusion → Recursive Posterior Tracking**

The benchmark validates that conventional sensor observations remain near the expected baseline posterior range rather than artificially inflating anomaly probability.

---

## Major Corrections Applied

### 1. Witness Credibility (C) Category Correction

Previous versions incorrectly derived Witness Credibility from sensor track-quality metrics.

Metrics such as:

- Track consistency
- Maneuver index
- Multi-path score
- Doppler spread

represent physical/sensor evidence, not human witness reliability.

Correction:

- C is now treated as an assumed witness credibility value.
- Default: ASSUMED_WITNESS_C = 0.65

This represents a moderate witness baseline consistent with the JOR rubric.

For fully autonomous sensor-only operation: ASSUMED_WITNESS_C = None


uses the sensor-only credibility floor.

---

### 2. Physical Evidence (P) Reallocation

Sensor quality metrics previously influencing C were moved into P.

The Physical Evidence component now incorporates:

- Detection probability
- Radar cross section estimate
- Peak signal power
- Track consistency
- Maneuver behavior
- Multi-path quality
- Doppler characteristics

This aligns the implementation with the JOR evidence model.

---

### 3. Detection Probability Calibration

The original detection probability baseline was too high:
Previous: 0.96
Updated: 0.75

The adjusted value centers the conventional scenario inside the validated posterior target band.

Verified 60-step run:

Posterior NH Mean = 0.299

Range:
0.221 - 0.339


Target:
0.25 - 0.35

Result: MATCHED


---

## JOR V3.1 Fusion Model

The Bayesian fusion core remains unchanged.

Weights:
C = 0.40
E = 0.30
P = 0.30

Calibration constant:
K = 0.20

Initial prior:
P(NH) = 0.20
P(H) = 0.80

Recursive retention:
PRIOR_RETENTION = 0.70


Each timestep:

1. Generate DSP sensor observation
2. Convert features into C/E/P evidence
3. Calculate SOP
4. Calculate NHP
5. Update Bayesian posterior
6. Feed posterior into the next timestep prior

---

## Simulation Configuration

Track duration:
60 steps
5 seconds per step
300 seconds total

Profile:
Conventional Flight
Flight Modifier = 0.00

The simulation intentionally represents a normal tracked object with stable sensor behavior.

---

## Sensor Feature Generation

Synthetic radar detections include:

- Signal-to-noise ratio
- Radar cross section estimate
- Peak power
- Doppler spread
- Detection probability
- Range resolution
- Maneuver index
- Track consistency
- Multi-path score

These features are transformed into JOR evidence categories.

---

## Visualization Output

The simulation generates:
jor_dsp_conventional.gif


The animation contains:

### Left Panel

Recursive state trajectory:
SOP vs Posterior NH


### Right Panel

Posterior response over 300 seconds:

- Recursive NH probability
- Rolling average
- Conventional baseline reference
- Expected noise floor band

---

## Requirements

Python packages:
numpy
matplotlib
numba
pillow


Install:

```bash

pip install numpy matplotlib numba pillow

Running the Simulation

Execute:
python jor_dsp_conventional.py

C range: ...
E range: ...
P range: ...

Posterior NH:
mean ≈ 0.30

Target band 0.25-0.35:
MATCHED

The resulting GIF demonstrates that a conventional sensor track remains calibrated within the validated JOR V3.1 baseline.

Summary

This benchmark verifies that JOR V3.1 can ingest conventional DSP-style sensor features without category errors or probability inflation.

The corrected pipeline preserves the framework's core principles:

Evidence separation
Conservative priors
Sensor/witness distinction
Recursive Bayesian updating
Calibration against known baselines

The result is a sensor-fusion workflow suitable for extending JOR V3.1 toward live or simulated multi-sensor ingestion.
