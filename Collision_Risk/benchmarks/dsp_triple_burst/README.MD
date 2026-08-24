# JOR V3.1 DSP-Driven Triple-Burst Accumulation Test

## Overview

This benchmark evaluates recursive evidence accumulation in the JOR V3.1 Bayesian fusion framework using a synthetic DSP sensor stream.

The test combines:

1. A corrected DSP-to-rubric sensor mapping pipeline.
2. A validated triple-burst anomaly schedule.

The objective is to determine how repeated elevated sensor returns influence the recursive Bayesian posterior when evidence arrives in separate bursts rather than as one continuous event.

---

## Purpose

The simulation demonstrates:

DSP Sensor Features
|
v
JOR C/E/P Evidence Mapping
|
v
Bayesian Fusion Update
|
v
Recursive Posterior Accumulation


Unlike isolated detection tests, this benchmark evaluates temporal evidence persistence and whether repeated observations accumulate in a controlled manner.

---

## DSP-to-JOR Mapping Corrections

### Witness Credibility (C)

C is not derived from sensor quality.

Sensor characteristics cannot determine witness reliability.

The model uses: ASSUMED_WITNESS_C = 0.65

representing a moderate witness baseline.

Sensor-only deployments can disable this assumption and use the sensor credibility floor.

---

### Physical Evidence (P)

Sensor-derived metrics are correctly assigned to P.

Included features:

- Detection probability
- Radar cross-section estimate
- Peak power
- Track consistency
- Maneuver characteristics
- Multi-path behavior
- Doppler spread

This prevents sensor quality from incorrectly inflating witness credibility.

---

## Triple-Burst Schedule

The anomaly event consists of three separate bursts:

Burst Length:
8 steps

Step Duration:
5 seconds

Burst Duration:
40 seconds

Gap Between Bursts:
3 steps (15 seconds)

Schedule:

Burst 1
30s - 65s

Burst 2
85s - 120s

Burst 3
140s - 175s


The gaps are intentionally placed inside the recursive decay window so each burst begins while the previous posterior remains elevated.

---

## Burst Effects

During burst periods:

Detection probability is increased:

Baseline:
0.75

Burst:
0.90

Radar cross-section estimate is increased:

Baseline:
3.0

Burst:
5.0

Flight Characteristics Modifier:

Major Anomaly:
0.05


These changes simulate stronger and more anomalous physical returns passing through the DSP pipeline.

---

## JOR V3.1 Bayesian Configuration

Weights:

Witness Credibility (C): 0.40
Environmental (E): 0.30
Physical Evidence (P): 0.30

Calibration:

K = 0.20

Initial Prior:

P(NH) = 0.20
P(H) = 0.80

Recursive retention:

PRIOR_RETENTION = 0.70


Each posterior update becomes the next timestep's prior with controlled retention toward the original baseline.

---

## Validation Goals

The benchmark evaluates:

### 1. Conventional Baseline Stability

Non-burst periods should remain near the established JOR conventional baseline:

Posterior NH:
0.25 - 0.35


---

### 2. Evidence Accumulation

The test measures:

Burst 2 Peak - Burst 1 Peak

Burst 3 Peak - Burst 2 Peak


Expected behavior:

- Burst 1 establishes elevated evidence.
- Burst 2 increases the posterior.
- Burst 3 shows bounded accumulation.

The system should demonstrate accumulation without runaway probability inflation. The validation sweep confirmed positive Burst1→Burst2 gain and a plateauing Burst2→Burst3 response. :contentReference[oaicite:1]{index=1}

---

## Visualization Output

Generated file:
jor_dsp_triple_burst.gif

*For a live, frame-by-frame variant with a real-time engine-timing readout, see [Triple_Burst_Microsecond_Fusion.py](#live-variant-microsecond-fusion-readout), which exports `triple_burst_microsecond_fusion.mp4`.*

The animation contains:

### Left Panel

Recursive state trajectory:
SOP vs Posterior NH


### Right Panel

Posterior NH response over time:

- Recursive posterior track
- Rolling average
- Conventional baseline reference
- Noise floor band
- Highlighted anomaly burst windows

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

Execute: python jor_dsp_triple_burst.py

Expected outputs:

Baseline posterior check

Burst accumulation summary

jor_dsp_triple_burst.gif

---

## Live Variant: Microsecond Fusion Readout

`Triple_Burst_Microsecond_Fusion.py` is a companion script built on the same
validated triple-burst DSP pipeline above, restructured to run live rather
than precomputed:

- Each frame computes its Bayesian fusion step in real time during
  animation, one step per frame, instead of being batch-precomputed
  beforehand.
- A third panel is added below the existing two: a terminal-style scrolling
  readout showing genuine per-step engine compute time and throughput,
  measured with `time.perf_counter()` around the actual numba-JIT fusion
  call.
- All three panels render into a single figure and export to one `.mp4`
  (`triple_burst_microsecond_fusion.mp4`), suitable for direct upload.

Summary

This benchmark demonstrates that JOR V3.1 can process repeated DSP-derived anomaly evidence through a corrected C/E/P evidence model while maintaining recursive Bayesian stability.

The test verifies:

Correct evidence category separation
Sensor-driven physical evidence accumulation
Recursive prior retention behavior
Controlled multi-event escalation
Prevention of runaway probability growth

The result is a foundation for future multi-sensor and live-stream ingestion experiments within the JOR V3.1 framework.
