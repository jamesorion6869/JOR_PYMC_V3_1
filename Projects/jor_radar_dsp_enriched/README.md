# JOR V3.1-enriched — Full-Scale Baseline & Robustness Benchmark

## Overview

This benchmark provides a full-scale synthetic sensor simulation and evaluation environment for the **JOR V3.1 Bayesian Fusion Framework**.

The benchmark integrates an enriched DSP-style sensor-processing layer with the JOR C/E/P evidence model and evaluates the resulting Bayesian posterior across a large set of independently seeded scenarios.

The purpose is to test:

- Sensor-feature processing and smoothing
- Missing and degraded sensor data
- Environmental and sensor-condition changes
- Multiple event phenotypes
- False-event conditions
- Temporal detection behavior
- Threshold selection
- False-positive control
- Detection latency
- Generalization across independent TRAIN, VALIDATION, and TEST partitions
- Reproducibility through deterministic seed partitioning

The implementation is intentionally synthetic. It is a **methodological and robustness benchmark**, not a claim about the nature of any real-world UAP event.

---

## Benchmark Design

The benchmark uses three completely isolated random-seed partitions:

| Partition | Scenarios | Seed Range | Purpose |
|---|---:|---:|---|
| TRAIN | 1,000 | 0–999 | Threshold optimization |
| VAL | 500 | 10,000–10,499 | Independent validation |
| TEST | 1,000 | 20,000–20,999 | Final held-out evaluation |

**Total:** 2,500 independently generated scenarios.

The code performs explicit integrity checks to verify that the three partitions contain unique, mutually exclusive seeds before evaluation begins.

The threshold is optimized using the TRAIN partition only. The resulting threshold is then frozen and evaluated independently on the VAL and TEST partitions.

---

## Simulation Structure

Each scenario contains a 60-step synthetic sensor track.

The simulation models changing operating conditions using a state-transition system containing:

- `NOMINAL`
- `DEGRADED`
- `SEVERELY_DEGRADED`
- `RECOVERING`

Each condition modifies factors such as:

- Signal-to-noise ratio
- Sensor variance
- Sensor dropout probability
- Detection probability

This allows the fusion pipeline to be evaluated under both nominal and degraded sensing conditions.

---

## Synthetic Sensor Model

The benchmark generates multiple sensor-derived quantities, including:

- SNR
- Radar cross section (RCS)
- Peak power
- Doppler
- Range resolution
- Detection probability
- Multipath effects

Autoregressive sensor processes are used to introduce temporal correlation rather than treating every observation as an independent random sample.

RCS generation also includes a Swerling-style stochastic component. 

---

## Event Phenotypes

True event scenarios can contain several synthetic event phenotypes:

- `A_STRONG_MULTI`
- `C_INTERMITTENT`
- `G_GRADUAL_ONSET`
- `E_CONFLICTING`
- `F_SPARSE`

These represent different temporal and sensor-signature behaviors rather than a single idealized event profile.

The benchmark also generates false-event conditions:

- `CLUTTER`
- `MULTIPATH`
- `TRANSIENT_SPIKE`
- `TRACK_INSTABILITY`

These are intentionally introduced to test false-positive behavior and robustness.

---

## Enriched DSP Processing

The `EnrichedDSP` processing layer performs temporal feature conditioning before the features are mapped into the JOR evidence structure.

The DSP layer includes:

- Rolling sensor buffers
- RCS smoothing
- Innovation tracking
- Innovation variance estimation
- Innovation gating
- SNR history
- RCS history
- Detection-probability history
- Sensor-validity tracking
- SNR trend estimation
- Innovation trend estimation
- Cross-feature consistency estimation
- Track-consistency estimation
- Maneuver-index estimation

The resulting processed features are converted into normalized evidence values for the JOR fusion engine.

---

# JOR C/E/P Evidence Mapping

The benchmark maps processed sensor information into the JOR evidence framework:

### C — Witness / Credibility

For this synthetic sensor benchmark, the credibility component is represented by an assumed baseline:

```text
C ≈ 0.65
