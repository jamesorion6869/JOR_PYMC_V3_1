# JOR V3.1-enriched — Synthetic Baseline & Robustness Benchmark

## Overview

This project is part of the jor-bayesian-fusion repository. It provides a full-scale synthetic sensor simulation and evaluation environment for the JOR V3.1 Bayesian Fusion Framework.

The benchmark combines an enriched DSP-style sensor-processing layer with the JOR C/E/P evidence model and evaluates the resulting Bayesian posterior across 2,500 independently seeded scenarios.

It is designed to exercise:

- Sensor-feature processing and smoothing
- Missing and degraded sensor data
- Changing sensor and environmental conditions
- Multiple event phenotypes
- False-event conditions
- Temporal detection behavior
- Threshold selection under a false-positive budget
- Detection latency
- Generalization across independent TRAIN, VAL, and TEST partitions
- Reproducibility through deterministic seed partitioning

> **Scope.** The implementation is intentionally synthetic. It is a **methodological and robustness benchmark** of the fusion pipeline. It makes no claim about the nature of any real-world UAP event and is not validation of JOR on real sensor data. See [Limitations and Scope](#limitations-and-scope).

---

## Benchmark Design

The benchmark uses three isolated random-seed partitions:

| Partition | Scenarios | Seed Range | Purpose |
|---|---:|---:|---|
| TRAIN | 1,000 | 0–999 | Threshold optimization |
| VAL | 500 | 10,000–10,499 | Independent validation |
| TEST | 1,000 | 20,000–20,999 | Final held-out evaluation |

**Total:** 2,500 scenarios.

Before evaluation, the code asserts that each partition has no duplicate seeds, that the three partitions do not overlap, and that the total unique seed count equals the sum of the partition sizes.

Each scenario is generated from its seed via `numpy.random.SeedSequence`, which spawns separate random streams for true events, false events, operating condition, sensor values, and dropout. A given seed therefore always produces the same scenario.

The detection threshold is optimized on TRAIN only. It is then frozen and evaluated on VAL and TEST. All other fusion parameters are fixed defaults and are **not** tuned.

---

## Simulation Structure

Each scenario is a 60-step synthetic sensor track. At each step the sensor operating condition follows a Markov chain:

| Condition | SNR shift (dB) | Variance mult. | Dropout prob. | Det. prob. shift |
|---|---:|---:|---:|---:|
| `NOMINAL` | 0.0 | 0.55 | 0.008 | 0.00 |
| `DEGRADED` | −2.5 | 1.50 | 0.07 | −0.06 |
| `SEVERELY_DEGRADED` | −7.0 | 3.00 | 0.22 | −0.20 |
| `RECOVERING` | −1.0 | 0.85 | 0.02 | −0.02 |

Transitions: `NOMINAL` → `NOMINAL` 0.92 / `DEGRADED` 0.08. `DEGRADED` → itself 0.55 / `SEVERELY_DEGRADED` 0.15 / `RECOVERING` 0.30. `SEVERELY_DEGRADED` → itself 0.55 / `RECOVERING` 0.45. `RECOVERING` → itself 0.35 / `NOMINAL` 0.65. Every scenario starts in `NOMINAL`.

---

## Synthetic Sensor Model

Each step generates SNR, radar cross section (RCS), peak power, Doppler, range resolution, detection probability, and a multipath term.

- SNR, RCS, peak power, Doppler, and range resolution each include an autoregressive (AR(1)) component, so observations are temporally correlated rather than independent draws.
- RCS includes a Swerling-style stochastic component (a mixture of exponential and Gaussian-perturbed draws around a mean).
- Peak power is derived from RCS and SNR.
- Six channels (SNR, RCS, power, Doppler, range resolution, detection probability) can independently drop out at each step. Dropout probability depends on the operating condition and, for some phenotypes, on the event.

---

## Event Phenotypes

Each scenario contains 0, 1, or 2 true events, each lasting 5–11 steps. True events are drawn from five synthetic phenotypes:

| Phenotype | Behavior |
|---|---|
| `A_STRONG_MULTI` | Sustained boost to RCS, SNR, detection probability, and track consistency |
| `C_INTERMITTENT` | The same style of boost, switching on and off during the event |
| `G_GRADUAL_ONSET` | Boost ramps up over roughly the first 60% of the event |
| `E_CONFLICTING` | RCS rises while SNR and detection probability fall |
| `F_SPARSE` | Weak boost combined with heavy extra dropout |

Each scenario also contains 0, 1, or 2 **false events**, each lasting 2–5 steps:

| False event | Behavior |
|---|---|
| `CLUTTER` | RCS and SNR rise, track consistency drops slightly |
| `MULTIPATH` | Multipath term rises, track consistency drops, Doppler shifts |
| `TRANSIENT_SPIKE` | Short, strong RCS/SNR/Doppler spike |
| `TRACK_INSTABILITY` | Track consistency drops sharply |

False events are independent of true events and can occur in scenarios with or without a true event.

---

## Enriched DSP Processing

The `EnrichedDSP` layer conditions the raw features before they are mapped into the JOR evidence structure. It maintains:

- Rolling buffers (window of 7) for SNR, power, Doppler, and range resolution
- A smoothed RCS estimate with innovation tracking, innovation-variance estimation, and innovation gating
- Short histories (length 5) of SNR, innovation, RCS, and detection probability
- Sensor-validity tracking (fraction of channels available)
- SNR-trend and innovation-trend estimates
- A cross-feature consistency score (agreement in direction of change among RCS, SNR, and detection probability)
- Track-consistency and maneuver-index estimates

---

## JOR C/E/P Evidence Mapping

Processed features are mapped to three evidence components, each clipped to a fixed range. When a feature is unavailable at a step (dropout), the weighted average is taken over the remaining features. If nothing is available, the previous step's value is carried forward.

### C — Witness / Credibility

This is a sensor benchmark, so there is no witness. C is an assumed baseline:

> C = 0.65 + N(0, 0.02), clipped to [0.30, 0.85]

C is the same distribution whether or not an event is present, so it carries no discriminating information in this benchmark (see Limitations).

### E — Environmental / Sensor Evidence

Weighted average, clipped to [0.30, 0.85]:

| Feature | Normalization | Weight |
|---|---|---:|
| SNR | 10–30 dB | 0.30 |
| Peak power | 800–2000 | 0.20 |
| RCS (smoothed) | 0.5–10 | 0.20 |
| Range resolution | 10–25, inverted | 0.10 |
| SNR trend | rescaled from [−1, 1] to [0, 1] | 0.10 |
| Valid-channel fraction | none | 0.10 |

### P — Physical / Behavioral Evidence

Weighted average, clipped to [0.30, 0.95]:

| Feature | Normalization | Weight |
|---|---|---:|
| Detection probability | none | 0.18 |
| Track consistency | none | 0.15 |
| RCS (smoothed) | 0.5–10 | 0.12 |
| Peak power | 800–2000 | 0.12 |
| Cross-feature consistency | none | 0.12 |
| Maneuver index | inverted | 0.08 |
| Multipath | inverted | 0.08 |
| Doppler | 5–25, inverted | 0.05 |
| Innovation magnitude | inverted | 0.05 |
| Valid-channel fraction | none | 0.05 |

---

## Bayesian Fusion

At each step:

```
SOP = W_C·C + W_E·E + W_P·P
NHP = W_C·C + W_E·E + W_P·min(max(P + modifier, 0), 0.95)
P(E | NH) = NHP
P(E | H)  = clip(1 − NHP + K·SOP, 0, 1)
posterior = P(E|NH)·prior_NH / (P(E|NH)·prior_NH + P(E|H)·prior_H)
```

Default parameters: `W_C = 0.40`, `W_E = 0.30`, `W_P = 0.30`, `K = 0.20`. The initial prior is `P(NH) = 0.20`, `P(H) = 0.80`. The posterior is exponentially smoothed (`POSTERIOR_ALPHA = 0.30`), and the prior for the next step is a blend of the current posterior and the initial prior (`PRIOR_RETENTION = 0.70`).

The per-step `modifier` (the paper's Flight Characteristics Modifier) is zero throughout this benchmark, so NHP equals SOP. The code path exists but is not exercised.

---

## Detection Rule and Threshold Selection

A scenario is flagged when the smoothed posterior meets either of two conditions:

1. **Level branch:** the posterior stays above the threshold for 3 consecutive steps.
2. **Rise branch:** the posterior rises by at least 0.04 over a 4-step window while staying above (threshold − 0.06) for 3 consecutive steps.

The first step at which either condition is met ends the search. The reported flag step is the **start** of the sustained 3-step window.

The threshold is swept from 0.28 to 0.61 in steps of 0.01 on TRAIN. The selected threshold is the one with the highest recall subject to FPR ≤ 0.12 (ties broken by F1). If no threshold meets the budget, the lowest-FPR threshold is used.

---

## Scoring

Scoring is at the scenario level:

| Outcome | Condition |
|---|---|
| TP | Scenario has a true event, a flag is raised, and the flag lies within 3 steps (`LAG_ALLOWANCE`) of some true-event interval |
| FN | Scenario has a true event and either no flag or a flag that matches no event |
| FP | Scenario has no true event and a flag is raised |
| TN | Scenario has no true event and no flag |

**Latency** is `max(0, flag_step − matched_event_onset)` for true positives, reported as a mean over TPs.

Reported metrics: recall (TPR), FPR, precision, F1, and mean latency.

---

## Results

Frozen threshold selected on TRAIN: **0.47** (FPR budget 0.12)

| Partition | Scenarios (with / without true event) | Recall (TPR) | FPR | F1 | Mean latency (steps) |
|---|---:|---:|---:|---:|---:|
| TRAIN | 636 / 364 | 0.434 | 0.099 | 0.582 | 5.27 |
| VAL | 325 / 175 | 0.406 | 0.109 | 0.555 | 5.18 |
| TEST | 646 / 354 | 0.435 | 0.082 | 0.588 | 5.10 |

Environment: Python 3.11.14, numpy 2.3.5, matplotlib 3.10.8 (Windows). Because the benchmark is fully seeded, the numbers should reproduce exactly on the same library versions.

**Reading these numbers.**

- Recall is modest. At the FPR budget of 0.12, the selected threshold detects a little over 40% of scenarios that contain a true event. Under the scoring rules above, a scenario is also lost if the first flag fires on a false event or outside the 3-step lag allowance.
- VAL and TEST agree with TRAIN within sampling noise. For scale, the standard error on TEST recall is about 0.02 (646 positive scenarios), and on TEST FPR about 0.015 (354 negative scenarios).
- Mean latency is computed over true positives only, so it is biased toward early detections. Events last 5–11 steps, so a mean of about 5 steps means detections typically arrive well into the event.

---

## Limitations and Scope

This benchmark is synthetic and should be read as a test of the fusion pipeline's mechanics, not as evidence about real sensor data or real UAP reports.

**Generator and detector are coupled.** The scenario generator injects each event phenotype by shifting RCS, SNR, detection probability, and track consistency. The E and P evidence terms are then computed from those same quantities (SNR, RCS, peak power, which is derived from RCS and SNR, detection probability, and track consistency). Recall on TEST therefore measures how well the pipeline recovers signals the simulator planted, which is an optimistic ceiling rather than an estimate of performance on real data. It does not show that the pipeline would find real events in real data. The independent TRAIN/VAL/TEST partitions protect against threshold overfitting. They do not protect against this coupling, because all three partitions come from the same generative model.

**The credibility term carries no information.** C is fixed at 0.65 plus small Gaussian noise in every scenario, regardless of whether an event is present. With `W_C = 0.40`, a large share of the SOP/NHP weight sits on a term that cannot discriminate between events and non-events. Reported performance comes entirely from E and P.

**Fusion parameters are not tuned.** `W_C`, `W_E`, `W_P`, `K`, prior retention, and posterior smoothing use fixed defaults. Only the detection threshold is optimized, on TRAIN, under a false-positive budget of 0.12. Results are specific to these defaults.

**Scoring is scenario-level and first-flag only.** A scenario counts as a true positive if the first flag falls within 3 steps of any true event. Later flags in the same scenario are not scored, and a first flag that falls outside every event window counts as a miss even if the detector would later have fired correctly.

**False events are simple.** The four false-event categories are parametric perturbations of the same features and do not capture the diversity of real confounders.

**No baseline.** No comparison detector (for example, a plain SNR threshold) is run on the same scenarios, so the absolute numbers have no reference point.

---

## Reproducing the Results

```bash
pip install -r requirements.txt
python jor_dsp_integration_demo.py
```

The script generates all 2,500 scenarios, runs the partition integrity checks and dataset summary, selects the threshold on TRAIN, evaluates VAL and TEST, and writes example track figures and an animated MP4. The MP4 step requires an `ffmpeg` binary.

---

## License

[MIT](LICENSE.txt) — Copyright (c) 2026 Jake James
