# JOR — Maritime/Acoustic Sonar Concept Demo

A concept demonstration extending the **JOR (James Orion Report) framework** — a Bayesian evidence-fusion architecture originally built for scoring UAP sighting reports — into a maritime active-sonar contact-detection setting.

**What this is:** a synthetic-data proof of concept showing that JOR's fusion/scoring architecture behaves sensibly on maritime-shaped inputs, with a properly held-out TRAIN/VAL/TEST evaluation.

**What this is not:** a validated maritime sensor model, and not a claim about real-world sonar detection performance. Every number in this README comes from synthetic scenarios with illustrative sensor constants. See [Limitations](#limitations) before drawing any conclusion beyond "the architecture works as intended on this synthetic data."

---

## Background

JOR fuses three evidence dimensions — **C** (witness/operator confidence), **E** (environmental/signal evidence), **P** (physical/track evidence) — into a weighted score, then runs that through a Bayesian posterior update to produce a confidence-over-time track for a candidate contact. The original framework scores UAP sighting reports from human witness testimony; this demo retargets the same fusion math to an active-sonar setting, where C becomes a sensor/operator-confidence signal (no human eyewitness involved) and E/P are built from simulated acoustic returns instead of testimony.

The core Bayesian fusion math is **unchanged** from the validated JOR architecture. What's new here is the sensor-simulation layer feeding it.

## What's in the maritime adaptation

- **Sonar-domain sensor model**, replacing the original radar-domain one: target strength (TS) in place of radar cross-section, received level in place of peak power, sea-state/ambient-noise conditions in place of atmospheric conditions, and maritime-specific false-alarm categories (marine biologics, surface/bottom acoustic bounce, transient noise, thermocline-driven track instability) in place of radar clutter categories.
- **An independent secondary evidence channel** (`kin_consistency`, representing multi-ping/kinematic consistency) with its own random-noise stream, decoupled from the sea-state-driven degradation that the other channels share. This exists specifically to test whether JOR's fusion layer can exploit genuinely independent evidence, since a naive single-channel detector structurally cannot.
- **Tiered alerting**: a high-confidence CONFIRM tier (auto-flag) and a lower-confidence REVIEW tier (routed to a human instead of auto-actioned), rather than forcing one threshold to serve both "obviously a contact" and "maybe, hard to tell."
- **Physically grounded ambient-noise modeling**: the SNR penalty from rougher sea states is derived from Urick's analytic approximation to the Knudsen/Wenz deep-water ambient-noise spectrum (`NL(f, sea_state) = 10·log10(f^-5/3) + 94.5 + 30·log10(sea_state+1)`), not an invented number.

## Results

All headline numbers below are from a single **frozen, held-out TEST evaluation** — every parameter (fusion weights, detection-logic settings, operating thresholds) was selected on TRAIN, confirmed on VAL, and frozen before TEST was touched. See [Methodology](#methodology) for why that distinction matters and how it was enforced.

| | TRAIN (300) | VAL (150) | TEST (300, held-out) |
|---|---|---|---|
| Recall (CONFIRM tier) | 0.503 | 0.528 | **0.466** |
| False Positive Rate | 0.097 | 0.114 | **0.081** |
| F1 | — | 0.671 | 0.615 |
| Mean detection latency | — | 4.12 steps | 3.82 steps |

**JOR fusion vs. a naive single-channel SNR threshold**, both selected the same way (TRAIN → VAL → frozen → TEST once):

| | Threshold | Recall | FPR |
|---|---|---|---|
| JOR fusion | 0.38 | **0.466** | 0.081 |
| Naive SNR threshold | 19.5 dB | 0.376 | 0.027 |
| **Recall delta (JOR − naive)** | | **+0.090** | |

### Tiered alerting

| | Recall | Notes |
|---|---|---|
| CONFIRM (auto-flag) | 0.466 | Observed |
| CONFIRM + REVIEW combined | 0.513 | **Upper bound** — assumes a human reviewer correctly confirms every true contact routed to review; not an observed automated figure |
| Review workload | 6.0% of all TEST scenarios | 9 real contacts confirm missed, 9 false alarms, out of 300 |
| Missed by both tiers | 92 scenarios (30.7%) | Concentrated in specific phenotypes — see below |

### Where it succeeds and where it doesn't

Recall by contact "phenotype" (synthetic signature shape) shows the shortfall is concentrated, not uniform:

| Phenotype | CONFIRM recall | Combined recall |
|---|---|---|
| A_STRONG_MULTI (strong, multi-feature) | 0.846 | 0.865 |
| G_GRADUAL_ONSET | 0.648 | 0.648 |
| C_INTERMITTENT | 0.510 | 0.627 |
| E_CONFLICTING (internally ambiguous) | 0.349 | 0.381 |
| F_SPARSE (weak signal) | 0.127 | 0.218 |

The system reliably catches clear, strong contacts and struggles most with deliberately ambiguous or weak ones — a believable failure *pattern*, even though the overall miss rate is an artifact of how this synthetic test set's difficulty mix was constructed (see [Limitations](#limitations)), not a real-world estimate.

## Methodology

This demo enforces a disjoint **TRAIN (seeds 0–299) → VAL (10,000–10,149) → TEST (20,000–20,299)** split, with independent random streams per scenario component (events, false alarms, environmental conditions, sensor noise, dropout, the independent evidence channel) so the partitions are genuinely separated, not just differently-seeded copies of the same process.

Every tuned component — fusion weights, detection-logic parameters, the CONFIRM threshold, the REVIEW threshold, and the naive baseline's own threshold — follows the same discipline: **search on TRAIN → confirm on VAL → freeze → evaluate once on TEST.** A tuned configuration is only adopted over its default if it both meets the FPR budget and beats the default's VAL performance; otherwise the default is kept.

This script went through **three rounds of external methodology review**, each of which surfaced a real, verified issue that was fixed and re-validated:
1. The naive-baseline comparison was initially selecting operating thresholds directly from TEST (leakage) — fixed to use the same TRAIN→VAL→freeze discipline as JOR.
2. The fusion-hyperparameter search was adopting its TRAIN-subsample winner without checking it against VAL — fixed to gate adoption on VAL performance, matching what detection-logic tuning already did.
3. A code comment justifying *why* detection-logic tuning was worth trying cited a diagnostic that had been run on TEST — a subtler form of leakage (TEST influencing the experimental design, not the final numbers). Fixed by re-running the diagnostic on TRAIN; the same qualitative pattern held, confirming the original rationale was sound, just improperly sourced.

In all three cases, fixing the leak did not change the headline numbers — which is a meaningfully different thing than "the leaks didn't matter." The methodology needed fixing regardless of whether the numbers moved, and now it's been checked from outside its own blind spots, not just self-certified.

## Limitations

Read this section before citing any number above out of context.

- **Synthetic data throughout.** No real acoustic returns, no real platform, no real target signatures. Every SNR value, target-strength baseline, dropout rate, and sea-state transition probability is either an illustrative placeholder or, at best, grounded in a general acoustics formula rather than measured data.
- **The independent evidence channel (`kin_consistency`) is a deliberate modeling choice, not a measured property.** It's built so genuine contacts get a positive reading and false alarms get a negative one — useful for testing *whether JOR can exploit* an orthogonal evidence stream, not evidence that a real kinematic-consistency signal would behave this way on an actual platform.
- **C, E, and P are not three equally informative channels.** C is a confidence/context signal centered on a fixed baseline with noise — it is not independently discriminative between real and false contacts in this simulator. That's realistic (sensor/operator confidence often isn't decisive on its own) but shouldn't be read as "three equal votes."
- **The synthetic test set's difficulty mix is arbitrary**, not representative of any real contact-rate distribution. The ~63% positive rate and even phenotype split were chosen for evaluation coverage, not to model how often real contacts occur or how hard they typically are. The phenotype-level breakdown is more informative than the aggregate recall number for exactly this reason.
- **Target-strength and ambient-noise baselines are placeholders pending real sensor/platform data**, even where the underlying *formula* (ambient noise vs. sea state) is a real, citable acoustics relationship.
- **The "combined" tiered recall is an explicit upper bound**, not an observed figure — it assumes perfect human review-queue triage.

This demo shows that the JOR fusion architecture does something real when given independent evidence to work with, and that a disciplined TRAIN/VAL/TEST methodology holds up under external review. It does not show, and does not claim to show, real-world maritime sonar detection performance.

## Files

| File | Description |
|---|---|
| `jor_dsp_maritime_acoustic_sonar_demo.py` | Full script: scenario generation, fusion/detection tuning, tiered evaluation, all figures and the video |
| `jor_vs_naive_baseline_recall_fpr.png` | JOR vs. naive SNR-threshold, frozen operating points + exploratory TEST curves |
| `jor_maritime_example_tracks_overview.png` | Six example TEST scenarios: 4 confirmed, 1 review-tier catch, 1 honest miss |
| `jor_maritime_seed_20011_poster.png` | Single-scenario detail view |
| `jor_maritime_curated_examples.mp4` | 60-second animated walkthrough of the same six curated scenarios |

## Running it

```bash
python jor_dsp_maritime_acoustic_sonar_demo.py
```

Requires `numpy`, `matplotlib`, and `ffmpeg` (for the MP4; the script degrades gracefully to PNG-only output if `ffmpeg` isn't available). Full run — scenario generation, hyperparameter search, detection-logic search, tiered evaluation, all figures and the video — takes a few minutes.
