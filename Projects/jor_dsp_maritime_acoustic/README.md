# JOR — Maritime/Acoustic Sonar Concept Demo

A concept demonstration extending the **JOR (James Orion Report) framework** — a Bayesian evidence-fusion architecture originally built for scoring UAP sighting reports — into a maritime active-sonar contact-detection setting.

**What this is:** a synthetic-data proof of concept showing that JOR's fusion/scoring architecture behaves sensibly on maritime-shaped inputs, with a properly held-out TRAIN/VAL/TEST evaluation.

**What this is not:** a validated maritime sensor model, and not a claim about real-world sonar detection performance. Every number in this README comes from synthetic scenarios with illustrative sensor constants. See [Limitations](#limitations) before drawing any conclusion beyond "the architecture works as intended on this synthetic data."

This repo intentionally keeps **both** the original demo and its corrected successor, rather than quietly replacing one with the other — see [Version History](#version-history) for why, and for a real, verified methodology fix that changed the headline numbers.

---

## Background

JOR fuses three evidence dimensions — **C** (witness/operator confidence), **E** (environmental/signal evidence), **P** (physical/track evidence) — into a weighted score, then runs that through a Bayesian posterior update to produce a confidence-over-time track for a candidate contact. The original framework scores UAP sighting reports from human witness testimony; this demo retargets the same fusion math to an active-sonar setting, where C becomes a sensor/operator-confidence signal (no human eyewitness involved) and E/P are built from simulated acoustic returns instead of testimony.

The core Bayesian fusion math is **unchanged** from the validated JOR architecture, including the P(E|NH) = NHP and P(E|H) = min(1, 1 − NHP + K·SOP) likelihood formulation from JOR V3.1. What's new here is the sensor-simulation layer feeding it, and — in the current version — a recursive prior-retention mechanism that extends the paper's single-shot Bayesian update into a streaming, multi-step filter suited to a time-series detection problem.

## What's in the maritime adaptation

- **Sonar-domain sensor model**, replacing the original radar-domain one: target strength (TS) in place of radar cross-section, received level in place of peak power, sea-state/ambient-noise conditions in place of atmospheric conditions, and maritime-specific false-alarm categories (marine biologics, surface/bottom acoustic bounce, transient noise, thermocline-driven track instability) in place of radar clutter categories.
- **An independent secondary evidence channel** (`kin_consistency`, representing multi-ping/kinematic consistency) with its own random-noise stream, decoupled from the sea-state-driven degradation that the other channels share. This exists specifically to test whether JOR's fusion layer can exploit genuinely independent evidence, since a naive single-channel detector structurally cannot.
- **Tiered alerting**: a high-confidence CONFIRM tier (auto-flag) and a lower-confidence REVIEW tier (routed to a human instead of auto-actioned), rather than forcing one threshold to serve both "obviously a contact" and "maybe, hard to tell."
- **Physically grounded ambient-noise modeling**: the SNR penalty from rougher sea states is derived from Urick's analytic approximation to the Knudsen/Wenz deep-water ambient-noise spectrum (`NL(f, sea_state) = 10·log10(f^-5/3) + 94.5 + 30·log10(sea_state+1)`), not an invented number.

## Version history

Two scripts are kept in this repo on purpose:

| | `jor_dsp_maritime_acoustic_sonar_demo.py` (archived) | `jor_dsp_maritime_sonar_streaming.py` (current) |
|---|---|---|
| VAL set size | 150 scenarios | **400 scenarios** |
| CONFIRM threshold | 0.38 | 0.53 |
| TEST recall | 0.466 | **0.704** |
| TEST FPR | 0.081 | **0.027** |
| TEST F1 | 0.615 | 0.818 |
| Missed by both tiers | 92 / 300 (30.7%) | 40 / 300 (13.3%) |

The path from one to the other went through an intermediate iteration that isn't preserved as its own file, but is worth documenting because it's the most instructive part of the story:

1. **Original** (archived here): recall 0.466, FPR 0.081 — comfortably under the 0.12 FPR budget, but recall was the real limitation, concentrated in weak/sparse and internally-conflicting contact types.
2. **Intermediate iteration** (not separately preserved): strengthened the independent `kin_consistency` channel and added a `persist_score` feature aimed at the weak-signal failure modes. Recall jumped to 0.783 — a large, genuine improvement — but TEST FPR rose to 0.180, **exceeding** the 0.12 budget it had passed on VAL (0.114). Diagnosis: the 150-scenario VAL set wasn't large enough to reliably certify the selected threshold; a follow-up 5-partition repeat experiment confirmed the VAL→TEST FPR gap varied from −0.015 to +0.067 across reshuffled partitions of the same size, consistent with sampling noise rather than a structural flaw in the tuning logic.
3. **Current** (`jor_dsp_maritime_sonar_streaming.py`): same fusion architecture, VAL widened from 150 → 400 scenarios. This closed the FPR gap (TEST FPR 0.180 → 0.027) and, in the process, revealed that the 0.783 recall figure from step 2 had itself been inflated by VAL sampling luck — the true, validated recall at this operating point is 0.704. Confirmed stable: re-running with VAL widened further to 600 scenarios produced bit-for-bit identical results, indicating 400 is past the point of diminishing returns for this dataset.

Both the low-recall original and the correction of an inflated recall are kept visible here rather than only showing the final number, because a repo that only ever shows a clean success story is less credible than one that shows the mistake, the fix, and the discipline that caught it.

## Results (current version)

All headline numbers below are from a single **frozen, held-out TEST evaluation** on `jor_dsp_maritime_sonar_streaming.py` — every parameter (fusion weights, detection-logic settings, operating thresholds) was selected on TRAIN, confirmed on VAL, and frozen before TEST was touched. See [Methodology](#methodology) for why that distinction matters and how it was enforced.

| | TRAIN (300) | VAL (400) | TEST (300, held-out) |
|---|---|---|---|
| Recall (CONFIRM tier) | 0.754 | 0.681 | **0.704** |
| False Positive Rate | 0.062 | 0.095 | **0.027** |
| F1 | — | 0.787 | 0.818 |
| Mean detection latency | 3.99 steps | 3.77 steps | 3.74 steps |

**JOR fusion vs. a naive single-channel SNR threshold**, both selected the same way (TRAIN → VAL → frozen → TEST once):

| | Threshold | Recall | FPR |
|---|---|---|---|
| JOR fusion | 0.53 | **0.704** | 0.027 |
| Naive SNR threshold | 19.5 dB | 0.381 | 0.027 |
| **Recall delta (JOR − naive), matched FPR** | | **+0.323** | |

Unlike the original version's comparison (which had JOR and the naive baseline sitting at different FPRs), this result lands at the **same FPR (0.027) for both** — so the +0.323 recall gap is a genuinely matched-operating-point comparison, not an artifact of comparing two different false-alarm rates.

### Tiered alerting

| | Recall | Notes |
|---|---|---|
| CONFIRM (auto-flag) | 0.704 | Observed |
| CONFIRM + REVIEW combined | 0.788 | **Upper bound** — assumes a human reviewer correctly confirms every true contact routed to review; not an observed automated figure |
| Review workload | 11.0% of all TEST scenarios (33 scenarios) | 16 real contacts confirm missed, 17 false alarms, out of 300 |
| Missed by both tiers | 40 scenarios (13.3%) | Concentrated in specific phenotypes — see below |

Framed as a per-100-scenario breakdown (rounded, sums to 100): **44** real contacts auto-confirmed, **5** more caught only by review, **14** still missed by both tiers, **1** false alarm that slips through to auto-confirm, **6** false alarms sent to review, **30** correctly ignored as noise. In other words, roughly 89% of scenarios are resolved without a human ever looking at them; the remaining 11% split close to evenly between real misses worth catching and false alarms worth clearing.

### Where it succeeds and where it doesn't

Recall by contact "phenotype" (synthetic signature shape) shows the shortfall is concentrated, not uniform:

| Phenotype | CONFIRM recall | Combined recall |
|---|---|---|
| A_STRONG_MULTI (strong, multi-feature) | 0.904 | 0.942 |
| C_INTERMITTENT | 0.706 | 0.902 |
| G_GRADUAL_ONSET | 0.741 | 0.778 |
| E_CONFLICTING (internally ambiguous) | 0.667 | 0.746 |
| F_SPARSE (weak signal) | 0.636 | 0.727 |

Every phenotype improved over the original version, and the two hardest categories (E_CONFLICTING and F_SPARSE) improved the most — consistent with the `kin_consistency` weighting and `persist_score` feature added between versions, both aimed specifically at ambiguous and weak-signal cases. As before, this is a believable failure *pattern*, not a real-world estimate — see [Limitations](#limitations).

## Methodology

This demo enforces a disjoint **TRAIN (seeds 0–299) → VAL (10,000–10,399) → TEST (20,000–20,299)** split, with independent random streams per scenario component (events, false alarms, environmental conditions, sensor noise, dropout, the independent evidence channel) so the partitions are genuinely separated, not just differently-seeded copies of the same process.

Every tuned component — fusion weights, detection-logic parameters, the CONFIRM threshold, the REVIEW threshold, and the naive baseline's own threshold — follows the same discipline: **search on TRAIN → confirm on VAL → freeze → evaluate once on TEST.** A tuned configuration is only adopted over its default if it both meets the FPR budget and beats the default's VAL performance; otherwise the default is kept.

The original script went through **three rounds of external methodology review**, each of which surfaced a real, verified issue that was fixed and re-validated:
1. The naive-baseline comparison was initially selecting operating thresholds directly from TEST (leakage) — fixed to use the same TRAIN→VAL→freeze discipline as JOR.
2. The fusion-hyperparameter search was adopting its TRAIN-subsample winner without checking it against VAL — fixed to gate adoption on VAL performance, matching what detection-logic tuning already did.
3. A code comment justifying *why* detection-logic tuning was worth trying cited a diagnostic that had been run on TEST — a subtler form of leakage (TEST influencing the experimental design, not the final numbers). Fixed by re-running the diagnostic on TRAIN; the same qualitative pattern held, confirming the original rationale was sound, just improperly sourced.

A **fourth issue was found later**, in the successor version, and is the reason two versions exist: the VAL set (150 scenarios) was too small to reliably certify the FPR of a tuned configuration, letting a threshold pass VAL's budget check while exceeding that same budget on TEST. This was confirmed with a 5-partition repeat experiment (reshuffling which scenarios landed in TRAIN/VAL/TEST) before being fixed by widening VAL to 400 scenarios, and confirmed stable by further widening to 600 with no change in outcome.

In every case, fixing the leak or gap did not change the fundamental architecture — which is a meaningfully different thing than "the issues didn't matter." Each one needed fixing regardless of whether it moved the headline numbers, and the methodology has now been checked from outside its own blind spots multiple times over, not just self-certified once.

## Limitations

Read this section before citing any number above out of context.

- **Synthetic data throughout.** No real acoustic returns, no real platform, no real target signatures. Every SNR value, target-strength baseline, dropout rate, and sea-state transition probability is either an illustrative placeholder or, at best, grounded in a general acoustics formula rather than measured data.
- **The independent evidence channel (`kin_consistency`) is a deliberate modeling choice, not a measured property.** It's built so genuine contacts get a positive reading and false alarms get a negative one — useful for testing *whether JOR can exploit* an orthogonal evidence stream, not evidence that a real kinematic-consistency signal would behave this way on an actual platform.
- **C, E, and P are not three equally informative channels.** C is a confidence/context signal centered on a fixed baseline with noise — it is not independently discriminative between real and false contacts in this simulator. That's realistic (sensor/operator confidence often isn't decisive on its own) but shouldn't be read as "three equal votes."
- **The synthetic test set's difficulty mix is arbitrary**, not representative of any real contact-rate distribution. The phenotype split was chosen for evaluation coverage, not to model how often real contacts occur or how hard they typically are. The phenotype-level breakdown is more informative than the aggregate recall number for exactly this reason.
- **Target-strength and ambient-noise baselines are placeholders pending real sensor/platform data**, even where the underlying *formula* (ambient noise vs. sea state) is a real, citable acoustics relationship.
- **The "combined" tiered recall is an explicit upper bound**, not an observed figure — it assumes perfect human review-queue triage.
- **A validated VAL size is not the same as a validated model.** Widening VAL from 150 to 400 fixed a real generalization-certification problem, but every other limitation on this list still applies untouched — a larger validation sample makes the synthetic-data numbers more *trustworthy as synthetic-data numbers*, not more representative of real sonar performance.

This demo shows that the JOR fusion architecture does something real when given independent evidence to work with, that a disciplined TRAIN/VAL/TEST methodology holds up under repeated external and self-review, and that the review process itself is capable of catching its own mistakes, including ones several layers deep. It does not show, and does not claim to show, real-world maritime sonar detection performance.

## Files

| File | Description |
|---|---|
| `jor_dsp_maritime_sonar_streaming.py` | **Current.** Full script: scenario generation, fusion/detection tuning, tiered evaluation, all figures and the video. 400-scenario VAL set. |
| `jor_dsp_maritime_acoustic_sonar_demo.py` | **Archived, superseded.** Original version (150-scenario VAL, lower recall, no FPR issue). Kept in place so existing external links to it keep working; carries an in-file header pointing to the current version. |
| `jor_vs_naive_baseline_recall_fpr.png` | JOR vs. naive SNR-threshold, frozen operating points + exploratory TEST curves (current version) |
| `jor_maritime_example_tracks_overview.png` | Six example TEST scenarios: 4 confirmed, 1 review-tier catch, 1 honest miss (current version) |
| `jor_maritime_seed_20011_poster.png` | Single-scenario detail view |
| `jor_maritime_curated_examples.mp4` | 60-second animated walkthrough of the same six curated scenarios |

## Running it

```bash
python jor_dsp_maritime_sonar_streaming.py
```

Requires `numpy`, `matplotlib`, and `ffmpeg` (for the MP4; the script degrades gracefully to PNG-only output if `ffmpeg` isn't available). Full run — scenario generation, hyperparameter search, detection-logic search, tiered evaluation, all figures and the video — takes a few minutes; widening VAL further than 400 increases this roughly linearly with VAL size.

The archived `jor_dsp_maritime_acoustic_sonar_demo.py` still runs standalone the same way, for anyone comparing the two versions directly.
