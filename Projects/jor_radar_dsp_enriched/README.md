# JOR Radar/DSP Enriched Benchmark

A synthetic, reproducible benchmark that asks one narrow question: **when JOR (James Orion Report, V3.1) is placed on top of ordinary DSP outputs, does it score and fuse the evidence better than simpler scoring/fusion layers fed the same signals?**

- Author: Jake James (independent researcher)
- Licence: MIT, Copyright (c) 2026 Jake James (see `LICENSE.txt`)
- DOI: **[Zenodo DOI to be added once the record is published]**
- Status: synthetic benchmark only. Nothing here has been tested on real radar data.

---

## What this is, and what it is not

**JOR is not a detector and not a tracker.** It is not a detect-and-track system, it does not estimate target state, and it is not a replacement for CFAR, a Kalman/IMM filter, or track-before-detect.

JOR is an evidence-scoring and fusion layer. It is designed to **sit alongside the DSP chain**, consuming what the DSP already produces (CFAR-relative SNR, RCS, detection probability, innovation/consistency measures, dropout rate) and turning it into a single, auditable belief that "something real is happening" which a downstream decision rule can threshold.

For that reason this benchmark deliberately does **not** compare JOR against Kalman, IMM or track-before-detect. That would be a category mismatch. It compares **scoring and fusion layers that all consume the same DSP features**, with the same detection-rule family and the same false-alarm budget.

## What is compared

All methods see the same synthetic scenarios and the same enriched DSP features. Each is tuned on TRAIN data only, under the same false-positive-rate (FPR) budget (0.12), then scored once on fresh TEST seeds.

| Method (as labelled in output) | What it is |
|---|---|
| **JOR (tuned)** | C/E/P evidence into the JOR Bayesian update with memory (headline) |
| SNR (CFAR excess) | Causal CA-CFAR "excess SNR" with a threshold/consecutive-hit rule. Closest thing to current practice |
| C/E/P avg (tuned) | Same C/E/P inputs, tuned weighted average. No Bayes step, no memory |
| E+P avg (tuned) | E and P only, tuned weighted average. Internal ablation, not a standard radar method |
| JOR (default wts) | JOR with untuned default weights (reference) |
| E+P avg (0.5/0.5) | Untuned equal-weight average (reference) |
| JOR+adaptive (tuned) | JOR variant with adaptive decay (reference) |
| ref: JOR abs-SNR (tuned) | JOR fed absolute-SNR evidence instead of CFAR-relative (reference) |
| ref: SNR abs dB | Single-channel absolute-SNR tracker (reference) |

**Only these comparisons were run.** The results below do not show that JOR beats other combiners in general, on real data, or with other generators.

## Why tuned weights are the headline

JOR is modular: the channel weights (W_C, W_E, W_P) and the contradiction coefficient K are meant to be set for whatever system it is attached to, the way an antenna is turned to a particular station. An operator would not deploy it on defaults. So the headline result tunes JOR's weights and K on TRAIN data, and gives every baseline the same courtesy (tuned weights, tuned thresholds, same budget).

The earlier default-weight study is kept as an ablation and as the record of how the work started (see `archive/`).

Two consequences to keep in mind:

1. **The selected weights are specific to this generator.** In this synthetic setup the search drifted onto the P channel (e.g. stationary: W_C 0.02, W_E 0.02, W_P 0.96, K 0.12), because P is derived from RCS and absolute SNR via peak power. Do not read that as "P matters most in radar." What transfers is the *procedure* (tune on your own labelled data under a stated FPR budget), not the numbers.
2. **Retuning is needed when conditions change.** Settings tuned on the stationary noise floor and carried unchanged to a drifting one lost some FPR control (see carry-over below).

## Benchmark design (current generator)

- **Conditions:** STATIONARY noise floor, and DRIFTING noise floor (a linear SNR baseline ramp of ±U(3,6) dB over 60 steps, on its own random stream).
- **Scenarios:** synthetic sensor time series with a mix of event phenotypes (roughly equal shares) including intermittent, gradual-onset, conflicting-evidence and sparse events, plus false events. TEST: 1000 scenarios, 656 with a true event.
- **Seeds:** TRAIN 0–999, VAL 10000–10499, TEST 60000–60999. TEST seeds are fresh: the script checks they do not overlap any development seed range.
- **DSP front end:** causal CA-CFAR (guard 3, training 12, minimum 4 training cells) producing "excess SNR", which feeds the E and P evidence scores.
- **Evidence:** C is a fixed assumed baseline (0.65, non-informative). E and P are hand-weighted 0–1 scores of radar-like features. These are not standard radar methods; they are the benchmark's way of turning DSP features into evidence.
- **JOR update:** priors 0.20/0.80, prior retention 0.70, posterior alpha 0.30 (fixed). Tuned: W_C, W_E, W_P and K.
- **Weight search:** 150 candidates per condition (the default set, the author's earlier set, and 148 Dirichlet(1,1,1) draws with K ~ U(0.05, 0.40); search seed 2026). Selection uses TRAIN only: best recall subject to FPR ≤ 0.12, ties broken by F1 then earlier candidate. The threshold is then re-selected on the full grid.
- **Detection rule:** flag after 3 consecutive steps above a level threshold, or on a rise ≥ 0.04 over 4 steps with margin 0.06. Unit grid 0.200–0.898 in 0.002 steps (native dB grid and rise options for the SNR trackers).
- **Scoring:** scenario-level, first flag only, matched to a true event within a 3-step lag allowance. Recall, FPR, F1 and latency reported, with Wilson 95% intervals and paired bootstrap (2000 resamples, seed 12345) for differences. See the script for exact definitions.
- **Pre-declared claims:** 15 claims (T1–T15) were written and hashed before the TEST run (`PREDECLARED_CLAIMS_TUNED.md`, `LOCK_TUNED_v1.1.txt`).

## Results (TEST seeds 60000–60999, FPR budget 0.12)

Event recall / FPR on TEST, each method at its own TRAIN-selected setting:

| Method | Stationary recall / FPR | Drifting recall / FPR |
|---|---|---|
| **JOR (tuned)** | **0.491 / 0.119** | **0.479 / 0.119** |
| SNR (CFAR excess) | 0.239 / 0.113 | 0.220 / 0.119 |
| C/E/P avg (tuned) | 0.424 / 0.105 | 0.390 / 0.110 |
| E+P avg (tuned) | 0.436 / 0.113 | 0.354 / 0.108 |
| JOR (default wts) | 0.439 / 0.081 | 0.367 / 0.096 |
| ref: JOR abs-SNR (tuned) | 0.460 / 0.110 | 0.422 / 0.102 |
| ref: SNR abs dB | 0.497 / 0.090 | 0.070 / 0.081 |

Paired differences in recall, tuned JOR minus method (95% bootstrap interval):

| vs | Stationary | Drifting |
|---|---|---|
| SNR (CFAR excess) | +0.252 [+0.211, +0.292] | +0.259 [+0.219, +0.300] |
| C/E/P avg (tuned) | +0.067 [+0.038, +0.099] | +0.088 [+0.053, +0.121] |
| E+P avg (tuned) | +0.055 [+0.023, +0.088] | +0.125 [+0.093, +0.158] |
| JOR (default wts) | +0.052 [+0.023, +0.079] | +0.111 [+0.084, +0.140] |
| ref: SNR abs dB | −0.006 [−0.039, +0.027] | +0.409 [+0.370, +0.447] |

How to read this honestly:

- Tuned JOR beats the CFAR SNR tracker and the tuned C/E/P and E+P averages in both conditions; the intervals exclude zero. FPRs are comparable (differences are within noise).
- In the stationary case, tuned JOR is **statistically indistinguishable from the absolute-SNR tracker** (−0.006, interval spans zero). Absolute SNR is a strong cue in this generator because the generator ties RCS and peak power to it.
- Under drift the absolute-SNR tracker collapses (0.070 recall) while JOR holds up. That reference uses absolute dB, which a real system would not rely on under a drifting floor, so treat it as a stress reference, not a fair rival.
- The advantage of JOR over the averages appears to come mainly from **temporal accumulation** (memory). In a development ablation, JOR without memory fell to roughly E+P level. This is an interpretation of the dev runs, not a separately proven mechanism.
- The adaptive-decay variant did worse than tuned JOR (−0.064 stationary, −0.151 drifting).

**Carry-over (stationary-tuned settings applied unchanged to drifting TEST data):** tuned JOR FPR rose from 0.119 to 0.163 [0.128, 0.205], which is above its 0.12 budget but modest; the absolute-SNR tracker's FPR rose to 0.468. Retuning per condition restores the budget.

**Known weak spot:** the sparse-event phenotype (F_SPARSE) is essentially undetected by every method (first-detection recall ≤ 0.031 for all, stationary and drifting; Tables F1–F3 in the output). Its raw sensor observability is also low, so this is largely a property of the generator, not a JOR-specific finding.

**Claim check:** all 15 pre-declared claims (T1–T15) were evaluated on the reference run and held. The script prints each verdict.

The earlier default-weight study is kept in `archive/earlier_study_default_weights/` with its own pre-declared claims and results; it is superseded by this tuned study.

## Limitations

- **Synthetic and generator-coupled.** Results depend on how the scenarios, the E/P scoring functions and the noise/drift model were written. They are not evidence about real radar performance.
- **E and P are hand-built.** They are not a standard radar processing method. The E+P average is an internal ablation.
- **Tuned weights do not transfer.** Only the tuning procedure does. Retune on your own data and your own FPR budget.
- **One FPR budget for the headline (0.12)**; a budget sensitivity table is included in the output, but the pre-declared claims use 0.12.
- **Weights are tuned on TRAIN, not cross-validated across generators.** A different generator could rank methods differently.
- **Not detect-and-track.** No Kalman/IMM/TBD comparison is made or implied, and this benchmark says nothing about JOR versus those.
- Development seeds were examined while building the study; the TEST block (60000–60999) was not used for any design decision.

## Reproducing

Requirements: Python 3 and `numpy>=1.22` (`pip install -r requirements.txt`).

Full protocol (the one that produces the reported numbers):

```
python jor_tuned_study_v1.1.py > tuned_run.txt 2>&1
```

- Runtime is long: about 13 minutes on a fast machine with `--jobs 2`; about 1 h 50 min serial on a 2019 laptop.
- `--jobs N` parallelizes on platforms that support fork (Linux/macOS). On Windows it runs serial. `--jobs` needs a number.
- The full run refuses `--candidates` other than 150.

Quick mode (a smoke test, **not a result**):

```
python jor_tuned_study_v1.1.py --quick > quick_run.txt 2>&1
```

Uses fewer seeds and 20 candidates, labels its output "[QUICK MODE: not a result]", and does not evaluate the pre-declared claims. About 4 minutes on a fast machine; budget roughly 10 minutes on an older laptop.

Other flags: `--dev` (development seeds), `--candidates N` (only with `--dev` or `--quick`).

Reproduction check: the full run on seeds 60000–60999 was repeated on a separate Windows laptop (serial); both outputs are in `results/`. Compare your output against `results/tuned_study_seeds60000-60999.txt`.

## Files

| Path | What it is |
|---|---|
| `jor_tuned_study_v1.1.py` | The benchmark script (current) |
| `PREDECLARED_CLAIMS_TUNED.md` | Claims T1–T15, written before the TEST run |
| `LOCK_TUNED_v1.1.txt` | SHA-256 hashes locking the script and claims |
| `MANIFEST.txt` | File list with hashes |
| `requirements.txt` | `numpy>=1.22` |
| `LICENSE.txt` | MIT licence (the packaged copy differs from the repo copy only in line endings) |
| `results/tuned_study_seeds60000-60999.txt` | Reference full run |
| `results/tuned_study_reproduction_windows_serial.txt` | Independent reproduction on a Windows laptop |
| `results/quick_mode_example_not_a_result.txt` | Example quick-mode output |
| `archive/jor_tuned_study_v1.0.py`, `archive/LOCK_TUNED_v1.0.txt` | Previous script version (same numbers; v1.1 only added `--quick` and flag handling) |
| `archive/earlier_study_default_weights/` | The first study, run with default/author weights, kept as ablation |
| `archive/original_demo_2026-09/` | The original demo, `README_original.md` and requirements, kept for history |

## Citation

Zenodo DOI: **[placeholder, to be added once published]**

This benchmark accompanies the JOR V3.1 work (Zenodo DOI 10.5281/zenodo.18088931). Please cite the specific version of this record once it has a DOI.

## Licence

[MIT](LICENSE.txt) — Copyright (c) 2026 Jake James
