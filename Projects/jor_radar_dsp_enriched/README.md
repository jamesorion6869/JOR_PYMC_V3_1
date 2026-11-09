# JOR Radar/DSP Enriched Benchmark

A synthetic, reproducible benchmark that asks one narrow question: **when JOR (James Orion Report, V3.1) is placed on top of ordinary DSP outputs, does it score and fuse the evidence better than simpler scoring/fusion layers fed the same signals?**

- Author: Jake James (independent researcher)
- Background: JOR is described in the original paper, James Orion Report (JOR) V3.1, https://doi.org/10.5281/zenodo.18088931. This repository is a synthetic radar/DSP benchmark built on that framework; it does not repeat the framework's derivation.
- Licence: MIT, Copyright (c) 2026 Jake James (see `LICENSE.txt`)
- DOI: https://doi.org/10.5281/zenodo.23980949
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

**Only these comparisons were run** in the tuned study. Two later, separately pre-registered studies add CUSUM baselines on fresh seeds (see [CUSUM study v2.1](#cusum-comparison-study-v21-test-seeds-8000080999), the current one, and [v2.0](#cusum-comparison-study-v20-test-seeds-7000070999)). None of them shows that JOR beats other combiners in general, on real data, or with other generators.

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

## CUSUM comparison (study v2.1, TEST seeds 80000–80999)

This is the current CUSUM comparison. It repeats v2.0 (below) on a fresh TEST block, with one change in method: a stricter threshold rule, adopted because JOR's FPR overshot its budget in v2.0. v2.0 is kept as-is for the record.

- **Script:** `jor_cusum_study_v2.1.py`. It imports `jor_tuned_study_v1.1.py` unchanged as a library, as v2.0 did.
- **Seeds:** TRAIN 0–999 and VAL 10000–10499 as above; TEST 80000–80999 (new; the script refuses overlap with development seeds and with the spent blocks 60000–60999 and 70000–70999).
- **Selection rule (the change from v2.0):** a setting is eligible only if the upper 95% Wilson bound of its TRAIN FPR is at or below 0.12. It is applied identically to every method, during selection only. Reported FPRs are point estimates. In practice, chosen settings sit at a TRAIN FPR of about 0.074–0.085, not 0.12, and every method is compared at that operating point.
- **Detector form:** all three CUSUMs may fire on level, or on a rise over 4 steps, which is the detector form JOR uses.
- **Two descriptive rows (reported, not claims):** *CUSUM on JOR weights*, to check whether CUSUM was understated by running on averaging weights; and *C/E/P + memory*, the C/E/P average with JOR's posterior smoothing (EMA, alpha 0.30) and its own weight search, to check how much of JOR's lead is just memory.
- **Pre-registration:** claims V1–V7 (`PREDECLARED_CLAIMS_CUSUM_v2.1.md`), the same structure as v2.0. The script, claims and library were emailed by the author to himself at 07:49 EST on 2026-11-09, before the single full run began at 07:52:31 EST. Hashes and details are in `LOCK_CUSUM_v2.1.txt`.

Event recall / FPR on TEST:

| Method | Stationary recall / FPR | Drifting recall / FPR |
|---|---|---|
| **JOR (tuned)** | **0.475 / 0.125** | **0.422 / 0.118** |
| CUSUM on C/E/P | 0.454 / 0.102 | 0.342 / 0.083 |
| CUSUM-detrended | 0.288 / 0.109 | 0.281 / 0.131 |
| C/E/P avg (tuned) | 0.424 / 0.118 | 0.355 / 0.102 |
| SNR (CFAR excess) | 0.195 / 0.112 | 0.167 / 0.086 |
| CUSUM on JOR weights* | 0.456 / 0.096 | 0.255 / 0.013 |
| C/E/P + memory* | 0.301 / 0.105 | 0.243 / 0.089 |

\* descriptive row, not part of any claim.

Paired differences, tuned JOR minus method (95% bootstrap interval, 2000 resamples):

| vs | Stationary recall | Stationary FPR | Drifting recall | Drifting FPR |
|---|---|---|---|---|
| CUSUM on C/E/P | +0.020 [−0.006, +0.046] | +0.022 [−0.015, +0.060] | +0.080 [+0.054, +0.107] | +0.035 [+0.000, +0.073] |
| CUSUM-detrended | +0.186 [+0.145, +0.229] | +0.016 [−0.031, +0.063] | +0.141 [+0.101, +0.182] | −0.013 [−0.068, +0.037] |
| C/E/P avg (tuned) | +0.051 [+0.017, +0.086] | +0.006 [−0.039, +0.049] | +0.067 [+0.037, +0.100] | +0.016 [−0.026, +0.059] |
| C/E/P + memory* | +0.173 [+0.141, +0.207] | +0.019 [−0.024, +0.060] | +0.179 [+0.147, +0.212] | +0.029 [−0.013, +0.068] |

**Claim check: 6 of 7 held.** V2–V7 held. V1 did not hold: on the stationary floor JOR and plain CUSUM are statistically tied again. V7 held: JOR's TEST FPR was 0.125 (stationary) and 0.118 (drifting), under the pre-declared 0.15 ceiling. No selection fallbacks.

How to read this honestly:

- **The stricter rule brought JOR's FPR back near budget** (v2.0: 0.179 / 0.197; v2.1: 0.125 / 0.118), at a cost of about 0.02–0.05 in recall. This is one TEST block. The same JOR configuration earlier gave TEST FPR 0.119 on one block and 0.179 on another, so the rule held FPR in budget here; it does not guarantee it.
- **Stationary: still a tie with plain CUSUM** (recall +0.020, interval includes zero; F1 +0.015 [−0.011, +0.039]). Plain CUSUM again beat the C/E/P average on recall (+0.031 [+0.003, +0.061]). On this floor, much of JOR's edge over the averages can be matched by simple accumulation.
- **Drifting: JOR beat plain CUSUM by +0.080 recall, replicating v2.0's direction on fresh seeds.** JOR's FPR was also higher (+0.035; the interval's lower bound is at zero), so part of that lead comes with more false alarms. On F1 JOR still leads (+0.076 [+0.048, +0.105]). Unlike v2.0, plain CUSUM's barrier did not sit on its grid edge (h = 0.20).
- **Memory alone does not explain JOR's lead.** C/E/P + memory came in well below JOR (about −0.17 to −0.18 recall) and below the plain C/E/P average. As declared before the run, this does not separate the Bayes step, K and prior retention (0.70) from each other, because this row has none of them.
- **CUSUM on JOR weights** came close to plain CUSUM on the stationary floor (0.456 vs 0.454). On the drifting floor its chosen setting ran at a TEST FPR of 0.013, far below budget, so its drifting recall is not a like-for-like comparison.
- **Latency is mixed.** Mean latency on TEST: JOR 5.04 vs plain CUSUM 3.16 steps on the stationary floor (JOR later), 4.32 vs 5.16 on the drifting floor (JOR earlier). Detection counts a hit within ±3 steps of onset for every method.
- **Locking deviations (disclosed in `LOCK_CUSUM_v2.1.txt`):** the full `--dev` run on the laptop was skipped (only the header hashes and first JOR line were checked); the claims file was emailed without a line-by-line review by the author; and the last script change (error-path logging only) was checked in a separate sandbox, not in a laptop dev run.
- The claims file lists 13 disclosures, including forking paths (the Wilson rule, the CUSUM rise path and both descriptive rows were added after v2.0 and after development-seed checks), a track-consistency offset in the generator, and unequal search sizes between methods. Read them with the results.
- Same generator and same limits as the tuned study: synthetic, generator-coupled, not evidence about real radar.

## CUSUM comparison (study v2.0, TEST seeds 70000–70999)

Superseded by v2.1 above; kept as run.

CUSUM is the standard sequential change detector and the first baseline a reviewer would ask for. It also accumulates evidence over time, so it tests whether JOR's edge over the averages is more than plain accumulation. It was not part of the tuned study above, and it was designed after that study's TEST block had been seen. So it was run as a **new study on a fresh TEST block**, with its own pre-declared claims.

- **Script:** `jor_cusum_study_v2.0.py`. It imports `jor_tuned_study_v1.1.py` unchanged as a library (same generator, DSP, evidence, JOR update, detection rule, scoring and bootstrap).
- **Seeds:** TRAIN 0–999 and VAL 10000–10499 as above; TEST 70000–70999 (new; the script refuses overlap with development seeds and with 60000–60999).
- **Methods added:** *CUSUM on C/E/P* (one-sided CUSUM, S_t = max(0, S_{t−1} + x_t − ν), on the tuned C/E/P stream; grid over ν, barrier h and sustain) and *CUSUM-detrended* (the same on C/E/P minus an EWMA of earlier values; also tunes the EWMA rate). Every method was tuned on TRAIN only at the same FPR budget (0.12).
- **Pre-registration:** claims V1–V7 (`PREDECLARED_CLAIMS_CUSUM_v2.md`). The script, claims and library were emailed by the author to himself at 13:13 EST on 2026-11-06, before the single full run began at 13:18:57 EST. Hashes and details are in `LOCK_CUSUM_v2.0.txt`.

Event recall / FPR on TEST:

| Method | Stationary recall / FPR | Drifting recall / FPR |
|---|---|---|
| **JOR (tuned)** | **0.496 / 0.179** | **0.471 / 0.197** |
| CUSUM on C/E/P | 0.477 / 0.166 | 0.349 / 0.125 |
| CUSUM-detrended | 0.280 / 0.119 | 0.302 / 0.150 |
| C/E/P avg (tuned) | 0.442 / 0.122 | 0.423 / 0.166 |
| SNR (CFAR excess) | 0.248 / 0.116 | 0.203 / 0.113 |

Paired differences, tuned JOR minus method (95% bootstrap interval, 2000 resamples):

| vs | Stationary recall | Stationary FPR | Drifting recall | Drifting FPR |
|---|---|---|---|---|
| CUSUM on C/E/P | +0.019 [−0.009, +0.045] | +0.013 [−0.023, +0.047] | +0.122 [+0.088, +0.155] | +0.072 [+0.023, +0.122] |
| CUSUM-detrended | +0.216 [+0.173, +0.257] | +0.060 [+0.003, +0.112] | +0.169 [+0.124, +0.212] | +0.047 [−0.009, +0.105] |
| C/E/P avg (tuned) | +0.054 [+0.024, +0.086] | +0.056 [+0.009, +0.102] | +0.048 [+0.012, +0.086] | +0.031 [−0.022, +0.085] |

**Claim check: 5 of 7 held.** V2–V6 held. V1 did not hold: on the stationary floor JOR and plain CUSUM are statistically tied. V7 did not hold: JOR's TEST FPR was 0.179 (stationary) and 0.197 (drifting), above the pre-declared 0.15 ceiling.

How to read this honestly:

- **Stationary: a tie with plain CUSUM**, at comparable FPR (F1 difference +0.014 [−0.010, +0.038]). Plain CUSUM also beat the C/E/P average on recall (+0.035 [+0.003, +0.068]). So on this floor, much of JOR's edge over the averages can be matched by simple accumulation.
- **Drifting: JOR beat plain CUSUM by +0.122 recall, but at a higher FPR** (+0.072, interval excludes zero). On F1 it still leads (+0.106 [+0.073, +0.140]). Plain CUSUM's tuned barrier landed on the bottom of its grid (h = 0.01), so it may be under-tuned. The script warned about this. It is reported as-is, because re-running would spend the TEST block a second time.
- **JOR's FPR did not transfer.** Its threshold hit FPR 0.115 on TRAIN but 0.171 on VAL and 0.179 on TEST (stationary). The baselines drifted less, so several of JOR's recall wins here are not at matched FPR. In the earlier tuned study (TEST 60000–60999) JOR's TEST FPR was 0.119; this run shows that the threshold can overshoot on a new block.
- The CUSUM grids were widened after a quick-mode check on development seeds, and no `--dev` run was made with the final grids.
- Same generator and same limits as the tuned study: synthetic, generator-coupled, not evidence about real radar.

## Limitations

- **Synthetic and generator-coupled.** Results depend on how the scenarios, the E/P scoring functions and the noise/drift model were written. They are not evidence about real radar performance.
- **E and P are hand-built.** They are not a standard radar processing method. The E+P average is an internal ablation.
- **Tuned weights do not transfer.** Only the tuning procedure does. Retune on your own data and your own FPR budget.
- **One FPR budget for the headline (0.12)**; a budget sensitivity table is included in the output, but the pre-declared claims use 0.12.
- **Weights are tuned on TRAIN, not cross-validated across generators.** A different generator could rank methods differently.
- **Not detect-and-track.** No Kalman/IMM/TBD comparison is made or implied, and this benchmark says nothing about JOR versus those.
- Development seeds were examined while building the study; the TEST block (60000–60999) was not used for any design decision.
- **CUSUM studies:** JOR ties plain CUSUM on the stationary floor in both v2.0 and v2.1. In v2.0 JOR's FPR ran above budget (V7 not held); in v2.1 the stricter selection rule held it under 0.15 on one fresh block. JOR's drifting lead over CUSUM comes with a somewhat higher FPR. See the CUSUM sections.

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

CUSUM study v2.0 (keep `jor_tuned_study_v1.1.py` in the same folder):

```
python jor_cusum_study_v2.0.py
```

- Saves its own output next to the script as `jor_cusum_v2_results_<mode>_<date-time>.txt`. Compare against `results/jor_cusum_v2_results_TEST70000-70999_20261106-131857.txt`.
- About 69 minutes per condition (about 2.3 hours total) serial on a Windows laptop.
- `--quick` and `--dev` work as above and are not results. `--jobs N` as above.

CUSUM study v2.1 (keep `jor_tuned_study_v1.1.py` in the same folder):

```
python jor_cusum_study_v2.1.py
```

- Saves its own output next to the script as `jor_cusum_v2.1_results_<mode>_<date-time>.txt`. Compare against `results/jor_cusum_v2.1_results_TEST80000-80999_20261109-075231.txt`.
- About 64 minutes per condition (under 2.5 hours total) serial on a Windows laptop.
- `--quick` and `--dev` work as above and are not results. `--jobs N` as above.

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
| `jor_cusum_study_v2.0.py` | CUSUM comparison study (imports `jor_tuned_study_v1.1.py` unchanged) |
| `PREDECLARED_CLAIMS_CUSUM_v2.md` | Claims V1–V7, fixed before the v2.0 TEST run |
| `LOCK_CUSUM_v2.0.txt` | Pre-registration record and SHA-256 hashes for the v2.0 study |
| `results/jor_cusum_v2_results_TEST70000-70999_20261106-131857.txt` | CUSUM study v2.0 output, single full run |
| `jor_cusum_study_v2.1.py` | CUSUM comparison study v2.1, current (imports `jor_tuned_study_v1.1.py` unchanged) |
| `PREDECLARED_CLAIMS_CUSUM_v2.1.md` | Claims V1–V7 and disclosures, fixed before the v2.1 TEST run |
| `LOCK_CUSUM_v2.1.txt` | Pre-registration record and SHA-256 hashes for the v2.1 study |
| `results/jor_cusum_v2.1_results_TEST80000-80999_20261109-075231.txt` | CUSUM study v2.1 output, single full run |
| `archive/jor_tuned_study_v1.0.py`, `archive/LOCK_TUNED_v1.0.txt` | Previous script version (same numbers; v1.1 only added `--quick` and flag handling) |
| `archive/earlier_study_default_weights/` | The first study, run with default/author weights, kept as ablation |
| `archive/original_demo_2026-09/` | The original demo, `README_original.md` and requirements, kept for history |

## Citation

Zenodo DOI: https://doi.org/10.5281/zenodo.23980949

This benchmark accompanies the JOR V3.1 work (Zenodo DOI 10.5281/zenodo.18088931). Please cite the specific version of this record once it has a DOI.

## Licence

[MIT](LICENSE.txt) — Copyright (c) 2026 Jake James
