# Pre-declared claims: tuned-weights study

Written before the run on the held-out TEST block (seeds 60,000-60,999), which no earlier run,
search or look has touched. `jor_tuned_study.py` evaluates every claim below automatically and
prints HELD / NOT HELD next to the numbers. Results are reported whichever way each claim comes
out. No code, parameter, search-size or claim changes after the run.

## Honest context

- This is a second study. The first (default weights, TEST seeds 50,000-50,999, locked and run
  earlier) found JOR level with or below a plain E+P average. The author then supplied
  hand-picked weights (W_C 0.51, W_E 0.03, W_P 0.46, K 0.18) that raised JOR's recall on that
  block. That block is now spent for these questions and is not used here.
- The design below was written after development-range searches (TRAIN/VAL/dev blocks only) had
  shown that JOR's recall depends strongly on its weights, that tuned JOR stayed ahead of tuned
  weighted averages, and that the gap to a tuned E+P average was small. The claims are
  expectations from those searches, not blind predictions. T5 and T6 in particular could go
  either way.
- Weights, K and thresholds are selected on TRAIN only. VAL is reported as a check and is not
  used for selection. The author's hand-picked weights enter the search as one candidate
  alongside the original defaults and 148 fixed-seed random draws.
- Memory settings (prior retention 0.70, posterior alpha 0.30) are held at the original defaults.

## Protocol (identical for every method)

- Per condition, TRAIN-only selection: 150 weight/K candidates for JOR, the same 150 weight
  sets for the C/E/P weighted average (no Bayes step, no memory), 11 E/P weights for the E+P
  average. Each candidate is scored by TRAIN recall under FPR <= 0.12 on a coarse threshold
  grid; the best (ties: F1, then earlier) is kept and its threshold re-selected on the full grid.
- Scoring is scenario-level, first flag only, with a 3-step lag allowance.
- Intervals are 95% Wilson (single proportions) and a paired bootstrap over TEST scenarios
  (2,000 resamples, seed 12345) for differences.
- Two conditions: STATIONARY and DRIFTING (SNR baseline ramps linearly by +/- U(3, 6) dB).
- Carry-over: stationary-tuned weights and thresholds applied unchanged to drifting TEST data.

## Claims

| ID | Claim |
|---|---|
| T1 | Stationary: tuned JOR's recall exceeds the CFAR-excess SNR tracker's (paired 95% CI lower bound > 0). |
| T2 | Drifting: the same. |
| T3 | Stationary: tuned JOR's recall exceeds the tuned C/E/P weighted average's (CI lower bound > 0). |
| T4 | Drifting: the same. |
| T5 | Stationary: tuned JOR's recall exceeds the tuned E+P average's (CI lower bound > 0). |
| T6 | Drifting: the same. |
| T7 | Stationary: tuned JOR's recall exceeds default-weight JOR's (CI lower bound > 0). |
| T8 | Weights matter: JOR candidates' TRAIN recall, best minus median, is at least 0.03 in both conditions. |
| T9 | Stationary: tuned JOR's recall is no more than 0.03 below the absolute-SNR tracker's (point estimate). |
| T10 | Drifting: tuned JOR's recall exceeds the retuned absolute-SNR tracker's by more than 0.20 (point estimate). |
| T11 | Carried over to drift, the absolute-SNR tracker's FPR exceeds 0.25. |
| T12 | Carried over to drift, tuned JOR keeps FPR at or below 0.20. |
| T13 | Tuned JOR's TEST FPR is at or below 0.15 in both conditions. |
| T14 | Drifting: adaptive decay with tuned weights does not improve on tuned JOR (CI upper bound < 0). |
| T15 | Stationary F_SPARSE first-detection recall is below 0.10 for every method. |

## What the claims do not say

- They make no claim about real sensor data. The benchmark is synthetic and the generator
  injects the signals the evidence terms are computed from.
- Tuning to this generator shows what JOR can do when tuned to a data source, as intended by its
  design. It does not show that any particular weights transfer to another source.
- Weights are tuned separately per condition; a single set that is best in both is not claimed.
