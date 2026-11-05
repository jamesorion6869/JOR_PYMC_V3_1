# Pre-declared claims

Written before the final run on the held-out TEST block (seeds 50,000–50,999). The script
`jor_dsp_integration_demo.py` evaluates every claim below automatically and prints
HELD / NOT HELD next to the supporting numbers. Results will be reported whichever way each
claim comes out. No code, parameter or claim changes after the final run.

## Honest context

- These claims were written **after** the development seeds (0–999, 10,000–10,499,
  20,000–20,999, and exploratory looks at 30,000–30,999) had been examined. They are
  therefore expectations carried over from development, not blind predictions. The fresh
  block is the first time they are tested on data that played no part in forming them.
- The method set, the CFAR cell counts (guard 3, training 12), the drift model, the FPR
  budget (0.12) and the detection grids were fixed before the final run.
- The adaptive-decay JOR is a labeled variant. The original JOR is the headline method.

## Protocol (identical for every method)

- Detection settings are chosen on TRAIN only: highest recall subject to FPR ≤ 0.12, ties
  broken by F1, then by the earlier setting in the grid.
- Scoring is scenario-level, first flag only, with a 3-step lag allowance.
- Intervals are 95% Wilson (single proportions) and a paired bootstrap over TEST scenarios
  (2,000 resamples, seed 12345) for differences.
- Two conditions: STATIONARY noise floor, and DRIFTING noise floor (SNR baseline ramps
  linearly by ±U(3, 6) dB per scenario). The drifting condition changes nothing else.

## Claims

| ID | Claim |
|---|---|
| C1 | Stationary: JOR's recall exceeds the CFAR-excess SNR tracker's (paired 95% CI lower bound > 0). |
| C2 | Drifting: JOR's recall exceeds the CFAR-excess SNR tracker's (paired 95% CI lower bound > 0). |
| C3 | Stationary: JOR shows no detectable recall advantage over the E+P average (paired 95% CI includes 0). |
| C4 | Drifting: JOR shows no detectable recall advantage over the E+P average (paired 95% CI includes 0). |
| C5 | Stationary: the absolute-SNR tracker's recall is at least JOR's (point estimate). |
| C6 | Retuned on TRAIN under drift, the absolute-SNR tracker loses more than 0.20 recall relative to its stationary recall. |
| C7 | Settings tuned on the stationary TRAIN set and applied unchanged to drifting TEST data give the absolute-SNR tracker an FPR above 0.25. |
| C8 | In the same carry-over test, the CFAR E+P average keeps FPR at or below 0.15. |
| C9 | Drifting: the adaptive-decay variant does not improve on the original JOR (paired 95% CI upper bound of variant minus original < 0). |
| C10 | Stationary F_SPARSE first-detection recall is below 0.10 for every method. |

## What the claims do not say

- They make no claim about real sensor data. The benchmark is synthetic and the generator
  injects the signals the evidence terms are computed from.
- C3 and C4 are about the absence of a detectable difference. Failing to detect an advantage
  at this sample size does not prove there is none.
