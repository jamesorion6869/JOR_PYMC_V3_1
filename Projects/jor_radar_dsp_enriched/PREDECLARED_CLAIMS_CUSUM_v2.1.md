# Pre-declared claims: JOR vs CUSUM study v2.1

Declared before any run on the TEST block. Record the SHA-256 hashes of this file,
`jor_cusum_study_v2.1.py` and `jor_tuned_study_v1.1.py` (LF line endings; see
SHA256SUMS.txt) before the full run. One TEST run only; results reported as-is.

## Design
- Library: `jor_tuned_study_v1.1.py` (locked; SHA-256 0d1843bb...0132, LF), unmodified.
- Script: `jor_cusum_study_v2.1.py` (SHA-256 in SHA256SUMS.txt; printed at the start of the run).
- Seeds: TRAIN 0-999, VAL 10000-10499 (as in the locked study). TEST 80000-80999 (new; never used).
  Development runs used only TRAIN, VAL and TEST-substitute seeds 20000-20999 (`--dev`, `--quick`).
  Earlier TEST blocks 60000-60999 (v1.1) and 70000-70999 (v2.0) are not reused.
- Conditions: stationary noise floor; drifting noise floor (SNR, and the power derived from it, ramp +/-3 to 6 dB over 60 steps).
  Neither condition is named primary; both are reported.
- Methods: JOR (tuned), C/E/P avg (tuned), SNR (CFAR excess), CUSUM on C/E/P, CUSUM-detrended.
  Descriptive rows (reported, not claims): CUSUM on JOR weights; C/E/P + memory (EMA, alpha 0.30).
- All weights, thresholds and CUSUM settings chosen on TRAIN only. FPR budget 0.12.
- Selection rule (only change in method from v2.0): a setting is eligible only if the upper 95%
  Wilson bound of its TRAIN FPR is at or below 0.12. Applied identically to all methods, during
  selection only. Reported FPRs are point estimates.
- All three CUSUMs may fire on level, or on a rise of {0.05, 0.10, 0.20, 0.40} over 4 steps with
  margin {0.5, 1.0} x h (the detector form JOR uses).
- Detection: hit if the alarm falls within +/-3 steps of onset (LAG_ALLOWANCE = 3, all methods).
- Paired bootstrap on TEST, 2,000 resamples of scenarios, 95% percentile interval.

## Claims (recall; "exceeds" means the paired 95% CI lower bound is above 0)
- V1  Stationary: JOR exceeds CUSUM on C/E/P
- V2  Stationary: JOR exceeds CUSUM-detrended
- V3  Stationary: JOR exceeds C/E/P avg (replication of locked claim T3 on fresh seeds)
- V4  Drifting: JOR exceeds CUSUM on C/E/P
- V5  Drifting: JOR exceeds CUSUM-detrended
- V6  Drifting: JOR exceeds C/E/P avg (replication of locked claim T4 on fresh seeds)
- V7  JOR TEST FPR at or below 0.15 in both conditions

V3 and V6 compare the full JOR package (weights, Bayes update, K, prior retention, posterior
smoothing) with averaging. They do not attribute the difference to the Bayes step alone.

## How the descriptive rows will be read (fixed before the run)
- CUSUM on JOR weights: whether CUSUM was understated by running on averaging weights.
- C/E/P + memory: if near JOR, memory/smoothing explains most of JOR's lead; if well above
  C/E/P avg, smoothing explains much of V3/V6. If well below JOR, memory alone does not
  explain the lead, but the Bayes step, K and prior retention (0.70) still cannot be separated,
  because this row has none of them.

## Disclosures
1. Forking paths: the Wilson rule was adopted after v2.0's FPR overshoot (TRAIN 0.115 -> TEST 0.179).
   The CUSUM rise path and both descriptive rows were added after development-seed fairness checks.
2. Same JOR configuration gave TEST FPR 0.119 (60000s, v1.1) and 0.179 (70000s, v2.0):
   block-to-block FPR variation is large. V7 is the claim most exposed to it.
3. With the Wilson rule, chosen settings sit at TRAIN point FPR of about 0.074-0.085 (dev runs),
   not 0.12. All methods are compared at that effective operating point; no ROC or multi-FPR comparison.
4. Track-consistency offset: after DSP, the consistency feature sits at its 0.55 lower clip on about
   92% of steps, and the scenario shift for it reaches the evidence as a clean offset on about 11%.
   Development-seed check (not a result), shift zeroed: stationary JOR-CUSUM recall gap
   +0.031 (vs +0.017); drifting JOR-C/E/P +0.042 [+0.008, +0.076] (vs +0.060). All signs held.
5. Drift moves both SNR and the power derived from it. Development-seed check with undrifted
   power: gaps barely moved (JOR-CUSUM +0.079, JOR-C/E/P +0.057), so drifting results are
   conservative on level. A check, not a fix; the generator is unchanged.
6. P-dominant weights; C is near-constant in this simulator.
7. CUSUM has no weight search of its own (it runs on the tuned C/E/P stream; full search about
   160k evaluations); CUSUM-detrended runs only on C/E/P weights. Tie-breaks differ slightly
   between the library search and the CUSUM grid.
8. Search sizes differ (JOR/C/E/P: 150 weight sets x coarse thresholds; plain CUSUM about 9,600
   settings; detrended about 62,000). A larger TRAIN search can pick settings that do worse on
   TEST than TRAIN suggested, which tends to lower CUSUM's TEST numbers. Not quantified.
9. JOR memory settings (prior retention 0.70, posterior alpha 0.30) fixed at defaults, not tuned.
10. A faster CUSUM evaluator is used for TRAIN search only; each chosen setting is re-scored with
    the locked evaluator and asserted equal.
11. Bootstrap intervals are conditional on the selected settings (no TRAIN-selection variance).
    No multiplicity correction across the 7 claims.
12. VAL is report-only. Latency is max(0, alarm - onset).
13. One synthetic simulator, written by the same author as JOR. Results show soundness on
    synthetic radar data, not performance on live sensors.

All results are reported whether claims hold or not.
