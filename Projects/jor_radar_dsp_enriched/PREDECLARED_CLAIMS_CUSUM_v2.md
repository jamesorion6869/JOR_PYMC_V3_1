# Pre-declared claims: JOR vs CUSUM study v2.0

Declared before any run on the TEST block. Commit this file and
`jor_cusum_study_v2.0.py` (record both SHA-256 hashes) before the full run.

## Design
- Library: `jor_tuned_study_v1.1.py` (locked; SHA-256 0d1843bb...0132 with LF line endings), unmodified.
- Seeds: TRAIN 0-999, VAL 10000-10499 (as in the locked study). TEST 70000-70999 (new; never used before).
- Methods: JOR (tuned), C/E/P avg (tuned), SNR (CFAR excess), CUSUM on C/E/P, CUSUM-detrended.
- All weights, thresholds and CUSUM settings chosen on TRAIN only, FPR budget 0.12.
- CUSUM grids were fixed after development runs on seeds 20000-20999 only.
- Paired bootstrap on TEST, 2,000 resamples of scenarios, 95% percentile interval.

## Claims (recall; "exceeds" means the paired 95% CI lower bound is above 0)
- V1  Stationary: JOR exceeds CUSUM on C/E/P
- V2  Stationary: JOR exceeds CUSUM-detrended
- V3  Stationary: JOR exceeds C/E/P avg (replication of locked claim T3 on fresh seeds)
- V4  Drifting: JOR exceeds CUSUM on C/E/P
- V5  Drifting: JOR exceeds CUSUM-detrended
- V6  Drifting: JOR exceeds C/E/P avg (replication of locked claim T4 on fresh seeds)
- V7  JOR TEST FPR at or below 0.15 in both conditions

All results are reported whether claims hold or not.
