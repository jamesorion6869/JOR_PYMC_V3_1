# JOR vs Tuned CUSUM on LSS-HSR-L (Exploratory)

## What This Is

Scripts to reproduce the LSS-HSR-L runs: JOR (jor0/jor1/jor2) vs. tuned CUSUM vs. the raw classifier, flagging UAV segments among birds, rotating objects, and cars using Doppler-only inputs.

- **v1:** 24 Doppler-shape features (near ceiling).
- **v2:** 4 features (bandwidth, entropy, peak/median, bins > 0.5).

Rules were written before results:

- `PROTOCOL_hsr_v1.txt`
- `PROTOCOL_hsr_v2_weakP.txt`

## Data (Not Included)

Dataset not included. Download LSS-HSR-L from Science Data Bank:

https://doi.org/10.57760/sciencedb.radars.00063

License: CC BY-NC 4.0

Place the `LSS-HSR-L` folder (containing `air_routes.json`, `train/`, `validation/`, and `overflow/`) next to `run_all.py`.

### Citation

Tian Biao, Chen Junyan, Wan Yanyu, Huang Shilin, Zhang Yue. LSS-HSR-L: Low-Altitude target recognition dataset based on Holographic Staring Radar [DS/OL]. V2. Science Data Bank, 2026. https://doi.org/10.57760/sciencedb.radars.00063

Used under CC BY-NC 4.0. The dataset is not redistributed here; the scripts derive segments and features from it, and no original files were changed.

**LSS-HSR-L:** L-band holographic staring radar; Doppler waterfall and trajectories; urban, airport, and suburban sites in Shenzhen, Changsha, Chongqing, and other cities.

Nine classes in four groups:

- Rotary UAVs (four DJI models)
- Biological (sparrows, flocks, large migratory birds)
- Ground fixed rotating targets
- Cars

**Task:** UAV1-4 vs. Bird, Birds, Wing Bird, Rotate, and Car; route-grouped 5-fold.

## Setup (Anaconda Prompt or PowerShell)

```bash
conda create -n jorhsr python=3.11 -y
conda activate jorhsr
pip install -r requirements.txt
```

## Run

```bash
python run_all.py                    # v2, seeds 7 11 23 (15 folds)
python run_all.py --var v1 --seeds 7 # v1 ceiling run
```

Expect a few minutes per fold on a laptop (about 1-2 hours for all 15).

Finished folds are skipped on rerun.

**Output:**

- `hsr_results_v2_7_11_23.txt` (readable summary)
- Corresponding `.json` file
- Per-fold `.pkl` and log files

## Files

| File | Description |
|---|---|
| `build_routes.py` | Reads the `.mat` files into `routes.pkl` (route-level frames + Doppler spectra). |
| `hsr_prep.py` | Cuts 20-frame segments (max 10 per route) and extracts Doppler-shape features into `segs.pkl`. |
| `hsr_fold.py` | One fold: P model, JOR and CUSUM tuning on train, test scoring (`SEED`, `FOLD`, `VAR` environment variables). |
| `jor_tuned_study_v1.1.py` | JOR library (weight candidates, Wilson interval). |
| `combine_hsr.py` | Pools folds, recall/FPR, paired route-grouped bootstrap. |
| `results/` | Laptop results above, for comparison. |

## Results: v2 Weak Front End

**Seeds:** 7 / 11 / 23

**Metric:** Mean test recall / FPR

Run on a Windows laptop:

- scipy 1.16.3
- scikit-learn 1.8.0

Full summary: `results/hsr_results_v2_7_11_23_laptop.txt` (and `.json`).

### Performance Comparison

| Method | FPR Budget 0.12 | FPR Budget 0.08 | FPR Budget 0.05 |
|---|---:|---:|---:|
| JOR (jor2) | 0.810 / 0.106 | 0.717 / 0.073 | 0.615 / 0.044 |
| Tuned CUSUM | 0.802 / 0.109 | 0.719 / 0.068 | 0.605 / 0.042 |
| Raw classifier | 0.749 / 0.102 | 0.656 / 0.063 | 0.564 / 0.038 |

### Paired JOR2 - CUSUM Recall Gap

95% route-grouped bootstrap confidence interval:

- **FPR 0.12:**
  - Seed 7: +0.011 [-0.001, +0.024]
  - Seed 11: +0.013 [+0.002, +0.025]
  - Seed 23: +0.001 [-0.008, +0.009]

- **FPR 0.08 and 0.05:** Every confidence interval crosses zero.

JOR and CUSUM each add about 5-7 percentage points over the raw classifier; those confidence intervals exclude zero at every budget.

### Summary

JOR matched tuned CUSUM, slightly ahead at the 0.12 false-alarm budget (about +0.8 percentage points; one of three seed confidence intervals excludes zero) and tied at tighter budgets.

**These results are exploratory.**

An earlier run with newer libraries (scipy 1.18, scikit-learn 1.9) gave means within 1 percentage point of these, with the 0.12 gap at about +1.4 percentage points.

Other library versions may shift numbers slightly.

## Earlier v1 Run (24 Features, Seed 7 Only)

**Reported for disclosure.**

The front end was near its ceiling (frame AUC 0.957-0.970).

Recall at FPR budgets 0.12 / 0.08 / 0.05:

| Method | 0.12 | 0.08 | 0.05 |
|---|---:|---:|---:|
| JOR2 | 0.987 | 0.976 | 0.965 |
| CUSUM | 0.983 | 0.983 | 0.970 |
| Raw classifier | 0.976 | 0.965 | 0.949 |

The weaker 4-feature front end (v2) was chosen after v1 hit this ceiling.

That choice is a disclosed forking path; the v2 protocol was written before any v2 results.

## Caveats

- One radar type (L-band holographic staring radar), recorded at several sites in several cities.
- No site or time labels, so a site or session confound between classes cannot be ruled out.
- E uses SNR.
- Seeds reuse the same routes, so they are not independent.
- JOR's realised FPR ran slightly above CUSUM's at 0.08 and 0.05 (below it at 0.12).

**This is not a deployment UAV detector.**

## Earlier Real-Data Run (Not Included in This Folder)

JOR was also compared with tuned CUSUM on the Open Radar moving-target dataset (350 tracks, 50 UAV).

JOR tied tuned CUSUM at moderate false-alarm budgets, and CUSUM led at the tightest budget.

That result is exploratory and limited:

- The front-end classifier was weak.
- It covers a single sensor.
- UAV and vehicle tracks were recorded on different days, so the comparison carries a recording-session confound.

Code for that run is not published here.
