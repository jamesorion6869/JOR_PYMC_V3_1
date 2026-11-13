# JOR C-UAS Sensing-Assurance Monitor: Open Radar Evaluation

This folder replays **real radar tracks** from the Open Radar Initiative dataset through the JOR C-UAS sensing-assurance monitor, with **injected faults**, and scores how well the monitor warns that the detect-and-track picture is becoming unreliable.

JOR is **not** a detector or tracker. It does not detect, classify, or defeat drones. It is an analytical layer that tells an operator, or a higher-level C2 system, when the ability to detect and track is becoming unreliable, and which source is driving that.

The synthetic demo lives in the sibling folder `jor-counter-uas-assurance-demo`. This folder is the real-track evaluation.

> **Status: research prototype.** Real recorded tracks, but **every fault is injected**. The data has no real sensor failures. Nothing here has been validated against real sensor failures or live sensor feeds. Do not use it for operational decisions. See [Limitations](#limitations).

## Contents

1. [What is real and what is not](#what-is-real-and-what-is-not)
2. [Files](#files)
3. [Quick start](#quick-start)
4. [Results](#results)
5. [Compared with a simple per-sensor health check](#compared-with-a-simple-per-sensor-health-check)
6. [Limitations](#limitations)
7. [Data and citation](#data-and-citation)
8. [License](#license)

## What is real and what is not

| | |
|---|---|
| **Real** | Track timestamps and SNR values from the Open Radar Initiative Outdoor Moving Object dataset (350 tracks: vehicle 201, person 52, UAV 50, bicycle 47). |
| **Derived** | Update continuity (from timestamps) and confidence (from SNR), computed over a 40-sample window of each track. |
| **Injected** | Every fault: frozen feed, update loss, continuity/confidence loss, and EW-style noise/RF/load changes. |
| **Assumed** | The 8 "sources" are 8 unrelated tracks standing in for parallel sensors. Sensor-health and load numbers are fixed constants. |

## Files

| File | Purpose |
|---|---|
| `engine.py` | JOR fusion engine: calibration, recursive Bayesian update, state bands, status report |
| `cuas_adapter.py` | C-UAS adapter. This version adds a per-source self-baseline (alpha 0.04, deviation threshold 0.10, blend weight 0.55) to the fleet comparison, so it differs from the demo folder's adapter |
| `logger.py` | Append-only JSONL logger with a `run_id` per run |
| `open_radar_eval_v2.py` | The scored evaluation: 50 paired trials per condition, Wilson 95% intervals. Saves a timestamped text report and CSV on every run |
| `open_radar_demo.py` | One narrated timeline on real tracks, logged and rendered as a PNG and MP4 |
| `cuas_demo.py` | Needed by `open_radar_demo.py` (clock and detail helpers) |
| `viz.py` | Timeline PNG and replay video from the log |
| `moving_target_dataset.npy` | 350 Open Radar tracks trimmed to timestamps and SNR. **CC BY-NC 4.0, not MIT** (see [Data and citation](#data-and-citation)) |
| `open_radar_timeline.png`, `open_radar_replay.mp4` | Example demo outputs (seed 42) |

## Quick start

Python 3 with numpy. matplotlib is needed only for the PNG and video, and ffmpeg only for the MP4.

```
python open_radar_eval_v2.py              # scored evaluation, writes .txt report + .csv
python open_radar_eval_v2.py --sweep      # severity sweep for the drop fault
python open_radar_eval_v2.py --json out.json

python open_radar_demo.py                 # log + PNG + MP4
python open_radar_demo.py --no-video      # log + PNG only
```

Defaults: seed 42, 8 sources, 40-sample window, 50 trials, engine thresholds 0.69/0.59, persistence 4.

When the demo starts, it may print `(calibration suspect)`. This is the engine flagging that the calibration window contained real update gaps in the recorded tracks. It is not a code error.

In the demo timeline, the `src_2`, `src_3` and `src_1` faults are injected. `src_4` is unfaulted. Its flags come from natural gaps in the real recording, and the data cannot tell whether the sensor or the target caused them.

## Results

All faults injected. 50 trials per condition, seed 42.

**Scored evaluation (`open_radar_eval_v2.py`)**

| Condition | Faulty source named | `UNRELIABLE` | Median latency |
|---|---|---|---|
| Clean, 40 steps | healthy source flagged 0/50 | 0/50 | — |
| Frozen feed (`src_2`) | 50/50 | 0/50 (reaches `DEGRADED`) | 1 step |
| Sector jamming (`src_1`) | 50/50 | 5/50 | 5 steps |
| Wide-area jamming (context only) | state left `NOMINAL` 50/50 | 0/50 | — |
| Sector + wide jamming | 50/50 | 50/50 | 5 steps |

The `drop30` and `drop60` conditions in this script **scale** continuity and confidence. They do not remove real updates. A scaled drop is easier to detect (50/50 even at 30%), so those two rows overstate real performance.

**Real update removal (updates deleted from the track)**

| Updates removed | Faulty source named | `UNRELIABLE` | Median latency |
|---|---|---|---|
| 30% | 2/50 | 0/50 | 8.5 steps |
| 60% | 49/50 | 5/50 | 6 steps |
| 80% | 50/50 | 50/50 | 5 steps |

No healthy source was named in any removal condition.

**Long clean run** (50 trials × 400 steps = 20,000 steps): `UNRELIABLE` in 0 steps. `DEGRADED` in 23 steps (2 trials). A healthy source was named in 47 steps (4 trials). Steps are random window draws, not time, so no per-hour false-alarm rate can be stated.

The real-removal, long-clean and comparison results come from companion scripts that are not in this folder.

## Compared with a simple per-sensor health check

The baseline checks each sensor's median continuity against its own history, with the same smoothing and persistence. It has no fleet comparison and no fusion. It was tuned label-free to the same clean false-flag budget as JOR.

| | JOR | Simple check |
|---|---|---|
| Clean: trials with a healthy sensor named | 4/50 | 5/50 |
| 30% removed: named | 2/50 (8.5 steps) | 8/50 (5.5 steps) |
| 60% removed: named | 49/50 (6 steps) | 49/50 (4 steps) |
| 80% removed: named | 50/50 (5 steps) | 50/50 (4 steps) |
| Wrong sensor named during faults | 0/150 | 3/150 |

On single-sensor dropouts the two are roughly tied. The simple check is 1–2 steps faster and caught more 30% dropouts. That is expected, because each of these faults shows up in a sensor's own update stream, which is exactly what a per-sensor check watches.

**What JOR adds**

1. **One trust state for the whole picture** (`NOMINAL` / `DEGRADED` / `UNRELIABLE`), so an operator or C2 gets one answer instead of many sensor flags. *Tested:* 0 `UNRELIABLE` steps in 20,000 clean steps, and 50/50 trials at 80% update loss.
2. **Fleet context.** Each sensor is judged against the others, so an area-wide problem can be told apart from one bad sensor. *Partly tested:* wide-area jamming moved the state in 50/50 trials without blaming a sensor. This was not run against the simple check.
3. **Channel attribution.** It reports which kind of evidence is failing (sensor, load, observation), not only which sensor. *Design only:* this evaluation exercised one channel.
4. **Fewer wrong sensors blamed** during faults (0/150 vs 3/150). *Small sample, so treat this as a hint.*
5. **Explainable log.** Every state change is recorded with its inputs. *Design.*

**What it costs:** 1–2 steps slower to flag, fewer 30% dropouts caught, and more settings to tune.

**Not tested:** deceptive jamming or spoofing (false tracks, range or velocity gate pull-off), where a sensor's own stream looks healthy and only cross-sensor comparison could catch it. That is where JOR's design should matter most.

## Limitations

- **Injected faults only.** The dataset contains no real sensor failures.
- **One radar, 2021 recordings, 8 pseudo-sources.** The sources are unrelated tracks, not parallel sensors watching the same scene.
- **Thresholds tuned on the same dataset** (label-free, on a separate half), so the results are not fully independent.
- **Partial dropouts are weak.** Only 2/50 trials caught at 30% real update loss.
- **A frozen feed reaches `DEGRADED`, not `UNRELIABLE`**, by design of the current engine. Whether stale feeds should force the band is a doctrine decision.
- **Wide-area jamming and load are synthetic** settings, not measured.
- **NHP is an uncalibrated index**, not a probability.
- **Cause cannot be identified.** Low continuity can come from the sensor, the target, or an adversary. The monitor flags unreliable tracking, not why.
- **Lag in both directions.** Detection takes a few steps from onset, and states linger for a few steps after recovery.

## Data and citation

Data: Open Radar Initiative, Outdoor Moving Object dataset, https://github.com/openradarinitiative/open_radar_datasets. Licensed **CC BY-NC 4.0** (attribution, non-commercial). The trimmed `moving_target_dataset.npy` in this folder keeps that license and is not covered by this repository's MIT license.

Cite: D. Gusland, J. M. Christiansen, B. Torvik, F. Fioranelli, S. Z. Gurbuz, M. Ritchie, "Open Radar Initiative: Large Scale Dataset for Benchmarking of micro-Doppler Recognition Algorithms," 2021 IEEE Radar Conference (RadarConf21), pp. 1–6, doi:10.1109/RadarConf2147009.2021.9455239.

## License

Code: [MIT](LICENSE.txt), Copyright (c) 2026 Jake James.
Data: CC BY-NC 4.0, Open Radar Initiative (see above).
