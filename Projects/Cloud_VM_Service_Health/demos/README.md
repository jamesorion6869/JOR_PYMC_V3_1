# Real-Time Local Service Health Demos

This folder contains four live demonstrations of the **JOR 4.0 Cloud VM Service Health Fusion Engine** operating on *genuine* latency measurements. Unlike the synthetic harnesses in `tests/`, these scripts launch multiple lightweight HTTP instances on your local machine and measure **actual round-trip latency** using Python's standard library (plus NumPy).

No cloud account and no external services. The traffic, the latency, and the jitter are real; the degradation is a deliberate delay injected into one instance so the detector has something to find.

| Script | What it is |
|---|---|
| `real_time_service_test.py` | The original command-line test. Latency only (θₒ); θₛ and θ_c are fixed at 0.2. |
| `real_time_service_gui.py` | The same test plus a live status window, optional live laptop metrics (memory, CPU, disk) feeding θₛ / θ_c, and CLI labels that match the engine state. |
| `speed_test_service.py` | Speed test of the monitor itself: stage-by-stage timings (fusion step, adapter, laptop sampler, full cycle, logging) plus one real HTTP measurement round. Writes a results image and JSON. |
| `speed_test_service_scaling.py` | Fleet-size scaling test: compute cost of one monitor cycle for fleets of 5 up to 20,000 instances, and whether the one slow instance is still caught at every size. Compute only; opens no ports. Writes a chart and JSON. |

---

## What These Tests Demonstrate

The real-time harness validates the full end-to-end behavior of the Cloud VM Service Health pipeline:

- Live HTTP request latency ingestion
- Per-instance feature extraction
- Outlier detection and persistent-outlier tracking
- Bayesian fusion via `engine.py`
- Real-time alerting and recovery
- Continuous logging to `real_time_service_log.json`

It is the closest approximation to a real cloud fleet you can run locally.

---

## Important: `real_time_service_test.py` Exercises Only θₒ

The original harness intentionally focuses on **θₒ (observed service-health evidence)** — the latency-based signal produced by `ServiceHealthAdapter`.

It does **not** incorporate:

- **θₛ (system-state evidence)**
- **θ_c (operational-context evidence)**

Both are set to fixed constants in the script:

```python
theta_s = 0.2
theta_c = 0.2
```

`real_time_service_gui.py` can replace these constants with live readings from the machine running the demo (see below).

---

## JOR 4.0 Real-Time Service Health Demo (`real_time_service_gui.py`)

The intended integrated shape is a lightweight middleware daemon for existing telemetry, intended to reduce false alarms and catch single-instance gray failures early. This demo is a local proof of concept of the detection logic. It is not that daemon, and it does not measure real-world false-alarm reduction.

- Runs 5 real local HTTP instances and measures actual latency
- Gray failure inserted on `inst_2` at t = 25 s (900 ms delay)
- Gray failure ends at t = 45 s (`inst_2`)
- `inst_2` returns to normal latency
- Feeds live data into the JOR fusion engine + per-instance gray-failure detector
- Detects one slow instance (zero errors) via outlier EMA + IQR
- Self-calibrating baseline + hysteresis alert (NHP)
- Optional live laptop metrics (memory, CPU, disk) → θₛ / θ_c
- Live GUI: status banner, evidence chart, instance bars, system gauges
- CLI phase labels match engine state (CALIBRATING / HEALTHY / WATCH / DEGRADED)
- Latency baseline set to 8 ms

### What the window shows

- **Status banner** — what the *engine* concluded, not what the test injected:

  | State | Meaning |
  |---|---|
  | `CALIBRATING` | The engine is still learning its baseline (first 20 one-second steps). Alerts are suppressed. |
  | `HEALTHY` | Calibrated, no alert, no flagged instance. |
  | `WATCH` | An instance is flagged as a persistent outlier, or NHP is above the clear line (0.45), but the alert has not been raised. |
  | `DEGRADED` | The engine's hysteresis alert is active (raised above NHP 0.55, cleared below 0.45). |

- **Evidence chart** — NHP, θₒ, θₛ and θ_c over time, with the alert and clear thresholds. The calibration period and the injected fault window are shaded, so detection lag is visible.
- **Per-instance latency bars** — median latency per instance on a log scale, shown red and labelled `FLAGGED` when the adapter flags that instance.
- **Laptop readings** — memory, disk activity and CPU gauges, with the resulting θₛ / θ_c.
- **Main driver line** — which evidence channel is pushing the fused score furthest above its calibrated level.

The same state labels are printed in the terminal and written to the log file, so the CLI output and the window always agree.

### Evidence channels

| Channel | Source |
|---|---|
| θₒ | Measured latency (and errors) of the 5 local instances — always live. |
| θₛ | Laptop memory % and estimated disk activity, through `adapter.normalize_system_state`. Needs `psutil`. |
| θ_c | Laptop CPU load, through `adapter.normalize_context(cpu%, 100)`. Needs `psutil`. |

Without `psutil` (or with `--no-system`), θₛ and θ_c stay fixed at 0.2, exactly like the original test.

---

## Running

**Requirements**

- Python 3 and `numpy`
- `psutil` — *optional*, for the live laptop readings (`pip install psutil`)
- `tkinter` — for the window. It ships with the standard Python installers on Windows and macOS. On Debian/Ubuntu: `sudo apt install python3-tk`. Without it, the script falls back to command-line output only.

**From the `Cloud_VM_Service_Health` folder** (the one containing `engine.py`), tell Python where to find the modules:

```bash
# macOS / Linux
PYTHONPATH=. python demos/real_time_service_gui.py
PYTHONPATH=. python demos/real_time_service_test.py
```

```powershell
# Windows PowerShell
$env:PYTHONPATH = "."
python demos\real_time_service_gui.py
python demos\real_time_service_test.py
```

```bat
:: Windows cmd
set PYTHONPATH=.
python demos\real_time_service_gui.py
```

Alternatively, copy the script next to `engine.py`, `service_adapter.py` and `logger.py` and run it there with no extra setup.

Close the window or press `Ctrl+C` in the terminal to stop. Ports 9001–9005 must be free.

**Options for `real_time_service_gui.py`**

| Option | Effect |
|---|---|
| `--headless` | Command-line output only, no window. |
| `--duration N` | Stop automatically after N seconds. |
| `--no-system` | Do not read laptop metrics; keep θₛ = θ_c = 0.2. |
| `--no-warmup` | Skip the throwaway warm-up round (reproduces the original script's cold-start behavior). |
| `--calibration-steps N` | Number of calibration steps (default 20). |
| `--bad-after S` / `--recover-after S` | When the injected fault starts and ends, in seconds (defaults 25 and 45). |

**Speed tests**

```bash
PYTHONPATH=. python demos/speed_test_service.py
```

It starts the 5 local test servers itself for the live HTTP round, so ports 9001–9005 must be free (`--no-live` skips that round). It imports `real_time_service_gui.py`, so run it with the same `PYTHONPATH` setup as the other scripts. `psutil` (laptop sampler stage) and `matplotlib` (results image) are optional. Timings are single-thread, offline concept-demo numbers, not a real-time or production claim.

**Scaling test**

```bash
PYTHONPATH=. python demos/speed_test_service_scaling.py
```

Times one monitor cycle (per-instance scoring, θₛ / θ_c normalization and the fusion step) on pre-generated data for fleets of 5, 50, 500, 5,000 and 20,000 instances, and checks that the one slow instance is still flagged at every size (`--sizes` changes the fleet sizes, `--samples` the latency samples per instance, default 8). It needs only `engine.py` and `service_adapter.py`, opens no ports, and measures the monitor's compute only: collecting telemetry from a real fleet is not included. `matplotlib` is optional (chart). Cost grows roughly in a straight line with fleet size; the figures are single-thread, concept-demo timings, not a production claim.

---

## Things to Know

- **Let the machine idle for the first ~20 seconds.** The baseline is learned from the first calibration steps. The GUI script makes one throwaway warm-up round first so that cold-start latency does not land in the baseline.
- **Recovery is not instant.** After `inst_2` recovers, it stays flagged and the alert stays up for roughly 15 seconds in the default run. The per-instance outlier score is an exponential moving average and has to decay; this is a side effect of that smoothing, not a configurable hold-down.
- **CPU alone can raise an alert.** θ_c carries a 0.30 weight in the fused score, so sustained high CPU on the machine running the demo can trigger `DEGRADED` with zero flagged instances.
- **Disk activity is a rough estimate** derived from OS disk counters and may stay near zero on some systems.
- **The 8 ms latency baseline** in the GUI script matches the healthy latency of these local instances (about 5–10 ms). `real_time_service_test.py` uses the adapter default (50 ms). With the lower baseline, a single instance may occasionally be flagged for a step or two on a busy machine.

---

## Output Files

`real_time_service_test.py` and `real_time_service_gui.py` append to `real_time_service_log.json` in the directory you run them from (one JSON record per step). The GUI script also records the phase label, θₛ / θ_c and the raw laptop readings in each record. `speed_test_service.py` writes `speed_test_service_results.json` and `speed_test_service_results.png` there instead, and `speed_test_service_scaling.py` writes `speed_test_service_scaling_results.json` and `speed_test_service_scaling_results.png`. All of these are run output and do not need to be committed.
