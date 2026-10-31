# JOR — Bayesian Sequential Fusion for Tactical Maneuver Simulation

> **PROJECT UPDATE (JOR 4.0 Advanced C2 Platform)**
> 
> This repository hosts the JOR tactical decision-support framework across two tiers:
> * **[`advanced_c2_demo/`](./advanced_c2_demo/) — Interactive C2 Decision Support Platform (Latest)**  
>   Features a real-time web dashboard (Leaflet spatial heatmaps, MIL-STD-2525 symbols), Server-Sent Events (SSE) telemetry streaming, decision timers, a graded severity gauge, and an automated cause-and-effect post-mission debrief engine.
> * **Root-Level Files — Baseline Sequential Fusion Engine (Original)**  
>   The initial command-line testbed and Matplotlib time-series log analyzer detailed below.

---

## 📁 Repository Directory Structure

```text
jor_tactical_sim/
├── README.md                          # Unified project overview (this file)
├── requirements.txt                   # Core Python dependencies
├── engine.py                          # Initial JOREngine core
├── logger.py                          # JSON-lines telemetry logger
├── maneuver_adapter.py                # Input-to-signal converter (theta_s, theta_c, theta_o)
├── maneuver_sim_interactive.py        # Original interactive CLI driver
├── plot_maneuver_log.py               # Matplotlib log visualizer
├── maneuver_log_logistics_constrained.png # Reference log visualization chart
│
└── advanced_c2_demo/                  # JOR 4.0 Advanced C2 Decision Support Platform
    ├── README.md                      # Detailed platform documentation & image assets
    ├── tactical image.png             # Tactical C2 Dashboard Screenshot
    ├── engine.py                      # Dynamic Bayesian network & state update core
    ├── severity.py                    # Severity band and threshold calculations
    ├── maneuver_adapter.py            # Tactical maneuver dynamics & force coupling
    ├── force_structure.py             # Unit identity, echelon, and role definitions
    ├── theater_layout.py              # Abstract theater spatial initialization
    ├── decision_catalog.py            # Decision prompts, interventions, & options
    ├── map_telemetry.py               # Per-tick frame builder & JSON-lines sink
    ├── map_view.py                    # Geospatial coordinate transforms (WGS84) & MIL-STD-2525 rendering
    ├── stream_server.py               # HTTP Server-Sent Events (SSE) (port 8766) & REST decision server
    ├── maneuver_sim_interactive.py    # Interactive CLI & main engine harness
    ├── headless_runner.py             # Automated headless execution & policy drivers
    ├── run_demo.py                    # One-command synthetic demo launcher
    ├── vignettes.py                   # Pre-scripted operational vignettes & debt modeling
    ├── logger.py                      # Append-only fusion logger
    ├── plot_maneuver_log.py           # Matplotlib telemetry & NHP plot generator
    ├── aar_summary.py                 # After-Action Report metrics harvester
    ├── debrief.py                     # Structured cause-and-effect debrief builder
    ├── dashboard.html                 # Leaflet & MIL-STD-2525 web GUI dashboard
    ├── demo_map.py                    # Visual test harness for geospatial rendering
    ├── test_suite.py                  # Platform unit and integration test suite
    ├── test_regression.py             # Historical regression & parity test runner
    ├── DEMO_SCRIPT.txt                # Presenter talk track & demo instructions
    ├── README_JOR40.txt               # Quickstart CLI reference guide
    └── aar_demo.txt                   # Sample generated after-action summary
```

---

## 🚀 Quick Start

### Running the Advanced C2 Platform (`advanced_c2_demo`)
```bash
cd advanced_c2_demo
pip install -r requirements.txt
python run_demo.py
```
Open `http://127.0.0.1:8766/` to interact with the real-time C2 dashboard.

### Running the Root Baseline Simulator
```bash
pip install -r requirements.txt
python maneuver_sim_interactive.py
```

---

# JOR — Bayesian Sequential Fusion for Wargame-Style Anomaly Detection

JOR is a Bayesian posterior-fusion engine that tracks force health over time from
multiple noisy signals — readiness, operational pressure, and mission outcomes —
and produces a single, continuously updated probability that a force is in a
non-healthy state (NHP: non-healthy posterior).

It's built around a pattern that shows up broadly in sensor fusion and anomaly
detection: rather than alerting on a simple threshold ("if X > 0.5, alert"), it
treats each new reading as evidence, updates a posterior belief with it, and
blends that against prior state with a retention factor. This makes it resistant
to single noisy readings while still able to react quickly to a genuine,
sustained shift — including a catastrophic-override path for deviations large
enough that resisting them further would just delay a real alert.

The wargame scenario (readiness/morale/enemy posture/commander skill) is the
testbed used to exercise the fusion logic under realistic, noisy, multi-factor
conditions — not the point of the project on its own. The interesting part is
the fusion engine and how it behaves under stress, not the scenario dressing.

## How it works

1. **Calibration** — the engine observes the first N ticks of a run and learns
   a baseline "SOP" (signal object processing) score before making any health judgments,
   so the same absolute reading means different things in a high-tempo scenario
   versus a quiet one.
2. **Bayesian update** — each tick's deviation from baseline feeds a logistic
   likelihood, which updates a posterior via Bayes' rule.
3. **Retention blending** — the posterior is blended with its prior state
   (an exponential-smoothing-like retention factor) to avoid single-tick noise
   causing false alarms.
4. **Catastrophic override** — if deviation from baseline crosses a large
   threshold, retention increases sharply so the posterior can saturate quickly
   instead of lagging behind a real collapse.
5. **Hysteresis alerting** — the alert state uses separate upper/lower
   thresholds (rather than one cutoff) to avoid flapping in and out of alert on
   borderline readings.
6. **Near-collapse proximity tracking** — a strict "full collapse" signature
   requires several conditions to align on the exact same tick, which is
   appropriately hard to trigger and avoids false positives. But a strict
   AND-condition can also mask a near miss — a run where 3 of 4 conditions
   aligned and the 4th just missed looks identical to a run where nothing
   happened at all, unless you're also tracking how close it got. This layer
   reports that proximity explicitly instead of collapsing it into a single
   detected/not-detected result.

## Root Baseline Files

| File | Purpose |
|---|---|
| `engine.py` | The core `JOREngine` — calibration, Bayesian update, retention, catastrophic override, hysteresis alerting |
| `maneuver_adapter.py` | Converts raw scenario inputs (unit readiness, pressure, per-unit outcomes) into the three fused signals (theta_s, theta_c, theta_o), including outlier detection via median/IQR + persistent EMA |
| `logger.py` | Appends structured JSON-lines logs of every tick's fusion output and metadata |
| `maneuver_sim_interactive.py` | The interactive scenario driver — scenario presets, commander skill levels, enemy posture model, decision points, mission summary with collapse/near-collapse reporting |
| `plot_maneuver_log.py` | Reads a log file and renders a chart of the fused signals, NHP posterior, hysteresis band, and scripted event markers over time |

## Running the Baseline Simulator

```bash
pip install -r requirements.txt
python maneuver_sim_interactive.py
```

You'll be prompted to pick a scenario (1-5) and, as the run progresses, to make
periodic decisions in response to fatigue/pressure/outcome warnings. At the end
you'll get a mission summary including the collapse signature and near-collapse
proximity report.

To generate a chart from the resulting log:

```bash
python plot_maneuver_log.py maneuver_log_<scenario>.json
```

**Note:** the logger appends to its file rather than overwriting it, so running
the same scenario twice without deleting/renaming the log file in between will
concatenate two runs into one file. `plot_maneuver_log.py` detects this
automatically and plots only the most recent run by default (with a printed
note), but if you want a clean single-run log, delete the file before
re-running.

## Example output

A logistics-constrained run with a scripted coordination failure at tick 110:

![JOR fusion chart](maneuver_log_logistics_constrained.png)

```text
Baseline SOP (t=0-24):  0.355
Final SOP (t=119):      0.932
Final NHP:              0.953
ALERT Duration:         92 steps (76.7%)
Collapse Signature:     FRIENDLY COLLAPSE
Enemy Proximity:        NEAR-COLLAPSE — 2/3 conditions aligned at t=26
```

## What this is, and isn't

This is a working demonstration of Bayesian sequential fusion applied to a
multi-signal health-monitoring problem — calibration, posterior updating,
retention-based smoothing, override behavior for genuine catastrophic
deviations, and hysteresis alerting are all real, functioning components, not
a simplified stand-in for them.

What it is *not* is a calibrated or validated model: the fusion weights
(0.40/0.30/0.30), thresholds (0.50/0.40), and scenario parameters are
illustrative rather than derived from historical data or domain expertise. A
production version of this pattern would calibrate those weights against real
outcome data — the way, for example, defense readiness reporting systems tie
resource and mission-capability ratings back to actual historical performance
rather than hand-set constants. The fusion *architecture* here is the real
contribution; the specific numbers plugged into it are a starting point for
tuning, not a finished result.

[MIT](LICENSE.txt) — Copyright (c) 2026 [Jake James]
