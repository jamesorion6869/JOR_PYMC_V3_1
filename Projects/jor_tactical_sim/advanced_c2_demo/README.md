# James Orion Report (JOR 4.0) — Advanced C2 Decision Support Platform

![JOR 4.0 Tactical Simulation Interface](<tactical image.png>)

## Overview

The **James Orion Report (JOR 4.0)** is an advanced Command and Control (C2) tactical maneuver simulation and decision support platform. It integrates dynamic Bayesian network state estimation, Non-Healthy Posterior (NHP) process failure modeling, Signal Object Processing (SOP) metric updates, multi-component force readiness tracking, and geospatial MIL-STD-2525 unit mapping.

### Core Abbreviations & Terminology

* **JOR**: **J**ames **O**rion **R**eport
* **SOP**: **S**ignal **O**bject **P**rocessing (real-time signal processing and telemetry state tracking)
* **NHP**: **N**on-**H**ealthy **P**osterior (Bayesian posterior probability of system degradation or failure)

The platform is designed to:

* **Decouple Simulation State from Rendering:** Run headless simulations while outputting structured per-tick telemetry frames via Server-Sent Events (SSE / HTTP `text/event-stream`) or streaming to JSON-lines files for interactive playback.
* **Support Dynamic Commander Decision Loops:** Prompt human operators or automated agents with decision choices under simulated operational pressure, tracking feedback, fatigue, and outcomes.
* **Provide Full After-Action Debriefs:** Generate automated AAR summaries, statistical degradation breakdowns, and timeline event logs.

## Repository Architecture

```
JOR_PYMC_V3_1/
└── Projects/
    └── jor_tactical_sim/
        └── advanced_c2_demo/
            ├── engine.py                   # Dynamic Bayesian network & state update core
            ├── severity.py                 # Severity band and threshold calculations
            ├── maneuver_adapter.py         # Tactical maneuver dynamics & force coupling
            ├── force_structure.py          # Unit identity, echelon, and role definitions
            ├── theater_layout.py           # Abstract theater spatial initialization
            ├── decision_catalog.py         # Decision prompts, interventions, & options
            ├── map_telemetry.py            # Per-tick frame builder & JSON-lines sink
            ├── map_view.py                 # Geospatial coordinate transforms (WGS84)
            ├── stream_server.py            # HTTP Server-Sent Events (SSE) & REST decision server
            ├── maneuver_sim_interactive.py # Interactive CLI & main engine harness
            ├── maneuver_sim_original.py    # Legacy baseline reference engine
            ├── headless_runner.py          # Automated headless execution & policy drivers
            ├── run_demo.py                 # One-command synthetic demo launcher
            ├── vignettes.py                # Pre-scripted operational vignettes
            ├── logger.py                   # Append-only fusion logger
            ├── plot_maneuver_log.py        # Matplotlib telemetry & NHP plot generator
            ├── aar_summary.py              # After-Action Report metrics harvester
            ├── debrief.py                  # Structured debrief builder
            ├── dashboard.html              # Leaflet & MIL-STD-2525 web GUI dashboard
            ├── demo_map.py                 # Visual test harness for geospatial rendering
            ├── test_suite.py               # Platform unit and integration test suite
            ├── test_regression.py          # Historical regression & parity test runner
            ├── DEMO_SCRIPT.txt             # Presenter talk track & demo instructions
            ├── README_JOR40.txt            # Quickstart CLI reference guide
            ├── aar_demo.txt                # Sample generated after-action summary
            └── tactical image.png          # Reference UI dashboard screenshot
```

## Quickstart & Installation

### 1. Clone the Repository

Clone the main repository and navigate to the `jor_tactical_sim` project directory:

```bash
git clone https://github.com/jamesorion6869/JOR_PYMC_V3_1.git
cd JOR_PYMC_V3_1/Projects/jor_tactical_sim
```

### 2. Install Dependencies

The core platform relies on Python 3.9+ standard libraries alongside `numpy` and `matplotlib`:

```bash
pip install numpy matplotlib
```

## Running the Platform & Dashboard Controls

### Server Ports & Address Bindings

The application hosts the interactive UI dashboard and backend streaming servers locally:
* **Live Streaming Backend (SSE):** Defaults to `http://127.0.0.1:8766/`
* **Static Playback & Offline Web Server:** Defaults to `http://127.0.0.1:8765/`

### Dashboard Interactive Hotkeys

When using `dashboard.html` in the browser, quick diagnostic and UI focus hotkeys are available:
* **`M`**: Toggle **Map Focus** (expands geospatial MIL-STD-2525 theater view to fullscreen).
* **`[`**: Reveal/Focus **Theater Readiness Matrix** (left-hand sidebar unit readiness metrics).
* **`]`**: Reveal/Focus **Decision Support Engine Rail** (right-hand sidebar real-time Bayesian NHP risk indicators).

---

### Option A: Launch the Canonical Live Demo

Launch the interactive web demo featuring real-time SSE streaming telemetry, terminal presenter guidance, and live decision points:

```bash
python run_demo.py
```

Open your browser to `http://127.0.0.1:8766/`, click **Live stream** in the footer, and interact with decision prompts as they appear.

### Option B: Headless Offline Run & AAR Generation

Execute a full headless run with automatic After-Action Report (AAR) text generation while serving the static dashboard on port 8765:

```bash
python run_demo.py --offline
```

### Option C: Playback Existing Telemetry Runs

Host a static HTTP server (`http://127.0.0.1:8765/`) to inspect or replay pre-recorded JSON-lines telemetry runs (`map_frames_baseline.jsonl`):

```bash
python run_demo.py --playback
```

## Operational Scenarios & CLI Commands

You can run custom scenarios directly via `maneuver_sim_interactive.py`:

```bash
# Baseline coastal scenario
python maneuver_sim_interactive.py --scenario baseline --seed 42 --n-units 6 --commander expert

# Urban dense scenario
python maneuver_sim_interactive.py --scenario urban --seed 7 --n-units 6 --commander expert

# Deep maneuver open approaches
python maneuver_sim_interactive.py --scenario deep_maneuver --seed 42 --n-units 6 --commander expert

# Logistics constrained environment
python maneuver_sim_interactive.py --scenario logistics_constrained --seed 7 --n-units 6 --commander expert

# High enemy pressure scenario
python maneuver_sim_interactive.py --scenario high_enemy_pressure --seed 7 --n-units 6 --commander expert
```

## Scripted Operational Vignettes

Execute specific operational stress-test vignettes using `vignettes.py`:

```bash
python vignettes.py run shock
python vignettes.py run tempo
python vignettes.py run early_warning
```

> **Note on Vignette Dynamics:** Running scripted vignettes activates advanced modeling layers—such as carry-over `push_cost` debt and `neglect_cost` friction penalties—which are bypassed during standard scenario executions to simulate severe operational stress.

## Verification & Testing Suite

Run unit tests and regression verification scripts:

```bash
# Full platform unit test suite
python test_suite.py

# Parity & regression test runner vs legacy baseline
python test_regression.py
```

## License

This project is licensed under the [MIT License](LICENSE).
