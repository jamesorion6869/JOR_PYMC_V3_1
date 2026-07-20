# Real‑Time Local Service Health Test

This module provides a fully live demonstration of the **JOR 4.0 Cloud VM Service Health Fusion Engine** operating on *genuine* latency measurements. Unlike the synthetic harnesses in `tests/`, this script launches multiple lightweight HTTP instances on your local machine and measures **actual round‑trip latency** using Python’s standard library.

No cloud account, no external dependencies, and no simulation — this is real traffic, real jitter, and real degradation.

---

## What This Test Demonstrates

This real‑time harness validates the full end‑to‑end behavior of the Cloud VM Service Health pipeline:

- Live HTTP request latency ingestion  
- Per‑instance feature extraction  
- Outlier detection and persistent‑outlier tracking  
- Bayesian fusion via `engine.py`  
- Real‑time alerting and recovery  
- Continuous logging to `real_time_service_log.json`  

It is the closest approximation to a real cloud fleet you can run locally.

---

## Important: This Test Exercises Only θₒ

The real‑time harness intentionally focuses on **θₒ (operational evidence)** — the latency‑based signal produced by `ServiceHealthAdapter`.

It does **not** incorporate:

- **θₛ (sensor evidence)**  
- **θ_c (contextual evidence)**  

Both are set to fixed constants in the script:

```python
theta_s = 0.2
theta_c = 0.2
