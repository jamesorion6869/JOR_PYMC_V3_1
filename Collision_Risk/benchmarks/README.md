# JOR v3.1 Benchmarks

This directory contains scripts used to evaluate the performance, scalability, and stress-tolerance of the JOR v3.1 vectorized pipeline and core logic.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18088931.svg)](https://doi.org/10.5281/zenodo.18088931)

### Test Environment:

AMD Ryzen 5 3550H @ 2.10 GHz, 8 GB DDR4 RAM.

### Included Benchmark Scripts

* **`jor_sensor_flight_simulation_njit.py`**: A recursive, time-series simulation of the JOR V3.1 Bayesian Fusion core tracking a single object over a 60-second window (12 steps of 5s each), carrying the posterior forward as each step's prior rather than resetting to the fixed baseline every evaluation. Identifies and corrects a runaway-convergence failure mode in naive recursive updating (via `PRIOR_RETENTION` blending) across four flight profiles (Conventional, Minor/Moderate/Major Anomaly). Numba JIT-compiled, boundary-validated every step, with configurable retention factor for sensitivity analysis.
* **`jor_bayesian_batch_ew.py`**: A multi-batch scalability benchmark for the JOR V3.1 Bayesian Fusion core, sweeping batch sizes from 50,000 to 500,000 simulated profiles under 15% EW jamming and 15% sensor dropout. Numba JIT-compiled (`njit`, `fastmath=True`, `parallel=True`) with 10 warm-up iterations to exclude compilation overhead, reporting median latency, per-profile throughput, and validated posterior boundary integrity ([0,1] range, no NaN/inf) at each scale.
* **`jor_bayesian_resilience_stress_test.py`**: A high-performance benchmarking suite for the JOR V3.1 Bayesian Fusion core. It utilizes Numba's `njit` and `parallel=True` to process 100,000 profiles, measuring latency and verifying mathematical boundary integrity under simulated electronic warfare (EW) jamming and sensor dropout conditions.
* **`jor_v31_integrity_test.py`**: A high-performance stress-test that visualizes the JOR v3.1 pipeline integrity under 30% combined sensor degradation (15% EW jamming and 15% dropout). It employs a dual-panel animation to contrast "Raw Weighted NHP" with "Bayesian Posterior NHP," demonstrating the framework’s conservative bias, resistance to probability inflation in noisy environments, and graceful degradation behavior.
* **`jor_stage1_degraded_test.py`**: Executes an automated stress test simulating degraded airspace environments. It evaluates the core JOR v3.1 logic under conditions of Electronic Warfare (EW) jamming and sensor dropout, verifying that the mathematical boundary integrity and zero-state constraints hold true.
* **`jor_stage1_test.py`**: Provides an automated stress-testing suite for JOR v3.1 Stage 1 core logic. It tests high-volume throughput, kinematic step-function boundary limits, and saturation caps to ensure mathematical containment.
* **`jor_v31_multibatch_benchmark.py`**: Performs vectorized multi-batch scaling benchmarks to assess real-time ingestion viability. This script measures the performance of the full pipeline—including memory decoding, type promotion, normalization, SOP/NHP computation, and Bayesian posterior fusion—across varied batch sizes.
* **`jor_vec_pipeline.py`**: Tests the single-file vectorized pipeline performance, measuring full pipeline execution time (memory decoding, type casting, vector transforms, and JOR core math). It calculates amortized throughput to evaluate the efficiency of the core model embedded within the pipeline.
