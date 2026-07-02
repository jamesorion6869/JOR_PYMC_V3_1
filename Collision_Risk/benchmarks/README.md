# JOR v3.1 Benchmarks

This directory contains scripts used to evaluate the performance, scalability, and stress-tolerance of the JOR v3.1 vectorized pipeline and core logic.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18088931.svg)](https://doi.org/10.5281/zenodo.18088931)

### Test Environment:

AMD Ryzen 5 3550H @ 2.10 GHz, 8 GB DDR4 RAM.

### Included Benchmark Scripts

* **`jor_stage1_degraded_test.py`**: Executes an automated stress test simulating degraded airspace environments. It evaluates the core JOR v3.1 logic under conditions of Electronic Warfare (EW) jamming and sensor dropout, verifying that the mathematical boundary integrity and zero-state constraints hold true.
* **`jor_stage1_test.py`**: Provides an automated stress-testing suite for JOR v3.1 Stage 1 core logic. It tests high-volume throughput, kinematic step-function boundary limits, and saturation caps to ensure mathematical containment.
* **`jor_v31_multibatch_benchmark.py`**: Performs vectorized multi-batch scaling benchmarks to assess real-time ingestion viability. This script measures the performance of the full pipeline—including memory decoding, type promotion, normalization, SOP/NHP computation, and Bayesian posterior fusion—across varied batch sizes.
* **`jor_vec_pipeline.py`**: Tests the single-file vectorized pipeline performance, measuring full pipeline execution time (memory decoding, type casting, vector transforms, and JOR core math). It calculates amortized throughput to evaluate the efficiency of the core model embedded within the pipeline.
