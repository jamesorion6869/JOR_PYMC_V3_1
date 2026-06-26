import numpy as np

import time


# ---------------------------------------------------------------

# Benchmarking Parameters & Constants

# ---------------------------------------------------------------

NUM_PROFILES = 100000

W_C, W_E, W_P = 0.40, 0.30, 0.30


# Simulation Tuning: 15% EW Jamming, 15% Sensor Dropout

EW_JAM_RATE = 0.15

DROPOUT_RATE = 0.15

BENCHMARK_ITERATIONS = 100  


print("==============================================================")

print(f"RUNNING JOR V3.1 STAGE 1 DEGRADED AIRSPACE SIMULATION ({NUM_PROFILES} ROWS)")

print("==============================================================")


# 1. Stochastic Generation of Base Scores (0.30 to 0.95)

np.random.seed(42)  

C_base = np.random.uniform(0.30, 0.85, NUM_PROFILES)

E_base = np.random.uniform(0.30, 0.85, NUM_PROFILES)

P_base = np.random.uniform(0.30, 0.95, NUM_PROFILES)

Flight_mod = np.random.choice([0.00, 0.02, 0.04, 0.05], size=NUM_PROFILES, p=[0.4, 0.3, 0.2, 0.1])


# ---------------------------------------------------------------

# Injecting Mutually Exclusive EW and Sensor Dropouts

# ---------------------------------------------------------------

ew_mask = np.random.random(NUM_PROFILES) < EW_JAM_RATE


# Adjusted threshold to ensure final dropout pool hits exactly 15% of total population

adjusted_dropout_rate = DROPOUT_RATE / (1.0 - EW_JAM_RATE)

raw_dropout_mask = np.random.random(NUM_PROFILES) < adjusted_dropout_rate

dropout_mask = raw_dropout_mask & (~ew_mask)  


# Apply EW Jamming: Caps P_base at ceiling, applies modifier degradation

P_base[ew_mask] = 0.95

Flight_mod[ew_mask] = -0.07  


# Apply Sensor Dropouts: Wipes all base metrics completely to floor levels

C_base[dropout_mask] = 0.00

E_base[dropout_mask] = 0.00

P_base[dropout_mask] = 0.00

Flight_mod[dropout_mask] = 0.00


# ---------------------------------------------------------------

# Multi-Iteration Benchmarking Loop (Isolates Pure Throughput)

# ---------------------------------------------------------------

latencies = []


# Warm up execution pass once to clear cold-start cache distortions

_ = (W_C * C_base) + (W_E * E_base) + (W_P * P_base)


for _ in range(BENCHMARK_ITERATIONS):

    t_start = time.perf_counter()


    # Core Math Vectorization Block

    SOP_raw = (W_C * C_base) + (W_E * E_base) + (W_P * P_base)

    P_final = P_base + Flight_mod

    P_final = np.clip(P_final, 0.00, 0.95)  

    NHP_raw = (W_C * C_base) + (W_E * E_base) + (W_P * P_final)


    t_end = time.perf_counter()

    latencies.append((t_end - t_start) * 1000)


# Calculate Median Latency Metrics

median_latency_ms = np.median(latencies)

avg_latency_us = (median_latency_ms * 1000) / NUM_PROFILES


print(f"[TEST 1] Processing {NUM_PROFILES} Degraded Tracking Profiles ({BENCHMARK_ITERATIONS}x Loop)...")

print(f"--> EW Jamming Active On:  {np.sum(ew_mask)} profiles (Target: ~15,000)")

print(f"--> Sensor Dropouts On:   {np.sum(dropout_mask)} profiles (Target: ~15,000)")

print(f"--> Execution Complete over {BENCHMARK_ITERATIONS} runs.")

print(f"--> Median Processing Latency: {median_latency_ms:.2f} ms")

print(f"--> True Latency Per Profile:  {avg_latency_us:.4f} μs\n")


print("[TEST 2] Verifying True Mathematical Boundary Integrity...")

# Explicitly recompute arrays once from static inputs post-loop to decouple from loop leakage

SOP_validate = (W_C * C_base) + (W_E * E_base) + (W_P * P_base)

P_final_validate = np.clip(P_base + Flight_mod, 0.00, 0.95)

NHP_validate = (W_C * C_base) + (W_E * E_base) + (W_P * P_final_validate)


# Asserting on raw unrounded validation arrays to prevent precision leaks

assert np.all(SOP_validate <= 1.0), "SOP raw breached maximum logical bounds!"

assert np.all(NHP_validate <= 0.95), "NHP raw breached hard-cap ceiling boundary!"

assert np.all(NHP_validate >= 0.00), "Math allowed a sub-zero negative drop!"


# Explicit Zero-State Check for Sensor Dropouts

assert np.all(SOP_validate[dropout_mask] == 0.0), "Dropout profile failed to clear baseline SOP to zero!"

assert np.all(NHP_validate[dropout_mask] == 0.0), "Dropout profile failed to clear final NHP to zero!"


print("--> Boundary Status: PASS (All raw precision layers and zero-state constraints verified.)")

print("==============================================================")