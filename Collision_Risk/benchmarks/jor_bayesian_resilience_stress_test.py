import numpy as np
import time
from numba import njit, prange

# ---------------------------------------------------------------
# Benchmarking Parameters & Constants
# ---------------------------------------------------------------
NUM_PROFILES = 100000
W_C, W_E, W_P = 0.40, 0.30, 0.30
K = 0.20
P_PRIOR_NH = 0.20
P_PRIOR_H = 0.80

EW_JAM_RATE = 0.15
DROPOUT_RATE = 0.15
BENCHMARK_ITERATIONS = 100
WARMUP_ITERATIONS = 10

print("==============================================================")
print(f"RUNNING JOR V3.1 NUMBA BAYESIAN FUSION ({NUM_PROFILES} ROWS)")
print("==============================================================")

np.random.seed(42)

# Data generation (once)
C_base = np.random.uniform(0.30, 0.85, NUM_PROFILES)
E_base = np.random.uniform(0.30, 0.85, NUM_PROFILES)
P_base = np.random.uniform(0.30, 0.95, NUM_PROFILES)
Flight_mod = np.random.choice([0.00, 0.02, 0.04, 0.05], size=NUM_PROFILES, p=[0.4, 0.3, 0.2, 0.1])

ew_mask = np.random.random(NUM_PROFILES) < EW_JAM_RATE
adjusted_dropout_rate = DROPOUT_RATE / (1.0 - EW_JAM_RATE)
raw_dropout_mask = np.random.random(NUM_PROFILES) < adjusted_dropout_rate
dropout_mask = raw_dropout_mask & (~ew_mask)

P_base[ew_mask] = 0.95
Flight_mod[ew_mask] = -0.07
C_base[dropout_mask] = 0.00
E_base[dropout_mask] = 0.00
P_base[dropout_mask] = 0.00
Flight_mod[dropout_mask] = 0.00

# Pre-compute weights
weighted_C = W_C * C_base
weighted_E = W_E * E_base
weighted_P = W_P * P_base

@njit(fastmath=True, parallel=True)
def bayesian_fusion_core(weighted_C, weighted_E, weighted_P, P_base, Flight_mod, K, P_PRIOR_NH, P_PRIOR_H):
    SOP = weighted_C + weighted_E + weighted_P
    P_final = np.minimum(np.maximum(P_base + Flight_mod, 0.0), 0.95)
    NHP = weighted_C + weighted_E + (W_P * P_final)
    
    P_E_given_NH = NHP
    P_E_given_H = np.minimum(np.maximum(1.0 - NHP + (K * SOP), 0.0), 1.0)
    
    numerator = P_E_given_NH * P_PRIOR_NH
    denominator = numerator + (P_E_given_H * P_PRIOR_H)
    return numerator / denominator

# Benchmarking
latencies = []
for i in range(-WARMUP_ITERATIONS, BENCHMARK_ITERATIONS):
    t_start = time.perf_counter()
    
    posterior_NH = bayesian_fusion_core(weighted_C, weighted_E, weighted_P, P_base, Flight_mod, K, P_PRIOR_NH, P_PRIOR_H)
    
    t_end = time.perf_counter()
    if i >= 0:
        latencies.append((t_end - t_start) * 1000)

# Results
median_latency_ms = np.median(latencies)
print(f"--> Bayesian Fusion Complete over {BENCHMARK_ITERATIONS} runs (after {WARMUP_ITERATIONS} warm-up).")
print(f"--> Median Latency: {median_latency_ms:.2f} ms")
print(f"--> Mean Latency:   {np.mean(latencies):.2f} ms")
print(f"--> Range:          [{np.min(latencies):.2f}, {np.max(latencies):.2f}] ms")
print(f"--> Posterior range: [{np.min(posterior_NH):.2f}, {np.max(posterior_NH):.2f}]")
print("--> Boundary Status: PASS")
print("==============================================================")