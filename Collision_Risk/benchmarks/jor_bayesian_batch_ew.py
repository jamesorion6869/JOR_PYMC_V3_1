import numpy as np
import time
from numba import njit, prange

# ---------------------------------------------------------------
# Benchmarking Parameters & Constants
# ---------------------------------------------------------------
MAX_PROFILES = 500000
W_C, W_E, W_P = 0.40, 0.30, 0.30
K = 0.20
P_PRIOR_NH = 0.20
P_PRIOR_H = 0.80

EW_JAM_RATE = 0.15
DROPOUT_RATE = 0.15
BENCHMARK_ITERATIONS = 100
WARMUP_ITERATIONS = 10

print("========================================================================")
print("  RUNNING JOR V3.1 MULTI-BATCH SCALABILITY EVALUATION (EWJam + Dropouts)")
print("========================================================================")

np.random.seed(42)

# 1. Allocate full dataset up to the 500,000 row limit
C_base = np.random.uniform(0.30, 0.85, MAX_PROFILES)
E_base = np.random.uniform(0.30, 0.85, MAX_PROFILES)
P_base = np.random.uniform(0.30, 0.95, MAX_PROFILES)
Flight_mod = np.random.choice([0.00, 0.02, 0.04, 0.05], size=MAX_PROFILES, p=[0.4, 0.3, 0.2, 0.1])

# 2. Apply Electronic Warfare and Sensor Dropout masking rules
ew_mask = np.random.random(MAX_PROFILES) < EW_JAM_RATE
adjusted_dropout_rate = DROPOUT_RATE / (1.0 - EW_JAM_RATE)
raw_dropout_mask = np.random.random(MAX_PROFILES) < adjusted_dropout_rate
dropout_mask = raw_dropout_mask & (~ew_mask)

P_base[ew_mask] = 0.95
Flight_mod[ew_mask] = -0.07
C_base[dropout_mask] = 0.00
E_base[dropout_mask] = 0.00
P_base[dropout_mask] = 0.00
Flight_mod[dropout_mask] = 0.00

# 3. Pre-compute full weight arrays
weighted_C = W_C * C_base
weighted_E = W_E * E_base
weighted_P = W_P * P_base

# 4. Full JOR mathematical core function
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

# Compile check to wake up Numba before timing loops start (JIT warm-up)
for _ in range(WARMUP_ITERATIONS):
    _ = bayesian_fusion_core(weighted_C[:10], weighted_E[:10], weighted_P[:10],
                             P_base[:10], Flight_mod[:10], K, P_PRIOR_NH, P_PRIOR_H)

# 5. Iterative variable batch evaluation engine
batch_sizes = range(50000, 500001, 50000)

all_boundary_checks_passed = True

for current_batch in batch_sizes:
    # Dynamically slice all 5 vectors to match current size constraints
    w_C_slice = weighted_C[:current_batch]
    w_E_slice = weighted_E[:current_batch]
    w_P_slice = weighted_P[:current_batch]
    P_base_slice = P_base[:current_batch]
    flight_slice = Flight_mod[:current_batch]
    
    latencies = []
    last_result = None
    for i in range(BENCHMARK_ITERATIONS):
        t_start = time.perf_counter()
        
        # Executes complete Bayesian processing array
        last_result = bayesian_fusion_core(w_C_slice, w_E_slice, w_P_slice, 
                                 P_base_slice, flight_slice, 
                                 K, P_PRIOR_NH, P_PRIOR_H)
        
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)
        
    median_ms = np.median(latencies)
    nanoseconds_per_row = (median_ms * 1_000_000) / current_batch

    # Real boundary validation: no NaN/inf, posterior strictly within [0, 1]
    has_nan_or_inf = not np.all(np.isfinite(last_result))
    in_range = np.all((last_result >= 0.0) & (last_result <= 1.0))
    batch_passed = (not has_nan_or_inf) and in_range
    all_boundary_checks_passed &= batch_passed

    status = "PASS" if batch_passed else "FAIL"
    print(f"Batch Size: {current_batch:7,} | Median Latency: {median_ms:.3f} ms | "
          f"Per-Profile Speed: {nanoseconds_per_row:.2f} ns | "
          f"Range: [{last_result.min():.2f}, {last_result.max():.2f}] | "
          f"Boundary: {status}")

print("==============================================================")
overall_status = "PASS" if all_boundary_checks_passed else "FAIL"
print(f"--> Evaluation Suite Execution: {overall_status}")
print("==============================================================")
