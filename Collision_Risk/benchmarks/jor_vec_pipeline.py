import numpy as np
import time

# ============================
# JOR v3.1 CORE CONSTANTS
# ============================
WEIGHT_C, WEIGHT_E, WEIGHT_P = 0.4, 0.3, 0.3
TEST_SIZE = 1_000_000
ITERS = 10

c_cap, e_cap, p_cap = 0.50, 0.70, 0.75

dtype = [('c','f4'),('e','f4'),('p','f4'),('mod','u1'),('pad','u1',3)]

np.random.seed(42)

# ============================
# STATIC MEMORY SETUP
# ============================
data = np.zeros(TEST_SIZE, dtype=dtype)
data['c'] = np.random.uniform(0.3, 0.85, TEST_SIZE).astype(np.float32)
data['e'] = np.random.uniform(0.3, 0.85, TEST_SIZE).astype(np.float32)
data['p'] = np.random.uniform(0.3, 0.95, TEST_SIZE).astype(np.float32)
data['mod'] = np.random.choice([0,1,2,3], TEST_SIZE).astype(np.uint8)

raw = data.tobytes()
mod_lookup = np.array([0.00, 0.02, 0.04, 0.05], dtype=np.float64)

# ============================
# SINGLE EXECUTION FUNCTION
# ============================
def run_once():
    start = time.perf_counter()

    # ----------------------------------------------------
    # PIPELINE LAYER: MEMORY DECODE
    # ----------------------------------------------------
    view = np.frombuffer(raw, dtype=dtype)

    # ----------------------------------------------------
    # PIPELINE LAYER: TYPE PROMOTION (MAJOR COST DRIVER)
    # ----------------------------------------------------
    c = view['c'].astype(np.float64)
    e = view['e'].astype(np.float64)
    p = view['p'].astype(np.float64)
    m = mod_lookup[view['mod']]

    # ----------------------------------------------------
    # PIPELINE LAYER: NORMALIZATION / CLIPPING
    # ----------------------------------------------------
    c = np.minimum(c, c_cap)
    e = np.minimum(e, e_cap)
    p = np.minimum(p, p_cap)

    p_adj = np.minimum(p + m, 0.95)

    # ----------------------------------------------------
    # JOR CORE MODEL (SCORING EQUATIONS ONLY)
    # ----------------------------------------------------
    sop = WEIGHT_C * c + WEIGHT_E * e + WEIGHT_P * p
    nhp = WEIGHT_C * c + WEIGHT_E * e + WEIGHT_P * p_adj

    end = time.perf_counter()
    return (end - start) * 1000  # ms


# ============================
# BENCHMARK LOOP
# ============================
_ = run_once()  # warmup
times = [run_once() for _ in range(ITERS)]

times = np.array(times)

# ============================
# STATS
# ============================
min_v = np.min(times)
med_v = np.median(times)
mean_v = np.mean(times)
std_v = np.std(times)

# ============================
# OUTPUT
# ============================
print("\n==========================================================")
print("JOR V3.1 SINGLE-FILE VECTORIZED PIPELINE BENCHMARK")
print("==========================================================\n")

print("NOTE:")
print("- This benchmark measures FULL pipeline execution:")
print("  memory decode + dtype casting + vector transforms + JOR math")
print("- It is NOT an isolated microbenchmark of the JOR equations alone.\n")

print(f"Min    : {min_v:.2f} ms")
print(f"Median : {med_v:.2f} ms")
print(f"Mean   : {mean_v:.2f} ms")
print(f"Std    : {std_v:.2f} ms")

print("\n----------------------------------------------------------")
print("JOR CORE MODEL (embedded inside pipeline execution)")
print("----------------------------------------------------------")

print("SOP = 0.4C + 0.3E + 0.3P")
print("NHP = 0.4C + 0.3E + 0.3(P + mod)")

print("\n----------------------------------------------------------")
print("DERIVED THROUGHPUT (AMORTIZED OVER BATCH SIZE)")
print("----------------------------------------------------------")

print(f"Batch size: {TEST_SIZE:,}")

print(f"Amortized per profile (median): {(med_v / TEST_SIZE) * 1e6:.2f} ns")
print(f"Amortized per profile (mean)  : {(mean_v / TEST_SIZE) * 1e6:.2f} ns")
print(f"Amortized per profile (best)  : {(min_v / TEST_SIZE) * 1e6:.2f} ns")