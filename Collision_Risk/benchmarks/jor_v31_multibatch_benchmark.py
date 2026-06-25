import time
import numpy as np

# ==========================================================
# JOR v3.1 Vectorized Multi-Batch Scaling Benchmark
# ==========================================================

# JOR Core Constants
WEIGHT_C = 0.40
WEIGHT_E = 0.30
WEIGHT_P = 0.30
K = 0.20  # Bayesian calibration constant

# Benchmark Settings
BATCH_SIZES = [
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
]

ITERS = 5

# Hard Caps
C_CAP = 0.50
E_CAP = 0.70
P_CAP = 0.75

# Packed binary layout
DTYPE = [
    ("c", "f4"),
    ("e", "f4"),
    ("p", "f4"),
    ("mod", "u1"),
    ("pad", "u1", 3),
]

# Flight modifier lookup
MOD_LOOKUP = np.array([0.00, 0.02, 0.04, 0.05], dtype=np.float64)

np.random.seed(42)


# ==========================================================
# Synthetic Batch Generator
# ==========================================================

def make_batch(n):
    data = np.zeros(n, dtype=DTYPE)

    data["c"] = np.random.uniform(0.30, 0.85, n).astype(np.float32)
    data["e"] = np.random.uniform(0.30, 0.85, n).astype(np.float32)
    data["p"] = np.random.uniform(0.30, 0.95, n).astype(np.float32)
    data["mod"] = np.random.choice([0, 1, 2, 3], n).astype(np.uint8)

    return data.tobytes()


# ==========================================================
# One Complete Vectorized Pipeline Pass
# ==========================================================

def run_once(raw):
    start = time.perf_counter()

    # Memory decode
    view = np.frombuffer(raw, dtype=DTYPE)

    # Type promotion
    c = view["c"].astype(np.float64)
    e = view["e"].astype(np.float64)
    p = view["p"].astype(np.float64)
    m = MOD_LOOKUP[view["mod"]]

    # Normalization / clipping
    c = np.minimum(c, C_CAP)
    e = np.minimum(e, E_CAP)
    p = np.minimum(p, P_CAP)
    p_adj = np.minimum(p + m, 0.95)

    # JOR SOP / NHP
    sop = WEIGHT_C * c + WEIGHT_E * e + WEIGHT_P * p
    nhp = WEIGHT_C * c + WEIGHT_E * e + WEIGHT_P * p_adj

    # Bayesian fusion
    pe_nh = nhp
    pe_h = np.minimum(1.0, 1.0 - nhp + K * sop)

    p_nh = 0.20
    p_h = 0.80

    numerator = pe_nh * p_nh
    denominator = numerator + pe_h * p_h

    post_nh = numerator / denominator
    post_h = 1.0 - post_nh

    # Prevent optimization from eliminating work
    _ = (post_nh.mean(), post_h.mean())

    end = time.perf_counter()

    return (end - start) * 1000.0  # milliseconds


# ==========================================================
# Scaling Benchmark
# ==========================================================

def run_scaling_test():

    print("=" * 60)
    print("JOR v3.1 MULTI-BATCH INGESTION SCALING BENCHMARK")
    print("=" * 60)
    print()
    print("Pipeline:")
    print("  • Memory Decode")
    print("  • dtype Promotion")
    print("  • Normalization / Clipping")
    print("  • SOP / NHP Computation")
    print("  • Bayesian Posterior Fusion")
    print()
    print("Synthetic benchmark using the same vectorized")
    print("processing architecture as JOR v3.1.\n")

    for n in BATCH_SIZES:

        raw = make_batch(n)

        # Warm-up pass
        run_once(raw)

        times = np.array([run_once(raw) for _ in range(ITERS)])

        min_time = times.min()
        median_time = np.median(times)
        mean_time = times.mean()
        std_time = times.std()

        print("-" * 60)
        print(f"Batch Size : {n:,}")
        print("-" * 60)

        print(f"Minimum : {min_time:8.2f} ms")
        print(f"Median  : {median_time:8.2f} ms")
        print(f"Mean    : {mean_time:8.2f} ms")
        print(f"Std Dev : {std_time:8.2f} ms")
        print()

        print(f"Best    : {(min_time / n) * 1e6:8.2f} ns/profile")
        print(f"Median  : {(median_time / n) * 1e6:8.2f} ns/profile")
        print(f"Mean    : {(mean_time / n) * 1e6:8.2f} ns/profile")
        print()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    run_scaling_test()