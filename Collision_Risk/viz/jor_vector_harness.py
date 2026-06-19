import numpy as np
import time

WEIGHT_C = 0.4
WEIGHT_E = 0.3
WEIGHT_P = 0.3

def run_optimized_e2e_benchmark():
    print("==========================================================")
    print("JOR V3.1 STAGE 1: SYSTEM BOUNDARY PERFORMANCE LOG")
    print("==========================================================\n")
    
    test_size = 1000000
    c_cap, e_cap, p_cap = 0.50, 0.70, 0.75
    
    # 1. Structured Data Layout Definition
    dtype = [('c', 'f4'), ('e', 'f4'), ('p', 'f4'), ('mod', 'u1'), ('pad', 'u1', 3)]
    
    # Generate clean synthetic telemetry matching the expected memory structure
    np.random.seed(42)
    byte_stream = np.zeros(test_size, dtype=dtype)
    byte_stream['c'] = np.random.uniform(0.3, 0.85, test_size).astype(np.float32)
    byte_stream['e'] = np.random.uniform(0.3, 0.85, test_size).astype(np.float32)
    byte_stream['p'] = np.random.uniform(0.3, 0.95, test_size).astype(np.float32)
    byte_stream['mod'] = np.random.choice([0, 1, 2, 3], test_size).astype(np.uint8)
    
    raw_bytes_in_memory = byte_stream.tobytes()
    
    mod_lookup_table = np.array([0.00, 0.02, 0.04, 0.05], dtype=np.float64)
    
    def run_one_pass():
        # ----------------------------------------------------
        # PHASE A: ZERO-COPY MEMORY REINTERPRETATION & CASTING
        # ----------------------------------------------------
        start_parse = time.perf_counter()
        
        parsed_view = np.frombuffer(raw_bytes_in_memory, dtype=dtype)
        c_bases = parsed_view['c'].astype(np.float64)
        e_bases = parsed_view['e'].astype(np.float64)
        p_bases = parsed_view['p'].astype(np.float64)
        flight_mods = mod_lookup_table[parsed_view['mod']]
        
        end_parse = time.perf_counter()
        parse_time_ms = (end_parse - start_parse) * 1000
        
        # ----------------------------------------------------
        # PHASE B: ALGORITHMIC SCORING & BOUNDARY CLAMPING
        # ----------------------------------------------------
        start_math = time.perf_counter()
        
        c_final = np.round(np.minimum(c_bases, c_cap), 2)
        e_final = np.round(np.minimum(e_bases, e_cap), 2)  
        p_final_raw = np.round(np.minimum(p_bases, p_cap), 2)
        
        p_final_anomalous = np.round(np.minimum(p_final_raw + flight_mods, 0.95), 2)
        
        sop_array = np.round(WEIGHT_C * c_final + WEIGHT_E * e_final + WEIGHT_P * p_final_raw, 2)
        nhp_array = np.round(WEIGHT_C * c_final + WEIGHT_E * e_final + WEIGHT_P * p_final_anomalous, 2)
        
        end_math = time.perf_counter()
        math_time_ms = (end_math - start_math) * 1000
        
        return parse_time_ms, math_time_ms
    
    # ----------------------------------------------------
    # WARMUP PASS (excluded from reported stats)
    # ----------------------------------------------------
    print("[WARMUP] Running 1 untimed warmup iteration to settle caches/allocator...")
    run_one_pass()
    
    # ----------------------------------------------------
    # 10 TIMED ITERATIONS
    # ----------------------------------------------------
    n_iters = 10
    print(f"[TIMED RUNS] Executing {n_iters} timed iterations...\n")
    parse_times = []
    math_times = []
    total_times = []
    for i in range(n_iters):
        p_ms, m_ms = run_one_pass()
        parse_times.append(p_ms)
        math_times.append(m_ms)
        total_times.append(p_ms + m_ms)
        print(f"  Iter {i+1:2d}: parse={p_ms:7.2f} ms | math={m_ms:7.2f} ms | total={p_ms + m_ms:7.2f} ms")
    
    def stats(values):
        return {
            "min": np.min(values),
            "median": np.median(values),
            "mean": np.mean(values),
            "std": np.std(values),
        }
    
    parse_stats = stats(parse_times)
    math_stats = stats(math_times)
    total_stats = stats(total_times)
    
    # ----------------------------------------------------
    # METRIC LOGGING
    # ----------------------------------------------------
    print("\n==========================================================")
    print(f"RESULTS ACROSS {n_iters} TIMED ITERATIONS (1 warmup excluded)")
    print("==========================================================")
    print(f"{'Phase':<20}{'Min (ms)':>12}{'Median (ms)':>14}{'Mean (ms)':>12}{'Std (ms)':>12}")
    print(f"{'Parse':<20}{parse_stats['min']:>12.2f}{parse_stats['median']:>14.2f}{parse_stats['mean']:>12.2f}{parse_stats['std']:>12.2f}")
    print(f"{'Math':<20}{math_stats['min']:>12.2f}{math_stats['median']:>14.2f}{math_stats['mean']:>12.2f}{math_stats['std']:>12.2f}")
    print(f"{'Total':<20}{total_stats['min']:>12.2f}{total_stats['median']:>14.2f}{total_stats['mean']:>12.2f}{total_stats['std']:>12.2f}")
    print()
    print(f"--> Latency Per Profile (median total) : {(total_stats['median'] / test_size) * 1000:.4f} microseconds")
    print(f"--> Latency Per Profile (mean total)   : {(total_stats['mean'] / test_size) * 1000:.4f} microseconds")
    print(f"--> Latency Per Profile (best-case min): {(total_stats['min'] / test_size) * 1000:.4f} microseconds")

if __name__ == "__main__":
    run_optimized_e2e_benchmark()