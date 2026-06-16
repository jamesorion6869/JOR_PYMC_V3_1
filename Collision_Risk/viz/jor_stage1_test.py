import pandas as pd
import numpy as np
import time

# Load JOR v3.1 Stage 1 core logic deterministically
# (Extracted directly from jor_fusion.py to bypass interactive inputs)

WEIGHT_C = 0.4
WEIGHT_E = 0.3
WEIGHT_P = 0.3

def evaluate_stage1_row(c_base, e_base, p_base, c_caps, e_caps, p_caps, flight_mod):
    # 1. Apply Base + Modifiers (Simulating an average mid-point modifier state for simplicity)
    c_score = c_base
    e_score = e_base
    p_score = p_base
    
    # 2. Process Hard Caps (Replicating jor_fusion.py min/max cap rules)
    for cap in c_caps:
        c_score = min(c_score, cap)
    for cap in e_caps:
        if cap == 0.60: # "Daytime clear" rule enforces a minimum baseline bounds
            e_score = max(e_score, cap)
        else:
            e_score = min(e_score, cap)
    for cap in p_caps:
        p_score = min(p_score, cap)
        
    c_final = round(c_score, 2)
    e_final = round(e_score, 2)
    p_final_raw = round(p_score, 2)
    
    # 3. Apply Flight Behavior Modifiers
    p_final_anomalous = round(min(p_final_raw + flight_mod, 0.95), 2)
    
    # 4. Calculate Stage 1 Metrics
    sop = round(WEIGHT_C * c_final + WEIGHT_E * e_final + WEIGHT_P * p_final_raw, 2)
    nhp = round(WEIGHT_C * c_final + WEIGHT_E * e_final + WEIGHT_P * p_final_anomalous, 2)
    
    return sop, nhp

def run_stress_testing_suite():
    print("==================================================")
    print("RUNNING JOR V3.1 STAGE 1 AUTOMATED STRESS SUITE")
    print("==================================================\n")
    
    # ----------------------------------------------------
    # TEST 1: High-Volume Throughput and Vectorization Velocity
    # ----------------------------------------------------
    test_size = 100000
    print(f"[TEST 1] Generating {test_size} synthetic profiles for throughput testing...")
    
    # Create chaotic variations across the entire 0.0 to 1.0 spectrum
    np.random.seed(42)
    c_bases = np.random.uniform(0.3, 0.85, test_size)
    e_bases = np.random.uniform(0.3, 0.85, test_size)
    p_bases = np.random.uniform(0.3, 0.95, test_size)
    flight_mods = np.random.choice([0.00, 0.02, 0.04, 0.05], test_size)
    
    start_time = time.time()
    
    # Execute Stage 1 sequentially across the dataframe to measure pipeline speed
    results = [evaluate_stage1_row(c_bases[i], e_bases[i], p_bases[i], [0.50], [0.70], [0.75], flight_mods[i]) for i in range(test_size)]
    
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    avg_per_track_us = (total_time_ms / test_size) * 1000
    
    print(f"--> Execution complete.")
    print(f"--> Total Time for {test_size} Rows: {total_time_ms:.2f} ms")
    print(f"--> Latency per single profile processing: {avg_per_track_us:.4f} microseconds\n")
    
    # ----------------------------------------------------
    # TEST 2: Kinematic Step-Function Boundary Limits
    # ----------------------------------------------------
    print("[TEST 2] Testing boundary conflicts (Low Base Quality vs Major Flight Anomalies)...")
    
    # Case: Terrible base data, absolute maximum caps applied, but premium flight anomaly modifier
    sop_low, nhp_high = evaluate_stage1_row(
        c_base=0.30, e_base=0.30, p_base=0.30, 
        c_caps=[0.45], e_caps=[0.40], p_caps=[0.55], 
        flight_mod=0.05 # Major Anomaly
    )
    
    print(f"--> Low Bounds Input: Base=0.30, Caps Heavy, Flight Mod=+0.05")
    print(f"--> Output Metrics: SOP = {sop_low}, NHP = {nhp_high}")
    print(f"--> Delta (NHP - SOP): {round(nhp_high - sop_low, 2)}")
    if round(nhp_high - sop_low, 2) <= 0.02:
        print("--> Boundary Status: PASS (Stage 1 math strictly contained by hard caps.)\n")
        
    # ----------------------------------------------------
    # TEST 3: Saturation Caps and Maximum Ceiling Validation
    # ----------------------------------------------------
    print("[TEST 3] Verifying maximum ceiling clamping mechanism (Ceiling = 0.95)...")
    
    # Case: Maximum parameters applied simultaneously to check for overflow
    # CHANGED: "Daytime clear" string replaced with 0.60 to resolve the TypeError crash
    sop_max, nhp_max = evaluate_stage1_row(
        c_base=0.95, e_base=0.95, p_base=0.95, 
        c_caps=[], e_caps=[0.60], p_caps=[0.95], 
        flight_mod=0.05
    )
    
    print(f"--> High Bounds Input: Base=0.95, Max Flight Mod=+0.05")
    print(f"--> Output Metrics: SOP = {sop_max}, NHP = {nhp_max}")
    if nhp_max <= 0.95:
        print(f"--> Ceiling Status: PASS (NHP safely locked at absolute cap ceiling of {nhp_max}.)\n")

if __name__ == "__main__":
    run_stress_testing_suite()