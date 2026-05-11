"""
JOR Bayesian Fusion Framework (v3.1)
Methodology by: James Orion
Implementation: Jake James
Calibration: K=0.20 (Aligned with AARO 2024 Uncertainty Standards)

CONFIGURATION NOTICE: Global constants (PRIOR_NH, CALIBRATION_K, WEIGHTS) are defined at the top of jor_fusion.py. To change them for sensitivity analysis with PyMC sampling, edit jor_fusion.py directly, then run jor_pymc_runner.py from a fresh process/session. Remember to restore the original values after sensitivity analysis is complete.
Interactive tweak (standalone use only): Answer 'y' when prompted to adjust constants at runtime. WARNING: these changes will NOT carry over when running jor_pymc_runner.py afterward.

PRIOR_NH = 0.20
CALIBRATION_K = 0.20 
WEIGHT_C = 0.40 
WEIGHT_E = 0.30 
WEIGHT_P = 0.30
"""

# ---------------------------------------------------------------
# IMPORTANT: Global Constants & JOR Runner Compatibility
# The values below represent the "Conservative Standard" for UAP analysis.
# To restore to default: PRIOR=0.20, K=0.20, Weights=(0.4, 0.3, 0.3)
#
# CONFIGURATION BEHAVIOR:
#
# OPTION 1 - Manual edit (recommended for jor_pymc_runner.py):
#   Edit PRIOR_NH, CALIBRATION_K, WEIGHT_C/E/P directly in this file.
#   Then re-run jor_pymc_runner.py from a fresh process/session.
#
#   jor_pymc_runner.py imports constants from this module at runtime,
#   so file-level edits WILL be used if the script is re-executed cleanly.
#
# OPTION 2 - Interactive tweak (standalone use only):
#   Answer 'y' when prompted to adjust constants at runtime.
#   WARNING: These changes exist only in memory for this session.
#
#   DO NOT use this mode if you plan to run jor_pymc_runner.py afterward,
#   as it will re-import constants from this file and overwrite them.
#
# RECOMMENDATION:
#   Use Option 1 for reproducible runs and sensitivity testing.
# ---------------------------------------------------------------

import csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import subprocess

# -------------------------------
# Colorama Setup (auto-install if missing)
# -------------------------------
try:
    from colorama import init, Fore, Style
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import init, Fore, Style

init(autoreset=True)

# -------------------------------
# Global Parameters / Constants
# Configuration (EDIT HERE ONLY)
# Sensitivity / Calibration Parameters
# -------------------------------
PRIOR_NH = 0.20               # Default prior probability for non-human hypothesis
CALIBRATION_K = 0.20          # Calibration constant for Bayesian fusion
WEIGHT_C = 0.4                # Weight for Witness Credibility
WEIGHT_E = 0.3                # Weight for Environmental / Observation Conditions
WEIGHT_P = 0.3                # Weight for Physical / Sensor Evidence

# -------------------------------
# Helper functions
# -------------------------------
def choose_category(prompt, options):
    print(Fore.CYAN + f"\n{prompt}")
    for i, (key, desc) in enumerate(options.items(), 1):
        print(Fore.YELLOW + f"{i}: {key} - {desc}")
    while True:
        try:
            choice = int(input(Fore.GREEN + "Enter number: "))
            if 1 <= choice <= len(options):
                return list(options.keys())[choice - 1]
        except ValueError:
            pass
        print(Fore.RED + "Invalid input. Please enter a number from the list.")

def yes_no(prompt):
    while True:
        ans = input(Fore.CYAN + prompt + " (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print(Fore.RED + "Please enter 'y' or 'n'.")

def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(Fore.CYAN + prompt))
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(Fore.RED + f"Value must be between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print(Fore.RED + "Invalid input. Please enter a numeric value (e.g., 0.75).")

# -------------------------------
# Rubrics
# -------------------------------
C_base_scores = {
    "Weak": "0.30-0.50: Single untrained civilian; anonymous; no supporting accounts",
    "Moderate": "0.55-0.65: 2–3 civilians OR one trained observer, partial corroboration",
    "Strong": "0.70-0.80: Multiple trained observers OR multiple corroborating civilians",
    "Very Strong": "0.81-0.85: Trained personnel + multiple independent civilian accounts + documentation"
}

C_modifiers = {
    "Independent written reports or time-stamped logs": 0.03,
    "Witnesses from >2 independent positions": 0.02,
    "Witness inconsistencies": -0.03,
    "Known misidentification / unreliable": -0.05
}

C_hard_caps = {
    "Single untrained civilian": 0.50,
    "No trained observer": 0.70,
    "Anonymous witness": 0.45
}

E_base_scores = {
    "Weak": "0.30-0.45: Fog, heavy cloud, night with no illumination, brief duration (<10s)",
    "Moderate": "0.50-0.60: Light cloud, partially obstructed view, nighttime with some illumination, moderate duration (10-30s)",
    "Strong": "0.65-0.85: Clear sky OR controlled environment; multiple viewing angles, long duration (>30s)"
}

E_modifiers = {
    "Multiple vantage points": 0.03,
    "Weather officially documented": 0.02,
    "Object >1 km away": -0.03,
    "Observation <5 seconds": -0.05
}

E_hard_caps = {
    "Heavy fog": 0.40,
    "Nighttime OR Single perspective": 0.70,
    "Daytime clear": 0.60 
}

P_base_scores = {
    "Weak": "0.30-0.45: No physical traces, anecdotal only",
    "Moderate": "0.50-0.65: One sensor type or weak trace evidence",
    "Strong": "0.70-0.85: Two sensor types or confirmed anomalies",
    "Very Strong": "0.86-0.95: Multi-sensor + physical interaction"
}

P_modifiers = {
    "EMP / interference / shutdown": 0.05,
    "Multi-frame imagery / long video": 0.03,
    "Independent lab analysis": 0.02,
    "Ambiguous/poor video quality": -0.05,
    "Inconsistent sensor readings": -0.07,
}

P_hard_caps = {
    "No sensor data": 0.55,
    "Only video": 0.75,
    "Multi-sensor MAX": 0.95
}

Flight_modifiers = {
    "None / Conventional Flight": 0.00,
    "Minor Anomaly": 0.02,
    "Moderate Anomaly": 0.04,
    "Major Anomaly": 0.05
}

Flight_descriptors = {
    "None / Conventional Flight": "Standard flight behavior; within expected aerodynamics",
    "Minor Anomaly": "Slightly unusual maneuvers or speed; could be explainable",
    "Moderate Anomaly": "Clearly abnormal movement, speed, or trajectory; limited explanation",
    "Major Anomaly": "Highly unusual or impossible maneuvers; defies conventional physics"
}

# -------------------------------
# Core Logic
# -------------------------------
def calculate_posterior(SOP, NHP_in, prior_nh=None, calibration_k=None):
    if prior_nh is None: prior_nh = PRIOR_NH
    if calibration_k is None: calibration_k = CALIBRATION_K

    NHP = round(NHP_in, 2)
    p_e_given_nh = NHP
    p_e_given_h = round(max(0, min(1, 1 - NHP + (calibration_k * SOP))), 2)

    numerator = p_e_given_nh * prior_nh
    denominator = (p_e_given_nh * prior_nh) + (p_e_given_h * (1 - prior_nh))

    posterior = round(numerator / denominator, 2) if denominator != 0 else 0
    return SOP, NHP, posterior, p_e_given_nh, p_e_given_h

def score_factor(base_scores, modifiers, hard_caps, name):
    category = choose_category(f"Select {name} base category:", base_scores)
    score = get_float_input(f"Enter numeric base value for {category}: ")
    print(Fore.CYAN + f"\nApply optional modifiers for {name}:")
    for mod, val in modifiers.items():
        if yes_no(f"Apply modifier '{mod}'?"):
            score += val
    for cap, cap_val in hard_caps.items():
        if yes_no(f"Apply hard cap rule '{cap}'?"):
            if cap == "Daytime clear":
                score = max(score, cap_val)   # enforce minimum for daytime clear
            else:
                score = min(score, cap_val)   # enforce maximum for other caps
    return round(score, 2)

def choose_flight_category(prompt, options, descriptors):
    print(Fore.CYAN + f"\n{prompt}\n")
    for i, key in enumerate(options.keys(), 1):
        print(Fore.YELLOW + f"{i}: {key}")
        print(Fore.LIGHTBLACK_EX + f"    ({descriptors[key]})")
    while True:
        try:
            choice = int(input(Fore.GREEN + "Enter number: "))
            if 1 <= choice <= len(options):
                return list(options.keys())[choice - 1]
        except ValueError:
            pass
        print(Fore.RED + "Invalid input. Please enter a number from the list.")

def log_to_csv(case, C, E, P_raw, flight_mod_val, P_final, SOP, NHP, posterior, file="jor_scores.csv"):
    headers = ["Case", "C", "E", "P", "Flight_Mod", "P_final", "SOP", "NHP", "Posterior_NH"]
    file_exists = False
    try:
        with open(file, 'r', encoding="utf-8") as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([case, C, E, P_raw, flight_mod_val, P_final, SOP, NHP, posterior])

# -------------------------------
# Visual Output
# -------------------------------
def plot_probabilities(SOP, NHP, posterior, prior, case_name, show_chart=True):
    labels = ['Prior NH', 'Posterior NH']
    values = [prior, posterior]

    plt.figure(figsize=(5,4))
    bars = plt.bar(labels, values, color=['skyblue', 'orange'])
    plt.ylim(0,1)
    plt.title(f"Prior vs Posterior: {case_name}")
    plt.ylabel("Probability")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{value:.2f}", ha='center', va='bottom', fontsize=10)
    if show_chart:
        plt.show()
    else:
        plt.savefig(f"chart_{case_name.replace(' ','_')}.png")
    plt.close()

# -------------------------------
# Constants Adjustment
# -------------------------------
def tweak_constants():
    global PRIOR_NH, CALIBRATION_K, WEIGHT_C, WEIGHT_E, WEIGHT_P
    print(Fore.CYAN + "\n--- Adjust Global Parameters ---")
    if yes_no(f"Current PRIOR_NH = {PRIOR_NH}. Change it?"):
        PRIOR_NH = get_float_input("Enter new PRIOR_NH (0.0 - 1.0): ", 0.0, 1.0)
    if yes_no(f"Current CALIBRATION_K = {CALIBRATION_K}. Change it?"):
        CALIBRATION_K = get_float_input("Enter new CALIBRATION_K (0.0 - 1.0): ", 0.0, 1.0)

    print(Fore.CYAN + "\nNote: Evidence weights should sum to 1.0.")
    if yes_no(f"Current Weights (C:{WEIGHT_C}, E:{WEIGHT_E}, P:{WEIGHT_P}). Change them?"):
        WEIGHT_C = get_float_input("Enter new WEIGHT_C: ", 0.0, 1.0)
        WEIGHT_E = get_float_input("Enter new WEIGHT_E: ", 0.0, 1.0)
        WEIGHT_P = get_float_input("Enter new WEIGHT_P: ", 0.0, 1.0)

        total = WEIGHT_C + WEIGHT_E + WEIGHT_P
        if abs(total - 1.0) > 0.001:
            print(Fore.RED + f"Warning: Weights sum to {total}. Normalizing to 1.0...")
            WEIGHT_C /= total
            WEIGHT_E /= total
            WEIGHT_P /= total

# -------------------------------
# Main Application
# -------------------------------
def run():
    print(Fore.BLUE + "James Orion Report (JOR) Interactive Scoring Interface")

    print(Fore.GREEN + "\n" + "="*50)
    print(Fore.GREEN + "JOR CONFIG SNAPSHOT (runtime values)")
    print(Fore.GREEN + f"PRIOR_NH        = {PRIOR_NH}")
    print(Fore.GREEN + f"CALIBRATION_K   = {CALIBRATION_K}")
    print(Fore.GREEN + f"WEIGHT_C        = {WEIGHT_C}")
    print(Fore.GREEN + f"WEIGHT_E        = {WEIGHT_E}")
    print(Fore.GREEN + f"WEIGHT_P        = {WEIGHT_P}")
    print(Fore.GREEN + "="*50 + "\n")

    # ---------------------------------------------------------------
    # CONFIGURATION NOTICE (DO NOT REMOVE)
    # ---------------------------------------------------------------
    print(Fore.GREEN + """
CONFIGURATION NOTICE:
Global constants (PRIOR_NH, CALIBRATION_K, WEIGHTS) are defined at the top of jor_fusion.py.
To change them for sensitivity analysis with PyMC sampling, edit jor_fusion.py directly,
then run jor_pymc_runner.py from a fresh process/session.
Remember to restore the original values after sensitivity analysis is complete.

Interactive tweak (standalone use only):
Answer 'y' when prompted to adjust constants at runtime.
WARNING: these changes will NOT carry over when running jor_pymc_runner.py afterward.

REFERENCE (file baseline defaults — NOT runtime values):
These are the values defined in jor_fusion.py source at load time.
See CONFIG SNAPSHOT above for active runtime state.

PRIOR_NH = 0.20
CALIBRATION_K = 0.20
WEIGHT_C = 0.40
WEIGHT_E = 0.30
WEIGHT_P = 0.30
""")

    if yes_no("Do you want to tweak global constants before scoring?"):
        tweak_constants()

    show_charts = yes_no("Do you want to display charts interactively? (No = save only)")

    while True:
        print(Fore.MAGENTA + "\n" + "="*40)
        case = input(Fore.CYAN + "Enter case name: ").strip()

        C = score_factor(C_base_scores, C_modifiers, C_hard_caps, "Witness Credibility")
        E = score_factor(E_base_scores, E_modifiers, E_hard_caps, "Environmental Conditions")
        P_raw = score_factor(P_base_scores, P_modifiers, P_hard_caps, "Physical Evidence")

        flight_category = choose_flight_category("Select Flight Behavior Classification:", Flight_modifiers, Flight_descriptors)
        flight_mod_val = Flight_modifiers[flight_category]
        P_final = round(min(P_raw + flight_mod_val, 0.95), 2)

        SOP_val = round(WEIGHT_C*C + WEIGHT_E*E + WEIGHT_P*P_raw, 2)
        NHP_val = round(WEIGHT_C*C + WEIGHT_E*E + WEIGHT_P*P_final, 2)
        SOP, NHP, posterior, pnh, ph = calculate_posterior(SOP_val, NHP_val)

        log_to_csv(case, C, E, P_raw, flight_mod_val, P_final, SOP, NHP, posterior)

        print(Fore.YELLOW + "\n" + "-"*15 + " RESULTS " + "-"*15)
        print(Fore.CYAN + f"Case:             {case}")
        print(Fore.CYAN + f"SOP (Baseline):   {SOP}")
        print(Fore.CYAN + f"NHP (Anomalous):  {NHP}")
        print(Fore.CYAN + f"P(E|NH):          {pnh}")
        print(Fore.CYAN + f"P(E|H):           {ph}")
        print(Fore.CYAN + f"Posterior NH:     {posterior}")
        print(Fore.YELLOW + "-" * 39)

        plot_probabilities(SOP, NHP, posterior, PRIOR_NH, case, show_chart=show_charts)

        if not yes_no("Score another case?"):
            print(Fore.GREEN + "Exiting. Results saved to 'jor_scores.csv'.")
            break

if __name__ == "__main__":
    run()
