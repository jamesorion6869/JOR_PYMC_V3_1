import os

# Aggressive fix for OpenMP warnings - MUST come before any other imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import warnings
import pandas as pd

# 1. IMPORT the engine function
from jor_pymc import run_jor_pymc_safe

# 2. IMPORT the tuned constants from your fusion script
# This ensures that if you changed PRIOR_NH or WEIGHTS in the UI,
# the runner sees those changes before starting the sampler.
from jor_fusion import PRIOR_NH, CALIBRATION_K, WEIGHT_C, WEIGHT_E, WEIGHT_P


def main():

    # Suppress warnings in the main thread
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

    input_csv = "jor_scores.csv"

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Please run jor_fusion.py first to score cases.")
        return

    df_original = pd.read_csv(input_csv)

    # Prepare the data for the PyMC model
    df_for_pymc = pd.DataFrame({
        'case_name': df_original['Case'],
        'C_score': df_original['C'],
        'E_score': df_original['E'],
        'P_score': df_original['P'],
        'flight_mod': df_original['Flight_Mod']
    })

    print("--- JOR-V3 Sampling Session ---")
    print(f"Prior NH: {PRIOR_NH}")
    print(f"Calibration K: {CALIBRATION_K}")
    print(f"Weights: C={WEIGHT_C}, E={WEIGHT_E}, P={WEIGHT_P}")
    print("Starting vectorized PyMC sampling...")

    # 3. CALL the engine with the passed parameters
    results_df = run_jor_pymc_safe(
        df_for_pymc,
        prior_mu=PRIOR_NH,
        k_val=CALIBRATION_K,
        weights=[WEIGHT_C, WEIGHT_E, WEIGHT_P],
        chains=4,
        cores=4,
        target_accept=0.95,
        draws=1000,
        tune=1000
    )

    # Merge results back into the original dataset
    df_merged = df_original.merge(
        results_df,
        left_on="Case",
        right_on="case_name",
        how="left"
    )
    df_merged.drop(columns=['case_name'], inplace=True)

    # Round results for the final CSV save
    decimal_cols = ['Posterior_Mean', 'CI_2.5%', 'CI_97.5%']
    for col in decimal_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].round(3)

    df_merged.to_csv(input_csv, index=False)
    print(f"\nSuccess! Bayesian inference complete. CSV updated: {input_csv}")


if __name__ == "__main__":
    main()