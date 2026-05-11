import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.calibration import calibration_curve

def generate_safety_report(file="jor_scores.csv"):
    # Load the processed data
    df = pd.read_csv(file)
    
    if 'Aero_Safety_Risk' not in df.columns:
        print("Error: Aero_Safety_Risk not found. Run jor_pymc_runner.py first.")
        return

    # 1. Print Executive Summary
    print("\n" + "="*40)
    print("JOR-V3.1 AEROSPACE SAFETY AUDIT")
    print("="*40)
    
    summary = df['Hazard_Level'].value_counts()
    for level in ['Critical', 'Elevated', 'Low']:
        count = summary.get(level, 0)
        print(f"{level:10}: {count} cases identified")
    
    colors = {'Critical': '#d62728', 'Elevated': '#ff7f0e', 'Low': '#2ca02c'}

    # --- CHART 1: COLLISION RISK INDEX (BAR CHART) ---
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values('Aero_Safety_Risk', ascending=False).head(15)
    
    sns.barplot(
        data=df_sorted, 
        x='Aero_Safety_Risk', 
        y='Case', 
        palette=[colors[x] for x in df_sorted['Hazard_Level']]
    )
    
    plt.axvline(0.75, color='red', linestyle='--', label='Critical Threshold')
    plt.axvline(0.45, color='orange', linestyle='--', label='Elevated Threshold')
    
    plt.title("Collision Risk Index (CRI) - Top 15 High-Risk Cases")
    plt.xlabel("Aero Safety Risk Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("jor_safety_audit_cri.png")

    # --- CHART 2: QUADRANT MAP (PHYSICALITY VS RISK) ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='SOP_Mean', y='Aero_Safety_Risk', 
        hue='Hazard_Level', palette=colors, s=100
    )
    plt.axhline(0.75, color='red', linestyle='--', alpha=0.6)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.6)
    
    plt.title("JOR-V3.1 Threat Intelligence: Physicality vs. Risk")
    plt.xlabel("Solid Object Probability (SOP_Mean)")
    plt.ylabel("Aero Safety Risk Score")
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("jor_quadrant_map.png")

    # --- CHART 3: POSTERIOR FOREST PLOT (UNCERTAINTY) ---
    plt.figure(figsize=(10, 6))
    top_10 = df.sort_values('Posterior_Mean', ascending=False).head(10)
    
    # Calculate error bars from 95% Credible Intervals
    y_err = [
        top_10['Posterior_Mean'] - top_10['CI_2.5%'],
        top_10['CI_97.5%'] - top_10['Posterior_Mean']
    ]
    
    plt.errorbar(
        x=top_10['Posterior_Mean'], y=top_10['Case'], 
        xerr=y_err, fmt='o', color='black', capsize=5
    )
    plt.title("Posterior Probability & 95% Credible Intervals (Top 10)")
    plt.xlabel("Probability of Anomalous Origin (NHP)")
    plt.tight_layout()
    plt.savefig("jor_forest_plot.png")

    print("\nAll audit visuals saved successfully (CRI, Quadrant, Forest Plot).")

if __name__ == "__main__":
    generate_safety_report()