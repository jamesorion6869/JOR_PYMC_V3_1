import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_buffered_statistical_audit(csv_path, output_plot='buffered_hazard_distribution.png'):
    """
    Ingests JOR telemetry data, establishes a 0.04 Investigation Buffer Zone (0.73-0.77),
    re-triages edge cases, executes an optimized Welch's t-test on the clean groups,
    and exports an advanced distribution visualization documenting structural separation.
    This script has been added to the Analytical_Deep_Dives.py in /pages.
    """
    if not os.path.exists(csv_path):
        print(f"[Error] Target data file '{csv_path}' not found.")
        return

    # 1. Load telemetry source
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Define clean threshold constraints
    lower_buffer = 0.73
    upper_buffer = 0.77

    # 2. Programmatically apply the 3-Tier safety buffer
    df['Optimized_Tier'] = 'Investigation Zone'
    df.loc[df['Aero_Safety_Risk'] < lower_buffer, 'Optimized_Tier'] = 'Elevated'
    df.loc[df['Aero_Safety_Risk'] >= upper_buffer, 'Optimized_Tier'] = 'Critical'

    # Isolate risk score groupings for clean validation math
    critical_group = df[df['Optimized_Tier'] == 'Critical']['Aero_Safety_Risk'].dropna()
    elevated_group = df[df['Optimized_Tier'] == 'Elevated']['Aero_Safety_Risk'].dropna()
    investigation_group = df[df['Optimized_Tier'] == 'Investigation Zone']['Aero_Safety_Risk'].dropna()

    if critical_group.empty or elevated_group.empty:
        print("[Error] Insufficient data groups to run independent t-test verification.")
        return

    # 3. Execute Two-Sample Independent T-Test (Welch's for unequal variances)
    t_stat, p_val = stats.ttest_ind(critical_group, elevated_group, equal_var=False)

    # 4. Print comprehensive structural report to terminal
    print("\n" + "="*70)
    print("         JOR FRAMEWORK: Optimized 2-Tier Collision Risk Index (CRI) with Safety Buffer")
    print("="*70)
    print(f"Clear Critical Tier (≥ {upper_buffer:.2f})  : {len(critical_group)} cases | Mean: {critical_group.mean():.4f} | StdDev: {critical_group.std():.4f}")
    print(f"Clear Elevated Tier (< {lower_buffer:.2f})  : {len(elevated_group)} cases | Mean: {elevated_group.mean():.4f} | StdDev: {elevated_group.std():.4f}")
    print(f"Investigation Buffer ({lower_buffer:.2f}-{upper_buffer:.2f}) : {len(investigation_group)} cases flagged for manual review")
    print("-"*70)
    print(f"Buffered Welch's T-Statistic    : {t_stat:.4f}")
    print(f"Calculated Operational P-Value  : {p_val:.4e}")
    print(f"Statistical Separation Safety   : {'OPTIMAL (Exceeds Criteria)' if p_val < 0.01 else 'MARGINAL'}")
    print("="*70)

    # 5. Output specific files caught in the buffer net for active tracking
    if not investigation_group.empty:
        print("\n[Flagged] The following cases require immediate manual technical audit:")
        buffer_cases = df[df['Optimized_Tier'] == 'Investigation Zone'][['Case', 'Aero_Safety_Risk']]
        for _, row in buffer_cases.iterrows():
            print(f"  • Risk Score: {row['Aero_Safety_Risk']:.3f} -> {row['Case']}")
        print("="*70 + "\n")

    # 6. Generate High-Fidelity Distribution Visualization
    plt.style.use('dark_background')  # Keeps consistency with Command Center aesthetics
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Kernel Density Estimate (KDE) for isolated outer distributions
    sns.kdeplot(critical_group, fill=True, color='#ff453a', alpha=0.35, linewidth=2.5, label='Clear Critical Tier (≥ 0.77)', ax=ax)
    sns.kdeplot(elevated_group, fill=True, color='#ff9f0a', alpha=0.35, linewidth=2.5, label='Clear Elevated Tier (< 0.73)', ax=ax)
    
    # Render the shaded vertical safety channel
    ax.axvspan(lower_buffer, upper_buffer, color='#5856d6', alpha=0.20, 
               linestyle='--', edgecolor='#5856d6', linewidth=1.5, label='Investigation Buffer (0.73 - 0.77)')

    # Differentiated Rug plots to show precise sample cluster density
    sns.rugplot(elevated_group, color='#ff9f0a', alpha=0.8, height=0.04, ax=ax, linewidth=1.2)
    sns.rugplot(critical_group, color='#ff453a', alpha=0.8, height=0.04, ax=ax, linewidth=1.2)
    sns.rugplot(investigation_group, color='#5856d6', alpha=1.0, height=0.06, ax=ax, linewidth=2.0)

    # Explicit textual callout directly inside the shaded channel
    ax.text((lower_buffer + upper_buffer) / 2, ax.get_ylim()[1] * 0.5, "BUFFER\nZONE", 
            color='#5856d6', fontsize=10, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#0a0a0a', edgecolor='none', alpha=0.75))

    # Extended Mathematical Metadata Annotation Block
    stats_box = (
        f"Buffered Welch's $t$-test:\n"
        f"$t$ = {t_stat:.4f}\n"
        f"$p$ = {p_val:.4e}\n\n"
        f"System Breakdown:\n"
        f"• Critical: {len(critical_group)} rows\n"
        f"• Elevated: {len(elevated_group)} rows\n"
        f"• Buffered: {len(investigation_group)} rows"
    )
    ax.text(0.04, 0.96, stats_box, transform=ax.transAxes,
            fontsize=9.5, fontfamily='monospace', color='#ffffff',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#1c1c1e', edgecolor='#444444', alpha=0.9))

    # Axis Formatting
    ax.set_title("Collision Risk Index (CRI) Density Distribution Analysis", fontsize=13, fontweight='bold', pad=18, color='#ffffff')
    ax.set_xlabel("Calculated Aerospace Safety Risk Score (`Aero_Safety_Risk`)", fontsize=10, labelpad=12, color='#cccccc')
    ax.set_ylabel("Probability Density Profile", fontsize=10, labelpad=12, color='#cccccc')
    
    # Intelligently bound x-axis with safety margins
    ax.set_xlim(0.4, 1.1)
    ax.grid(color='#222222', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='#0a0a0a', edgecolor='#444444', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"[Success] Distribution diagnostic plot saved to: '{output_plot}'")

if __name__ == "__main__":
    # Ingest the active version 3.1 spreadsheet
    run_buffered_statistical_audit("jor_scores.csv")
