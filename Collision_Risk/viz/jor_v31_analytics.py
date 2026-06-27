import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_combined_infographic(csv_path, top_n=10, output_image='jor_v31_master_analytics.png'):
    """
    Loads JOR scores, extracts operational and statistical metrics, and builds
    a single, vertically stacked master infographic with the summary report in the middle.
    This script has been added to the Analytical_Deep_Dives.py in /pages.
    """
    try:
        # 1. Load and clean data
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Calculate Interval Width for variance/uncertainty analysis
        df['CI_Width'] = df['CI_97.5%'] - df['CI_2.5%']
        
        total_cases = len(df)
        hazard_counts = df['Hazard_Level'].value_counts()
        avg_overall_risk = df['Aero_Safety_Risk'].mean()
        critical_cases = df[df['Hazard_Level'] == 'Critical']
        elevated_cases = df[df['Hazard_Level'] == 'Elevated']
        
        # Sort for Top Risk Chart
        top_cases = df.sort_values(by='Aero_Safety_Risk', ascending=True).tail(top_n)
        
        # Isolate reliability subsets (Top 5 highest variance and top 5 narrowest variance)
        highest_var = df.sort_values(by='CI_Width', ascending=False).head(5)
        lowest_var = df.sort_values(by='CI_Width', ascending=True).head(5)
        reliability_subset = pd.concat([highest_var, lowest_var]).drop_duplicates().sort_values(by='Posterior_Mean', ascending=True)
        
        # Set clean professional styling
        sns.set_theme(style="whitegrid")
        
        # Grid Setup: Top Chart (Row 0), Summary Panel (Row 1), Reliability Chart (Row 2)
        fig, (ax_chart, ax_text, ax_rel) = plt.subplots(3, 1, figsize=(11, 15.5), 
                                                         gridspec_kw={'height_ratios': [2.5, 0.9, 2.5]})
        
        # --- PANEL 1: Operational Risk Profiles (Top) ---
        bars = ax_chart.barh(top_cases['Case'], top_cases['Aero_Safety_Risk'], 
                             color='#2b5c8f', edgecolor='#1a3a5f', height=0.6)
        ax_chart.set_xlabel('Aerospace Safety Risk Score (0.0 - 1.0)\n(Calculated via: SOP_Mean, Flight_Mod)', 
                            fontsize=11, fontweight='bold', labelpad=8)
        ax_chart.set_title(f'Top {top_n} Highest Aerospace Safety Risk UAP Incidents\nQuantified via Bayesian Inference', 
                           fontsize=13, fontweight='bold', pad=12, color='#111111')
        ax_chart.set_xlim(0, 1.05)
        
        for bar in bars:
            width = bar.get_width()
            ax_chart.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                          va='center', ha='left', fontsize=9, fontweight='bold', color='#333333')
            
        # --- PANEL 2: Integrated Report Summary Block (Middle) ---
        ax_text.axis('off') # Hide lines, grids, ticks
        report_block = (
            "=========================================================================================\n"
            "                          JOR FRAMEWORK: METRIC & SENSOR SYSTEM REPORT\n"
            "=========================================================================================\n"
            f"Total Incidents Evaluated: {total_cases}   |   Overall Average Aerospace Safety Risk: {avg_overall_risk:.3f}\n\n"
            f"Hazard Level Breakdown:    • Critical: {hazard_counts.get('Critical', 0)} cases ({ (hazard_counts.get('Critical', 0)/total_cases)*100:.1f}%)    • Elevated: {hazard_counts.get('Elevated', 0)} cases ({ (hazard_counts.get('Elevated', 0)/total_cases)*100:.1f}%)\n"
            f"Framework Insights:        • Avg SOP (Critical): {critical_cases['SOP_Mean'].mean():.3f}                • Avg SOP (Elevated): {elevated_cases['SOP_Mean'].mean():.3f}\n"
            "========================================================================================="
        )
        ax_text.text(0.5, 0.5, report_block, transform=ax_text.transAxes,
                     fontsize=9.5, fontfamily='monospace', color='#222222',
                     va='center', ha='center',
                     bbox=dict(boxstyle='round,pad=0.6', facecolor='#f4f6f8', edgecolor='#cccccc', alpha=1.0))

        # --- PANEL 3: Statistical Reliability Error-Bars (Bottom) ---
        for idx, row in reliability_subset.iterrows():
            case_name = row['Case']
            p_mean = row['Posterior_Mean']
            lower_err = p_mean - row['CI_2.5%']
            upper_err = row['CI_97.5%'] - p_mean
            current_color = '#c94c4c' if case_name in highest_var['Case'].values else '#2e7d32'
            
            ax_rel.errorbar(p_mean, case_name, xerr=[[lower_err], [upper_err]], 
                            fmt='o', color=current_color, ecolor=current_color, 
                            elinewidth=2.5, capsize=4, markersize=7)
            ax_rel.text(row['CI_97.5%'] + 0.005, case_name, f"Spread: {row['CI_Width']:.3f}", 
                        va='center', ha='left', fontsize=8.5, color='#555555', weight='bold')

        ax_rel.set_xlabel('Latent Anomaly Posterior Probability Mean (with 95% Credible Intervals)', 
                      fontsize=11, fontweight='bold', labelpad=8)
        ax_rel.set_title('MCMC Uncertainty Auditing: Parameter Reliability & Data Stability\n(Isolating High-Variance vs. High-Stability Tracking Profiles)', 
                         fontsize=13, fontweight='bold', pad=12, color='#111111')
        
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='#c94c4c', marker='o', lw=2.5, label='High Variance (Needs Data/Sensor Auditing)'),
                        Line2D([0], [0], color='#2e7d32', marker='o', lw=2.5, label='High Stability (Tight Parameter Convergence)')]
        ax_rel.legend(handles=custom_lines, loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9)
        ax_rel.set_xlim(df['CI_2.5%'].min() - 0.03, df['CI_97.5%'].max() + 0.06)
        
        # 4. Save and export finalized layout image
        plt.tight_layout()
        plt.savefig(output_image, dpi=300)
        print(f"[Success] Reordered master infographic saved to: '{output_image}'")
        
    except FileNotFoundError:
        print(f"[Error] The file '{csv_path}' was not found.")
    except KeyError as e:
        print(f"[Error] Missing expected column in CSV: {e}")

if __name__ == "__main__":
    generate_combined_infographic("jor_scores.csv")
