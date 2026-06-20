import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

# Match the dark-theme terminal aesthetics of the Command Center
st.set_page_config(layout="wide", page_title="JOR-V3.1 Analytics Console")

# Establish core tactical color constants
DARK_BG = "#0b0b0d"
PANEL_BG = "#111114"
TEXT_WHITE = "#ffffff"
TEXT_MUTED = "#aaaaaa"
BORDER_GREY = "#2d2d33"

# Apply global dark styling templates to matplotlib
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': PANEL_BG,
    'text.color': TEXT_WHITE,
    'axes.labelcolor': TEXT_WHITE,
    'xtick.color': TEXT_WHITE,
    'ytick.color': TEXT_WHITE,
    'grid.color': BORDER_GREY
})

DATA_FILE = "jor_scores.csv"

def load_live_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Target data source '{file_path}' not found. Please verify the pipeline path.")
        st.stop()
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

# Header Context Ribbon - Refined for clear risk management framing
st.title("JOR-V3.1 Aerospace Hazard Triage & Analytical Console")
st.caption("Post-Event Evidentiary Auditing, Parameter Convergence & Threat-Tier Separation")
st.markdown("---")

# Read case profile data from the evidentiary pipeline
df = load_live_data(DATA_FILE)

# Establish distinct tabs for each report module
tab_infographic, tab_separation = st.tabs([
    "Master Infographic Pipeline", 
    "T-Tier Buffer Separation"
])

# --- TAB 1: MASTER INFOGRAPHIC PIPELINE ---
with tab_infographic:
    st.subheader("Integrated Evidentiary Matrix & Hazard Index Report")
    
    # User-adjustable slider for display depth
    top_n = st.slider("Select Display Range for Top Risks:", min_value=5, max_value=20, value=10)
    
    # 1. Math Prep
    df_infra = df.copy()
    df_infra['CI_Width'] = df_infra['CI_97.5%'] - df_infra['CI_2.5%']
    
    total_cases = len(df_infra)
    hazard_counts = df_infra['Hazard_Level'].value_counts()
    avg_overall_risk = df_infra['Aero_Safety_Risk'].mean()
    critical_cases = df_infra[df_infra['Hazard_Level'] == 'Critical']
    elevated_cases = df_infra[df_infra['Hazard_Level'] == 'Elevated']
    
    top_cases = df_infra.sort_values(by='Aero_Safety_Risk', ascending=True).tail(top_n)
    
    highest_var = df_infra.sort_values(by='CI_Width', ascending=False).head(5)
    lowest_var = df_infra.sort_values(by='CI_Width', ascending=True).head(5)
    reliability_subset = pd.concat([highest_var, lowest_var]).drop_duplicates().sort_values(by='Posterior_Mean', ascending=True)
    
    # 2. Render Canvas with sharp dark framing
    fig, (ax_chart, ax_text, ax_rel) = plt.subplots(3, 1, figsize=(12, 17.5), 
                                                     gridspec_kw={'height_ratios': [2.6, 0.7, 2.6]})
    fig.patch.set_facecolor(DARK_BG)
    
    # PANEL 1: Relative Hazard Index Profiles
    ax_chart.set_facecolor(PANEL_BG)
    bars = ax_chart.barh(top_cases['Case'], top_cases['Aero_Safety_Risk'], 
                         color='#2b5c8f', edgecolor='#1a3a5f', height=0.55)
    ax_chart.set_xlabel('Relative Aerospace Hazard Potential Index (0.0 - 1.0)\n[Compound Matrix: Solid Object Probability × Observed Flight Behavior Weighting (Flight_Mod)]', 
                        fontsize=10, fontweight='bold', labelpad=10, color=TEXT_WHITE)
    ax_chart.set_title(f'Top {top_n} Highest Calculated Aerospace Hazard Profiles\nQuantified via Bayesian Weighting Framework', 
                       fontsize=12, fontweight='bold', pad=14, color=TEXT_WHITE)
    ax_chart.set_xlim(0, 1.12)
    ax_chart.grid(color=BORDER_GREY, linestyle='--', linewidth=0.5)
    ax_chart.tick_params(axis='both', labelsize=9.5, colors=TEXT_WHITE)
    
    for bar in bars:
        width = bar.get_width()
        ax_chart.text(width + 0.015, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                      va='center', ha='left', fontsize=9, fontweight='bold', color=TEXT_WHITE)
        
    # PANEL 2: Integrated Report Summary Block (Scrubbed of Instrumentation Vocabulary)
    ax_text.axis('off')
    report_block = (
        "=========================================================================================\n"
        "                       JOR FRAMEWORK: CASE PROFILE EVALUATION REPORT\n"
        "=========================================================================================\n"
        f"Total Profiles Evaluated: {total_cases}   |   Mean Framework Hazard Score: {avg_overall_risk:.3f}\n\n"
        f"Triage Tier Breakdown:    * Critical: {hazard_counts.get('Critical', 0)} cases ({ (hazard_counts.get('Critical', 0)/total_cases)*100:.1f}%)    * Elevated: {hazard_counts.get('Elevated', 0)} cases ({ (hazard_counts.get('Elevated', 0)/total_cases)*100:.1f}%)\n"
        f"Framework Insights:       * Avg SOP (Critical): {critical_cases['SOP_Mean'].mean():.3f}                * Avg SOP (Elevated): {elevated_cases['SOP_Mean'].mean():.3f}\n"
        "========================================================================================="
    )
    ax_text.text(0.5, 0.5, report_block, transform=ax_text.transAxes,
                 fontsize=9.5, fontfamily='monospace', color='#00ff00',
                 va='center', ha='center',
                 bbox=dict(boxstyle='round,pad=0.7', facecolor=PANEL_BG, edgecolor=BORDER_GREY, alpha=1.0))

    # PANEL 3: Evidentiary Convergence Error-Bars
    ax_rel.set_facecolor(PANEL_BG)
    for idx, row in reliability_subset.iterrows():
        case_name = row['Case']
        p_mean = row['Posterior_Mean']
        lower_err = p_mean - row['CI_2.5%']
        upper_err = row['CI_97.5%'] - p_mean
        current_color = '#ff453a' if case_name in highest_var['Case'].values else '#30d158'
        
        ax_rel.errorbar(p_mean, case_name, xerr=[[lower_err], [upper_err]], 
                        fmt='o', color=current_color, ecolor=current_color, 
                        elinewidth=2.5, capsize=4, markersize=7)
        ax_rel.text(row['CI_97.5%'] + 0.008, case_name, f"Spread: {row['CI_Width']:.3f}", 
                    va='center', ha='left', fontsize=8.5, color=TEXT_MUTED, weight='bold')

    ax_rel.set_xlabel('Latent Anomaly Posterior Probability Mean (with 95% Credible Intervals)', 
                      fontsize=10, fontweight='bold', labelpad=10, color=TEXT_WHITE)
    ax_rel.set_title('MCMC Uncertainty Auditing: Parameter Reliability & Evidentiary Convergence\n(Isolating High-Variance vs. High-Stability Case Profiles)', 
                     fontsize=12, fontweight='bold', pad=14, color=TEXT_WHITE)
    ax_rel.grid(color=BORDER_GREY, linestyle='--', linewidth=0.5)
    ax_rel.tick_params(axis='both', labelsize=9.5, colors=TEXT_WHITE)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#ff453a', marker='o', lw=2.5, label='High Variance (Needs Data/Evidentiary Auditing)'),
                    Line2D([0], [0], color='#30d158', marker='o', lw=2.5, label='High Stability (Tight Parameter Convergence)')]
    ax_rel.legend(handles=custom_lines, loc='lower right', frameon=True, facecolor=DARK_BG, edgecolor=BORDER_GREY, fontsize=9)
    ax_rel.set_xlim(df_infra['CI_2.5%'].min() - 0.05, df_infra['CI_97.5%'].max() + 0.08)
    
    plt.subplots_adjust(left=0.25, right=0.93, top=0.94, bottom=0.06, hspace=0.35)
    st.pyplot(fig)
    plt.close(fig)

# --- TAB 2: T-TIER BUFFER SEPARATION ---
with tab_separation:
    st.subheader("Relative Hazard Index Density Distribution & Welch's T-Test Verification")
    
    c1, c2 = st.columns(2)
    with c1:
        lower_buffer = st.number_input("Lower Buffer Threshold Limit:", min_value=0.50, max_value=0.80, value=0.73, step=0.01)
    with c2:
        upper_buffer = st.number_input("Upper Buffer Threshold Limit:", min_value=0.70, max_value=0.90, value=0.77, step=0.01)
        
    df_sep = df.copy()
    df_sep['Optimized_Tier'] = 'Investigation Zone'
    df_sep.loc[df_sep['Aero_Safety_Risk'] < lower_buffer, 'Optimized_Tier'] = 'Elevated'
    df_sep.loc[df_sep['Aero_Safety_Risk'] >= upper_buffer, 'Optimized_Tier'] = 'Critical'

    critical_group = df_sep[df_sep['Optimized_Tier'] == 'Critical']['Aero_Safety_Risk'].dropna()
    elevated_group = df_sep[df_sep['Optimized_Tier'] == 'Elevated']['Aero_Safety_Risk'].dropna()
    investigation_group = df_sep[df_sep['Optimized_Tier'] == 'Investigation Zone']['Aero_Safety_Risk'].dropna()

    if critical_group.empty or elevated_group.empty:
        st.error("Insufficient mathematical sample distribution sizing to compute Welch's t-test validation matrix.")
    else:
        t_stat, p_val = stats.ttest_ind(critical_group, elevated_group, equal_var=False)
        
        st.markdown("### Framework Validation Report Summary")
        m_stat1, m_stat2, m_stat3, m_stat4 = st.columns(4)
        m_stat1.metric("CRITICAL CASES", len(critical_group), f"Mean: {critical_group.mean():.3f}")
        m_stat2.metric("ELEVATED CASES", len(elevated_group), f"Mean: {elevated_group.mean():.3f}")
        m_stat3.metric("WELCH'S T-STATISTIC", f"{t_stat:.4f}")
        m_stat4.metric("P-VALUE", f"{p_val:.4e}",
                       delta="OPTIMAL SEPARATION" if p_val < 0.01 else "MARGINAL SEPARATION",
                       delta_color="normal" if p_val < 0.01 else "off")

        col_plot, col_table = st.columns([3, 2])
        
        with col_plot:
            fig, ax = plt.subplots(figsize=(11, 6.5))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(PANEL_BG)
            
            sns.kdeplot(critical_group, fill=True, color='#ff453a', alpha=0.35, linewidth=2.5, label='Clear Critical Tier (>= Upper Buffer)', ax=ax)
            sns.kdeplot(elevated_group, fill=True, color='#ff9f0a', alpha=0.35, linewidth=2.5, label='Clear Elevated Tier (< Lower Buffer)', ax=ax)
            
            ax.axvspan(lower_buffer, upper_buffer, color='#5856d6', alpha=0.20, 
                       linestyle='--', linewidth=1.5, label='Investigation Buffer')

            sns.rugplot(elevated_group, color='#ff9f0a', alpha=0.8, height=0.04, ax=ax, linewidth=1.2)
            sns.rugplot(critical_group, color='#ff453a', alpha=0.8, height=0.04, ax=ax, linewidth=1.2)
            sns.rugplot(investigation_group, color='#5856d6', alpha=1.0, height=0.06, ax=ax, linewidth=2.0)

            ax.text((lower_buffer + upper_buffer) / 2, ax.get_ylim()[1] * 0.5, "BUFFER\nZONE", 
                    color='#5856d6', fontsize=10, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor=DARK_BG, edgecolor='none', alpha=0.75))

            stats_box = (
                f"Welch's t-test Validation:\n"
                f"t-stat = {t_stat:.4f}\n"
                f"p-val  = {p_val:.4e}\n\n"
                f"Active Slice:\n"
                f"* Critical: {len(critical_group)} lines\n"
                f"* Elevated: {len(elevated_group)} lines\n"
                f"* Buffered: {len(investigation_group)} lines"
            )
            ax.text(0.04, 0.96, stats_box, transform=ax.transAxes,
                    fontsize=9.5, fontfamily='monospace', color=TEXT_WHITE,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#1c1c1e', edgecolor=BORDER_GREY, alpha=0.9))

            ax.set_title("Relative Hazard Index Density Distribution Analysis", fontsize=12, fontweight='bold', pad=18, color=TEXT_WHITE)
            ax.set_xlabel("Calculated Relative Aerospace Hazard Potential Score", fontsize=10, labelpad=12, color=TEXT_WHITE)
            ax.set_ylabel("Probability Density Profile", fontsize=10, labelpad=12, color=TEXT_WHITE)
            ax.set_xlim(0.4, 1.1)
            ax.grid(color=BORDER_GREY, linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', colors=TEXT_WHITE)
            ax.legend(loc='upper right', frameon=True, facecolor=DARK_BG, edgecolor=BORDER_GREY, fontsize=9.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_table:
            st.markdown("### Active Investigation Buffer Net")
            if not investigation_group.empty:
                st.warning(f"Detected {len(investigation_group)} profiles flagged for manual technical review:")
                buffer_cases = df_sep[df_sep['Optimized_Tier'] == 'Investigation Zone'][['Case', 'Aero_Safety_Risk']].reset_index(drop=True)
                
                st.dataframe(
                    buffer_cases, 
                    width='stretch',
                    column_config={
                        "Case": "Flagged Case Profile",
                        "Aero_Safety_Risk": st.column_config.NumberColumn("Hazard Score", format="%.3f")
                    }
                )
            else:
                st.success("No cases currently fall inside the investigation buffer zone.")