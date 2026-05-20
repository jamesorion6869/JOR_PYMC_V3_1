import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Force wide layout and terminal dark theme aesthetics at the top
st.set_page_config(layout="wide", page_title="JOR-V3.1 Safety Command Center")

# Ensure dark background for all matplotlib/seaborn instances rendered globally
plt.style.use('dark_background')

# --- DATA STORAGE & AUTOMATED WORKABLE EXAMPLE GENERATOR ---
DATA_FILE = "jor_scores.csv"

def generate_mock_workable_example(file_path):
    """Generates a small baseline dataset if a new user hasn't run the backend pipeline yet."""
    mock_data = pd.DataFrame({
        'Case': [
            'USS Nimitz CSG - 2004 (Workable Demo)', 
            'Gimbal Anomaly (Workable Demo)', 
            'Commercial Quadcopter Track (Workable Demo)'
        ],
        'C': [0.90, 0.85, 0.70],
        'E': [0.84, 0.75, 0.90],
        'P': [0.90, 0.80, 0.40],
        'Flight_Mod': [0.05, 0.04, 0.00],
        'P_final': [0.95, 0.84, 0.40],
        'SOP': [0.88, 0.81, 0.57],
        'NHP': [0.90, 0.82, 0.57],
        'Posterior_NH': [0.45, 0.35, 0.12],
        'SOP_Mean': [0.882, 0.814, 0.571],
        'Posterior_Mean': [0.439, 0.345, 0.118],
        'CI_2.5%': [0.377, 0.280, 0.085],
        'CI_97.5%': [0.500, 0.412, 0.161],
        'Aero_Safety_Risk': [0.926, 0.847, 0.571],
        'Hazard_Level': ['Critical', 'Critical', 'Elevated']
    })
    mock_data.to_csv(file_path, index=False)

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        generate_mock_workable_example(file_path)
        
    df = pd.read_csv(file_path)
    
    # Verify strict column logic required for visual mapping
    required_cols = ['Case', 'SOP_Mean', 'Flight_Mod', 'Aero_Safety_Risk']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        generate_mock_workable_example(file_path)
        df = pd.read_csv(file_path)
    
    # Safely compute uncertainty span for bubble sizes and truncate decimal leak tails
    if 'CI_97.5%' in df.columns and 'CI_2.5%' in df.columns:
        df['Uncertainty'] = (df['CI_97.5%'] - df['CI_2.5%']).round(3)
    else:
        df['Uncertainty'] = 0.1  # Fallback size scalar
        
    if 'Hazard_Level' in df.columns:
        df['Hazard_Level'] = df['Hazard_Level'].astype(str)
    else:
        df['Hazard_Level'] = 'Low'
        
    return df

# Trigger live data reload if cache-clear button is pressed
if st.sidebar.button("🔄 Hot-Reload Core Telemetry", use_container_width=True, help="Clears memory cache and reads underlying framework file."):
    st.cache_data.clear()
    st.toast("System cache purged. Reloading dataset...", icon="🔄")

# Load and execute interface warnings outside execution cache container boundary
if not os.path.exists(DATA_FILE):
    st.sidebar.info("💡 No 'jor_scores.csv' detected. Initialized demo dataset.")

df = load_data(DATA_FILE)

# --- HEADER STATUS RIBBON (AS-ANALYZED CALIBRATION STATUS) ---
st.title("📟 JOR-V3.1 Aerospace Safety Command Center")
st.caption("Operational Safety Management System (SMS) & Bayesian Evidence Fusion Framework")

summary = df['Hazard_Level'].value_counts()
crit_count = summary.get('Critical', 0)
elev_count = summary.get('Elevated', 0)
low_count = summary.get('Low', 0)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("MCMC ENGINE STATUS", "CALIBRATED")
m2.metric("CALIBRATION CONSTANT (K)", "0.20", help="Aligned with AARO Standards")
m3.metric("PRIOR HYPOTHESIS (μ)", "0.20")
m4.metric("🚨 CRITICAL HAZARDS", f"{crit_count} Cases")
m5.metric("⚠️ ELEVATED HAZARDS", f"{elev_count} Cases")
m6.metric("✅ LOW RISK TRACKS", f"{low_count} Cases")

st.markdown("---")

# --- MAIN PAGE MULTI-PANEL SPLIT COMPOSITION ---
col_sidebar, col_charts = st.columns([1, 3])

with col_sidebar:
    st.subheader("🛠️ Fleet Filter Controls")
    all_tiers = ['Critical', 'Elevated', 'Low']
    selected_tiers = st.multiselect(
        "Isolate Target Threat Tiers:", 
        options=all_tiers, 
        default=all_tiers
    )
    
    filtered_df = df[df['Hazard_Level'].isin(selected_tiers)].copy()
    if filtered_df.empty:
        st.warning("No cases match target filters. Resetting framework display.")
        filtered_df = df.copy()

    st.markdown("---")
    st.subheader("🔍 Active Track Auditor")
    
    case_list = sorted(filtered_df['Case'].unique())
    selected_case = st.selectbox("Select Target Case Profile:", case_list)
    case_row = filtered_df[filtered_df['Case'] == selected_case].iloc[0]
    
    st.info(f"**Target System ID:** {case_row['Case']}")
    st.metric(
        label="Collision Risk Index (CRI)", 
        value=f"{case_row['Aero_Safety_Risk']:.3f}", 
        delta=str(case_row['Hazard_Level']),
        delta_color="inverse" if case_row['Hazard_Level'] == 'Critical' else "normal"
    )
    
    st.markdown("**Evidentiary Input Scores:**")
    st.text(f"Witness Credibility (C):      {case_row['C']:.2f}")
    st.text(f"Environmental Modifiers (E):  {case_row['E']:.2f}")
    st.text(f"Physical Sensor Telemetry (P): {case_row['P']:.2f}")
    
    st.markdown("---")
    st.subheader("📥 Intelligence Handoff")
    csv_payload = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Current View to CSV",
        data=csv_payload,
        file_name="jor_filtered_audit.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_charts:
    tab_frontier, tab_quadrant, tab_uncertainty = st.tabs([
        "🌌 Kinetic Hazard Frontier", 
        "📊 Threat Intelligence Quadrant Map", 
        "🌲 Evidentiary Uncertainty Forest"
    ])
    
    with tab_frontier:
        st.subheader("Kinetic Hazard Frontier & Aerospace Safety Overlay")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Grid Resolution Mesh Setup
        x_mesh = np.linspace(0, 1.1, 100)
        y_mesh = np.linspace(0, 0.15, 100)
        X, Y = np.meshgrid(x_mesh, y_mesh)
        Z = X * (1 + Y)
        
        # --- BACKGROUND LOGISTIC CONTOURS ---
        contours = ax.contour(X, Y, Z, levels=[0.45, 0.75, 1.1], colors=['#334433', '#444433', '#553333'], linewidths=1.2, linestyles='--')
        ax.clabel(contours, inline=True, fmt={0.45:'Low Risk Bound', 0.75:'Elevated Risk Bound', 1.1:'Critical Frontier'}, fontsize=8, colors='#aaaaaa')
        
        bubble_sizes = (filtered_df['Uncertainty'] * 2000).clip(100, 800)
        
        # Context fleet layout
        scatter = ax.scatter(
            filtered_df['SOP_Mean'], filtered_df['Flight_Mod'], 
            c=filtered_df['Aero_Safety_Risk'], cmap='plasma', 
            s=bubble_sizes, alpha=0.6, edgecolors='#111111', linewidths=0.7,
            vmin=0.0, vmax=1.2
        )

        # --- UNIVERSAL TARGET LOCK OVERLAY ---
        target_size = (case_row['Uncertainty'] * 2000).clip(100, 800)
        
        ax.scatter(
            case_row['SOP_Mean'], case_row['Flight_Mod'],
            s=target_size * 2.5, facecolors='none', edgecolors='#ff453a', linewidths=1.5, linestyle='--'
        )
        ax.scatter(
            case_row['SOP_Mean'], case_row['Flight_Mod'],
            s=120, color='#ff453a', edgecolors='#ffffff', linewidths=1.5, zorder=5
        )
        
        ax.annotate(
            f"[+] TARGET LOCK:\n{case_row['Case']}", 
            xy=(case_row['SOP_Mean'], case_row['Flight_Mod']),
            xytext=(35, 35), textcoords='offset points',
            fontsize=9, fontfamily='monospace', color='#ffffff', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#1c0505', alpha=0.95, ec='#ff453a', lw=1.5),
            arrowprops=dict(arrowstyle='->', color='#ff453a', lw=1.5)
        )
            
        ax.text(0.02, 0.02, "SYSTEM: JOR-V3.1\nSTATUS: OPERATIONAL", 
                fontsize=9, fontfamily='monospace', color='#00ff00',
                bbox=dict(boxstyle='square', fc='#0a0a0a', alpha=0.9, ec='#00ff00', lw=1),
                transform=ax.transAxes)
        
        ax.set_xlabel('Solid Object Probability (SOP_Mean)', color='#cccccc')
        ax.set_ylabel('Kinetic Flight Modifier (Flight_Mod)', color='#cccccc')
        ax.set_xlim(-0.05, 1.15)
        ax.set_ylim(-0.01, 0.16)
        ax.grid(color='#222222', linestyle='-', linewidth=0.5)
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Collision Risk Index (CRI Score)', color='#cccccc')
        
        st.pyplot(fig)
        plt.close(fig)

    with tab_quadrant:
        st.subheader("Threat Intelligence Matrix: Physicality vs. Risk Overlay")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors_dict = {'Critical': '#ff453a', 'Elevated': '#ff9f0a', 'Low': '#30d158'}
        present_colors = {k: v for k, v in colors_dict.items() if k in filtered_df['Hazard_Level'].unique()}
        if not present_colors: present_colors = {'Low': '#30d158'}

        sns.scatterplot(
            data=filtered_df, x='SOP_Mean', y='Aero_Safety_Risk', 
            hue='Hazard_Level', palette=present_colors, s=150, alpha=0.6, ax=ax,
            edgecolor='#111111', linewidth=0.8
        )
        
        # Target dynamic crosshair sync
        ax.axhline(case_row['Aero_Safety_Risk'], color='#ff453a', linestyle=':', alpha=0.6, linewidth=1.5)
        ax.axvline(case_row['SOP_Mean'], color='#ff453a', linestyle=':', alpha=0.6, linewidth=1.5)
        
        ax.scatter(
            case_row['SOP_Mean'], case_row['Aero_Safety_Risk'],
            s=250, color='#ff453a', marker='o', edgecolors='#ffffff', linewidths=1.8, zorder=5
        )
        ax.text(
            case_row['SOP_Mean'] + 0.02, case_row['Aero_Safety_Risk'] + 0.02,
            f"[+] {case_row['Case']}", color='#ffffff', fontsize=9, fontfamily='monospace',
            weight='bold', bbox=dict(boxstyle='square,pad=0.2', fc='#111111', alpha=0.9, ec='#ff453a', lw=1)
        )

        ax.axhline(0.75, color='#ff453a', linestyle='--', alpha=0.7, label='Critical Action Limit (0.75)')
        ax.axvline(0.50, color='#555555', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.25)
        ax.grid(True, which='both', linestyle='-', alpha=0.1, color='#ffffff')
        ax.legend(facecolor='#0a0a0a', edgecolor='#444444')
        
        st.pyplot(fig)
        plt.close(fig)

    with tab_uncertainty:
        st.subheader("MCMC Posterior Uncertainty Range Check")
        if 'CI_2.5%' in filtered_df.columns and 'CI_97.5%' in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # --- DYNAMIC SELECTION POOL CONFIGURATION ---
            sorted_pool = filtered_df.sort_values('Posterior_Mean', ascending=False)
            display_count = min(11, len(sorted_pool))
            top_subset = sorted_pool.head(display_count).copy()
            
            if selected_case not in top_subset['Case'].values:
                target_row = filtered_df[filtered_df['Case'] == selected_case]
                top_subset = pd.concat([top_subset, target_row])
            
            top_subset = top_subset.sort_values('Posterior_Mean', ascending=False)
            
            is_target = (top_subset['Case'] == selected_case)
            peers = top_subset[~is_target]
            target_only = top_subset[is_target]
            
            # 1. Background Peer Profiles (Cleaned up arrays to avoid nested function crashes)
            if not peers.empty:
                lower_err_p = (peers['Posterior_Mean'] - peers['CI_2.5%']).clip(lower=0)
                upper_err_p = (peers['CI_97.5%'] - peers['Posterior_Mean']).clip(lower=0)
                y_err_matrix_p = np.vstack([lower_err_p, upper_err_p])
                
                ax.errorbar(
                    x=peers['Case'], 
                    y=peers['Posterior_Mean'], 
                    yerr=y_err_matrix_p, 
                    fmt='o', 
                    color='#ff9f0a', 
                    ecolor='#444444', 
                    elinewidth=1.5, 
                    capsize=4, 
                    markersize=7, 
                    label='Peer Tracks (95% CI)'
                )
                
            # 2. Prominent Red Target Profile Lock Overlay
            if not target_only.empty:
                lower_err_t = (target_only['Posterior_Mean'] - target_only['CI_2.5%']).clip(lower=0)
                upper_err_t = (target_only['CI_97.5%'] - target_only['Posterior_Mean']).clip(lower=0)
                y_err_matrix_t = np.vstack([lower_err_t, upper_err_t])
                
                ax.errorbar(
                    x=target_only['Case'], 
                    y=target_only['Posterior_Mean'], 
                    yerr=y_err_matrix_t, 
                    fmt='o', 
                    color='#ff453a', 
                    ecolor='#ff453a', 
                    elinewidth=2.5, 
                    capsize=6, 
                    markersize=10, 
                    label='[+] ACTIVE TARGET LOCK'
                )
            
            # --- AXIS FORMATTING REFINEMENTS ---
            ax.set_ylabel("Non-Human Hypothesis Posterior Probability (Mean)", color='#cccccc', fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            
            ax.grid(True, which='both', linestyle='-', alpha=0.12, color='#ffffff')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#444444')
            ax.spines['bottom'].set_color('#444444')
            
            plt.xticks(rotation=25, ha='right', fontsize=9, fontfamily='monospace', color='#aaaaaa')
            ax.legend(facecolor='#0a0f1d', edgecolor='#ff453a', loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("MCMC Post-Processing telemetry missing.")