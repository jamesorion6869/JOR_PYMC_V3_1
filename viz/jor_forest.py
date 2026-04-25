import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 1. Load the data
# Ensure 'jor_scores.csv' is in your working directory
df = pd.read_csv('jor_scores.csv')

# 2. Sort by Posterior_Mean (Ascending for the plot's bottom-to-top rendering)
df = df.sort_values('Posterior_Mean', ascending=True).reset_index(drop=True)

# 3. Define Aesthetic Color Scheme
bg_color = '#121720'      # Dark background
low_bg = '#2a221f'        # Dark reddish-brown
monitor_bg = '#223831'    # Lightened Green (for better contrast)
priority_bg = '#1b263b'   # Dark Blue

low_point = '#f7d1a2'     # Peach/Orange dots
monitor_point = '#b9f4bc' # Light Green dots
priority_point = '#a6c6f7'# Light Blue dots

baseline_color = '#d64c4c'# Red dashed line
grid_color = '#30363d'

# 4. Create high-resolution figure
fig, ax = plt.subplots(figsize=(16, 24), facecolor=bg_color)
ax.set_facecolor(bg_color)

# Set axes limits
ax.set_xlim(0.1, 0.6)
ax.set_ylim(-1, len(df))

# 5. Draw Background Priority Zones
ax.axvspan(0.1, 0.25, color=low_bg, alpha=1, zorder=0)
ax.axvspan(0.25, 0.35, color=monitor_bg, alpha=1, zorder=0)
ax.axvspan(0.35, 0.6, color=priority_bg, alpha=1, zorder=0)

# 6. Draw Skeptical Baseline (at 0.20)
ax.axvline(x=0.20, color=baseline_color, linestyle='--', linewidth=2, zorder=1)

# 7. Plot each case
for i, row in df.iterrows():
    mean = row['Posterior_Mean']
    low = row['CI_2.5%']
    high = row['CI_97.5%']
    
    # Choose point color based on tier
    if mean >= 0.35:
        p_color = priority_point
    elif mean >= 0.25:
        p_color = monitor_point
    else:
        p_color = low_point
        
    # Draw horizontal Credible Interval (CI) line
    ax.hlines(i, low, high, color=p_color, linewidth=2.5, alpha=0.9, zorder=2)
    # Draw end-cap ticks for the CI
    ax.vlines(low, i-0.2, i+0.2, color=p_color, linewidth=2.5, alpha=0.9, zorder=2)
    ax.vlines(high, i-0.2, i+0.2, color=p_color, linewidth=2.5, alpha=0.9, zorder=2)
    
    # Draw the central dot
    ax.scatter(mean, i, color=p_color, s=150, edgecolors='white', linewidths=0.8, zorder=3)
    
    # Add numerical text label next to the plot
    ax.text(high + 0.006, i, f"{mean:.3f}", color='white', 
            va='center', ha='left', fontsize=12, fontweight='bold')

# 8. Styling Labels and Ticks
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['Case'], color='white', fontsize=13)

ax.set_xlabel('Posterior P(NH|E)', color='white', fontsize=16, labelpad=20)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.tick_params(axis='x', colors='white', labelsize=14)

# Remove chart borders for a clean look
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('white')

# Add subtle grid lines
ax.xaxis.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.4)

# 9. Title and Legend
plt.title('JOR-Bayesian Fusion: UAP Case Forest Plot (n=50)\nPosterior P(NH|E) with 95% Credible Intervals', 
          color='white', fontsize=22, pad=40)

legend_elements = [
    Patch(facecolor=priority_bg, label='Priority review ($\geq 0.35$)'),
    Patch(facecolor=monitor_bg, label='Monitor ($0.25 - 0.35$)'),
    Patch(facecolor=low_bg, label='Low priority ($< 0.25$)'),
    Line2D([0], [0], color=baseline_color, linestyle='--', label='Skeptical baseline (0.20)'),
    Line2D([0], [0], marker='o', color='white', markerfacecolor='white', markersize=12, label='Posterior mean (95% CI)')
]

ax.legend(handles=legend_elements, loc='lower right', facecolor=bg_color, 
           edgecolor=grid_color, fontsize=13, labelcolor='white', framealpha=0.95)

plt.tight_layout()

# 10. Save the final image at 600 DPI for ultra-sharp quality
plt.savefig('UAP_Forest_Plot_Final.png', dpi=600, bbox_inches='tight', facecolor=bg_color)
print("Plot successfully saved as UAP_Forest_Plot_Final.png")