import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 1. Load data
# Ensure 'jor_scores.csv' is in your working directory
df = pd.read_csv('jor_scores.csv')

# 2. Robust Year Extraction (Corrects for flight numbers like 1628)
def extract_year(case_str):
    matches = re.findall(r'\d{4}', case_str)
    return int(matches[-1]) if matches else None

df['Year'] = df['Case'].apply(extract_year)
df = df.sort_values('Year').reset_index(drop=True)

# 3. Calculate Trend Line (Moving Average)
df['MA'] = df['Posterior_Mean'].rolling(window=5, center=True).mean()

# 4. Identify Top 5 cases for annotation
top_5 = df.nlargest(5, 'Posterior_Mean')

# 5. Define Aesthetic Color Scheme
bg_color = '#121720'
low_bg = '#2a221f'
monitor_bg = '#223831'
priority_bg = '#1b263b'

low_point = '#f7d1a2'
monitor_point = '#b9f4bc'
priority_point = '#a6c6f7'

baseline_color = '#d64c4c'
grid_color = '#30363d'

# 6. Create high-resolution figure
fig, ax = plt.subplots(figsize=(18, 12), facecolor=bg_color)
ax.set_facecolor(bg_color)

ax.set_ylim(0.1, 0.6)
ax.set_xlim(df['Year'].min() - 5, df['Year'].max() + 5)

# 7. Draw Horizontal Background Priority Zones
ax.axhspan(0.1, 0.25, color=low_bg, alpha=1, zorder=0)
ax.axhspan(0.25, 0.35, color=monitor_bg, alpha=1, zorder=0)
ax.axhspan(0.35, 0.6, color=priority_bg, alpha=1, zorder=0)

# 8. Draw Skeptical Baseline
ax.axhline(y=0.20, color=baseline_color, linestyle='--', linewidth=2, zorder=1)

# 9. Draw Era Markers
eras = [
    (1947, 1969, "Project Blue Book Era"),
    (1970, 2003, "Post-Condon / Hiatus"),
    (2004, 2024, "Modern Sensor / AARO Era")
]

for start, end, label in eras:
    ax.axvline(x=start, color='white', linestyle=':', alpha=0.2, zorder=1)
    ax.text((start + end)/2, 0.11, label, color='white', alpha=0.4, 
            ha='center', fontsize=10, fontstyle='italic')

# 10. Plot each case
for i, row in df.iterrows():
    year = row['Year']
    mean = row['Posterior_Mean']
    low = row['CI_2.5%']
    high = row['CI_97.5%']
    p_color = priority_point if mean >= 0.35 else (monitor_point if mean >= 0.25 else low_point)
    
    ax.vlines(year, low, high, color=p_color, linewidth=1.5, alpha=0.3, zorder=2)
    ax.scatter(year, mean, color=p_color, s=130, edgecolors='white', linewidths=0.5, alpha=0.7, zorder=3)

# 11. Plot Trend Line
ax.plot(df['Year'], df['MA'], color='white', linewidth=3, alpha=0.4, label='5-Case Trend', zorder=4)

# 12. Annotate Top 5 cases (Labels name without the year)
for i, row in top_5.iterrows():
    name_clean = row['Case'].split(' - ')[0]
    ax.annotate(name_clean, xy=(row['Year'], row['Posterior_Mean']), xytext=(5, 5), 
                textcoords='offset points', color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc=bg_color, alpha=0.5, ec='none'))

# 13. Labeling and Styling
ax.set_xlabel('Year', color='white', fontsize=14, labelpad=15)
ax.set_ylabel('Posterior P(NH|E)', color='white', fontsize=14, labelpad=15)
ax.tick_params(axis='both', colors='white', labelsize=12)
for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.yaxis.grid(True, color=grid_color, linestyle='-', linewidth=0.7, alpha=0.4)

plt.title('Chronological UAP Bayesian Fusion: Trend Analysis (n=50)\nPosterior P(NH|E) with Historical Context', 
          color='white', fontsize=22, pad=40)

# Custom Legend
legend_elements = [
    Patch(facecolor=priority_bg, label='Priority review ($\geq 0.35$)'),
    Patch(facecolor=monitor_bg, label='Monitor ($0.25 - 0.35$)'),
    Patch(facecolor=low_bg, label='Low priority ($< 0.25$)'),
    Line2D([0], [0], color='white', alpha=0.4, linewidth=3, label='5-Case Moving Average'),
    Line2D([0], [0], color=baseline_color, linestyle='--', label='Skeptical baseline (0.20)'),
    Line2D([0], [0], marker='o', color='white', markerfacecolor='white', markersize=10, label='Posterior mean')
]
ax.legend(handles=legend_elements, loc='upper left', facecolor=bg_color, edgecolor=grid_color, 
           fontsize=11, labelcolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig('Enhanced_Chronological_UAP_Plot.png', dpi=400, bbox_inches='tight', facecolor=bg_color)
print("Plot successfully saved as Enhanced_Chronological_UAP_Plot.png")