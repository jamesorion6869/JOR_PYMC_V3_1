import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Load data
if not os.path.exists('jor_scores.csv'):
    print("Error: jor_scores.csv not found.")
else:
    df = pd.read_csv('jor_scores.csv')
    df['Uncertainty'] = df['CI_97.5%'] - df['CI_2.5%']

    # 2. Setup Theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 10))

    # 3. Risk Gradient Background
    x = np.linspace(0, 1.1, 100)
    y = np.linspace(0, 0.15, 100)
    X, Y = np.meshgrid(x, y)
    Z = X * (1 + Y) 
    plt.contourf(X, Y, Z, levels=[0, 0.45, 0.75, 1.3], 
                 colors=['#1a2a1a', '#2d2d1a', '#3d1a1a'], alpha=1)

    # 4. Plot Bubbles
    scatter = ax.scatter(
        df['SOP_Mean'], df['Flight_Mod'], 
        c=df['Aero_Safety_Risk'], cmap='viridis', 
        s=df['Uncertainty']*2800, edgecolor='white', linewidth=0.3, alpha=0.9
    )

    # 5. RADIAL TACTICAL ANNOTATION (Priority 5)
    # This prevents the "one case showing" issue by spreading labels in a circle
    priority_5_df = df.sort_values('Aero_Safety_Risk', ascending=False).head(5)
    
    # Angles for label placement: 45, 135, 225, 315, 90 degrees
    angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4, np.pi/2]
    dist = 60  # Distance in pixels from the dot

    for idx, (i, row) in enumerate(priority_5_df.iterrows()):
        angle = angles[idx % len(angles)]
        # Calculate X/Y offset
        off_x = dist * np.cos(angle)
        off_y = dist * np.sin(angle)
        
        label_text = f"{row['Case']}\nCRI:{row['Aero_Safety_Risk']:.2f}"
        
        ax.annotate(
            label_text,
            xy=(row['SOP_Mean'], row['Flight_Mod']),
            xytext=(off_x, off_y), 
            textcoords='offset points',
            fontsize=8,
            fontfamily='monospace',
            color='#00ff00',
            bbox=dict(boxstyle='round,pad=0.3', fc='#000000', alpha=0.9, ec='#00ff00', lw=1),
            # The 'Leader Line' connects the label to the specific bubble
            arrowprops=dict(arrowstyle='->', color='#00ff00', lw=1, alpha=0.8, 
                            connectionstyle="arc3,rad=0.1"),
            ha='center',
            va='center'
        )

    # 6. Metadata Box
    plt.text(0.02, 0.02, "SYSTEM: JOR-V3.1\nBRIER: 0.1520\nSTATUS: CALIBRATED\nLOC: TALLAHASSEE_FL", 
             fontsize=10, fontfamily='monospace', color='#00ff00',
             bbox=dict(boxstyle='square', fc='#000000', alpha=0.9, ec='#00ff00', lw=1),
             transform=ax.transAxes)

    # 7. Final Polish
    plt.grid(color='#333333', linestyle='-', linewidth=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('CRI (Aviation Risk Index)', color='white')
    
    plt.title('JOR-V3.1 KINETIC HAZARD FRONTIER\nTACTICAL SENSOR FUSION MAP', 
              fontsize=18, color='white', fontweight='bold', pad=20)
    plt.xlabel('PHYSICALITY (SOP_Mean)', color='white')
    plt.ylabel('KINETIC MULTIPLIER (Flight_Mod)', color='white')

    plt.xlim(0, 1.05)
    plt.ylim(-0.01, 0.13)

    plt.savefig('terminal_hazard_frontier.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Success: terminal_hazard_frontier.png generated.")