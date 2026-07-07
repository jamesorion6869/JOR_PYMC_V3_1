import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# --- Updated Parameters for LinkedIn Demo ---
NUM_PROFILES = 100000
PLOT_SUBSET = 2000
PASS_THRESHOLD = 0.60          # threshold for raw weighted NHP (left panel)
POSTERIOR_PASS_THRESHOLD = 0.30  # threshold for Bayesian posterior NHP (right panel)
# Posterior NHP is capped much lower than raw NHP by the K=0.20 calibration constant
# (in this simulation, posterior NHP tops out around ~0.43). 0.30 is chosen to align
# with the paper's own Socorro (Tier 2) posterior value of 0.31, giving a meaningful
# non-degenerate pass rate rather than a threshold that's structurally unreachable.
# Realistic "Stressed" Environment (Synchronized with Integrity Test)
EW_JAM_RATE = 0.15
DROPOUT_RATE = 0.15
ITERATIONS = 100
INTERVAL_MS = 100  # Faster playback for demo impact

# --- JOR-Bayesian Fusion constants (from JOR_Bayesian_Fusion_V3.1 paper) ---
K = 0.20          # calibration constant scaling SOP into P(E|H)
P_NH = 0.20       # conservative prior: probability of non-human explanation
P_H = 0.80        # conservative prior: probability of human explanation

# --- Tactical Theme (toned down: no sweep, no CRT scanlines) ---
RADAR_GREEN = '#39ff14'
RADAR_GREEN_DIM = '#1a4d1a'
RADAR_BG = '#050a07'
EW_RED = '#ff4433'
CENTER = (0.5, 0.5)
MAX_RING_RADIUS = 0.62  # keep rings inside the 0-1 axes

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
fig.patch.set_facecolor(RADAR_BG)


def style_axis(ax, title_text, xlabel, ylabel, threshold):
    ax.set_facecolor(RADAR_BG)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, color=RADAR_GREEN, fontfamily='monospace', fontsize=10)
    ax.set_ylabel(ylabel, color=RADAR_GREEN, fontfamily='monospace', fontsize=10)
    ax.tick_params(colors=RADAR_GREEN_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(RADAR_GREEN_DIM)

    # Concentric rings
    NUM_RINGS = 4
    for i in range(1, NUM_RINGS + 1):
        r = MAX_RING_RADIUS * i / NUM_RINGS
        ring = Circle(CENTER, r, fill=False, edgecolor=RADAR_GREEN_DIM,
                      linewidth=1, linestyle='-', alpha=0.7, zorder=0)
        ax.add_patch(ring)

    # Spokes
    for angle_deg in range(0, 360, 45):
        angle = np.radians(angle_deg)
        x_end = CENTER[0] + MAX_RING_RADIUS * np.cos(angle)
        y_end = CENTER[1] + MAX_RING_RADIUS * np.sin(angle)
        ax.plot([CENTER[0], x_end], [CENTER[1], y_end],
                color=RADAR_GREEN_DIM, linewidth=1, alpha=0.6, zorder=0)

    # Pass threshold reference line
    ax.axhline(y=threshold, color=RADAR_GREEN, linestyle='--',
               linewidth=1.2, alpha=0.6, zorder=1)
    ax.text(0.015, threshold + 0.015, f"THRESHOLD ({threshold:.2f})",
            fontsize=7.5, color=RADAR_GREEN, fontfamily='monospace', alpha=0.8)

    ax.set_title(title_text, color=RADAR_GREEN, fontfamily='monospace', fontsize=11)


style_axis(ax1, "Raw Weighted NHP", "SOP (Baseline)", "NHP (Raw, Weighted Avg)", PASS_THRESHOLD)
style_axis(ax2, "Bayesian Posterior NHP", "SOP (Baseline)", "NHP (Posterior, Bayes-Fused)", POSTERIOR_PASS_THRESHOLD)

# --- Plotting elements for LEFT panel (raw NHP) ---
scat_opt_1 = ax1.scatter([], [], s=5, c=RADAR_GREEN, alpha=0.75, label='Optimal', zorder=2, linewidths=0)
scat_ew_1 = ax1.scatter([], [], s=8, c=EW_RED, alpha=0.85, label='EW Jamming', zorder=3, linewidths=0)
dropout_marker_1, = ax1.plot([], [], 'x', color='white', markersize=12,
                             markeredgewidth=2, label='Dropout', zorder=10)
ax1.legend(loc='upper left', frameon=True, facecolor=RADAR_BG,
           edgecolor=RADAR_GREEN_DIM, labelcolor=RADAR_GREEN, fontsize=8)
stats_text_1 = ax1.text(0.30, 0.05, "", fontsize=11, color=RADAR_GREEN,
                         fontfamily='monospace',
                         bbox=dict(facecolor=RADAR_BG, edgecolor=RADAR_GREEN, alpha=0.9))

# --- Plotting elements for RIGHT panel (Bayesian posterior NHP) ---
scat_opt_2 = ax2.scatter([], [], s=5, c=RADAR_GREEN, alpha=0.75, label='Optimal', zorder=2, linewidths=0)
scat_ew_2 = ax2.scatter([], [], s=8, c=EW_RED, alpha=0.85, label='EW Jamming', zorder=3, linewidths=0)
dropout_marker_2, = ax2.plot([], [], 'x', color='white', markersize=12,
                             markeredgewidth=2, label='Dropout', zorder=10)
ax2.legend(loc='upper left', frameon=True, facecolor=RADAR_BG,
           edgecolor=RADAR_GREEN_DIM, labelcolor=RADAR_GREEN, fontsize=8)
stats_text_2 = ax2.text(0.30, 0.05, "", fontsize=11, color=RADAR_GREEN,
                         fontfamily='monospace',
                         bbox=dict(facecolor=RADAR_BG, edgecolor=RADAR_GREEN, alpha=0.9))

suptitle_obj = fig.suptitle("JOR_BAYESIAN_FUSION v3.1 | Simulation Active", color=RADAR_GREEN,
                             fontfamily='monospace', fontsize=13)

# --- Pre-calculation for Speed ---
print("Pre-generating simulation data for high-speed playback...")
ALL_C = np.random.uniform(0.3, 0.85, (ITERATIONS, NUM_PROFILES))
ALL_E = np.random.uniform(0.3, 0.85, (ITERATIONS, NUM_PROFILES))
ALL_P = np.random.uniform(0.3, 0.95, (ITERATIONS, NUM_PROFILES))
ALL_Mod = np.random.choice([0.0, 0.02, 0.04, 0.05], (ITERATIONS, NUM_PROFILES))
ALL_EW = np.random.random((ITERATIONS, NUM_PROFILES)) < EW_JAM_RATE
ALL_Drop = (np.random.random((ITERATIONS, NUM_PROFILES)) < (DROPOUT_RATE / (1.0 - EW_JAM_RATE))) & (~ALL_EW)


def update(frame):
    # Pull pre-generated data
    C, E, P, Mod = ALL_C[frame], ALL_E[frame], ALL_P[frame], ALL_Mod[frame]
    ew, drop = ALL_EW[frame], ALL_Drop[frame]

    # Math Logic (unchanged from original)
    P[ew] = 0.95
    Mod[ew] = -0.07

    SOP = (0.4 * C) + (0.3 * E) + (0.3 * P)
    NHP_raw = np.clip((0.4 * C) + (0.3 * E) + (0.3 * (P + Mod)), 0, 0.95)

    # --- JOR-Bayesian Fusion: Steps 4-7 from the paper ---
    # P(E|NH) = NHP (raw weighted score)
    P_E_given_NH = NHP_raw
    # P(E|H) = min(1, 1 - NHP + K*SOP)
    P_E_given_H = np.clip(1 - NHP_raw + K * SOP, 0, 1)
    # Bayes' theorem: P(NH|E) = [P(E|NH)*P(NH)] / [P(E|NH)*P(NH) + P(E|H)*P(H)]
    numerator = P_E_given_NH * P_NH
    denominator = numerator + (P_E_given_H * P_H)
    NHP_posterior = numerator / denominator

    # --- LEFT panel: raw NHP ---
    opt_mask = ~ew & ~drop
    scat_opt_1.set_offsets(np.c_[SOP[opt_mask][:PLOT_SUBSET], NHP_raw[opt_mask][:PLOT_SUBSET]])
    scat_ew_1.set_offsets(np.c_[SOP[ew][:PLOT_SUBSET], NHP_raw[ew][:PLOT_SUBSET]])

    # --- RIGHT panel: Bayesian posterior NHP ---
    scat_opt_2.set_offsets(np.c_[SOP[opt_mask][:PLOT_SUBSET], NHP_posterior[opt_mask][:PLOT_SUBSET]])
    scat_ew_2.set_offsets(np.c_[SOP[ew][:PLOT_SUBSET], NHP_posterior[ew][:PLOT_SUBSET]])

    # Dropout Flicker (same random flicker position on both panels)
    if np.any(drop):
        dx, dy = np.random.uniform(0.05, 0.25), np.random.uniform(0.05, 0.25)
        dropout_marker_1.set_data([dx], [dy])
        dropout_marker_2.set_data([dx], [dy])
        dropout_marker_1.set_visible(True)
        dropout_marker_2.set_visible(True)
    else:
        dropout_marker_1.set_visible(False)
        dropout_marker_2.set_visible(False)

    # UI Updates
    pass_rate_raw = np.mean(NHP_raw > PASS_THRESHOLD) * 100
    pass_rate_posterior = np.mean(NHP_posterior > POSTERIOR_PASS_THRESHOLD) * 100
    suptitle_obj.set_text(f"JOR_BAYESIAN_FUSION v3.1 | Iteration {frame + 1}/{ITERATIONS}")

    if frame == ITERATIONS - 1:
        stats_text_1.set_text(f"Final Pass Rate: {pass_rate_raw:.1f}%")
        stats_text_2.set_text(f"Final Pass Rate: {pass_rate_posterior:.1f}%")

    return (scat_opt_1, scat_ew_1, dropout_marker_1, stats_text_1,
            scat_opt_2, scat_ew_2, dropout_marker_2, stats_text_2, suptitle_obj)


# blit=False: fig.suptitle is a figure-level artist and isn't compatible with blit's
# per-axes partial-redraw mechanism. With only ~4000 points across two panels,
# full redraws at 100ms/frame are still smooth.
ani = animation.FuncAnimation(fig, update, frames=ITERATIONS, interval=INTERVAL_MS,
                               blit=False, repeat=False)
plt.tight_layout()
plt.show()
