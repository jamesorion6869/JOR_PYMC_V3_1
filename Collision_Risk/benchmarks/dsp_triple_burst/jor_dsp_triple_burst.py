"""
JOR V3.1 -- DSP-Driven Triple-Burst Accumulation Test
----------------------------------------------------------------
Combines two pieces of validated work from this project:

  1. The CORRECTED DSP-to-rubric mapping (jor_dsp_conventional.py):
     - C represents an assumed-moderate-witness baseline (~0.65), not
       sensor-quality metrics (fixes the category error / saturation bug)
     - Sensor/track-quality metrics (track_consistency, maneuver_index,
       multi_path_score, doppler_spread) correctly feed P, not C
     - detection_probability baseline calibrated to 0.75 so the
       Conventional posterior centers at ~0.30, matching every other
       validated baseline in this project

  2. The validated TRIPLE-BURST schedule (jor_dsp_triple_burst.py):
     - 3 bursts, 8 steps (40s) each, 3-step (15s) gaps between bursts --
       inside the recursive decay window, so each new burst catches the
       posterior still elevated from the last, testing evidence
       ACCUMULATION rather than independent isolated spikes
     - Confirmed via a 10-seed sweep: Burst1->Burst2 gain is reliably
       positive, Burst2->Burst3 plateaus (bounded, not runaway,
       accumulation)

During a burst, the DSP generator's detection_probability and RCS get
boosted (simulating a stronger/more anomalous physical return) and the
Flight Characteristics Modifier is applied, exactly as in the earlier
hand-coded triple-burst test -- just now running through the corrected,
DSP-realistic C/E/P pipeline instead of directly-set baseline values.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Change to "QtAgg" for local interactive display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from numba import njit

# ---------------------------------------------------------------
# Constants (validated JOR V3.1 math)
# ---------------------------------------------------------------
W_C, W_E, W_P = 0.40, 0.30, 0.30
K = 0.20
P_PRIOR_NH_INITIAL = 0.20
P_PRIOR_H_INITIAL = 0.80

TIME_STEPS = 60
STEP_SECONDS = 5
PRIOR_RETENTION = 0.70

ASSUMED_WITNESS_C = 0.65  # matches the Conventional baseline

# ---------------------------------------------------------------
# Triple-Burst Schedule (validated: 8-step bursts, 3-step gaps
# inside the decay window)
# ---------------------------------------------------------------
BURST_LEN = 8
GAP_LEN = 3
FIRST_BURST_START = 5  # burst 1 begins at t=30s

BURST_WINDOWS = []
_cursor = FIRST_BURST_START
for _i in range(3):
    BURST_WINDOWS.append((_cursor, _cursor + BURST_LEN - 1))
    _cursor += BURST_LEN
    if _i < 2:
        _cursor += GAP_LEN

ANOMALY_MODIFIER = 0.05     # "Major Anomaly" flight characteristics modifier
ANOMALY_DET_PROB_BOOST = 0.15   # stronger/more consistent physical return during a burst
ANOMALY_RCS_BOOST = 2.0         # larger apparent radar cross-section during a burst

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# Bayesian Fusion Step (validated JOR V3.1 core)
# ---------------------------------------------------------------
@njit(fastmath=True)
def bayesian_fusion_step(
    C, E, P_raw, modifier, K, prior_NH, prior_H,
    W_C=0.40, W_E=0.30, W_P=0.30,
):
    SOP = W_C * C + W_E * E + W_P * P_raw
    P_for_NHP = min(max(P_raw + modifier, 0.0), 0.95)
    NHP = W_C * C + W_E * E + W_P * P_for_NHP
    P_E_given_NH = NHP
    P_E_given_H = min(max(1.0 - NHP + K * SOP, 0.0), 1.0)
    numerator = P_E_given_NH * prior_NH
    denominator = numerator + (P_E_given_H * prior_H)
    posterior_NH = numerator / denominator if denominator > 0 else 0.0
    return SOP, NHP, posterior_NH

# ---------------------------------------------------------------
# DSP-style helpers
# ---------------------------------------------------------------
def normalize(value, v_min, v_max):
    v = max(min(value, v_max), v_min)
    return (v - v_min) / (v_max - v_min)

def weighted_average(*components):
    total_w = sum(w for _, w in components)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in components) / total_w

def in_any_burst(step):
    return any(s <= step <= e for s, e in BURST_WINDOWS)

# ---------------------------------------------------------------
# Synthetic DSP Radar Detection (calibrated baseline, boosted during bursts)
# ---------------------------------------------------------------
def generate_radar_detection(in_burst):
    det_prob_baseline = 0.75 + (ANOMALY_DET_PROB_BOOST if in_burst else 0.0)
    rcs_baseline = 3.0 + (ANOMALY_RCS_BOOST if in_burst else 0.0)
    return {
        "snr_db": 18.0 + rng.normal(0, 1.5),
        "rcs_estimate": rcs_baseline + rng.normal(0, 0.4),
        "peak_power": 1400.0 + rng.normal(0, 80.0),
        "doppler_spread": 10.0 + rng.normal(0, 2.0),
        "detection_probability": np.clip(det_prob_baseline + rng.normal(0, 0.01), 0.0, 1.0),
        "range_resolution": 15.0 + rng.normal(0, 0.5),
        "maneuver_index": 0.10 + rng.normal(0, 0.02),
        "track_consistency": 0.93 + rng.normal(0, 0.02),
        "multi_path_score": max(0.0, 0.03 + rng.normal(0, 0.01)),
    }

# ---------------------------------------------------------------
# DSP -> JOR Mapping
# ---------------------------------------------------------------
def sensor_step(step):
    in_burst = in_any_burst(step)
    rd = generate_radar_detection(in_burst)

    # C: Witness Credibility -- assumed-moderate-witness baseline,
    # NOT derived from sensor data (fixes the category error).
    C = float(np.clip(ASSUMED_WITNESS_C + rng.normal(0, 0.03), 0.30, 0.85))

    # E: Environmental / Observation Clarity -- unchanged mapping.
    E = weighted_average(
        (normalize(rd["snr_db"], 10.0, 30.0), 0.35),
        (normalize(rd["peak_power"], 800.0, 2000.0), 0.25),
        (normalize(rd["rcs_estimate"], 0.5, 10.0), 0.25),
        (1.0 - normalize(rd["range_resolution"], 10.0, 25.0), 0.15),
    )

    # P: Physical / Sensor Evidence -- includes the reallocated
    # track-quality metrics (fix #2), boosted RCS/detection_probability
    # during a burst simulate stronger/more anomalous physical returns.
    P_raw = weighted_average(
        (rd["detection_probability"], 0.25),
        (normalize(rd["rcs_estimate"], 0.5, 10.0), 0.15),
        (normalize(rd["peak_power"], 800.0, 2000.0), 0.15),
        (rd["track_consistency"], 0.20),
        (1.0 - rd["maneuver_index"], 0.10),
        (1.0 - rd["multi_path_score"], 0.10),
        (1.0 - normalize(rd["doppler_spread"], 5.0, 25.0), 0.05),
    )

    modifier = ANOMALY_MODIFIER if in_burst else 0.0

    return (
        float(np.clip(C, 0.30, 0.85)),
        float(np.clip(E, 0.30, 0.85)),
        float(np.clip(P_raw, 0.30, 0.95)),
        modifier,
        in_burst,
    )

# ---------------------------------------------------------------
# Run Simulation
# ---------------------------------------------------------------
def run_extended_conventional():
    prior_NH = P_PRIOR_NH_INITIAL
    prior_H = P_PRIOR_H_INITIAL

    sop_track, posterior_track, rolling_avg_track = [], [], []
    C_track, E_track, P_track, burst_flags = [], [], [], []

    for step in range(TIME_STEPS):
        C, E, P_raw, modifier, in_burst = sensor_step(step)
        C_track.append(C); E_track.append(E); P_track.append(P_raw)
        burst_flags.append(in_burst)

        SOP, NHP, posterior_NH = bayesian_fusion_step(
            C, E, P_raw, modifier, K, prior_NH, prior_H
        )

        sop_track.append(SOP)
        posterior_track.append(posterior_NH)
        rolling_avg_track.append(np.mean(posterior_track[-20:]))

        prior_NH = (
            PRIOR_RETENTION * posterior_NH
            + (1 - PRIOR_RETENTION) * P_PRIOR_NH_INITIAL
        )
        prior_H = 1.0 - prior_NH

    return (
        np.array(sop_track), np.array(posterior_track), np.array(rolling_avg_track),
        np.array([(s + 1) * STEP_SECONDS for s in range(TIME_STEPS)]),
        np.array(C_track), np.array(E_track), np.array(P_track), np.array(burst_flags),
    )

# ---------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------
print("Pre-computing DSP-driven triple-burst track (mapping)...")
SOP_TRACK, POST_TRACK, ROLLING_TRACK, TIME_AXIS, C_TRACK, E_TRACK, P_TRACK, BURST_FLAGS = run_extended_conventional()

BURST_TIMES = [((s + 1) * STEP_SECONDS, (e + 1) * STEP_SECONDS) for s, e in BURST_WINDOWS]

print("\n--- Baseline check (non-burst steps) ---")
non_burst_post = POST_TRACK[~BURST_FLAGS]
print(f"Non-burst posterior mean: {non_burst_post.mean():.3f} (target: 0.25-0.35)")

print("\n--- Burst Accumulation Summary ---")
peaks = []
for i, (s, e) in enumerate(BURST_WINDOWS, 1):
    peak = POST_TRACK[s:e + 1].max()
    peaks.append(peak)
    print(f"Burst {i} (t={BURST_TIMES[i-1][0]}-{BURST_TIMES[i-1][1]}s): peak posterior = {peak:.4f}")
print(f"Burst 2 - Burst 1 gain: {peaks[1]-peaks[0]:+.4f}")
print(f"Burst 3 - Burst 2 gain: {peaks[2]-peaks[1]:+.4f}")
print("-----------------------------------\n")

# ---------------------------------------------------------------
# Visualization Theme
# ---------------------------------------------------------------
RADAR_BG = "#050a07"
RADAR_GREEN_DIM = "#1a4d1a"
CONVENTIONAL_COLOR = "#39ff14"
ANOMALY_SHADE_COLOR = "#ff3b30"
ANOMALY_TEXT_COLOR = "#ff6b61"

plt.style.use("dark_background")

fig = plt.figure(figsize=(15, 6.5))
fig.patch.set_facecolor(RADAR_BG)
gs = fig.add_gridspec(1, 2, wspace=0.28)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Radar Panel
ax1.set_facecolor(RADAR_BG)
ax1.set_title("Recursive State Trajectory: SOP vs Posterior NH (DSP)", color=CONVENTIONAL_COLOR)

sop_min = max(0.0, SOP_TRACK.min() - 0.05)
sop_max = min(1.0, SOP_TRACK.max() + 0.05)
post_min = max(0.0, POST_TRACK.min() - 0.05)
post_max = min(1.0, POST_TRACK.max() + 0.05)

CENTER = ((sop_min + sop_max) / 2, (post_min + post_max) / 2)
MAX_RING_RADIUS = min(sop_max - sop_min, post_max - post_min) / 2 * 0.95

ax1.set_xlim(sop_min, sop_max)
ax1.set_ylim(post_min, post_max)
ax1.set_aspect("equal")
ax1.set_xlabel("SOP (Baseline)", color=CONVENTIONAL_COLOR)
ax1.set_ylabel("Posterior NH", color=CONVENTIONAL_COLOR)
ax1.tick_params(colors=RADAR_GREEN_DIM)

for spine in ax1.spines.values():
    spine.set_color(RADAR_GREEN_DIM)

for i in range(1, 5):
    r = MAX_RING_RADIUS * i / 4
    ax1.add_patch(Circle(CENTER, r, fill=False, edgecolor=RADAR_GREEN_DIM, alpha=0.7))

for angle_deg in range(0, 360, 45):
    angle = np.radians(angle_deg)
    ax1.plot([CENTER[0], CENTER[0] + MAX_RING_RADIUS * np.cos(angle)],
             [CENTER[1], CENTER[1] + MAX_RING_RADIUS * np.sin(angle)],
             color=RADAR_GREEN_DIM, alpha=0.6)

radar_line, = ax1.plot([], [], color=CONVENTIONAL_COLOR, linewidth=1.5)
radar_marker, = ax1.plot([], [], "o", color=CONVENTIONAL_COLOR, markersize=6)

# Stability Panel
ax2.set_facecolor(RADAR_BG)
ax2.set_title("Recursive Posterior NH Response -- DSP Triple Burst (300s)", color=CONVENTIONAL_COLOR)
ax2.set_xlim(0, TIME_AXIS[-1])
ax2.set_ylim(0, 1.0)
ax2.set_xlabel("Time (s)", color=CONVENTIONAL_COLOR)
ax2.set_ylabel("Posterior NH", color=CONVENTIONAL_COLOR)
ax2.tick_params(colors=RADAR_GREEN_DIM)

for spine in ax2.spines.values():
    spine.set_color(RADAR_GREEN_DIM)

ax2.grid(color=RADAR_GREEN_DIM, alpha=0.3)

ax2.axhline(y=0.30, color=CONVENTIONAL_COLOR, linestyle="--", alpha=0.6)
ax2.text(5, 0.315, "Normal Flight Baseline (0.30)", color=CONVENTIONAL_COLOR, fontsize=8)

ax2.fill_between(TIME_AXIS, 0.25, 0.35, color=RADAR_GREEN_DIM, alpha=0.15, label="Noise Floor Band (0.25-0.35)")

for i, (start_t, end_t) in enumerate(BURST_TIMES):
    ax2.axvspan(start_t, end_t, color=ANOMALY_SHADE_COLOR, alpha=0.10,
                label="Major Anomaly Bursts" if i == 0 else None)
    ax2.text((start_t + end_t) / 2, 0.92, f"BURST {i + 1}", color=ANOMALY_TEXT_COLOR,
             fontsize=8, fontweight="bold", ha="center", va="center")

post_line, = ax2.plot([], [], color=CONVENTIONAL_COLOR, linewidth=1.5)
post_marker, = ax2.plot([], [], "o", color=CONVENTIONAL_COLOR, markersize=6)
rolling_line, = ax2.plot([], [], color="#00ffaa", linewidth=1.2, alpha=0.8, label="Rolling Avg (100s)")

ax2.legend(loc="lower right", fontsize=8, facecolor=RADAR_BG, edgecolor=RADAR_GREEN_DIM)

title_text = ax2.text(0.98, 0.96, "", transform=ax2.transAxes, ha="right", va="top",
                      color=CONVENTIONAL_COLOR, fontsize=9,
                      bbox=dict(facecolor=RADAR_BG, edgecolor=RADAR_GREEN_DIM, alpha=0.85))

def update(frame):
    step = frame + 1
    radar_line.set_data(SOP_TRACK[:step], POST_TRACK[:step])
    radar_marker.set_data([SOP_TRACK[step-1]], [POST_TRACK[step-1]])
    post_line.set_data(TIME_AXIS[:step], POST_TRACK[:step])
    post_marker.set_data([TIME_AXIS[step-1]], [POST_TRACK[step-1]])
    rolling_line.set_data(TIME_AXIS[:step], ROLLING_TRACK[:step])
    title_text.set_text(f"t={TIME_AXIS[step-1]}s ({step}/{TIME_STEPS})")
    return radar_line, radar_marker, post_line, post_marker, rolling_line, title_text

ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, interval=600, blit=True, repeat=False)

fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.28)
ani.save("jor_dsp_triple_burst.gif", writer='pillow', fps=10)
print("Animation saved as jor_dsp_triple_burst.gif")
