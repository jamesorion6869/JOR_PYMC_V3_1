"""
JOR V3.1 -- DSP Triple-Burst Demo with Live Processing Readout (single video)
------------------------------------------------------------------------------
Same validated triple-burst DSP pipeline as jor_dsp_triple_burst.py, restructured
so the simulation computes live, one step per animation frame (not precomputed
in a batch), and a third panel is added underneath the two existing charts: a
terminal-style scrolling readout showing genuine per-step engine compute time
and throughput, measured with time.perf_counter() around the actual numba-JIT
fusion call -- not a simulated or decorative number.

Everything renders into ONE matplotlib figure and exports to a single .mp4.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from numba import njit

# ---------------------------------------------------------------
# Constants (validated JOR V3.1 math -- unchanged from the source file)
# ---------------------------------------------------------------
W_C, W_E, W_P = 0.40, 0.30, 0.30
K = 0.20
P_PRIOR_NH_INITIAL = 0.20
P_PRIOR_H_INITIAL = 0.80

TIME_STEPS = 60
STEP_SECONDS = 5
PRIOR_RETENTION = 0.70

ASSUMED_WITNESS_C = 0.65

BURST_LEN = 8
GAP_LEN = 3
FIRST_BURST_START = 5

BURST_WINDOWS = []
_cursor = FIRST_BURST_START
for _i in range(3):
    BURST_WINDOWS.append((_cursor, _cursor + BURST_LEN - 1))
    _cursor += BURST_LEN
    if _i < 2:
        _cursor += GAP_LEN

ANOMALY_MODIFIER = 0.05
ANOMALY_DET_PROB_BOOST = 0.15
ANOMALY_RCS_BOOST = 2.0

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# Bayesian Fusion Step (validated JOR V3.1 core, unchanged)
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

# Warm up the JIT compiler BEFORE any timing happens, so the displayed
# per-step engine time reflects compiled speed, not first-call compilation.
_ = bayesian_fusion_step(0.5, 0.5, 0.5, 0.0, K, 0.2, 0.8)

# ---------------------------------------------------------------
# DSP-style helpers (unchanged)
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

def sensor_step(step):
    in_burst = in_any_burst(step)
    rd = generate_radar_detection(in_burst)

    C = float(np.clip(ASSUMED_WITNESS_C + rng.normal(0, 0.03), 0.30, 0.85))

    E = weighted_average(
        (normalize(rd["snr_db"], 10.0, 30.0), 0.35),
        (normalize(rd["peak_power"], 800.0, 2000.0), 0.25),
        (normalize(rd["rcs_estimate"], 0.5, 10.0), 0.25),
        (1.0 - normalize(rd["range_resolution"], 10.0, 25.0), 0.15),
    )

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

BURST_TIMES = [((s + 1) * STEP_SECONDS, (e + 1) * STEP_SECONDS) for s, e in BURST_WINDOWS]

# ---------------------------------------------------------------
# Live simulation state (computed frame-by-frame inside the animation,
# not precomputed in a batch
# ---------------------------------------------------------------
class LiveState:
    def __init__(self):
        self.prior_NH = P_PRIOR_NH_INITIAL
        self.prior_H = P_PRIOR_H_INITIAL
        self.sop_track = []
        self.post_track = []
        self.rolling_track = []
        self.time_axis = []
        self.log_lines = []
        self.engine_times_us = []

    def step(self, step_idx):
        C, E, P_raw, modifier, in_burst = sensor_step(step_idx)

        t0 = time.perf_counter()
        SOP, NHP, posterior_NH = bayesian_fusion_step(
            C, E, P_raw, modifier, K, self.prior_NH, self.prior_H
        )
        t1 = time.perf_counter()
        engine_us = (t1 - t0) * 1e6

        self.sop_track.append(SOP)
        self.post_track.append(posterior_NH)
        self.rolling_track.append(np.mean(self.post_track[-20:]))
        t_sec = (step_idx + 1) * STEP_SECONDS
        self.time_axis.append(t_sec)
        self.engine_times_us.append(engine_us)

        steps_per_sec = 1e6 / engine_us if engine_us > 0 else float("inf")
        flag = " [ANOMALY BURST]" if in_burst else ""
        line = (f"t={t_sec:>3d}s  step {step_idx+1:02d}/{TIME_STEPS}  "
                f"C={C:.3f} E={E:.3f} P={P_raw:.3f}  "
                f"SOP={SOP:.3f} NHP={NHP:.3f}  POST={posterior_NH:.3f}  "
                f"| engine {engine_us:6.1f}us ({steps_per_sec:,.0f} steps/s){flag}")
        self.log_lines.append(line)

        self.prior_NH = (
            PRIOR_RETENTION * posterior_NH
            + (1 - PRIOR_RETENTION) * P_PRIOR_NH_INITIAL
        )
        self.prior_H = 1.0 - self.prior_NH

        return SOP, posterior_NH, in_burst

STATE = LiveState()

# ---------------------------------------------------------------
# Figure layout: top row = existing two panels, bottom row = terminal
# ---------------------------------------------------------------
RADAR_BG = "#050a07"
RADAR_GREEN_DIM = "#1a4d1a"
CONVENTIONAL_COLOR = "#39ff14"
ANOMALY_SHADE_COLOR = "#ff3b30"
ANOMALY_TEXT_COLOR = "#ff6b61"
TERMINAL_TEXT_COLOR = "#39ff14"

plt.style.use("dark_background")

fig = plt.figure(figsize=(15, 9.5))
fig.patch.set_facecolor(RADAR_BG)
gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.4], hspace=0.35, wspace=0.28)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])  # terminal panel spans full width

# --- Panel 1: radar/state trajectory ---
ax1.set_facecolor(RADAR_BG)
ax1.set_title("Recursive State Trajectory: SOP vs Posterior NH (DSP)", color=CONVENTIONAL_COLOR)
sop_axis_min, sop_axis_max = 0.35, 0.75
post_axis_min, post_axis_max = 0.15, 0.55
CENTER = ((sop_axis_min + sop_axis_max) / 2, (post_axis_min + post_axis_max) / 2)
MAX_RING_RADIUS = min(sop_axis_max - sop_axis_min, post_axis_max - post_axis_min) / 2 * 0.95
ax1.set_xlim(sop_axis_min, sop_axis_max)
ax1.set_ylim(post_axis_min, post_axis_max)
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

# --- Panel 2: stability / time series ---
ax2.set_facecolor(RADAR_BG)
ax2.set_title("Recursive Posterior NH Response -- DSP Triple Burst (300s)", color=CONVENTIONAL_COLOR)
ax2.set_xlim(0, TIME_STEPS * STEP_SECONDS)
ax2.set_ylim(0, 1.0)
ax2.set_xlabel("Time (s)", color=CONVENTIONAL_COLOR)
ax2.set_ylabel("Posterior NH", color=CONVENTIONAL_COLOR)
ax2.tick_params(colors=RADAR_GREEN_DIM)
for spine in ax2.spines.values():
    spine.set_color(RADAR_GREEN_DIM)
ax2.grid(color=RADAR_GREEN_DIM, alpha=0.3)
ax2.axhline(y=0.30, color=CONVENTIONAL_COLOR, linestyle="--", alpha=0.6)
ax2.text(5, 0.315, "Normal Flight Baseline (0.30)", color=CONVENTIONAL_COLOR, fontsize=8)
ax2.fill_between([0, TIME_STEPS * STEP_SECONDS], 0.25, 0.35, color=RADAR_GREEN_DIM, alpha=0.15,
                  label="Noise Floor Band (0.25-0.35)")
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

# --- Panel 3: terminal-style live engine readout ---
ax3.set_facecolor("#000000")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis("off")
for spine in ax3.spines.values():
    spine.set_color(RADAR_GREEN_DIM)
    spine.set_visible(True)
N_LOG_LINES = 9
terminal_text = ax3.text(
    0.012, 0.95, "", transform=ax3.transAxes, ha="left", va="top",
    color=TERMINAL_TEXT_COLOR, fontsize=9.3, family="monospace", linespacing=1.65,
)
header_text = ax3.text(
    0.012, 0.985, "JOR-V3.1 ENGINE  |  live per-step fusion timing (numba JIT, compiled)",
    transform=ax3.transAxes, ha="left", va="bottom",
    color="#00ffaa", fontsize=8.5, family="monospace", fontweight="bold",
)
avg_text = ax3.text(
    0.988, 0.985, "", transform=ax3.transAxes, ha="right", va="bottom",
    color="#00ffaa", fontsize=8.5, family="monospace", fontweight="bold",
)

def update(frame):
    SOP, posterior_NH, in_burst = STATE.step(frame)

    step = frame + 1
    radar_line.set_data(STATE.sop_track[:step], STATE.post_track[:step])
    radar_marker.set_data([STATE.sop_track[step - 1]], [STATE.post_track[step - 1]])
    post_line.set_data(STATE.time_axis[:step], STATE.post_track[:step])
    post_marker.set_data([STATE.time_axis[step - 1]], [STATE.post_track[step - 1]])
    rolling_line.set_data(STATE.time_axis[:step], STATE.rolling_track[:step])
    title_text.set_text(f"t={STATE.time_axis[step - 1]}s ({step}/{TIME_STEPS})")

    visible_lines = STATE.log_lines[-N_LOG_LINES:]
    terminal_text.set_text("\n".join(visible_lines))

    if STATE.engine_times_us:
        mean_us = float(np.mean(STATE.engine_times_us))
        avg_text.set_text(f"avg: {mean_us:5.1f}us/step  ({1e6/mean_us:,.0f} steps/sec)")

    return (radar_line, radar_marker, post_line, post_marker, rolling_line,
            title_text, terminal_text, avg_text)

ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, interval=600, blit=True, repeat=False)

fig.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.05)

writer = animation.FFMpegWriter(fps=1000/600, bitrate=3000)
ani.save("triple_burst_microsecond_fusion.mp4", writer=writer)
print("Video saved as triple_burst_microsecond_fusion.mp4")

print("\n--- Engine timing summary ---")
print(f"Mean per-step compute time: {np.mean(STATE.engine_times_us):.2f} us")
print(f"Mean throughput: {1e6/np.mean(STATE.engine_times_us):,.0f} steps/sec")
