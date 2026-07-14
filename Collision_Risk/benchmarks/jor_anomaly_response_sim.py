"""
JOR V3.1 Sensor & Flight Characteristics -- Hybrid Radar + Stability Visualization
---------------------------------------------------------------------------------
This visualization animates the extended 300s (60-step) Conventional-only
flight profile using the validated JOR V3.1 recursive Bayesian fusion math.

Left Panel:  Radar-style SOP vs Posterior NH track (60 steps)
Right Panel: Posterior NH vs Time, including:
    - 0.30 baseline (normal flight behavior)
    - rolling 100s average (20-step window)
    - noise-floor band (0.25–0.35)
    - Major Anomaly injection window (155s–230s)

This script matches the extended stability analysis mathematically and
provides a visual confirmation of long-term equilibrium behavior.
"""

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
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

PROFILE_NAME = "Conventional"
PROFILE_CFG = {"modifier": 0.00, "p_boost": 0.00}

# Major anomaly injection:
# Python steps 30 through 45 inclusive = 16 observations.
ANOMALY_START_STEP = 30
ANOMALY_END_STEP = 45

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# Bayesian Fusion Step
# ---------------------------------------------------------------
@njit(fastmath=True)
def bayesian_fusion_step(
    C,
    E,
    P_raw,
    modifier,
    K,
    prior_NH,
    prior_H,
    W_C=0.40,
    W_E=0.30,
    W_P=0.30,
):
    SOP = W_C * C + W_E * E + W_P * P_raw

    P_for_NHP = min(max(P_raw + modifier, 0.0), 0.95)
    NHP = W_C * C + W_E * E + W_P * P_for_NHP

    P_E_given_NH = NHP
    P_E_given_H = min(
        max(1.0 - NHP + K * SOP, 0.0),
        1.0,
    )

    numerator = P_E_given_NH * prior_NH
    denominator = numerator + (P_E_given_H * prior_H)

    posterior_NH = (
        numerator / denominator
        if denominator > 0
        else 0.0
    )

    return SOP, NHP, posterior_NH


# ---------------------------------------------------------------
# Sensor Model
# ---------------------------------------------------------------
def sensor_step():
    base_C = 0.65 + rng.normal(0, 0.03)
    base_E = 0.60 + rng.normal(0, 0.03)
    base_P = (
        0.55
        + PROFILE_CFG["p_boost"]
        + rng.normal(0, 0.03)
    )

    return (
        np.clip(base_C, 0.30, 0.85),
        np.clip(base_E, 0.30, 0.85),
        np.clip(base_P, 0.30, 0.95),
    )


# ---------------------------------------------------------------
# Run Extended Conventional Profile
# Major Anomaly Injected for 16 Steps
# ---------------------------------------------------------------
def run_extended_conventional():
    prior_NH = P_PRIOR_NH_INITIAL
    prior_H = P_PRIOR_H_INITIAL

    sop_track = []
    posterior_track = []
    rolling_avg_track = []

    for step in range(TIME_STEPS):

        # Major anomaly injection:
        # steps 30–45 inclusive = 16 observations
        if ANOMALY_START_STEP <= step <= ANOMALY_END_STEP:
            anomaly_modifier = 0.05
            anomaly_pboost = 0.15
        else:
            anomaly_modifier = PROFILE_CFG["modifier"]
            anomaly_pboost = PROFILE_CFG["p_boost"]

        C, E, P_raw = sensor_step()

        P_raw = np.clip(
            P_raw + anomaly_pboost,
            0.30,
            0.95,
        )

        SOP, NHP, posterior_NH = bayesian_fusion_step(
            C,
            E,
            P_raw,
            anomaly_modifier,
            K,
            prior_NH,
            prior_H,
        )

        sop_track.append(SOP)
        posterior_track.append(posterior_NH)

        # Rolling 100-second average = 20 observations
        rolling_avg = np.mean(
            posterior_track[-20:]
        )
        rolling_avg_track.append(rolling_avg)

        # Recursive prior regularization
        prior_NH = (
            PRIOR_RETENTION * posterior_NH
            + (1 - PRIOR_RETENTION)
            * P_PRIOR_NH_INITIAL
        )

        prior_H = 1.0 - prior_NH

    return (
        np.array(sop_track),
        np.array(posterior_track),
        np.array(rolling_avg_track),
        np.array(
            [
                (s + 1) * STEP_SECONDS
                for s in range(TIME_STEPS)
            ]
        ),
    )


# ---------------------------------------------------------------
# Precompute Tracks
# ---------------------------------------------------------------
print(
    "Pre-computing extended 300s Conventional track "
    "with Major Anomaly injection..."
)

SOP_TRACK, POST_TRACK, ROLLING_TRACK, TIME_AXIS = (
    run_extended_conventional()
)

# Displayed observation times for injected steps 30–45
ANOMALY_START_TIME = (
    ANOMALY_START_STEP + 1
) * STEP_SECONDS

ANOMALY_END_TIME = (
    ANOMALY_END_STEP + 1
) * STEP_SECONDS


# ---------------------------------------------------------------
# Visualization Theme
# ---------------------------------------------------------------
RADAR_BG = "#050a07"
RADAR_GREEN_DIM = "#1a4d1a"
CONVENTIONAL_COLOR = "#39ff14"

ANOMALY_SHADE_COLOR = "#ff3b30"
ANOMALY_TEXT_COLOR = "#ff6b61"

plt.style.use("dark_background")


# ---------------------------------------------------------------
# Create Figure
# ---------------------------------------------------------------
fig = plt.figure(figsize=(15, 6.5))

try:
    mngr = plt.get_current_fig_manager()
    fig.canvas.draw()

    screen = mngr.window.screen().availableGeometry()
    size = mngr.window.size()

    x = (screen.width() - size.width()) // 2
    y = (screen.height() - size.height()) // 2

    mngr.window.move(x, y)

except Exception:
    pass

fig.patch.set_facecolor(RADAR_BG)

gs = fig.add_gridspec(
    1,
    2,
    wspace=0.28,
)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])


# ---------------------------------------------------------------
# Radar Panel
# ---------------------------------------------------------------
ax1.set_facecolor(RADAR_BG)

ax1.set_title(
    "Recursive State Trajectory: "
    "SOP vs Posterior NH",
    color=CONVENTIONAL_COLOR,
)

sop_min = max(
    0.0,
    SOP_TRACK.min() - 0.05,
)

sop_max = min(
    1.0,
    SOP_TRACK.max() + 0.05,
)

post_min = max(
    0.0,
    POST_TRACK.min() - 0.05,
)

post_max = min(
    1.0,
    POST_TRACK.max() + 0.05,
)

CENTER = (
    (sop_min + sop_max) / 2,
    (post_min + post_max) / 2,
)

MAX_RING_RADIUS = (
    min(
        sop_max - sop_min,
        post_max - post_min,
    )
    / 2
    * 0.95
)

ax1.set_xlim(
    sop_min,
    sop_max,
)

ax1.set_ylim(
    post_min,
    post_max,
)

ax1.set_aspect("equal")

ax1.set_xlabel(
    "SOP (Baseline)",
    color=CONVENTIONAL_COLOR,
)

ax1.set_ylabel(
    "Posterior NH",
    color=CONVENTIONAL_COLOR,
)

ax1.tick_params(
    colors=RADAR_GREEN_DIM
)

for spine in ax1.spines.values():
    spine.set_color(
        RADAR_GREEN_DIM
    )

# Radar rings
for i in range(1, 5):

    r = (
        MAX_RING_RADIUS
        * i
        / 4
    )

    ax1.add_patch(
        Circle(
            CENTER,
            r,
            fill=False,
            edgecolor=RADAR_GREEN_DIM,
            alpha=0.7,
        )
    )

# Radar spokes
for angle_deg in range(
    0,
    360,
    45,
):

    angle = np.radians(
        angle_deg
    )

    ax1.plot(
        [
            CENTER[0],
            CENTER[0]
            + MAX_RING_RADIUS
            * np.cos(angle),
        ],
        [
            CENTER[1],
            CENTER[1]
            + MAX_RING_RADIUS
            * np.sin(angle),
        ],
        color=RADAR_GREEN_DIM,
        alpha=0.6,
    )

radar_line, = ax1.plot(
    [],
    [],
    color=CONVENTIONAL_COLOR,
    linewidth=1.5,
)

radar_marker, = ax1.plot(
    [],
    [],
    "o",
    color=CONVENTIONAL_COLOR,
    markersize=6,
)


# ---------------------------------------------------------------
# Stability Panel
# ---------------------------------------------------------------
ax2.set_facecolor(
    RADAR_BG
)

ax2.set_title(
    "Recursive Posterior NH Response "
    "(300s)",
    color=CONVENTIONAL_COLOR,
)

ax2.set_xlim(
    0,
    TIME_AXIS[-1],
)

ax2.set_ylim(
    0,
    1.0,
)

ax2.set_xlabel(
    "Time (s)",
    color=CONVENTIONAL_COLOR,
)

ax2.set_ylabel(
    "Posterior NH",
    color=CONVENTIONAL_COLOR,
)

ax2.tick_params(
    colors=RADAR_GREEN_DIM
)

for spine in ax2.spines.values():
    spine.set_color(
        RADAR_GREEN_DIM
    )

ax2.grid(
    color=RADAR_GREEN_DIM,
    alpha=0.3,
)


# ---------------------------------------------------------------
# Normal Flight Baseline
# ---------------------------------------------------------------
ax2.axhline(
    y=0.30,
    color=CONVENTIONAL_COLOR,
    linestyle="--",
    alpha=0.6,
)

ax2.text(
    5,
    0.315,
    "Normal Flight Baseline (0.30)",
    color=CONVENTIONAL_COLOR,
    fontsize=8,
)


# ---------------------------------------------------------------
# Noise Floor Band
# ---------------------------------------------------------------
ax2.fill_between(
    TIME_AXIS,
    0.25,
    0.35,
    color=RADAR_GREEN_DIM,
    alpha=0.15,
    label="Noise Floor Band (0.25–0.35)",
)


# ---------------------------------------------------------------
# Major Anomaly Injection Window
# ---------------------------------------------------------------
ax2.axvspan(
    ANOMALY_START_TIME,
    ANOMALY_END_TIME,
    color=ANOMALY_SHADE_COLOR,
    alpha=0.10,
    label=(
        "Major Anomaly Injection "
        f"({ANOMALY_START_TIME}–{ANOMALY_END_TIME}s)"
    ),
)

ax2.text(
    (
        ANOMALY_START_TIME
        + ANOMALY_END_TIME
    )
    / 2,
    0.92,
    "MAJOR ANOMALY\nINJECTED",
    color=ANOMALY_TEXT_COLOR,
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------
# Animated Stability Tracks
# ---------------------------------------------------------------
post_line, = ax2.plot(
    [],
    [],
    color=CONVENTIONAL_COLOR,
    linewidth=1.5,
)

post_marker, = ax2.plot(
    [],
    [],
    "o",
    color=CONVENTIONAL_COLOR,
    markersize=6,
)

rolling_line, = ax2.plot(
    [],
    [],
    color="#00ffaa",
    linewidth=1.2,
    alpha=0.8,
    label="Rolling Avg (100s)",
)

ax2.legend(
    loc="upper left",
    fontsize=8,
    facecolor=RADAR_BG,
    edgecolor=RADAR_GREEN_DIM,
)


# ---------------------------------------------------------------
# Animation Status Box
# ---------------------------------------------------------------
title_text = ax2.text(
    0.98,
    0.96,
    "",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    color=CONVENTIONAL_COLOR,
    fontsize=9,
    bbox=dict(
        facecolor=RADAR_BG,
        edgecolor=RADAR_GREEN_DIM,
        alpha=0.85,
    ),
)


# ---------------------------------------------------------------
# Animation Update
# ---------------------------------------------------------------
def update(frame):

    step = frame + 1

    # Radar panel
    radar_line.set_data(
        SOP_TRACK[:step],
        POST_TRACK[:step],
    )

    radar_marker.set_data(
        [SOP_TRACK[step - 1]],
        [POST_TRACK[step - 1]],
    )

    # Stability panel
    post_line.set_data(
        TIME_AXIS[:step],
        POST_TRACK[:step],
    )

    post_marker.set_data(
        [TIME_AXIS[step - 1]],
        [POST_TRACK[step - 1]],
    )

    rolling_line.set_data(
        TIME_AXIS[:step],
        ROLLING_TRACK[:step],
    )

    title_text.set_text(
        f"t={TIME_AXIS[step - 1]}s "
        f"({step}/{TIME_STEPS})"
    )

    return (
        radar_line,
        radar_marker,
        post_line,
        post_marker,
        rolling_line,
        title_text,
    )


# ---------------------------------------------------------------
# Run Animation
# ---------------------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=TIME_STEPS,
    interval=600,
    blit=True,
    repeat=False,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.show()