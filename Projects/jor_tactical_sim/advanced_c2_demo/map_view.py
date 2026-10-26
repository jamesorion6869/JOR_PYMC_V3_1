"""
JOR theater map renderer.

Visual backlog features:
  - MIL-STD-2525 / APP-6 inspired friendly land-unit frames (simplified)
  - Spatial NHP risk as a continuous heat field (Gaussian kernels), not only circles
  - Multi-frame animation (GIF) for mission playback
  - Optional live figure updates during a running simulation

Usage:
    python3 map_view.py map_frames_baseline.jsonl [out.png]
    python3 map_view.py map_frames_baseline.jsonl out.png --t 110
    python3 map_view.py map_frames_baseline.jsonl mission.gif --animate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib import animation

from theater_layout import theater_bounds


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_frames(path: str) -> List[Dict[str, Any]]:
    frames = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _readiness_fill(r: float) -> Tuple[float, float, float]:
    """Interior fill: green (healthy) → yellow → red (exhausted)."""
    r = float(np.clip(r, 0.0, 1.0))
    if r >= 0.5:
        t = (r - 0.5) / 0.5
        return (1.0 - t, 1.0, 0.15)
    t = r / 0.5
    return (1.0, t, 0.1)


# ---------------------------------------------------------------------------
# MIL-STD-2525 / APP-6 inspired symbology (simplified friendly land units)
# ---------------------------------------------------------------------------
#
# Full MIL-STD-2525 is a large icon library. For this synthetic demo we use the
# doctrinal *frame + affiliation color* pattern commanders scan first:
#   - Friendly affiliation → blue frame
#   - Land unit → rectangular frame with slightly concave top (APP-6 style approx)
#   - Main icon: infantry "X" or armor ellipse depending on unit role heuristic
#   - Degraded: black diagonal slash + thicker frame

FRIENDLY_FRAME = "#2b5cff"       # affiliation blue
FRIENDLY_FRAME_ALERT = "#8b0000"  # dark red when force is in ALERT
FRIENDLY_FILL_DEFAULT = "#d6e4ff"


def _unit_role(unit_id: int) -> str:
    """Cheap role assignment for demo variety (not true TO&E)."""
    roles = ("infantry", "infantry", "armor", "infantry", "recon", "armor")
    return roles[unit_id % len(roles)]


def draw_mil_symbol(ax, x: float, y: float, readiness: float, unit_id: int,
                    degraded: bool = False, alert: bool = False, scale: float = 0.55):
    """
    Draw a simplified friendly land-unit symbol centered at (x, y).

    Frame follows APP-6 friendly rectangle; interior tint tracks readiness;
    icon marks role. Returns nothing (draws on ax).
    """
    w, h = scale * 1.15, scale * 0.85
    frame_color = FRIENDLY_FRAME_ALERT if (alert and readiness < 0.45) else FRIENDLY_FRAME
    fill = _readiness_fill(readiness)

    # Main frame (rounded rectangle approximates land unit box)
    frame = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.8 if not degraded else 2.4,
        edgecolor=frame_color,
        facecolor=fill,
        zorder=5,
    )
    ax.add_patch(frame)

    role = _unit_role(unit_id)
    ic = frame_color
    if role == "infantry":
        # Saltire (X) — infantry
        s = scale * 0.28
        ax.plot([x - s, x + s], [y - s * 0.7, y + s * 0.7], color=ic, lw=1.6, zorder=6)
        ax.plot([x - s, x + s], [y + s * 0.7, y - s * 0.7], color=ic, lw=1.6, zorder=6)
    elif role == "armor":
        # Horizontal oval — armor
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((x, y), w * 0.55, h * 0.35, fill=False,
                             edgecolor=ic, linewidth=1.6, zorder=6))
    else:  # recon
        # Chevron
        s = scale * 0.22
        ax.plot([x - s, x, x + s], [y - s * 0.5, y + s * 0.6, y - s * 0.5],
                color=ic, lw=1.6, zorder=6)

    if degraded:
        # Battle-damage slash
        ax.plot([x - w * 0.4, x + w * 0.4], [y - h * 0.35, y + h * 0.35],
                color="black", lw=2.0, zorder=7, solid_capstyle="round")

    # Unit designator
    label = f"U{unit_id}"
    if degraded:
        label += "*"
    ax.text(x, y - h * 0.95, label, ha="center", va="top", fontsize=7,
            fontweight="bold", color="#222222", zorder=8)


# ---------------------------------------------------------------------------
# Spatial NHP heat field
# ---------------------------------------------------------------------------

def nhp_heat_field(units: List[Dict[str, Any]], nhp: float, alert: bool,
                   bounds: Tuple[float, float, float, float],
                   grid: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a continuous risk field by placing Gaussian kernels at each unit.

    Kernel amplitude rises with force NHP and local readiness deficit; degraded
    units contribute a stronger local peak. Returns (X, Y, Z) mesh grids.
    """
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(ymin, ymax, grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X, dtype=float)

    # Base spatial scale in theater units
    sigma = 1.15 + 1.6 * float(nhp)
    for u in units:
        ux, uy = float(u["x"]), float(u["y"])
        readiness = float(u.get("readiness", 0.7))
        local = float(nhp) * (0.45 + 0.55 * (1.0 - readiness))
        if u.get("degraded"):
            local *= 1.45
        if alert:
            local *= 1.15
        local = max(local, 0.02)
        dist2 = (X - ux) ** 2 + (Y - uy) ** 2
        Z += local * np.exp(-dist2 / (2.0 * sigma ** 2))

    # Normalize so the peak relates to force NHP for stable color scales
    peak = float(Z.max()) if Z.max() > 0 else 1.0
    target_peak = max(0.08, min(1.0, 0.25 + 0.9 * float(nhp)))
    Z *= target_peak / peak
    return X, Y, Z


# ---------------------------------------------------------------------------
# Frame render
# ---------------------------------------------------------------------------

def render_frame(frame: Dict[str, Any], out_path: Optional[str] = None,
                 title: Optional[str] = None, ax=None, fig=None,
                 show_colorbar: bool = True) -> Tuple[Any, Any]:
    """
    Render one telemetry frame. If ax/fig are provided, draw into them
    (for live / animation). If out_path is set, save a PNG.
    """
    units = frame["units"]
    if not units:
        raise ValueError("frame has no units")

    xs = np.array([u["x"] for u in units], dtype=float)
    ys = np.array([u["y"] for u in units], dtype=float)
    positions = np.column_stack([xs, ys])
    bounds = theater_bounds(positions, pad=2.5)
    xmin, xmax, ymin, ymax = bounds

    own_fig = fig is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=140)
    assert ax is not None and fig is not None
    ax.clear()

    nhp = float(frame.get("nhp", 0.0))
    alert = bool(frame.get("alert", False))

    # --- Continuous NHP risk field ---
    X, Y, Z = nhp_heat_field(units, nhp, alert, bounds)
    levels = np.linspace(0.0, max(0.15, float(Z.max()) + 1e-6), 12)
    cmap = plt.get_cmap("YlOrRd")
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.55, zorder=1)
    ax.contour(X, Y, Z, levels=levels[1::2], colors="#8b0000", linewidths=0.4,
               alpha=0.35, zorder=2)

    # Soft halo rings still mark degraded units for quick scan
    for u in units:
        if u.get("degraded"):
            ax.add_patch(Circle(
                (u["x"], u["y"]), 0.9 + 1.2 * nhp,
                facecolor=(0.8, 0.05, 0.05, 0.12 + 0.25 * nhp),
                edgecolor=(0.5, 0, 0, 0.35), linewidth=1.0, zorder=3,
            ))

    # --- MIL-style unit symbols ---
    for u in units:
        draw_mil_symbol(
            ax, u["x"], u["y"],
            readiness=float(u["readiness"]),
            unit_id=int(u["id"]),
            degraded=bool(u.get("degraded")),
            alert=alert,
        )

    state = frame.get("mission_state", "")
    scen = frame.get("scenario", "")
    t = frame.get("t", 0)
    default_title = f"JOR Theater Map — {scen}  t={t}  [{state}]"
    ax.set_title(title or default_title, fontsize=12, fontweight="bold", pad=12)

    metrics = (
        f"NHP={nhp:.3f}  {'ALERT' if alert else 'Normal'}  |  "
        f"θs={frame.get('theta_s', 0):.2f}  θc={frame.get('theta_c', 0):.2f}  "
        f"θo={frame.get('theta_o', 0):.2f}  morale={frame.get('morale', 0):.2f}"
    )
    ax.text(0.5, -0.08, metrics, transform=ax.transAxes, ha="center", fontsize=9,
            color="#333333")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Theater X")
    ax.set_ylabel("Theater Y")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_facecolor("#f4f5f2")

    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2b5cff",
               markeredgecolor="#2b5cff", markersize=9, label="Friendly frame (APP-6 style)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=(0.2, 1.0, 0.15),
               markersize=9, label="High readiness fill"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=(1.0, 0.25, 0.1),
               markersize=9, label="Low readiness fill"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.9, 0.3, 0.1),
               alpha=0.5, markersize=12, label="NHP risk field"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7.5, framealpha=0.92)

    if show_colorbar and own_fig:
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Spatial NHP risk", fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, facecolor="white", bbox_inches="tight")
        print(f"Saved map to {out_path}")
    if own_fig and out_path:
        plt.close(fig)
    return fig, ax


def render_animation(frames: List[Dict[str, Any]], out_path: str,
                     stride: int = 2, fps: int = 8) -> None:
    """Write a GIF (or MP4 if extension is .mp4) stepping through frames."""
    if not frames:
        raise ValueError("no frames")
    subset = frames[:: max(1, stride)]
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=110)

    def _update(i):
        render_frame(subset[i], out_path=None, ax=ax, fig=fig, show_colorbar=False)
        return []

    anim = animation.FuncAnimation(
        fig, _update, frames=len(subset), interval=1000 / max(fps, 1), blit=False
    )
    out_path = str(out_path)
    if out_path.lower().endswith(".mp4"):
        anim.save(out_path, writer="ffmpeg", fps=fps)
    else:
        anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved animation to {out_path} ({len(subset)} frames)")


# ---------------------------------------------------------------------------
# Live map (optional interactive / headless tick callback)
# ---------------------------------------------------------------------------

class LiveMapDisplay:
    """
    Holds a matplotlib figure and redraws each telemetry frame.

    Use from the sim loop:
        live = LiveMapDisplay()
        ...
        live.update(frame)
        live.close()
    """

    def __init__(self, pause: float = 0.05):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 7.5), dpi=110)
        self.pause = pause

    def update(self, frame: Dict[str, Any]) -> None:
        render_frame(frame, out_path=None, ax=self.ax, fig=self.fig, show_colorbar=False)
        self.fig.canvas.draw_idle()
        plt.pause(self.pause)

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description="Render JOR map telemetry")
    parser.add_argument("frames_file", help="JSON-lines map telemetry file")
    parser.add_argument("out", nargs="?", default=None, help="output PNG or GIF path")
    parser.add_argument("--t", type=int, default=None, help="render specific tick (default: last)")
    parser.add_argument("--animate", action="store_true",
                        help="write an animation from all frames (GIF/MP4)")
    parser.add_argument("--stride", type=int, default=3, help="keep every Nth frame in animation")
    parser.add_argument("--fps", type=int, default=6, help="animation frames per second")
    args = parser.parse_args()

    frames = load_frames(args.frames_file)
    if not frames:
        print("No frames in file", file=sys.stderr)
        sys.exit(1)

    if args.animate:
        out = args.out or str(Path(args.frames_file).with_suffix(".gif"))
        render_animation(frames, out, stride=args.stride, fps=args.fps)
        return

    if args.t is not None:
        chosen = next((f for f in frames if f.get("t") == args.t), None)
        if chosen is None:
            print(f"No frame with t={args.t}", file=sys.stderr)
            sys.exit(1)
    else:
        chosen = frames[-1]

    out = args.out or str(Path(args.frames_file).with_suffix(".png"))
    render_frame(chosen, out)


if __name__ == "__main__":
    main()
