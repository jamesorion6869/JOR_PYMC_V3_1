"""
plot_maneuver_log.py

Reads a maneuver_log_*.json file produced by maneuver_sim_interactive.py
(one JSON object per line) and renders a chart showing:

  - theta_s / theta_c / theta_o over time (thin lines)
  - NHP posterior over time (bold line)
  - the ALERT hysteresis band (0.40 - 0.50) shaded
  - scripted event markers (local strain @ t=30, catastrophic event @ t=110)
  - ALERT region shaded lightly across the timeline

Usage:
    python3 plot_maneuver_log.py maneuver_log_urban.json [output.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(path, run="last"):
    """
    Loads a maneuver_log_*.json file.

    Because FusionLogger appends to the log file rather than overwriting it,
    a single file can contain multiple concatenated simulation runs (each
    restarting its own tick counter at t=0). Naively plotting every line in
    file order draws a spurious connecting line from the end of one run to
    the start of the next. This function detects run boundaries (wherever
    t decreases instead of increasing) and, by default, returns only the
    most recent run.

    run: "last" (default) - only the most recent run
         "all"            - every run concatenated (may show connector artifacts)
         an integer index  - a specific run, 0-indexed in file order
    """
    all_entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_entries.append(json.loads(line))

    # Split into runs: a new run starts whenever t resets to a value
    # not greater than the previous entry's t.
    runs = []
    current_run = []
    prev_t = None
    for entry in all_entries:
        t = entry.get("metadata", {}).get("t")
        if prev_t is not None and t is not None and t <= prev_t:
            runs.append(current_run)
            current_run = []
        current_run.append(entry)
        prev_t = t
    if current_run:
        runs.append(current_run)

    if len(runs) > 1:
        print(f"Note: {len(runs)} separate runs detected in this log file "
              f"(it's append-only, so re-running the sim adds to it rather "
              f"than replacing it).")

    if run == "all":
        selected = all_entries
    elif run == "last":
        selected = runs[-1]
        if len(runs) > 1:
            print("Plotting only the most recent run. "
                  "Delete/rename the log file before re-running the sim "
                  "to avoid this, or pass run=<index> to pick a specific run.")
    else:
        selected = runs[int(run)]

    ticks, sop, nhp, alert = [], [], [], []
    theta_s, theta_c, theta_o = [], [], []

    for entry in selected:
        meta = entry.get("metadata", {})
        ticks.append(meta.get("t"))
        sop.append(entry.get("sop"))
        nhp.append(entry.get("nhp"))
        alert.append(entry.get("alert"))
        theta_s.append(meta.get("theta_s"))
        theta_c.append(meta.get("theta_c"))
        theta_o.append(meta.get("theta_o"))

    return {
        "t": np.array(ticks),
        "sop": np.array(sop),
        "nhp": np.array(nhp),
        "alert": np.array(alert),
        "theta_s": np.array(theta_s),
        "theta_c": np.array(theta_c),
        "theta_o": np.array(theta_o),
    }


def plot_log(data, title, out_path):
    t = data["t"]

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

    # --- Hysteresis band (0.40 - 0.50) ---
    ax.axhspan(0.40, 0.50, color="#f5b400", alpha=0.15, label="Hysteresis band (0.40-0.50)")

    # --- ALERT shading across the timeline ---
    alert = data["alert"]
    in_alert = False
    start = None
    for i, a in enumerate(alert):
        if a and not in_alert:
            in_alert = True
            start = t[i]
        elif not a and in_alert:
            in_alert = False
            ax.axvspan(start, t[i], color="#d62728", alpha=0.06)
    if in_alert:
        ax.axvspan(start, t[-1], color="#d62728", alpha=0.06)

    # --- Component signals (thin lines) ---
    ax.plot(t, data["theta_s"], color="#1f77b4", linewidth=1.3, alpha=0.85, label=r"$\theta_s$ readiness")
    ax.plot(t, data["theta_c"], color="#ff7f0e", linewidth=1.3, alpha=0.85, label=r"$\theta_c$ pressure")
    ax.plot(t, data["theta_o"], color="#2ca02c", linewidth=1.3, alpha=0.85, label=r"$\theta_o$ outcomes")

    # --- NHP posterior (bold) ---
    ax.plot(t, data["nhp"], color="#7b2cbf", linewidth=2.6, label="NHP posterior (JOR fusion output)")

    # --- Threshold reference lines ---
    ax.axhline(0.50, color="#d62728", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(0.40, color="#f5b400", linewidth=1.0, linestyle="--", alpha=0.6)

    # --- Scripted event markers ---
    event_ticks = {30: "Local strain (Unit 1)", 110: "Catastrophic event"}
    for et, label in event_ticks.items():
        if t.min() <= et <= t.max():
            ax.axvline(et, color="black", linewidth=1.0, linestyle=":")
            ax.text(et, 1.005, label, ha="center", va="bottom", fontsize=8.5,
                    transform=ax.get_xaxis_transform())

    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Simulation tick (t)")
    ax.set_ylabel("Value (0-1)")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=28)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    print(f"Saved chart to {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_maneuver_log.py <log_file.json> [output.png]")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.with_suffix(".png")

    data = load_log(log_path)
    scenario_name = log_path.stem.replace("maneuver_log_", "").replace("_", " ").title()
    title = f"JOR Bayesian Fusion — {scenario_name} Scenario"

    plot_log(data, title, out_path)


if __name__ == "__main__":
    main()
