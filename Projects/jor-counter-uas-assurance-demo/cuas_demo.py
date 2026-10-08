"""
Prototype: JOR sensing-assurance meta-layer for Counter-UAS (C-UAS).

This layer does NOT detect, classify or defeat drones. It reports how much
the detect/track picture can currently be trusted, so an operator or
higher-level C2 knows when that ability is becoming unreliable.

Demonstrates:
  - self-calibrating baseline across the normal load range
  - quiet sky (no targets, feeds alive) stays NOMINAL
  - a single degraded source (gray failure) is flagged and attributed
  - lost sensor feeds (stale heartbeats) are flagged via sensor integrity
  - congestion + noise drives the state to UNRELIABLE
  - recovery when conditions return to normal

Operator panel expansions:
  SOP = System Operating Picture
  NHP = Non-Healthy Posterior  (uncalibrated index that detect/track is degraded)
"""

import argparse
import json
import os
import uuid
from datetime import datetime

import numpy as np

from engine import JOREngine
from cuas_adapter import CUASHealthAdapter
from logger import FusionLogger

STATE_FILE = "cuas_engine_state.json"
PANEL_W = 62


# ----------------------------------------------------------------------
# State persistence
# ----------------------------------------------------------------------
def load_engine_state(engine):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        engine.p_final = state.get("p_final", engine.prior_nh)
        engine.alert_status = state.get("alert_status", False)
        engine.baseline_sop = state.get("baseline_sop", engine.baseline_sop)
        engine.baseline_channels = state.get("baseline_channels", engine.baseline_channels)
        engine.is_calibrated = state.get("is_calibrated", engine.is_calibrated)
        engine.calibrating = not engine.is_calibrated
        engine.degraded_status = engine.p_final >= engine.degraded_th
        print("--- Engine state restored from disk ---")


def save_engine_state(engine):
    state = {
        "p_final": engine.p_final,
        "alert_status": engine.alert_status,
        "baseline_sop": engine.baseline_sop,
        "baseline_channels": engine.baseline_channels,
        "is_calibrated": engine.is_calibrated,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


class Clock:
    """Simulated 1 Hz clock so status reports carry a timestamp."""
    def __init__(self):
        self.t = 0.0

    def tick(self):
        self.t += 1.0
        return self.t


# ----------------------------------------------------------------------
# Simulation helpers
# ----------------------------------------------------------------------
def simulate_source_tracks(num_sources=8, bad_source_idx=None,
                           base_continuity=0.94, bad_continuity=0.48,
                           base_confidence=0.88, n_tracks_per=25,
                           empty_sources=()):
    """Per-source track quality. One optional 'bad' source has low continuity.
    Sources in `empty_sources` report no tracks (quiet sky or lost feed)."""
    source_tracks = {}
    for i in range(num_sources):
        sid = f"src_{i}"
        if i in empty_sources:
            cont, conf = np.array([]), np.array([])
        elif i == bad_source_idx:
            cont = np.clip(np.random.normal(bad_continuity, 0.06, n_tracks_per), 0.05, 1.0)
            conf = np.clip(np.random.normal(0.55, 0.08, n_tracks_per), 0.05, 1.0)
        else:
            cont = np.clip(np.random.normal(base_continuity, 0.03, n_tracks_per), 0.05, 1.0)
            conf = np.clip(np.random.normal(base_confidence, 0.04, n_tracks_per), 0.05, 1.0)
        source_tracks[sid] = {"continuities": cont, "confidences": conf}
    return source_tracks


def simulate_heartbeats(num_sources=8, stale=()):
    return {f"src_{i}": (6.0 if i in stale else float(np.random.uniform(0.1, 0.8)))
            for i in range(num_sources)}


def _val(x, i, n):
    """Scalar, or (start, end) tuple linearly interpolated across the phase."""
    if isinstance(x, tuple):
        return x[0] + (x[1] - x[0]) * (i / max(n - 1, 1))
    return x


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


DETAIL_KEYS = ("per_source", "degraded_sources", "stale_sources", "malformed_sources", "idle_sources",
               "no_targets", "data_missing", "median_continuity", "median_confidence",
               "noise_floor", "persistent_outliers", "pending_outliers")


def _details(obs):
    return {k: _jsonable(obs[k]) for k in DETAIL_KEYS if k in obs}


# ----------------------------------------------------------------------
# Operator panel
# ----------------------------------------------------------------------
def print_operator_panel(status, theta_s, theta_c, theta_o, step_label=""):
    d = status["details"]

    def row(text=""):
        print("│" + f" {text}".ljust(PANEL_W)[:PANEL_W] + "│")

    calibrating = status["state"] == "CALIBRATING"
    base_txt = "--" if status["baseline_sop"] is None else f"{status['baseline_sop']:.3f}"
    nhp_txt = "  -- " if calibrating else f"{status['nhp'] * 100:5.1f}%"
    driver = status["dominant_channel"]
    if driver is None:
        # Not NOMINAL but no channel above baseline = the recursive state is still decaying
        driver = "none now" if status["state"] in ("DEGRADED", "UNRELIABLE") else "-"
    no_track_data = bool(d.get("no_targets") or d.get("data_missing"))
    cont_txt = "n/a " if no_track_data else f"{d.get('median_continuity', 0.0):.2f}"
    conf_txt = "n/a" if no_track_data else f"{d.get('median_confidence', 0.0):.2f}"
    marker = "●" if status["alert"] else "○"

    print("┌" + "─" * PANEL_W + "┐")
    if step_label:
        row(step_label)
    row("JOR C-UAS Sensing-Assurance Monitor")
    row("SOP = System Operating Picture   NHP = Non-Healthy Posterior")
    print("├" + "─" * PANEL_W + "┤")
    row(f"DETECT/TRACK ASSURANCE  {marker} {status['state']:<11} NHP {nhp_txt}")
    row(f"SOP {status['sop']:.3f} (baseline {base_txt})   driver: {driver}")
    print("├" + "─" * PANEL_W + "┤")
    row(f"θ_o  Track Quality     {theta_o:.2f}   cont {cont_txt}  conf {conf_txt}")
    row(f"θ_c  Sensing Load      {theta_c:.2f}   noise {d.get('noise_floor', 0.0):.2f}")
    row(f"θ_s  Sensor Integrity  {theta_s:.2f}")
    parts = []
    if d.get("degraded_sources"):
        parts.append("degraded " + ",".join(d["degraded_sources"]))
    if d.get("stale_sources"):
        parts.append("stale " + ",".join(d["stale_sources"]))
    if d.get("malformed_sources"):
        parts.append("malformed " + ",".join(d["malformed_sources"]))
    if d.get("per_source") is False:
        row("Sources: n/a (aggregate input, no per-source detail)")
    else:
        row("Sources: " + ("; ".join(parts) if parts else "none flagged"))
    if d.get("data_missing"):
        row("Mode: NO DATA (feeds stale/invalid)")
    elif d.get("no_targets"):
        row("Mode: no targets (feeds alive)")
    if status["data_fault"]:
        row("Monitor input fault: holding last state")
    print("└" + "─" * PANEL_W + "┘")
    print()


def fuse_and_report(engine, logger, clock, label, i, n_steps, obs,
                    theta_s, theta_c, show_panel):
    sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs["theta_o"])
    status = engine.status(ts=clock.tick(), details=_details(obs))
    if show_panel or i == n_steps - 1:
        print_operator_panel(status, theta_s, theta_c, obs["theta_o"],
                             step_label=f"{label}  step {i + 1}/{n_steps}")
    logger.log_state(sop, nhp, alert, metadata={
        "phase": label, "step": i,
        "theta_s": theta_s, "theta_c": theta_c, "theta_o": obs["theta_o"],
        "status": status,
    })
    return status


# ----------------------------------------------------------------------
# Phase runners
# ----------------------------------------------------------------------
def run_phase(engine, adapter, logger, clock, label, n_steps,
              continuity=0.94, confidence=0.88, noise=0.12, fa_rate=0.02,
              coop_tracks=15, total_volume=25, rf_density=0.20,
              sensor_health=None, platform_health=0.97, backlog=0.05,
              panel_every=2):
    """Aggregate-only phase (no per-source detail)."""
    print(f"\n=== Phase: {label} ===")
    if sensor_health is None:
        sensor_health = {f"src_{k}": 0.96 for k in range(8)}
    for i in range(n_steps):
        cont = np.clip(np.random.normal(continuity, 0.03, 40), 0.05, 1.0)
        conf = np.clip(np.random.normal(confidence, 0.04, 40), 0.05, 1.0)
        obs = adapter.extract_features(cont, conf, noise, fa_rate)
        theta_c = adapter.normalize_context(int(round(_val(coop_tracks, i, n_steps))),
                                            int(round(_val(total_volume, i, n_steps))),
                                            _val(rf_density, i, n_steps))
        theta_s = adapter.normalize_system_state(sensor_health, platform_health, backlog)
        fuse_and_report(engine, logger, clock, label, i, n_steps, obs,
                        theta_s, theta_c, show_panel=(i % panel_every == 0))


def run_source_phase(engine, adapter, logger, clock, label, n_steps,
                     bad_source=None, bad_onset=0, bad_continuity=0.42,
                     stale_sources=(), stale_onset=0, no_targets=False,
                     noise=(0.13, 0.13), fa_rate=0.03,
                     coop_tracks=20, total_volume=35, rf_density=0.25,
                     platform_health=0.94, backlog=0.08, panel_every=1):
    """Per-source phase with heartbeats: gray failure, feed loss, or quiet sky."""
    print(f"\n=== Phase: {label} ===")
    for i in range(n_steps):
        bad_idx = bad_source if (bad_source is not None and i >= bad_onset) else None
        stale = tuple(stale_sources) if i >= stale_onset else ()
        empty = set(range(8)) if no_targets else set(stale)
        source_tracks = simulate_source_tracks(bad_source_idx=bad_idx,
                                               bad_continuity=bad_continuity,
                                               empty_sources=empty)
        beats = simulate_heartbeats(stale=stale)
        # Noise steps from its first to its second value once a fault begins
        onset = bad_onset if bad_source is not None else stale_onset
        nf = noise[1] if (bad_source is not None or stale_sources) and i >= onset else noise[0]
        obs = adapter.extract_features_per_source(source_tracks, nf, fa_rate,
                                                  heartbeat_ages=beats)

        sensor_health = {f"src_{k}": 0.95 for k in range(8)}
        if bad_idx is not None:
            sensor_health[f"src_{bad_idx}"] = 0.55        # last-reported health
        theta_c = adapter.normalize_context(coop_tracks, total_volume, rf_density)
        theta_s = adapter.normalize_system_state(sensor_health, platform_health, backlog,
                                                 heartbeat_ages=beats)
        fuse_and_report(engine, logger, clock, label, i, n_steps, obs,
                        theta_s, theta_c, show_panel=(i % panel_every == 0))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="restore engine state from disk (default: start fresh)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz", nargs="?", const="cuas_timeline.png", metavar="PNG",
                        help="also render a static timeline PNG (needs matplotlib; optional)")
    parser.add_argument("--video", nargs="?", const="cuas_replay.mp4", metavar="MP4",
                        help="also render an animated replay (mp4 via ffmpeg, else gif; optional)")
    args = parser.parse_args(argv)
    np.random.seed(args.seed)

    print("=" * 64)
    print("  JOR C-UAS Sensing-Assurance Prototype")
    print("  A meta-layer: it does not detect or defeat drones.")
    print("  It reports when the ability to detect and track is")
    print("  becoming unreliable.")
    print("=" * 64)
    print("Evidence channels:")
    print("  θ_s  = sensor & platform integrity, feed liveness")
    print("  θ_c  = load on the sensing system (saturation, RF, cooperative mix)")
    print("  θ_o  = track continuity, classification confidence,")
    print("         sensor noise, per-source track anomalies")
    print()

    engine = JOREngine(prior_nh=0.05, steepness=12.0,
                       upper_th=0.55, lower_th=0.45,
                       retention=0.70, calibration_steps=20,
                       delta_deadband=0.02)
    print(f"NHP reachable range: {engine.nhp_floor * 100:.1f}% - {engine.nhp_ceiling * 100:.1f}%"
          f"  (set by retention={engine.retention})")
    if args.resume:
        load_engine_state(engine)

    adapter = CUASHealthAdapter()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    logger = FusionLogger("cuas_airspace_log.jsonl", run_id=run_id)
    clock = Clock()
    ctx = dict(engine=engine, adapter=adapter, logger=logger, clock=clock)

    # 1. Calibration across the normal load range
    if engine.is_calibrated:
        print(f"  >> Resumed calibrated baseline SOP = {engine.baseline_sop:.4f} "
              f"(calibration phase skipped)\n")
    else:
        run_phase(**ctx, label="Calibration (healthy)", n_steps=20,
                  continuity=0.94, confidence=0.90, noise=0.10, fa_rate=0.01,
                  coop_tracks=(10, 22), total_volume=(16, 36), rf_density=(0.13, 0.26),
                  platform_health=0.98, backlog=0.03, panel_every=5)
        print(f"  >> Calibrated baseline SOP = {engine.baseline_sop:.4f} "
              f"(spread {engine.baseline_spread:.4f})")
        if engine.calibration_suspect:
            print("  !! WARNING: calibration window contains outliers - baseline may be contaminated")
        print()

    # 2. Healthy operations
    run_phase(**ctx, label="Healthy operations", n_steps=8,
              continuity=0.93, confidence=0.88, noise=0.12, fa_rate=0.02,
              coop_tracks=18, total_volume=28, rf_density=0.18, panel_every=3)

    # 3. Quiet sky: feeds alive, nothing to track -> must stay NOMINAL
    run_source_phase(**ctx, label="Quiet sky (no targets)", n_steps=8, no_targets=True,
                     noise=(0.12, 0.12), fa_rate=0.0,
                     coop_tracks=2, total_volume=2, rf_density=0.12,
                     platform_health=0.97, backlog=0.03, panel_every=4)

    # 4. Gray failure: one source develops persistently low continuity
    run_source_phase(**ctx, label="Gray failure (src_3)", n_steps=14,
                     bad_source=3, bad_onset=5, bad_continuity=0.42,
                     noise=(0.13, 0.28), fa_rate=0.04)

    # 5. Recovery between faults
    run_source_phase(**ctx, label="Recovery A", n_steps=8,
                     noise=(0.11, 0.11), fa_rate=0.015,
                     coop_tracks=16, total_volume=24, rf_density=0.16,
                     platform_health=0.97, backlog=0.04, panel_every=4)

    # 6. Feed loss: three sensors stop sending heartbeats
    run_source_phase(**ctx, label="Feed loss (src_1,2,5)", n_steps=12,
                     stale_sources=(1, 2, 5), stale_onset=4,
                     noise=(0.12, 0.12), fa_rate=0.02,
                     coop_tracks=18, total_volume=28, rf_density=0.18,
                     platform_health=0.96, backlog=0.05, panel_every=2)

    # 7. Congested + noisy (no single bad source)
    run_phase(**ctx, label="Congested + high noise", n_steps=10,
              continuity=0.78, confidence=0.68, noise=0.55, fa_rate=0.12,
              coop_tracks=40, total_volume=95, rf_density=0.65,
              platform_health=0.90, backlog=0.25, panel_every=3)

    # 8. Recovery
    run_source_phase(**ctx, label="Recovery", n_steps=12,
                     noise=(0.11, 0.11), fa_rate=0.015,
                     coop_tracks=16, total_volume=24, rf_density=0.16,
                     platform_health=0.97, backlog=0.04, panel_every=3)

    save_engine_state(engine)

    final = engine.status(ts=clock.t)
    print("=" * 64)
    print("  SUMMARY")
    print(f"  Calibrated baseline SOP : {engine.baseline_sop:.4f}")
    print(f"  Final NHP               : {engine.p_final:.4f}")
    print(f"  Final state             : {final['state']}")
    print(f"  Log appended to         : cuas_airspace_log.jsonl (run {run_id})")
    print("  Final status report (C2 payload):")
    print("  " + json.dumps(final))
    print("=" * 64)

    if args.viz or args.video:
        try:
            import viz                      # optional; the core demo never needs it
            for path in viz.render("cuas_airspace_log.jsonl", png=args.viz, video=args.video,
                                   run_id=run_id):
                print(f"  wrote {path}")
        except (ImportError, SystemExit) as exc:
            print(f"  (visualisation skipped: {exc})")


if __name__ == "__main__":
    main()
