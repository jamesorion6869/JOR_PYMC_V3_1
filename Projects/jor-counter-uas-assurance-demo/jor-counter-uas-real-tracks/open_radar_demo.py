"""
JOR C-UAS sensing-assurance monitor replayed on REAL Open Radar tracks.

The monitor does not detect or track targets. This script feeds it quality
metadata (update continuity from timestamps, confidence from SNR) taken from
real Open Radar tracks, injects faults into that stream, logs every step, and
renders the log with viz.py as a PNG and an MP4.

What is real and what is not:
  real      : the track timestamps and SNR values (Open Radar Initiative,
              Outdoor Moving Object dataset)
  injected  : every fault (stale heartbeat, continuity/confidence loss, EW-style
              noise/RF/load). Open Radar contains no real sensor failures.
  assumed   : the 8 "sources" are 8 unrelated tracks standing in for parallel
              sensors, and the sensor-health / load numbers are fixed constants.

Usage (needs moving_target_dataset.npy, engine.py, cuas_adapter.py, logger.py,
cuas_demo.py, viz.py and open_radar_eval_v2.py in the same folder):

    python open_radar_demo.py                       # log + PNG + MP4
    python open_radar_demo.py --no-video            # log + PNG only
    python open_radar_demo.py --npy other.npy --seed 7

Data: Open Radar Initiative, CC BY-NC 4.0. Cite Gusland et al., "Open Radar
Initiative: Large Scale Dataset for Benchmarking of micro-Doppler Recognition
Algorithms," 2021 IEEE Radar Conference (RadarConf21).
"""
import argparse
import os
import uuid
from datetime import datetime

import numpy as np

import open_radar_eval_v2 as v            # feature mapping, engine/adapter config, windowing
import viz
from cuas_demo import Clock, _details
from logger import FusionLogger

LOG = "open_radar_demo_log.jsonl"
FOOTER = ("REAL Open Radar tracks (Gusland et al., IEEE RadarConf21; CC BY-NC 4.0)  ·  "
          "src_2/3/1 faults are INJECTED; src_4 flags are natural gaps in the real recording  ·  sensing-assurance meta-layer")


def log_step(engine, logger, clock, label, i, obs, theta_s, theta_c):
    sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs["theta_o"])
    status = engine.status(ts=clock.tick(), details=_details(obs))
    logger.log_state(sop, nhp, alert, metadata={
        "phase": label, "step": i,
        "theta_s": theta_s, "theta_c": theta_c, "theta_o": obs["theta_o"],
        "status": status,
    })
    return status


def calibrate(engine, adapter, logger, clock, cal_feats, rng, op, label):
    """Exactly calibration_steps steps, aggregate input, like the demo's calibration."""
    for k in range(engine.calibration_steps):
        c, f = v.windowed(cal_feats[k % len(cal_feats)], rng)
        obs = adapter.extract_features(c, f, op["noise"], op["far"])
        theta_c = adapter.normalize_context(op["coop"], op["volume"], op["rf"])
        theta_s = adapter.normalize_system_state(
            {f"src_{i}": op["health"] for i in range(6)}, op["platform"], op["backlog"])
        log_step(engine, logger, clock, label, k, obs, theta_s, theta_c)
    assert engine.is_calibrated


def run_phase(engine, adapter, logger, clock, label, n_steps, feats, rng,
              stale=None, ramp=None, jam=None, op=None):
    """One per-source phase.
    stale : source id whose heartbeat is dead for the whole phase
    ramp  : (source id, steps to full severity): continuity/confidence loss on that
            source growing from 0 to the 60%-dropout setting (x0.40 / x0.55)
    jam   : source id to break up, plus the jamming noise / RF / load context
    """
    base = dict(op or v.OP)
    for i in range(n_steps):
        cur, fault = dict(base), {}
        beats = {sid: 0.4 for sid in feats}
        if stale:
            beats[stale] = 10.0
        if ramp:
            sev = min(1.0, i / ramp[1])
            fault = dict(target=ramp[0], cs=1.0 - sev * 0.60, fs=1.0 - sev * 0.45)
        if jam:
            cur.update(v.JAM)
            fault = dict(target=jam, cs=0.45, fs=0.50)

        tracks = {}
        for sid, ft in feats.items():
            c, f = v.windowed(ft, rng)
            if sid == fault.get("target"):
                c = np.clip(c * fault["cs"], 0.05, 1.0)
                f = np.clip(f * fault["fs"], 0.05, 1.0)
            tracks[sid] = {"continuities": c, "confidences": f}
        obs = adapter.extract_features_per_source(
            tracks, noise_floor=cur["noise"], false_alarm_rate=cur["far"], heartbeat_ages=beats)
        theta_c = adapter.normalize_context(cur["coop"], cur["volume"], cur["rf"])
        theta_s = adapter.normalize_system_state(
            {sid: cur["health"] for sid in tracks}, cur["platform"], cur["backlog"],
            heartbeat_ages=beats)
        log_step(engine, logger, clock, label, i, obs, theta_s, theta_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default=v.NPY_PATH)
    ap.add_argument("--seed", type=int, default=v.SEED)
    ap.add_argument("--png", default="open_radar_timeline.png")
    ap.add_argument("--video", default="open_radar_replay.mp4")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    sigs = np.load(args.npy, allow_pickle=True)
    long_tracks = [s for s in sigs if len(s["ts"]) > 100]
    order = np.random.default_rng(args.seed)
    order.shuffle(long_tracks)
    cal_feats = [v.track_to_features(s) for s in long_tracks[:v.N_CAL_TRACKS]]
    src = long_tracks[v.N_CAL_TRACKS:v.N_CAL_TRACKS + v.N_SOURCES]       # held-out tracks
    feats = {f"src_{i}": v.track_to_features(s) for i, s in enumerate(src)}
    print(f"{len(sigs)} tracks, {len(long_tracks)} long; {len(cal_feats)} calibration, "
          f"{len(feats)} sources")

    rng = np.random.default_rng([args.seed, 7])
    engine, adapter = v.make_engine_adapter()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    logger = FusionLogger(LOG, run_id=run_id)
    clock = Clock()
    ctx = dict(engine=engine, adapter=adapter, logger=logger, clock=clock, feats=feats, rng=rng)

    calibrate(engine, adapter, logger, clock, cal_feats, rng, v.OP, "Calibration (healthy)")
    print(f"Calibrated baseline SOP = {engine.baseline_sop:.4f}"
          f"{'  (calibration suspect)' if engine.calibration_suspect else ''}")

    run_phase(label="Healthy operations", n_steps=10, **ctx)
    run_phase(label="Frozen feed (src_2)", n_steps=12, stale="src_2", **ctx)
    run_phase(label="Recovery A", n_steps=10, **ctx)
    run_phase(label="Dropout ramp (src_3)", n_steps=22, ramp=("src_3", 14), **ctx)
    run_phase(label="Recovery B", n_steps=14, **ctx)
    run_phase(label="EW jamming (src_1)", n_steps=14, jam="src_1", **ctx)
    run_phase(label="Recovery", n_steps=16, **ctx)

    final = engine.status(ts=clock.t)
    print(f"Final state {final['state']}, NHP {final['nhp']:.4f}; log -> {LOG} (run {run_id})")

    viz.FOOTER = FOOTER                      # honest footer instead of the synthetic-data one
    video = None if args.no_video else args.video
    for path in viz.render(LOG, png=args.png, video=video, run_id=run_id):
        print("wrote", path)


if __name__ == "__main__":
    main()
