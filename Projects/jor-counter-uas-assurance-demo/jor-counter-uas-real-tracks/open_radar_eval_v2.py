"""
Held-out Open Radar evaluation, v2 (diagnostic).

This scores the JOR C-UAS sensing-assurance monitor, which does not detect or
track targets. It measures how well the monitor warns an operator that the
detect/track picture is becoming unreliable, and names the degraded source.

What changed versus open_radar_eval.py, and why:

1. Inputs now vary step to step. v1 fed the *same arrays* every step, so the
   engine saw a constant signal and "8 steps" was really 1 step repeated. Here
   each step draws a random WINDOW-sample slice of every source's track.
2. The warning is split into three separate numbers instead of one OR:
     attributed : the faulty source itself was named (stale or degraded)
     state      : the global band left NOMINAL (says nothing about *which* source)
     UNRELIABLE : the hysteresis alert fired
   v1 counted a trial as a success if ANY of these happened.
3. Latency (steps until the first attributed warning) is reported.
4. Wrong-source flags are counted (healthy sources named as degraded/stale).
5. EW jamming is split into sector-only, wide-area-only and combined, so you can
   see which signal is doing the work. v1 only ran the combined case, where the
   ambient noise/RF/load change alone pushes the global state out of NOMINAL.
6. Calibration uses the same operating point as the test (override with
   --cal-mismatch to reproduce v1), and exactly calibration_steps steps. v1 fed
   30 tracks to a 20-step calibration, so 10 leftover steps ran through the
   Bayesian update before the baseline phase.
7. Trials are paired: the window draws are identical across conditions within a
   trial, so the injected fault is the only difference from "clean".
8. Clean runs are longer (CLEAN_STEPS) and report DEGRADED and UNRELIABLE
   separately; warm-up alarms before any fault are counted too.
9. Wilson 95% intervals on every rate.
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime

import numpy as np

from engine import JOREngine
from cuas_adapter import CUASHealthAdapter

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
NPY_PATH = "moving_target_dataset.npy"
N_TRIALS = 50
N_SOURCES = 8
SEED = 42
WINDOW = 40            # samples per source per step
BASELINE_STEPS = 8     # healthy steps before a fault is injected
HOLD_STEPS = 12        # steps with the fault active
CLEAN_STEPS = 40       # steps in the clean (no-fault) run
N_CAL_TRACKS = 30

# Operating point used for calibration, baselining and every test step.
OP = dict(noise=0.18, rf=0.22, volume=20, coop=8,
          health=0.94, platform=0.96, backlog=0.06, far=0.03)
# What v1 calibrated with (differs from the test operating point).
OP_CAL_LEGACY = dict(noise=0.05, rf=0.15, volume=12, coop=5,
                     health=0.95, platform=0.97, backlog=0.05, far=0.02)
JAM = dict(noise=0.65, rf=0.75, volume=55)

# expect: "attributed" -> the named source should be flagged
#         "state"      -> only the global state can respond (no single culprit)
#         "none"       -> nothing should fire
CONDITIONS = {
    "clean":        dict(target=None,    expect="none",  steps=CLEAN_STEPS),
    "frozen_feed":  dict(target="src_2", expect="attributed", stale=True),
    "drop60":       dict(target="src_3", expect="attributed", cs=0.40, fs=0.55),
    "drop30":       dict(target="src_3", expect="attributed", cs=0.70, fs=0.80),
    "jam_sector":   dict(target="src_1", expect="attributed", cs=0.45, fs=0.50),
    "jam_wide":     dict(target=None,    expect="state", ctx=JAM),
    "jam_combined": dict(target="src_1", expect="attributed", cs=0.45, fs=0.50, ctx=JAM),
}


# ------------------------------------------------------------------
# Feature mapping (unchanged from v1)
# ------------------------------------------------------------------
def track_to_features(signature):
    ts = np.asarray(signature["ts"], dtype=float)
    snr = np.asarray(signature["snr_db"], dtype=float)
    if len(ts) < 2:
        return np.array([]), np.array([])
    dt = np.diff(ts)
    median_dt = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.1
    cont = np.clip(1.0 - (dt - median_dt) / (3 * median_dt + 1e-6), 0.05, 1.0)
    conf = np.clip((snr[1:] - 5.0) / 20.0, 0.2, 0.98)
    return cont, conf


def make_engine_adapter():
    engine = JOREngine(
        prior_nh=0.05, steepness=12.0,
        upper_th=0.69, lower_th=0.59,
        retention=0.70, calibration_steps=20,
        delta_deadband=0.02,
    )
    adapter = CUASHealthAdapter(
        ema_alpha=0.30, persistence_steps=4, min_iqr=0.05,
        outlier_iqr_threshold=1.5, self_baseline_alpha=0.04,
        self_deviation_threshold=0.10, self_score_weight=0.55,
    )
    return engine, adapter


def windowed(feat, rng):
    cont, conf = feat
    n = min(len(cont), len(conf))
    if n <= WINDOW:
        return cont[:n], conf[:n]
    s = int(rng.integers(0, n - WINDOW + 1))
    return cont[s:s + WINDOW], conf[s:s + WINDOW]


def calibrate(engine, adapter, cal_feats, rng, op):
    """Exactly engine.calibration_steps steps, so no step leaks into the update."""
    for k in range(engine.calibration_steps):
        c, f = windowed(cal_feats[k % len(cal_feats)], rng)
        obs = adapter.extract_features(c, f, op["noise"], op["far"])
        theta_c = adapter.normalize_context(op["coop"], op["volume"], op["rf"])
        theta_s = adapter.normalize_system_state(
            {f"src_{i}": op["health"] for i in range(6)}, op["platform"], op["backlog"])
        engine.fusion_step(theta_s, theta_c, obs["theta_o"])
    assert engine.is_calibrated, "calibration did not complete"


def fuse_step(engine, adapter, feats, beats, rng, op, fault=None):
    fault = fault or {}
    tracks = {}
    for sid, ft in feats.items():                       # fixed order -> paired draws
        c, f = windowed(ft, rng)
        if sid == fault.get("target") and "cs" in fault:
            c = np.clip(c * fault["cs"], 0.05, 1.0)
            f = np.clip(f * fault["fs"], 0.05, 1.0)
        tracks[sid] = {"continuities": c, "confidences": f}
    obs = adapter.extract_features_per_source(
        tracks, noise_floor=op["noise"], false_alarm_rate=op["far"], heartbeat_ages=beats)
    theta_c = adapter.normalize_context(op["coop"], op["volume"], op["rf"])
    theta_s = adapter.normalize_system_state(
        {sid: op["health"] for sid in tracks}, op["platform"], op["backlog"],
        heartbeat_ages=beats)
    engine.fusion_step(theta_s, theta_c, obs["theta_o"])
    st = engine.status()
    return dict(state=st["state"], stale=list(obs["stale_sources"]),
                degraded=list(obs["degraded_sources"]))


def first(seq, pred):
    for i, x in enumerate(seq, start=1):
        if pred(x):
            return i
    return None


def run_trial(name, spec, feats, cal_feats, trial, cal_op):
    rng = np.random.default_rng([SEED, trial])          # same draws in every condition
    engine, adapter = make_engine_adapter()
    calibrate(engine, adapter, cal_feats, rng, cal_op)

    beats = {sid: 0.4 for sid in feats}
    warm = [fuse_step(engine, adapter, feats, beats, rng, OP) for _ in range(BASELINE_STEPS)]
    warmup_alarm = any(w["state"] != "NOMINAL" or w["stale"] or w["degraded"] for w in warm)

    fault = dict(spec)
    op = {**OP, **spec.get("ctx", {})}
    if spec.get("stale"):
        beats[spec["target"]] = 10.0
    steps = spec.get("steps", HOLD_STEPS)
    hold = [fuse_step(engine, adapter, feats, beats, rng, op, fault) for _ in range(steps)]

    tgt = spec["target"]
    flagged = lambda h: tgt is not None and (tgt in h["stale"] or tgt in h["degraded"])
    others = lambda h: [s for s in h["stale"] + h["degraded"] if s != tgt]
    return dict(
        attr=first(hold, flagged) if tgt else None,
        state=first(hold, lambda h: h["state"] in ("DEGRADED", "UNRELIABLE")),
        unrel=first(hold, lambda h: h["state"] == "UNRELIABLE"),
        wrong_src=any(others(h) for h in hold),
        warmup_alarm=warmup_alarm,
    )


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k:>2}/{n} [{lo:4.0%}-{hi:4.0%}]"


def summarize(rows):
    n = len(rows)
    lat = [r["attr"] for r in rows if r["attr"]]
    return dict(
        n=n,
        attributed=sum(r["attr"] is not None for r in rows),
        state=sum(r["state"] is not None for r in rows),
        unreliable=sum(r["unrel"] is not None for r in rows),
        median_latency=float(np.median(lat)) if lat else None,
        wrong_source=sum(r["wrong_src"] for r in rows),
        warmup_alarm=sum(r["warmup_alarm"] for r in rows),
    )


SWEEP_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def run_sweep(pool_feats, cal_feats, cal_op, n_trials):
    """Ramp a dropout-style fault on src_3 from none (0.0) to the drop60 setting (1.0).

    Severity k scales continuity by 1-k*(1-0.40) and confidence by 1-k*(1-0.55), so
    k=0.5 is roughly the old 30% dropout setting (x0.70 / x0.78 vs x0.70 / x0.80)
    and k=1.0 is exactly drop60. k=0 injects nothing: any flag on src_3 is a false
    attribution, so it is the sweep's own false-alarm row."""
    out = {}
    for k in SWEEP_LEVELS:
        spec = dict(target="src_3", expect="attributed",
                    cs=1.0 - k * (1.0 - 0.40), fs=1.0 - k * (1.0 - 0.55))
        rows = []
        for trial in range(n_trials):
            pick = np.random.default_rng([SEED, 1000 + trial]).choice(
                len(pool_feats), size=N_SOURCES, replace=False)
            feats = {f"src_{i}": pool_feats[j] for i, j in enumerate(pick)}
            rows.append(run_trial("sweep", spec, feats, cal_feats, trial, cal_op))
        out[k] = summarize(rows)
        print(f"  severity {k:.2f} done")

    print("\n===== Severity sweep (dropout-style fault on src_3) =====")
    print("severity 0.0 = no fault (false-alarm row); 1.0 = the 60% dropout setting\n")
    hdr = (f"{'sev':<5}{'cont x':<8}{'conf x':<8}{'attributed':<18}{'state':<18}"
           f"{'UNRELIABLE':<18}{'latency':<9}{'wrong-src'}")
    print(hdr)
    print("-" * len(hdr))
    for k, s in out.items():
        n = s["n"]
        lat = "-" if s["median_latency"] is None else f"{s['median_latency']:.0f} st"
        print(f"{k:<5.2f}{1 - k * 0.60:<8.2f}{1 - k * 0.45:<8.2f}"
              f"{rate(s['attributed'], n):<18}{rate(s['state'], n):<18}"
              f"{rate(s['unreliable'], n):<18}{lat:<9}{s['wrong_source']:>2}/{n}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="run the severity sweep instead of the fixed conditions")
    ap.add_argument("--npy", default=NPY_PATH)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--cal-mismatch", action="store_true",
                    help="calibrate at v1's (different) operating point")
    ap.add_argument("--json", metavar="PATH", help="also write the summary as JSON")
    args = ap.parse_args()
    cal_op = OP_CAL_LEGACY if args.cal_mismatch else OP

    sigs = np.load(args.npy, allow_pickle=True)
    long_tracks = [s for s in sigs if len(s["ts"]) > 100]
    rng = np.random.default_rng(SEED)
    rng.shuffle(long_tracks)
    cal_set, pool = long_tracks[:N_CAL_TRACKS], long_tracks[N_CAL_TRACKS:]
    cal_feats = [track_to_features(s) for s in cal_set]
    pool_feats = [track_to_features(s) for s in pool]
    print(f"Tracks: {len(sigs)} total, {len(long_tracks)} long, "
          f"{len(cal_set)} cal, {len(pool)} held-out pool"
          f"{'  [cal-mismatch]' if args.cal_mismatch else ''}")

    if args.sweep:
        res = run_sweep(pool_feats, cal_feats, cal_op, args.trials)
        LAST_RUN.update(mode="sweep", results=res)
        if args.json:
            with open(args.json, "w") as f:
                json.dump({str(k): v for k, v in res.items()}, f, indent=2)
            print(f"wrote {args.json}")
        return

    rows = {name: [] for name in CONDITIONS}
    for trial in range(args.trials):
        pick = np.random.default_rng([SEED, 1000 + trial]).choice(
            len(pool_feats), size=N_SOURCES, replace=False)
        feats = {f"src_{i}": pool_feats[j] for i, j in enumerate(pick)}
        for name, spec in CONDITIONS.items():
            rows[name].append(run_trial(name, spec, feats, cal_feats, trial, cal_op))
        if (trial + 1) % 10 == 0:
            print(f"  completed {trial + 1}/{args.trials}")

    out = {name: summarize(r) for name, r in rows.items()}
    LAST_RUN.update(mode="conditions", results=out)

    print("\n===== Open Radar held-out results, v2 =====")
    print("attributed = faulty source named | state = global band left NOMINAL |")
    print("UNRELIABLE = alert fired | wrong-src = a healthy source was flagged\n")
    hdr = (f"{'condition':<13}{'attributed':<18}{'state':<18}{'UNRELIABLE':<18}"
           f"{'latency':<9}{'wrong-src':<11}{'warm-up alarm'}")
    print(hdr)
    print("-" * len(hdr))
    for name, spec in CONDITIONS.items():
        s = out[name]
        n = s["n"]
        attr = rate(s["attributed"], n) if spec["target"] else "-"
        lat = "-" if s["median_latency"] is None else f"{s['median_latency']:.0f} st"
        print(f"{name:<13}{attr:<18}{rate(s['state'], n):<18}{rate(s['unreliable'], n):<18}"
              f"{lat:<9}{s['wrong_source']:>2}/{n:<8}{s['warmup_alarm']:>2}/{n}")

    c = out["clean"]
    print(f"\nClean ({CLEAN_STEPS} steps): left NOMINAL {c['state']}/{c['n']}, "
          f"UNRELIABLE {c['unreliable']}/{c['n']}, healthy source flagged {c['wrong_source']}/{c['n']}")
    jw, js = out["jam_wide"], out["jam_sector"]
    print(f"Jamming: wide-area alone -> state {jw['state']}/{jw['n']}; "
          f"sector breakup alone -> attributed {js['attributed']}/{js['n']}, "
          f"state {js['state']}/{js['n']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


# ------------------------------------------------------------------
# Saved report: every run writes a text report and a CSV (added; scoring unchanged)
# ------------------------------------------------------------------
LAST_RUN = {}


class _Tee:
    """Copy everything printed to the console into the text report."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _write_csv(path):
    res = LAST_RUN.get("results")
    if not res:
        return
    fields = ["n", "attributed", "state", "unreliable", "median_latency",
              "wrong_source", "warmup_alarm"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if LAST_RUN["mode"] == "sweep":
            w.writerow(["severity", "cont_x", "conf_x"] + fields)
            for k, s in res.items():
                w.writerow([k, round(1 - k * 0.60, 3), round(1 - k * 0.45, 3)]
                           + [s[x] for x in fields])
        else:
            w.writerow(["condition", "expect", "target"] + fields)
            for name, s in res.items():
                spec = CONDITIONS[name]
                w.writerow([name, spec["expect"], spec["target"] or ""]
                           + [s[x] for x in fields])


if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "sweep" if "--sweep" in sys.argv else "conditions"
    report_txt = f"open_radar_eval_v2_{mode}_{stamp}.txt"
    report_csv = f"open_radar_eval_v2_{mode}_{stamp}.csv"
    console = sys.stdout
    with open(report_txt, "w", encoding="utf-8") as rep:
        sys.stdout = _Tee(console, rep)
        try:
            print(f"Open Radar held-out evaluation v2 - run {stamp}")
            print(f"Command: python {' '.join(sys.argv)}")
            print(f"Settings: seed {SEED}, sources {N_SOURCES}, window {WINDOW}, "
                  f"baseline {BASELINE_STEPS}, hold {HOLD_STEPS}, clean {CLEAN_STEPS}, "
                  f"cal tracks {N_CAL_TRACKS}")
            print("Data: Open Radar Initiative, Outdoor Moving Object dataset "
                  "(Gusland et al., IEEE RadarConf21; CC BY-NC 4.0)")
            print("Faults are INJECTED, not real sensor failures.")
            print("drop30/drop60 SCALE continuity and confidence; they do not remove real updates.\n")
            main()
        finally:
            sys.stdout = console
    _write_csv(report_csv)
    print(f"report -> {report_txt}")
    if LAST_RUN.get("results"):
        print(f"csv    -> {report_csv}")
