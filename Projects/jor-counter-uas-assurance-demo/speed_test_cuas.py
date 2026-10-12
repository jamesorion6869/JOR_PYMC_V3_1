"""
JOR C-UAS Sensing-Assurance Demo - Offline Speed Test
=====================================================
Measures the cost of the C-UAS meta-layer, stage by stage:

  1. Engine fusion step         (JOREngine.fusion_step)
  2. Engine status() report     (JSON-safe C2 payload)
  3. Adapter stages             (theta_c, theta_s, theta_o aggregate / per-source)
  4. Full monitor cycle         (3 adapter calls + fusion + status), no I/O
  5. Full monitor cycle + log   (adds FusionLogger JSONL append, file I/O)
  6. End-to-end demo run        (cuas_demo.main, 92 steps, stdout suppressed)

Inputs are pre-generated so the timings isolate the monitor, not the
simulator.  Scenario inputs cycle through healthy / gray-failure /
feed-loss / congested conditions so every code path is exercised.

CONCEPT DEMO numbers - offline, single thread, not a real-time claim.
"""

import argparse
import contextlib
import datetime
import io
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import JOREngine                      # noqa: E402
from cuas_adapter import CUASHealthAdapter        # noqa: E402
from logger import FusionLogger                   # noqa: E402
import cuas_demo                                   # noqa: E402

N_SOURCES = 8
TRACKS_PER_SOURCE = 25


def best_of(fn, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), float(np.median(times))


def make_engine():
    # Same configuration as cuas_demo.main(), pre-calibrated so the timed
    # loop exercises the normal-operation path (likelihood + Bayes + bands).
    return JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                     retention=0.70, calibration_steps=20, delta_deadband=0.02,
                     baseline_sop=0.20)


# ---------------------------------------------------------------------------
# Input pools (pre-generated; cycle through the demo's operating conditions)
# ---------------------------------------------------------------------------
def build_step_inputs(n_steps, seed=7):
    np.random.seed(seed)
    steps = []
    for i in range(n_steps):
        mode = i % 4
        if mode == 0:       # healthy
            kw = dict()
        elif mode == 1:     # gray failure on src_3
            kw = dict(bad_source_idx=3)
        elif mode == 2:     # feed loss: src_1,2,5 stale
            kw = dict(empty_sources={1, 2, 5})
        else:               # quiet sky
            kw = dict(empty_sources=set(range(N_SOURCES)))
        stale = (1, 2, 5) if mode == 2 else ()
        tracks = cuas_demo.simulate_source_tracks(
            num_sources=N_SOURCES, n_tracks_per=TRACKS_PER_SOURCE, **kw)
        beats = cuas_demo.simulate_heartbeats(num_sources=N_SOURCES, stale=stale)
        health = {f"src_{k}": 0.95 for k in range(N_SOURCES)}
        if mode == 1:
            health["src_3"] = 0.55
        steps.append(dict(
            tracks=tracks, beats=beats, health=health,
            noise=0.13 if mode != 1 else 0.28,
            fa=0.02 if mode != 1 else 0.04,
            coop=18, vol=28 if mode != 3 else 2, rf=0.18,
            platform=0.96, backlog=0.05,
            agg_cont=np.clip(np.random.normal(0.93, 0.03, 40), 0.05, 1.0),
            agg_conf=np.clip(np.random.normal(0.88, 0.04, 40), 0.05, 1.0),
        ))
    return steps


# ---------------------------------------------------------------------------
# 1-2. Engine micro-benchmarks
# ---------------------------------------------------------------------------
def bench_engine(n_calls=200_000):
    rng = np.random.default_rng(1)
    # theta triples spanning healthy -> degraded
    thetas = [tuple(map(float, row)) for row in rng.uniform(0.0, 0.7, size=(1000, 3))]

    eng = make_engine()
    for t in thetas[:200]:
        eng.fusion_step(*t)                                   # warm-up

    eng = make_engine()
    n_th = len(thetas)
    t0 = time.perf_counter()
    for i in range(n_calls):
        eng.fusion_step(*thetas[i % n_th])
    el = time.perf_counter() - t0
    fusion_us = el / n_calls * 1e6

    # Fault path (non-finite input is rejected before any math)
    eng = make_engine()
    nan = float("nan")
    t0 = time.perf_counter()
    for _ in range(n_calls):
        eng.fusion_step(nan, 0.1, 0.1)
    fault_us = (time.perf_counter() - t0) / n_calls * 1e6

    # status() report
    eng = make_engine()
    for t in thetas[:100]:
        eng.fusion_step(*t)
    n_status = 50_000
    t0 = time.perf_counter()
    for _ in range(n_status):
        eng.status(ts=1.0, details={})
    status_us = (time.perf_counter() - t0) / n_status * 1e6

    print("\n--- Engine micro-benchmarks ---")
    print(f"  fusion_step (normal path)   : {fusion_us:7.2f} us/call  ({1e6 / fusion_us:,.0f} steps/s)")
    print(f"  fusion_step (fault path)    : {fault_us:7.2f} us/call")
    print(f"  status() C2 report          : {status_us:7.2f} us/call")
    return dict(fusion_us=fusion_us, fault_us=fault_us, status_us=status_us)


# ---------------------------------------------------------------------------
# 3. Adapter stages
# ---------------------------------------------------------------------------
def bench_adapter(steps, reps=2000):
    n = len(steps)

    def time_loop(fn):
        fn(steps[0])                                          # warm-up
        t0 = time.perf_counter()
        for i in range(reps):
            fn(steps[i % n])
        return (time.perf_counter() - t0) / reps * 1e6

    ad = CUASHealthAdapter()
    ctx_us = time_loop(lambda s: ad.normalize_context(s["coop"], s["vol"], s["rf"]))
    sys_us = time_loop(lambda s: ad.normalize_system_state(
        s["health"], s["platform"], s["backlog"], heartbeat_ages=s["beats"]))
    agg_us = time_loop(lambda s: ad.extract_features(
        s["agg_cont"], s["agg_conf"], s["noise"], s["fa"]))
    ps_us = time_loop(lambda s: ad.extract_features_per_source(
        s["tracks"], s["noise"], s["fa"], heartbeat_ages=s["beats"]))

    print("\n--- Adapter stages ---")
    print(f"  theta_c  normalize_context            : {ctx_us:8.2f} us/call")
    print(f"  theta_s  normalize_system_state       : {sys_us:8.2f} us/call")
    print(f"  theta_o  extract_features (40 tracks) : {agg_us:8.2f} us/call")
    print(f"  theta_o  per-source ({N_SOURCES} src x {TRACKS_PER_SOURCE} trk) : {ps_us:8.2f} us/call")
    return dict(ctx_us=ctx_us, sys_us=sys_us, agg_us=agg_us, ps_us=ps_us)


# ---------------------------------------------------------------------------
# 4-5. Full monitor cycle (per-source path), without and with logging
# ---------------------------------------------------------------------------
def monitor_cycle(engine, adapter, s, clock_t, logger=None):
    obs = adapter.extract_features_per_source(
        s["tracks"], s["noise"], s["fa"], heartbeat_ages=s["beats"])
    theta_c = adapter.normalize_context(s["coop"], s["vol"], s["rf"])
    theta_s = adapter.normalize_system_state(
        s["health"], s["platform"], s["backlog"], heartbeat_ages=s["beats"])
    sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs["theta_o"])
    status = engine.status(ts=clock_t, details=cuas_demo._details(obs))
    if logger is not None:
        logger.log_state(sop, nhp, alert, metadata={
            "phase": "bench", "step": int(clock_t),
            "theta_s": theta_s, "theta_c": theta_c, "theta_o": obs["theta_o"],
            "status": status,
        })
    return status


def bench_cycle(steps, n_cycles=4000, n_log_cycles=2000):
    n = len(steps)

    # Calibrate once from healthy-ish steps so the loop is in normal operation
    def fresh():
        eng = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                        retention=0.70, calibration_steps=20, delta_deadband=0.02)
        ad = CUASHealthAdapter()
        for i in range(20):
            monitor_cycle(eng, ad, steps[(i * 4) % n], float(i))   # mode-0 healthy steps
        return eng, ad

    eng, ad = fresh()
    for i in range(50):                                       # warm-up
        monitor_cycle(eng, ad, steps[i % n], float(i))

    eng, ad = fresh()
    t0 = time.perf_counter()
    for i in range(n_cycles):
        monitor_cycle(eng, ad, steps[i % n], float(i))
    el = time.perf_counter() - t0
    cyc_us = el / n_cycles * 1e6

    # With JSONL logging to a temp file
    with tempfile.TemporaryDirectory() as td:
        logger = FusionLogger(os.path.join(td, "bench_log.jsonl"), run_id="bench")
        eng, ad = fresh()
        t0 = time.perf_counter()
        for i in range(n_log_cycles):
            monitor_cycle(eng, ad, steps[i % n], float(i), logger=logger)
        el_log = time.perf_counter() - t0
        size = os.path.getsize(logger.log_file)
    cyc_log_us = el_log / n_log_cycles * 1e6

    print("\n--- Full monitor cycle (per-source path) ---")
    print(f"  No I/O (adapter x3 + fusion + status) : {cyc_us:8.1f} us/cycle  ({1e6 / cyc_us:,.0f} cycles/s)")
    print(f"  With JSONL log append                 : {cyc_log_us:8.1f} us/cycle  ({1e6 / cyc_log_us:,.0f} cycles/s)")
    print(f"  Log cost alone                        : {cyc_log_us - cyc_us:8.1f} us/cycle")
    print(f"  Avg log record size                   : {size / n_log_cycles:8.0f} bytes")
    print(f"  Headroom vs 1 Hz demo clock           : {1e6 / cyc_log_us:,.0f}x")
    return dict(cyc_us=cyc_us, cyc_log_us=cyc_log_us, rec_bytes=size / n_log_cycles)


# ---------------------------------------------------------------------------
# 6. End-to-end demo run
# ---------------------------------------------------------------------------
def bench_demo(repeats=5):
    times = []
    n_steps = None
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                buf = io.StringIO()
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(buf):
                    cuas_demo.main(["--seed", "42"])
                times.append(time.perf_counter() - t0)
                with open("cuas_airspace_log.jsonl") as f:
                    n_steps = sum(1 for _ in f)
            finally:
                os.chdir(cwd)
    best, med = min(times), float(np.median(times))
    print("\n--- End-to-end demo (cuas_demo.main, stdout suppressed) ---")
    print(f"  Steps simulated      : {n_steps}")
    print(f"  Best of {repeats} runs      : {best * 1000:.1f} ms")
    print(f"  Median of {repeats} runs    : {med * 1000:.1f} ms")
    print(f"  Per step (all-in)    : {best / n_steps * 1000:.2f} ms  "
          f"(simulation + panels + logging + fusion)")
    return dict(best=best, n_steps=n_steps)


# ---------------------------------------------------------------------------
# Results image (drawn from the exact numbers of THIS run)
# ---------------------------------------------------------------------------
THEMES = {
    "light": dict(surface="#fcfcfb", tile="#f6f6f3", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark":  dict(surface="#1a1a19", tile="#232322", ink="#f0efec", ink2="#c3c2b7",
                  muted="#898781", grid="#2c2c2a", axis="#383835",
                  series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}


def _fmt_us(v):
    if v < 10:
        return f"{v:.2f} µs"
    if v < 100:
        return f"{v:.1f} µs"
    return f"{v:,.0f} µs"


def render_png(R, path, theme="light"):
    """Render a results card from the numbers in R. Returns the path, or None."""
    try:
        import logging
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    except ImportError:
        print("\n  (PNG skipped: matplotlib is not installed - run `pip install matplotlib`,")
        print("   then `python speed_test_cuas.py --plot-only speed_test_cuas_results.json`)")
        return None

    T = THEMES[theme]
    S = T["series"]
    e, a, c, d, m = R["engine"], R["adapter"], R["cycle"], R["demo"], R["meta"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "SF Pro Text", "Helvetica Neue",
                                       "Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(13, 7.6), dpi=170, facecolor=T["surface"])

    def text(x, y, s, size=10, color=None, weight="normal", ha="left", va="baseline"):
        fig.text(x, y, s, fontsize=size, color=color or T["ink"], fontweight=weight,
                 ha=ha, va=va)

    def rect(x, y, w, h, fc, ec="none", lw=0.0):
        fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure,
                                 facecolor=fc, edgecolor=ec, linewidth=lw))

    def hline(x0, x1, y):
        fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                              color=T["grid"], linewidth=0.8))

    # ---- header -----------------------------------------------------------
    text(0.04, 0.945, "JOR C-UAS Sensing-Assurance Monitor — Offline Speed Test",
         size=20, weight="bold")
    proc = (m.get("processor") or "").strip()
    sub = (f"Run {m['date']}  ·  Python {m['python']}  ·  NumPy {m['numpy']}"
           f"  ·  {m['platform']}")
    text(0.04, 0.912, sub[:150], size=9.5, color=T["ink2"])
    if proc or m.get("cpus"):
        text(0.04, 0.890, f"{proc[:90]}  ·  {m.get('cpus', '?')} logical CPUs".strip(" ·"),
             size=9.5, color=T["muted"])

    # ---- KPI tiles ---------------------------------------------------------
    tiles = [
        ("Full monitor cycle", f"{c['cyc_us']:.0f} µs",
         f"{1e6 / c['cyc_us']:,.0f} cycles/s, no I/O"),
        ("With JSONL logging", f"{c['cyc_log_us']:.0f} µs",
         f"{1e6 / c['cyc_log_us']:,.0f} cycles/s"),
        ("JOR fusion step", f"{e['fusion_us']:.2f} µs",
         f"{1e6 / e['fusion_us']:,.0f} steps/s"),
        ("End-to-end demo", f"{d['best'] * 1000:.0f} ms",
         f"{d['n_steps']} steps, all-in"),
    ]
    tw, gap, ty, th = 0.2175, 0.0167, 0.735, 0.125
    for i, (lab, val, cap) in enumerate(tiles):
        x = 0.04 + i * (tw + gap)
        rect(x, ty, tw, th, T["tile"], T["grid"], 0.8)
        text(x + 0.014, ty + th - 0.030, lab, size=9.5, color=T["ink2"])
        text(x + 0.014, ty + 0.046, val, size=27, weight="bold")
        text(x + 0.014, ty + 0.014, cap, size=9, color=T["muted"])

    # ---- stage-cost bar chart ---------------------------------------------
    stages = [
        ("θ_o  per-source (8 × 25 tracks)", a["ps_us"], 0),
        ("θ_o  aggregate (40 tracks)", a["agg_us"], 0),
        ("θ_s  system state", a["sys_us"], 1),
        ("JSONL log append (derived)", max(c["cyc_log_us"] - c["cyc_us"], 0.0), 4),
        ("θ_c  context load", a["ctx_us"], 2),
        ("status() C2 report", e["status_us"], 3),
        ("JOR fusion step", e["fusion_us"], 3),
    ]
    stages.sort(key=lambda s: s[1])          # barh draws bottom-up -> largest on top
    names = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    cols = [S[s[2]] for s in stages]
    vmax = max(vals)

    text(0.04, 0.700, "Cost per stage (µs per call)", size=12, weight="bold")
    legend = [("θ_o track quality", 0), ("θ_s sensor integrity", 1),
              ("θ_c sensing load", 2), ("JOR engine", 3), ("JSONL logging", 4)]
    for (lab, idx), x in zip(legend, [0.04, 0.175, 0.325, 0.445, 0.545]):
        rect(x, 0.6665, 0.0085, 0.0145, S[idx])
        text(x + 0.0125, 0.6665, lab, size=9, color=T["ink2"])

    ax = fig.add_axes([0.30, 0.15, 0.39, 0.485], facecolor=T["surface"])
    ys = list(range(len(vals)))
    ax.barh(ys, vals, height=0.52, color=cols, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9.5, color=T["ink"])
    ax.set_xlim(0, vmax * 1.28)
    ax.xaxis.grid(True, color=T["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T["axis"])
    ax.tick_params(axis="x", colors=T["muted"], labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0, pad=6)
    for y, v in zip(ys, vals):
        ax.text(v + vmax * 0.018, y, _fmt_us(v), va="center", fontsize=9.5, color=T["ink"])

    ps_share = a["ps_us"] / c["cyc_us"] * 100.0
    eng_share = (e["fusion_us"] + e["status_us"]) / c["cyc_us"] * 100.0
    text(0.04, 0.085,
         f"Per-source θ_o scoring is {ps_share:.0f}% of a full monitor cycle; "
         f"the JOR engine (fusion + status) is {eng_share:.1f}%.",
         size=10, color=T["ink2"])

    # ---- results table -----------------------------------------------------
    tx0, tx1 = 0.735, 0.96
    text(tx0, 0.700, "All results", size=12, weight="bold")
    rows = [
        ("Engine fusion step", f"{e['fusion_us']:.2f} µs"),
        ("  fault path (NaN input)", f"{e['fault_us']:.2f} µs"),
        ("status() C2 report", f"{e['status_us']:.2f} µs"),
        ("θ_c  normalize_context", f"{a['ctx_us']:.1f} µs"),
        ("θ_s  normalize_system_state", f"{a['sys_us']:.1f} µs"),
        ("θ_o  aggregate (40 trk)", f"{a['agg_us']:.1f} µs"),
        ("θ_o  per-source (8×25)", f"{a['ps_us']:.1f} µs"),
        ("Full cycle, no I/O", f"{c['cyc_us']:.0f} µs · {1e6 / c['cyc_us']:,.0f}/s"),
        ("Full cycle + JSONL log", f"{c['cyc_log_us']:.0f} µs · {1e6 / c['cyc_log_us']:,.0f}/s"),
        ("Log record size", f"{c['rec_bytes']:.0f} bytes"),
        ("End-to-end demo", f"{d['best'] * 1000:.1f} ms / {d['n_steps']} steps"),
    ]
    y = 0.655
    for lab, val in rows:
        text(tx0, y, lab, size=9, color=T["ink2"])
        text(tx1, y, val, size=9, color=T["ink"], ha="right")
        hline(tx0, tx1, y - 0.014)
        y -= 0.0455

    text(0.04, 0.045,
         "Offline, single-thread timings of the monitor (adapter + JOR engine + report + log) "
         "on pre-generated inputs. Concept demo — not a real-time or fielded C-UAS claim.",
         size=8.5, color=T["muted"])

    fig.savefig(path, facecolor=T["surface"])
    plt.close(fig)
    return os.path.abspath(path)


def collect_meta():
    return dict(
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor() or "",
        cpus=os.cpu_count(),
    )


def main():
    ap = argparse.ArgumentParser(description="JOR C-UAS offline speed test")
    ap.add_argument("--png", default="speed_test_cuas_results.png",
                    help="where to write the results image")
    ap.add_argument("--json", default="speed_test_cuas_results.json",
                    help="where to save the raw numbers")
    ap.add_argument("--no-png", action="store_true", help="skip the results image")
    ap.add_argument("--dark", action="store_true", help="render the image in dark mode")
    ap.add_argument("--plot-only", metavar="JSON",
                    help="re-render the image from a saved results JSON (no benchmarks run)")
    args = ap.parse_args()
    theme = "dark" if args.dark else "light"

    if args.plot_only:
        with open(args.plot_only, encoding="utf-8") as f:
            R = json.load(f)
        out = render_png(R, args.png, theme)
        if out:
            print(f"wrote {out}")
        return

    print("=" * 66)
    print("JOR C-UAS Sensing-Assurance Demo - Offline Speed Test")
    print("=" * 66)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"NumPy           : {np.__version__}")
    print(f"Platform        : {platform.platform()}")
    print(f"Processor       : {platform.processor() or 'n/a'}")
    print("\nNote: pre-generated inputs, single thread, offline. CONCEPT DEMO numbers.")

    steps = build_step_inputs(400)

    eng = bench_engine()
    adp = bench_adapter(steps)
    cyc = bench_cycle(steps)
    demo = bench_demo()

    print("\n" + "=" * 66)
    print("Summary")
    print("=" * 66)
    print(f"  JOR fusion step            : {eng['fusion_us']:.1f} us")
    print(f"  status() report            : {eng['status_us']:.1f} us")
    print(f"  Adapter, per-source theta_o: {adp['ps_us']:.1f} us")
    print(f"  Full monitor cycle (no I/O): {cyc['cyc_us']:.0f} us  ({1e6 / cyc['cyc_us']:,.0f}/s)")
    print(f"  Full cycle + JSONL logging : {cyc['cyc_log_us']:.0f} us  ({1e6 / cyc['cyc_log_us']:,.0f}/s)")
    print(f"  End-to-end demo            : {demo['best'] * 1000:.0f} ms for {demo['n_steps']} steps")
    print("\nThese numbers measure the offline monitor only (adapter + JOR engine")
    print("+ report + log). They do not claim real-time or fielded C-UAS performance.")
    print("=" * 66)

    # Save the exact numbers, then draw the image from them
    R = dict(meta=collect_meta(), engine=eng, adapter=adp, cycle=cyc, demo=demo)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print(f"\nRaw numbers saved to {os.path.abspath(args.json)}")
    if not args.no_png:
        out = render_png(R, args.png, theme)
        if out:
            print(f"Results image saved to {out}")


if __name__ == "__main__":
    main()
