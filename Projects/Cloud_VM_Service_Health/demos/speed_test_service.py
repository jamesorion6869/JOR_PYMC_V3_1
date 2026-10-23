"""
JOR Real-Time Service Health Demo - Speed Test
==============================================
Measures what the monitor costs, stage by stage, for real_time_service_gui.py:

  1. Engine fusion step            (JOREngine.fusion_step)
  2. Adapter stages                (theta_c, theta_s, theta_o aggregate / per-instance)
  3. Laptop sampler                (psutil memory / CPU / disk read; optional)
  4. Banner state                  (the label shared by the window, CLI and log)
  5. Full monitor cycle            (adapter x3 + sampler + fusion + banner), no I/O
  6. Full monitor cycle + log      (adds FusionLogger append, file I/O)
  7. Live measurement round        (40 real HTTP requests to the 5 local instances)
  8. Monitor share of the 1 s step (compute cost of 5-6 vs the demo's 1 s clock)

Stages 1-6 use pre-generated latency inputs so they isolate the monitor itself.
Stage 7 is the real network/HTTP cost the demo pays to collect its evidence.
The window drawing (tkinter) is not timed: it needs a display.

CONCEPT DEMO numbers - single thread, local machine, not a real-time claim.

Usage (from the folder containing engine.py, service_adapter.py, logger.py
and real_time_service_gui.py):
    python speed_test_service.py
    python speed_test_service.py --dark
    python speed_test_service.py --no-live          (skip the HTTP round)
    python speed_test_service.py --plot-only speed_test_service_results.json
Needs numpy. Optional: psutil (sampler stage), matplotlib (results image).
Ports 9001-9005 must be free for the live-round stage.
"""

import argparse
import datetime
import json
import os
import platform
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import JOREngine                          # noqa: E402
from service_adapter import ServiceHealthAdapter      # noqa: E402
from logger import FusionLogger                       # noqa: E402
import real_time_service_gui as gui                   # noqa: E402

N_INSTANCES = gui.NUM_INSTANCES
REQS_PER_INSTANCE = gui.REQUESTS_PER_INSTANCE_PER_STEP
STEP_S = gui.STEP_INTERVAL_SECONDS
BASELINE_MS = 8.0          # same adapter baseline the GUI script uses


def make_engine(precalibrated=True):
    return JOREngine(prior_nh=0.05, steepness=12.0, upper_th=gui.UPPER_TH,
                     lower_th=gui.LOWER_TH, retention=0.70, calibration_steps=20,
                     baseline_sop=0.05 if precalibrated else None)


def make_adapter():
    return ServiceHealthAdapter(latency_baseline_ms=BASELINE_MS)


# ---------------------------------------------------------------------------
# Pre-generated inputs: healthy rounds, and rounds with inst_2 slowed to 900 ms
# ---------------------------------------------------------------------------
def build_inputs(n_steps=400, seed=7):
    rng = np.random.default_rng(seed)
    steps = []
    for i in range(n_steps):
        bad = (i % 4 == 1)                      # a quarter of the rounds carry the fault
        lats = {}
        for k in range(N_INSTANCES):
            base = 900.0 if (bad and k == gui.BAD_INSTANCE_INDEX) else 7.0
            lats[f"inst_{k}"] = base * rng.lognormal(0.0, 0.25, REQS_PER_INSTANCE)
        steps.append(dict(lats=lats, errors=0, total=N_INSTANCES * REQS_PER_INSTANCE,
                          mem=float(rng.uniform(35, 70)), cpu=float(rng.uniform(3, 40)),
                          disk=float(rng.uniform(0, 1.5))))
    return steps


def time_loop(fn, items, reps):
    fn(items[0])                                           # warm-up
    n = len(items)
    t0 = time.perf_counter()
    for i in range(reps):
        fn(items[i % n])
    return (time.perf_counter() - t0) / reps * 1e6


# ---------------------------------------------------------------------------
# 1. Engine
# ---------------------------------------------------------------------------
def bench_engine(n_calls=200_000):
    rng = np.random.default_rng(1)
    thetas = [tuple(map(float, r)) for r in rng.uniform(0.0, 0.7, size=(1000, 3))]
    eng = make_engine()
    for t in thetas[:200]:
        eng.fusion_step(*t)
    eng = make_engine()
    t0 = time.perf_counter()
    for i in range(n_calls):
        eng.fusion_step(*thetas[i % len(thetas)])
    fusion_us = (time.perf_counter() - t0) / n_calls * 1e6

    print("\n--- Engine ---")
    print(f"  fusion_step (calibrated) : {fusion_us:7.2f} us/call  ({1e6 / fusion_us:,.0f} steps/s)")
    return dict(fusion_us=fusion_us)


# ---------------------------------------------------------------------------
# 2-4. Adapter, sampler, banner
# ---------------------------------------------------------------------------
def bench_adapter(steps, reps=5000):
    ad = make_adapter()
    ctx = time_loop(lambda s: ad.normalize_context(s["cpu"], 100.0), steps, reps)
    sysu = time_loop(lambda s: ad.normalize_system_state(s["mem"], s["disk"]), steps, reps)
    agg = time_loop(lambda s: ad.extract_features(
        np.concatenate(list(s["lats"].values())), s["errors"], s["total"]), steps, reps)
    per = time_loop(lambda s: ad.extract_features_per_instance(
        s["lats"], s["errors"], s["total"]), steps, reps)

    ev = dict(calibrating=False, alert=False, outliers=1, nhp=0.2)
    ban = time_loop(lambda s: gui.banner_state(ev), steps, reps * 20)

    print("\n--- Adapter + banner ---")
    print(f"  theta_c  normalize_context            : {ctx:8.2f} us/call")
    print(f"  theta_s  normalize_system_state       : {sysu:8.2f} us/call")
    print(f"  theta_o  aggregate (40 samples)       : {agg:8.2f} us/call")
    print(f"  theta_o  per-instance ({N_INSTANCES} x {REQS_PER_INSTANCE})        : {per:8.2f} us/call")
    print(f"  banner_state                          : {ban:8.3f} us/call")
    return dict(ctx_us=ctx, sys_us=sysu, agg_us=agg, per_us=per, banner_us=ban)


def bench_sampler(reps=2000):
    if gui.psutil is None:
        print("\n--- Laptop sampler ---\n  psutil not installed: skipped")
        return None
    sm = gui.SystemSampler()
    sm.sample()
    t0 = time.perf_counter()
    for _ in range(reps):
        sm.sample()
    us = (time.perf_counter() - t0) / reps * 1e6
    print("\n--- Laptop sampler ---")
    print(f"  SystemSampler.sample (psutil mem/CPU/disk) : {us:8.1f} us/call")
    return dict(sampler_us=us)


# ---------------------------------------------------------------------------
# 5-6. Full monitor cycle, without and with logging
# ---------------------------------------------------------------------------
def monitor_cycle(engine, adapter, sampler, s, logger=None, step=0):
    obs = adapter.extract_features_per_instance(s["lats"], s["errors"], s["total"])
    if sampler is not None:
        mem, cpu, dq = sampler.sample()
    else:
        mem, cpu, dq = s["mem"], s["cpu"], s["disk"]
    theta_s = adapter.normalize_system_state(mem, dq or 0.0)
    theta_c = adapter.normalize_context(cpu, 100.0)
    sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs["theta_o"])
    outliers = obs.get("persistent_outliers", 0)
    phase = gui.banner_state({"calibrating": not engine.is_calibrated, "alert": alert,
                              "outliers": outliers, "nhp": nhp})
    if logger is not None:
        logger.log_state(sop, nhp, alert, metadata={
            "step": step, "elapsed_s": float(step), "phase": phase,
            "persistent_outliers": outliers,
            "theta_s": round(float(theta_s), 4), "theta_c": round(float(theta_c), 4),
            "mem_pct": mem, "cpu_pct": cpu, "disk_q": dq})
    return phase


def bench_cycle(steps, n_cycles=4000, n_log_cycles=2000, use_sampler=True):
    n = len(steps)
    sampler = gui.SystemSampler() if (use_sampler and gui.psutil is not None) else None

    def fresh():
        eng, ad = make_engine(precalibrated=False), make_adapter()
        for i in range(20):                       # healthy rounds -> real calibration
            monitor_cycle(eng, ad, sampler, steps[(i * 4) % n])
        return eng, ad

    eng, ad = fresh()
    for i in range(50):
        monitor_cycle(eng, ad, sampler, steps[i % n])

    eng, ad = fresh()
    t0 = time.perf_counter()
    for i in range(n_cycles):
        monitor_cycle(eng, ad, sampler, steps[i % n])
    cyc = (time.perf_counter() - t0) / n_cycles * 1e6

    with tempfile.TemporaryDirectory() as td:
        lg = FusionLogger(os.path.join(td, "bench_log.json"))
        eng, ad = fresh()
        t0 = time.perf_counter()
        for i in range(n_log_cycles):
            monitor_cycle(eng, ad, sampler, steps[i % n], logger=lg, step=i)
        cyc_log = (time.perf_counter() - t0) / n_log_cycles * 1e6
        rec = os.path.getsize(lg.log_file) / n_log_cycles

    print("\n--- Full monitor cycle ---")
    print(f"  Sampler: {'live psutil' if sampler else 'synthetic (psutil not installed)'}")
    print(f"  No I/O (adapter + sampler + fusion + banner) : {cyc:8.1f} us/cycle  ({1e6 / cyc:,.0f}/s)")
    print(f"  With JSON-lines log append                   : {cyc_log:8.1f} us/cycle  ({1e6 / cyc_log:,.0f}/s)")
    print(f"  Log cost alone                               : {cyc_log - cyc:8.1f} us/cycle")
    print(f"  Avg log record size                          : {rec:8.0f} bytes")
    print(f"  Share of the {STEP_S:g} s demo step (with log)         : "
          f"{cyc_log / (STEP_S * 1e6) * 100:.3f} %")
    return dict(cyc_us=cyc, cyc_log_us=cyc_log, rec_bytes=rec,
                live_sampler=sampler is not None)


# ---------------------------------------------------------------------------
# 7. Live HTTP measurement round
# ---------------------------------------------------------------------------
def bench_live(rounds=12):
    servers = []
    try:
        for i in range(N_INSTANCES):
            servers.append(gui.start_instance(i)[0])
        times = []
        with ThreadPoolExecutor(max_workers=N_INSTANCES) as ex:
            for _ in range(3):                                    # warm-up rounds
                gui.measure_all_instances(N_INSTANCES, gui.BASE_PORT, ex)
            for _ in range(rounds):
                t0 = time.perf_counter()
                lats, errs, tot = gui.measure_all_instances(N_INSTANCES, gui.BASE_PORT, ex)
                times.append((time.perf_counter() - t0) * 1000.0)
        med_ms = float(np.median([float(np.median(v)) for v in lats.values()]))
    finally:
        for s in servers:
            try:
                s.shutdown()
                s.server_close()
            except Exception:
                pass
    best, med = min(times), float(np.median(times))
    print("\n--- Live HTTP measurement round (real requests, healthy instances) ---")
    print(f"  Requests per round   : {N_INSTANCES * REQS_PER_INSTANCE}  "
          f"({N_INSTANCES} instances x {REQS_PER_INSTANCE})")
    print(f"  Best of {rounds} rounds     : {best:7.1f} ms")
    print(f"  Median of {rounds} rounds   : {med:7.1f} ms")
    print(f"  Median per-request latency seen : {med_ms:.1f} ms")
    return dict(best_ms=best, median_ms=med, per_request_ms=med_ms, errors=int(errs))


# ---------------------------------------------------------------------------
# Results image
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
    return f"{v:.2f} µs" if v < 10 else (f"{v:.1f} µs" if v < 100 else f"{v:,.0f} µs")


def render_png(R, path, theme="light"):
    try:
        import logging
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    except ImportError:
        print("\n  (PNG skipped: matplotlib is not installed - `pip install matplotlib`,")
        print("   then `python speed_test_service.py --plot-only speed_test_service_results.json`)")
        return None

    T, S = THEMES[theme], THEMES[theme]["series"]
    e, a, c, m = R["engine"], R["adapter"], R["cycle"], R["meta"]
    sm, lv = R.get("sampler"), R.get("live")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "SF Pro Text", "Helvetica Neue",
                                       "Arial", "DejaVu Sans"]
    fig = plt.figure(figsize=(13, 7.6), dpi=170, facecolor=T["surface"])

    def text(x, y, s, size=10, color=None, weight="normal", ha="left"):
        fig.text(x, y, s, fontsize=size, color=color or T["ink"], fontweight=weight, ha=ha)

    def rect(x, y, w, h, fc, ec="none", lw=0.0):
        fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure,
                                 facecolor=fc, edgecolor=ec, linewidth=lw))

    def hline(x0, x1, y):
        fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                              color=T["grid"], linewidth=0.8))

    text(0.04, 0.945, "JOR Real-Time Service Health Demo — Speed Test", size=20, weight="bold")
    text(0.04, 0.912, (f"Run {m['date']}  ·  Python {m['python']}  ·  NumPy {m['numpy']}"
                       f"  ·  {m['platform']}")[:150], size=9.5, color=T["ink2"])
    proc = (m.get("processor") or "").strip()
    text(0.04, 0.890, f"{proc[:90]}  ·  {m.get('cpus', '?')} logical CPUs".strip(" ·"),
         size=9.5, color=T["muted"])

    live_val = f"{lv['median_ms']:.0f} ms" if lv else "n/a"
    tiles = [
        ("Full monitor cycle", f"{c['cyc_us']:.0f} µs", f"{1e6 / c['cyc_us']:,.0f} cycles/s, no I/O"),
        ("With logging", f"{c['cyc_log_us']:.0f} µs",
         f"{c['cyc_log_us'] / (STEP_S * 1e6) * 100:.3f}% of a {STEP_S:g} s step"),
        ("JOR fusion step", f"{e['fusion_us']:.2f} µs", f"{1e6 / e['fusion_us']:,.0f} steps/s"),
        ("Live HTTP round", live_val,
         f"{N_INSTANCES * REQS_PER_INSTANCE} real requests (median)" if lv else "skipped"),
    ]
    tw, gap, ty, th = 0.2175, 0.0167, 0.735, 0.125
    for i, (lab, val, cap) in enumerate(tiles):
        x = 0.04 + i * (tw + gap)
        rect(x, ty, tw, th, T["tile"], T["grid"], 0.8)
        text(x + 0.014, ty + th - 0.030, lab, size=9.5, color=T["ink2"])
        text(x + 0.014, ty + 0.046, val, size=27, weight="bold")
        text(x + 0.014, ty + 0.014, cap, size=9, color=T["muted"])

    stages = [
        ("θ_o  per-instance (5 × 8)", a["per_us"], 0),
        ("θ_o  aggregate (40 samples)", a["agg_us"], 0),
        ("JSON log append (derived)", max(c["cyc_log_us"] - c["cyc_us"], 0.0), 4),
        ("θ_s  system state", a["sys_us"], 1),
        ("θ_c  context load", a["ctx_us"], 2),
        ("JOR fusion step", e["fusion_us"], 3),
        ("banner_state", a["banner_us"], 3),
    ]
    if sm:
        stages.append(("laptop sampler (psutil)", sm["sampler_us"], 1))
    stages.sort(key=lambda s: s[1])
    names, vals = [s[0] for s in stages], [s[1] for s in stages]
    cols = [S[s[2]] for s in stages]
    vmax = max(vals)

    text(0.04, 0.700, "Cost per stage (µs per call)", size=12, weight="bold")
    legend = [("θ_o latency", 0), ("θ_s memory/disk", 1), ("θ_c CPU load", 2),
              ("JOR engine / label", 3), ("JSON logging", 4)]
    for (lab, idx), x in zip(legend, [0.04, 0.135, 0.255, 0.355, 0.485]):
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

    per_share = a["per_us"] / c["cyc_us"] * 100.0
    text(0.04, 0.085,
         f"Per-instance θ_o scoring is {per_share:.0f}% of a full monitor cycle; "
         f"the JOR engine step is {e['fusion_us'] / c['cyc_us'] * 100:.1f}%.",
         size=10, color=T["ink2"])

    tx0, tx1 = 0.735, 0.96
    text(tx0, 0.700, "All results", size=12, weight="bold")
    rows = [
        ("Engine fusion step", f"{e['fusion_us']:.2f} µs"),
        ("θ_c  normalize_context", f"{a['ctx_us']:.1f} µs"),
        ("θ_s  normalize_system_state", f"{a['sys_us']:.1f} µs"),
        ("θ_o  aggregate (40 smp)", f"{a['agg_us']:.1f} µs"),
        ("θ_o  per-instance (5×8)", f"{a['per_us']:.1f} µs"),
        ("Laptop sampler (psutil)", f"{sm['sampler_us']:.1f} µs" if sm else "n/a"),
        ("Full cycle, no I/O", f"{c['cyc_us']:.0f} µs · {1e6 / c['cyc_us']:,.0f}/s"),
        ("Full cycle + log", f"{c['cyc_log_us']:.0f} µs · {1e6 / c['cyc_log_us']:,.0f}/s"),
        ("Log record size", f"{c['rec_bytes']:.0f} bytes"),
        ("Live HTTP round (40 req)", f"{lv['median_ms']:.1f} ms median" if lv else "skipped"),
    ]
    y = 0.655
    for lab, val in rows:
        text(tx0, y, lab, size=9, color=T["ink2"])
        text(tx1, y, val, size=9, color=T["ink"], ha="right")
        hline(tx0, tx1, y - 0.014)
        y -= 0.0455

    text(0.04, 0.045,
         "Single-thread timings of the monitor (adapter + JOR engine + label + log) on "
         "pre-generated inputs; live round uses real local HTTP. Concept demo — not a "
         "real-time or production claim.", size=8.5, color=T["muted"])
    fig.savefig(path, facecolor=T["surface"])
    plt.close(fig)
    return os.path.abspath(path)


def collect_meta():
    return dict(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                python=sys.version.split()[0], numpy=np.__version__,
                platform=platform.platform(), processor=platform.processor() or "",
                cpus=os.cpu_count())


def main():
    ap = argparse.ArgumentParser(description="JOR real-time service health speed test")
    ap.add_argument("--png", default="speed_test_service_results.png")
    ap.add_argument("--json", default="speed_test_service_results.json")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--no-live", action="store_true", help="skip the real HTTP round")
    ap.add_argument("--plot-only", metavar="JSON")
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
    print("JOR Real-Time Service Health Demo - Speed Test")
    print("=" * 66)
    print(f"Python    : {sys.version.split()[0]}")
    print(f"NumPy     : {np.__version__}")
    print(f"psutil    : {gui.psutil.__version__ if gui.psutil else 'not installed'}")
    print(f"Platform  : {platform.platform()}")
    print(f"Processor : {platform.processor() or 'n/a'}")
    print("\nNote: stages 1-6 use pre-generated inputs, single thread. CONCEPT DEMO numbers.")

    steps = build_inputs(400)
    eng = bench_engine()
    adp = bench_adapter(steps)
    smp = bench_sampler()
    cyc = bench_cycle(steps)
    live = None if args.no_live else bench_live()

    print("\n" + "=" * 66)
    print("Summary")
    print("=" * 66)
    print(f"  JOR fusion step              : {eng['fusion_us']:.1f} us")
    print(f"  Adapter, per-instance theta_o: {adp['per_us']:.1f} us")
    print(f"  Full monitor cycle (no I/O)  : {cyc['cyc_us']:.0f} us  ({1e6 / cyc['cyc_us']:,.0f}/s)")
    print(f"  Full cycle + log             : {cyc['cyc_log_us']:.0f} us  ({1e6 / cyc['cyc_log_us']:,.0f}/s)")
    if live:
        print(f"  Live HTTP round (40 requests): {live['median_ms']:.1f} ms median")
    print("\nThese numbers measure the offline monitor (and one local HTTP round).")
    print("They do not claim real-time or production performance.")
    print("=" * 66)

    R = dict(meta=collect_meta(), engine=eng, adapter=adp, sampler=smp, cycle=cyc, live=live)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print(f"\nRaw numbers saved to {os.path.abspath(args.json)}")
    if not args.no_png:
        out = render_png(R, args.png, theme)
        if out:
            print(f"Results image saved to {out}")


if __name__ == "__main__":
    main()
