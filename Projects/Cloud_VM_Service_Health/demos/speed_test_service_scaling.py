"""
JOR Service Health - Fleet-Size Scaling Speed Test
==================================================
How does the monitor's own compute cost grow as the fleet gets bigger?

For each fleet size (default 5, 50, 500, 5,000, 20,000 instances) this times one
monitor cycle on pre-generated latency data:

    ServiceHealthAdapter.extract_features_per_instance  (per-instance theta_o)
  + theta_s / theta_c normalization
  + JOREngine.fusion_step

and reports ms per cycle, microseconds per instance, and the largest fleet that
would fit inside a 1 s, 10 s and 60 s scrape interval (extrapolated from the
measured cost). One instance in every fleet is slowed (gray failure) so the
detector is doing real work, and the run checks that it is still flagged.

Measures the monitor's COMPUTE only. Collecting telemetry from a real fleet
(network, agents, missing data) is not included. Single thread, offline.
CONCEPT DEMO numbers - not a production or real-time claim.

Needs engine.py and service_adapter.py in the same folder (or on PYTHONPATH).
Needs numpy. matplotlib is optional (for the chart).

Usage:
    python speed_test_service_scaling.py
    python speed_test_service_scaling.py --sizes 5 100 1000 10000
    python speed_test_service_scaling.py --samples 8 --no-png
"""

import argparse
import datetime
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import JOREngine                          # noqa: E402
from service_adapter import ServiceHealthAdapter      # noqa: E402

BASELINE_MS = 8.0


def build_fleet(n_instances, samples, bad_idx, rng, bad=False):
    lats = {}
    for k in range(n_instances):
        base = 900.0 if (bad and k == bad_idx) else 7.0
        lats[f"inst_{k}"] = base * rng.lognormal(0.0, 0.25, samples)
    return lats


def bench_size(n, samples, seed=7):
    rng = np.random.default_rng(seed)
    bad_idx = n // 2
    total = n * samples
    # a few pre-generated rounds, cycled; every 4th round carries the gray failure
    rounds = [build_fleet(n, samples, bad_idx, rng, bad=(i % 4 == 1)) for i in range(8)]

    adapter = ServiceHealthAdapter(latency_baseline_ms=BASELINE_MS)
    engine = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                       retention=0.70, calibration_steps=20)

    def cycle(lats):
        obs = adapter.extract_features_per_instance(lats, 0, total)
        theta_s = adapter.normalize_system_state(45.0, 1.0)
        theta_c = adapter.normalize_context(30.0, 100.0)
        engine.fusion_step(theta_s, theta_c, obs["theta_o"])
        return obs

    healthy = [r for j, r in enumerate(rounds) if j % 4 != 1]
    for i in range(20):                       # calibrate on healthy rounds
        cycle(healthy[i % len(healthy)])
    for i in range(4):                        # warm-up
        cycle(rounds[i % 8])

    # time enough cycles for a stable number, but keep big fleets quick
    reps = int(np.clip(200_000 / max(n, 1), 12, 400))
    times = []
    for i in range(reps):
        t0 = time.perf_counter()
        cycle(rounds[i % 8])
        times.append(time.perf_counter() - t0)

    # sanity: the slow instance must still be flagged in a fault round
    # (fresh random rounds each time, as in real traffic; the same round repeated
    # would turn every instance's one-off noise into a "persistent" offset)
    adapter2 = ServiceHealthAdapter(latency_baseline_ms=BASELINE_MS)
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(10):                       # healthy lead-in
        adapter2.extract_features_per_instance(
            build_fleet(n, samples, bad_idx, rng2, bad=False), 0, total)
    for _ in range(10):                       # gray failure on one instance
        adapter2.extract_features_per_instance(
            build_fleet(n, samples, bad_idx, rng2, bad=True), 0, total)
    flagged_ids = {k for k, v in adapter2.instance_ema.items()
                   if v > adapter2.outlier_iqr_threshold}
    caught = f"inst_{bad_idx}" in flagged_ids
    false_flags = len(flagged_ids - {f"inst_{bad_idx}"})
    best = min(times)
    med = float(np.median(times))
    return dict(n=n, reps=reps, best_ms=best * 1e3, median_ms=med * 1e3,
                per_instance_us=best / n * 1e6, slow_instance_caught=bool(caught),
                other_flagged=int(false_flags))


def render_png(R, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (PNG skipped: matplotlib not installed)")
        return None
    rows = R["sizes"]
    x = [r["n"] for r in rows]
    y = [r["best_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    ax.loglog(x, y, marker="o", color="#2a78d6", linewidth=2)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:,.1f} ms", (xi, yi), textcoords="offset points",
                    xytext=(6, -14), fontsize=9)
    for limit, lab in ((1000, "1 s interval"), (10000, "10 s interval"), (60000, "60 s interval")):
        if limit < max(y) * 20:
            ax.axhline(limit, color="#c93a3a", linestyle="--", linewidth=0.8)
            ax.text(min(x), limit * 1.1, lab, fontsize=8, color="#c93a3a")
    ax.set_xlabel("Instances in the fleet (8 samples each)")
    ax.set_ylabel("Monitor compute per cycle (ms)")
    ax.set_title("JOR service health monitor: compute cost vs fleet size")
    ax.grid(True, which="both", alpha=0.25)
    fig.text(0.01, 0.01, "Single thread, pre-generated inputs, compute only. Concept demo, "
             "not a production claim.", fontsize=7, color="#777777")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return os.path.abspath(path)


def main():
    ap = argparse.ArgumentParser(description="JOR service health fleet-size scaling test")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 50, 500, 5000, 20000])
    ap.add_argument("--samples", type=int, default=8, help="latency samples per instance")
    ap.add_argument("--json", default="speed_test_service_scaling_results.json")
    ap.add_argument("--png", default="speed_test_service_scaling_results.png")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("JOR Service Health - Fleet-Size Scaling Speed Test")
    print("=" * 70)
    print(f"Python {sys.version.split()[0]}  NumPy {np.__version__}  {platform.platform()}")
    print(f"{args.samples} latency samples per instance; compute only, single thread.\n")
    print(f"{'instances':>10} {'ms/cycle':>10} {'us/instance':>12} {'slow one caught':>16} {'other flagged':>14}")

    rows = []
    for n in args.sizes:
        r = bench_size(n, args.samples)
        rows.append(r)
        print(f"{r['n']:>10,} {r['best_ms']:>10.2f} {r['per_instance_us']:>12.1f} "
              f"{('yes' if r['slow_instance_caught'] else 'NO'):>16} {r['other_flagged']:>14}")

    # extrapolate capacity from the largest measured size (cost is ~linear in fleet size)
    big = rows[-1]
    per_inst_s = big["best_ms"] / 1e3 / big["n"]
    print("\nApprox. fleet size that fits in one scrape interval "
          f"(from the {big['n']:,}-instance cost, using up to 50% of the interval):")
    cap = {}
    for interval in (1, 10, 60):
        n_fit = int(0.5 * interval / per_inst_s)
        cap[str(interval)] = n_fit
        print(f"  {interval:>2} s interval : ~{n_fit:,} instances")
    ok = all(r["slow_instance_caught"] for r in rows)
    print(f"\nDetector still flags the slow instance at every size: {'YES' if ok else 'NO'}")
    print("('other flagged' = healthy instances that were also flagged in that check)")
    print("\nCompute only. Real telemetry collection is not included. CONCEPT DEMO numbers.")
    print("=" * 70)

    R = dict(meta=dict(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                       python=sys.version.split()[0], numpy=np.__version__,
                       platform=platform.platform(), cpus=os.cpu_count()),
             samples_per_instance=args.samples, sizes=rows, capacity_estimate=cap)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print(f"Raw numbers saved to {os.path.abspath(args.json)}")
    if not args.no_png:
        out = render_png(R, args.png)
        if out:
            print(f"Chart saved to {out}")


if __name__ == "__main__":
    main()
