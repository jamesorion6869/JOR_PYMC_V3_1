"""
Real-time local service health test -- with a live status window.

Same test as real_time_service_test.py (real local HTTP "instances", real
latency measurements, the real engine.py + service_adapter.py), plus a small
window that runs alongside the CLI output:

  * a status banner:  CALIBRATING / HEALTHY / WATCH / DEGRADED
  * a live chart of NHP, theta_o, theta_s and theta_c with the alert thresholds
  * per-instance latency bars (red when the adapter flags that instance)
  * live laptop readings (memory, CPU, disk activity) that feed theta_s / theta_c

The banner shows what the ENGINE concluded, not what the test injected, so
you can see the detection lag against the "injected fault" shading.

    CALIBRATING  engine still learning its baseline (no alerts possible)
    HEALTHY      calibrated, no alert, no persistent outlier instance
    WATCH        an instance is flagged as a persistent outlier, or NHP is
                 above the clear line, but the alert is not raised
    DEGRADED     the engine's hysteresis alert is active

Evidence channels:
    theta_o  measured latency / errors of the local test instances (live)
    theta_s  laptop memory % + disk activity, through
             adapter.normalize_system_state (live, needs psutil)
    theta_c  laptop CPU load, through adapter.normalize_context(cpu%, 100)
             (live, needs psutil)
Without psutil (or with --no-system) theta_s and theta_c stay fixed at 0.2,
exactly like the original test.

Needs numpy. Live laptop readings need psutil:  pip install psutil
tkinter ships with Python; if it is not available, or you pass --headless,
it runs CLI-only.

Usage:
    python real_time_service_gui.py
    python real_time_service_gui.py --headless
    python real_time_service_gui.py --duration 90
    python real_time_service_gui.py --no-system      (fixed 0.2, original behavior)

Close the window or press Ctrl+C in the terminal to stop.
"""

import argparse
import http.server
import math
import queue
import socketserver
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from engine import JOREngine
from service_adapter import ServiceHealthAdapter
from logger import FusionLogger

try:
    import tkinter as tk
except Exception:  # tkinter missing -> CLI-only
    tk = None

try:
    import psutil
except Exception:  # psutil missing -> fixed theta_s / theta_c
    psutil = None

NUM_INSTANCES = 5
BASE_PORT = 9001
BAD_INSTANCE_INDEX = 2
BAD_AFTER_SECONDS = 25
RECOVER_AFTER_SECONDS = 45
BAD_DELAY_MS = 900.0
REQUESTS_PER_INSTANCE_PER_STEP = 8
STEP_INTERVAL_SECONDS = 1.0

UPPER_TH = 0.55
LOWER_TH = 0.45
CALIBRATION_STEPS = 20
WINDOW_SECONDS = 120.0  # width of the scrolling chart

FIXED_THETA = 0.2  # theta_s / theta_c when live system readings are off

# Shared state so handler threads know whether THEIR instance should be slow.
instance_delay_ms = {i: 0.0 for i in range(NUM_INSTANCES)}


# --------------------------------------------------------------------------
# Live laptop readings (psutil)
# --------------------------------------------------------------------------

class SystemSampler:
    """
    memory_pct -> theta_s (together with disk activity)
    cpu_pct    -> theta_c (as utilization, knee at 80%)
    disk_q     -> estimated average disk queue depth: the change in
                  (read_time + write_time) per elapsed millisecond. On Windows
                  this behaves more like a busy fraction per disk, so it stays
                  small there. None if the OS exposes no disk counters.
    """

    def __init__(self):
        psutil.cpu_percent(interval=None)  # first call is meaningless; prime it
        self._last_io = self._io_time()
        self._last_t = time.perf_counter()

    @staticmethod
    def _io_time():
        try:
            io = psutil.disk_io_counters()
            if io is None:
                return None
            return float(io.read_time + io.write_time)
        except Exception:
            return None

    def sample(self):
        mem = float(psutil.virtual_memory().percent)
        cpu = float(psutil.cpu_percent(interval=None))
        now = time.perf_counter()
        io = self._io_time()
        disk_q = None
        if io is not None and self._last_io is not None and now > self._last_t:
            disk_q = max(0.0, (io - self._last_io) / ((now - self._last_t) * 1000.0))
        self._last_io, self._last_t = io, now
        return mem, cpu, disk_q


# --------------------------------------------------------------------------
# Local "service instances" (unchanged from the original test)
# --------------------------------------------------------------------------

def make_handler(instance_idx):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            delay = instance_delay_ms.get(instance_idx, 0.0)
            if delay > 0:
                time.sleep(delay / 1000.0)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    return Handler


class InstanceServer(socketserver.ThreadingTCPServer):
    # Lets you re-run right after stopping (no "Address already in use"),
    # and keeps the listen queue big enough for 8 concurrent requests.
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64


def start_instance(instance_idx):
    port = BASE_PORT + instance_idx
    server = InstanceServer(("127.0.0.1", port), make_handler(instance_idx))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def measure_instance(port, n_requests=REQUESTS_PER_INSTANCE_PER_STEP, timeout=2.0):
    url = f"http://127.0.0.1:{port}/"

    def one_request():
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                resp.read()
            return (time.perf_counter() - start) * 1000.0, False
        except Exception:
            return timeout * 1000.0, True

    with ThreadPoolExecutor(max_workers=n_requests) as pool:
        results = list(pool.map(lambda _: one_request(), range(n_requests)))

    latencies = np.array([r[0] for r in results])
    errors = sum(1 for r in results if r[1])
    return latencies, errors


def measure_all_instances(num_instances, base_port, executor):
    futures = {
        i: executor.submit(measure_instance, base_port + i)
        for i in range(num_instances)
    }
    instance_latencies = {}
    total_errors = 0
    total_requests = 0

    for i, future in futures.items():
        lat, err = future.result()
        instance_latencies[f"inst_{i}"] = lat
        total_errors += err
        total_requests += len(lat)

    return instance_latencies, total_errors, total_requests


# --------------------------------------------------------------------------
# Monitor loop: same logic as the original, but it also emits one event dict
# per step for the window. Runs in a worker thread when the GUI is on.
# --------------------------------------------------------------------------

def monitor_loop(events, stop, args):
    engine = JOREngine(
        prior_nh=0.05,
        steepness=12.0,
        upper_th=UPPER_TH,
        lower_th=LOWER_TH,
        retention=0.70,
        calibration_steps=args.calibration_steps,
    )
    # Real healthy latency of these local test servers is ~5–10 ms.
    # The adapter default of 50 ms made the pure latency path to theta_o
    # silent; only the outlier boost was driving detection.  Aligning the
    # baseline with actual healthy latency makes the aggregate path meaningful.
    adapter = ServiceHealthAdapter(latency_baseline_ms=8.0)
    logger = FusionLogger("real_time_service_log.json")

    sampler = None
    if psutil is not None and not args.no_system:
        try:
            sampler = SystemSampler()
        except Exception as exc:
            print(f"(System readings unavailable: {exc})", flush=True)
    if sampler is not None:
        print("System readings: live memory / CPU / disk via psutil.", flush=True)
    else:
        why = "--no-system given" if args.no_system else "psutil not installed (pip install psutil)"
        print(f"System readings off ({why}): theta_s = theta_c = {FIXED_THETA}.",
              flush=True)

    step = 0

    with ThreadPoolExecutor(max_workers=NUM_INSTANCES) as executor:
        # Throwaway warm-up round: the very first connections and threads are
        # slow (hundreds of ms on some runs), which otherwise lands in the
        # engine's calibration average. Not fed to the engine or the log.
        if not args.no_warmup:
            measure_all_instances(NUM_INSTANCES, BASE_PORT, executor)

        if sampler is not None:
            # Re-prime the CPU/disk counters AFTER the warm-up burst, then let one
            # idle interval pass, so step 0's readings cover the same kind of
            # window as every later step (idle time + one measurement round).
            sampler.sample()
            stop.wait(STEP_INTERVAL_SECONDS)

        start_time = time.time()
        while not stop.is_set():
            elapsed = time.time() - start_time
            if args.duration and elapsed >= args.duration:
                break

            fault_active = args.bad_after <= elapsed < args.recover_after
            instance_delay_ms[BAD_INSTANCE_INDEX] = BAD_DELAY_MS if fault_active else 0.0

            instance_latencies, total_errors, total_requests = measure_all_instances(
                NUM_INSTANCES, BASE_PORT, executor)

            obs = adapter.extract_features_per_instance(
                instance_latencies, total_errors, total_requests)

            mem_pct = cpu_pct = disk_q = None
            theta_s = theta_c = FIXED_THETA
            if sampler is not None:
                mem_pct, cpu_pct, disk_q = sampler.sample()
                theta_s = adapter.normalize_system_state(mem_pct, disk_q or 0.0)
                theta_c = adapter.normalize_context(cpu_pct, 100.0)

            sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs['theta_o'])

            outliers = obs.get('persistent_outliers', 0)

            # CLI / log phase label: the same state the window banner shows
            # (what the ENGINE concluded), not what the test injected.
            phase = banner_state({
                "calibrating": not engine.is_calibrated,
                "alert": alert,
                "outliers": outliers,
                "nhp": nhp,
            })

            median = obs.get('median_latency_ms', obs.get('p95_latency_ms', 0))
            status = "ALERT" if alert else "Normal"

            sys_txt = ""
            if sampler is not None:
                dq = "n/a" if disk_q is None else f"{disk_q:.2f}"
                sys_txt = (f"  | mem={mem_pct:3.0f}% cpu={cpu_pct:3.0f}% disk_q={dq} "
                           f"s={theta_s:.2f} c={theta_c:.2f}")

            print(f"t={elapsed:5.1f}s  [{phase:11s}]  "
                  f"theta_o={obs['theta_o']:.3f}  "
                  f"outliers={outliers}  median={median:6.1f}ms  "
                  f"NHP={nhp:.3f}  {status}{sys_txt}", flush=True)

            logger.log_state(sop, nhp, alert, metadata={
                "step": step,
                "elapsed_s": round(elapsed, 1),
                "phase": phase,
                "persistent_outliers": outliers,
                "theta_s": round(float(theta_s), 4),
                "theta_c": round(float(theta_c), 4),
                "mem_pct": mem_pct,
                "cpu_pct": cpu_pct,
                "disk_q": None if disk_q is None else round(disk_q, 4),
            })

            if events is not None:
                flagged = sorted(
                    k for k, v in adapter.instance_ema.items()
                    if v > adapter.outlier_iqr_threshold
                )
                events.put({
                    "t": elapsed,
                    "step": step,
                    "calibrating": not engine.is_calibrated,
                    "cal_done": min(step + 1, engine.calibration_steps),
                    "cal_total": engine.calibration_steps,
                    "nhp": float(nhp),
                    "theta_o": float(obs['theta_o']),
                    "alert": bool(alert),
                    "outliers": int(outliers),
                    "median_ms": float(median),
                    "fault_active": bool(fault_active),
                    "medians": {k: float(np.median(v))
                                for k, v in instance_latencies.items()},
                    "flagged": flagged,
                    "baseline_ms": adapter.latency_baseline_ms,
                    "theta_s": float(theta_s),
                    "theta_c": float(theta_c),
                    "sys_live": sampler is not None,
                    "mem_pct": mem_pct,
                    "cpu_pct": cpu_pct,
                    "disk_q": disk_q,
                })

            step += 1
            stop.wait(STEP_INTERVAL_SECONDS)

    stop.set()  # lets the window close itself when a --duration run ends


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------

STATE_COLORS = {
    "CALIBRATING": "#3b6ea8",
    "HEALTHY": "#2e9e5b",
    "WATCH": "#c98a12",
    "DEGRADED": "#c93a3a",
}


def banner_state(ev):
    if ev["calibrating"]:
        return "CALIBRATING"
    if ev["alert"]:
        return "DEGRADED"
    if ev["outliers"] > 0 or ev["nhp"] > LOWER_TH:
        return "WATCH"
    return "HEALTHY"


class StatusWindow:
    def __init__(self, root, events, stop, args):
        self.root = root
        self.events = events
        self.stop = stop
        self.args = args
        self.history = []
        self.cal_end_t = None

        root.title("JOR 4.0 - Service Health (real-time local test)")
        root.geometry("920x800")
        root.minsize(640, 620)
        root.configure(bg="#111111")

        self.banner = tk.Label(root, text="STARTING", font=("Helvetica", 30, "bold"),
                               fg="white", bg="#555555", pady=14)
        self.banner.pack(fill="x")

        self.detail = tk.Label(root, text="", font=("Courier", 11), fg="#dddddd",
                               bg="#111111", anchor="w", padx=12, pady=6)
        self.detail.pack(fill="x")
        self.truth = tk.Label(root, text="", font=("Helvetica", 10), fg="#9a9a9a",
                              bg="#111111", anchor="w", padx=12)
        self.truth.pack(fill="x")
        self.driver = tk.Label(root, text="", font=("Helvetica", 10, "bold"),
                               fg="#cfcfcf", bg="#111111", anchor="w", padx=12)
        self.driver.pack(fill="x")

        self.chart = tk.Canvas(root, bg="#181818", highlightthickness=0, height=280)
        self.chart.pack(fill="both", expand=True, padx=10, pady=(8, 4))
        self.bars = tk.Canvas(root, bg="#181818", highlightthickness=0, height=170)
        self.bars.pack(fill="x", padx=10, pady=(4, 4))
        self.sysp = tk.Canvas(root, bg="#181818", highlightthickness=0, height=112)
        self.sysp.pack(fill="x", padx=10, pady=(4, 10))

        self.chart.bind("<Configure>", lambda e: self.draw_chart())
        self.bars.bind("<Configure>", lambda e: self.draw_bars())
        self.sysp.bind("<Configure>", lambda e: self.draw_sys())

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.banner.config(text="CALIBRATING", bg=STATE_COLORS["CALIBRATING"])
        self.detail.config(text="waiting for first measurement...")
        root.after(100, self.poll)

    # -- plumbing ---------------------------------------------------------
    def on_close(self):
        self.stop.set()
        self.root.destroy()

    def poll(self):
        got = False
        try:
            while True:
                ev = self.events.get_nowait()
                self.history.append(ev)
                if self.cal_end_t is None and not ev["calibrating"]:
                    self.cal_end_t = ev["t"]
                got = True
        except queue.Empty:
            pass

        if got:
            self.refresh()

        if self.stop.is_set() and self.events.empty():
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    # -- drawing ----------------------------------------------------------
    def refresh(self):
        ev = self.history[-1]
        state = banner_state(ev)
        self.banner.config(text=state, bg=STATE_COLORS[state])

        if state == "CALIBRATING":
            self.detail.config(
                text=f"learning baseline: step {ev['cal_done']}/{ev['cal_total']}   "
                     f"(alerts are suppressed until this finishes)")
        else:
            self.detail.config(
                text=f"t={ev['t']:5.1f}s   NHP {ev['nhp']:.3f}   "
                     f"theta_o {ev['theta_o']:.3f}   "
                     f"fleet median {ev['median_ms']:.1f} ms   "
                     f"flagged instances {ev['outliers']}")

        if ev["fault_active"]:
            self.truth.config(
                text=f"test is injecting: inst_{BAD_INSTANCE_INDEX} slowed to "
                     f"{BAD_DELAY_MS:.0f} ms (zero errors)")
        else:
            self.truth.config(text="test is injecting: nothing (all instances normal)")

        self.driver.config(text=self.driver_text(ev))

        self.draw_chart()
        self.draw_bars()
        self.draw_sys()

    def driver_text(self, ev):
        """Which evidence channel is pushing the fused SOP up vs. its calibrated level."""
        if ev["calibrating"]:
            return ""
        cal = [e for e in self.history if e["step"] < ev["cal_total"]]
        if not cal:
            return ""
        keys = ("theta_s", "theta_c", "theta_o")
        weights = {"theta_s": 0.40, "theta_c": 0.30, "theta_o": 0.30}  # engine.py weights
        names = {"theta_s": "memory / disk (theta_s)",
                 "theta_c": "CPU load (theta_c)",
                 "theta_o": "service latency (theta_o)"}
        base = {k: sum(e[k] for e in cal) / len(cal) for k in keys}
        dev = {k: weights[k] * (ev[k] - base[k]) for k in keys}
        top = max(dev, key=dev.get)
        if dev[top] < 0.02:
            return "main driver: none (all channels near their calibrated baseline)"
        return f"main driver: {names[top]}   (+{dev[top]:.2f} fused SOP vs. baseline)"

    def draw_chart(self):
        c = self.chart
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 160)
        left, right, top, bottom = 52, 16, 24, 28
        pw, ph = w - left - right, h - top - bottom

        hist = self.history
        t_last = hist[-1]["t"] if hist else 0.0
        t1 = max(t_last, WINDOW_SECONDS)
        t0 = t1 - WINDOW_SECONDS

        def X(ts):
            return left + (ts - t0) / (t1 - t0) * pw

        def Y(v):
            return top + (1.0 - min(max(v, 0.0), 1.0)) * ph

        def span(a, b, color, label):
            a, b = max(a, t0), min(b, t1)
            if b <= a:
                return
            c.create_rectangle(X(a), top, X(b), top + ph, fill=color, outline="")
            c.create_text(X(a) + 4, top + 3, text=label, anchor="nw",
                          fill="#8f8f8f", font=("Helvetica", 8))

        # background regions: calibration + injected fault
        cal_hi = self.cal_end_t if self.cal_end_t is not None else t_last
        if hist:
            span(0.0, cal_hi, "#16202e", "calibrating")
        span(self.args.bad_after, self.args.recover_after, "#2e1719", "injected fault")

        # grid + y labels
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            c.create_line(left, Y(v), left + pw, Y(v), fill="#2a2a2a")
            c.create_text(left - 6, Y(v), text=f"{v:.2f}", anchor="e",
                          fill="#8f8f8f", font=("Helvetica", 8))

        # x ticks every 10 s
        tick = int(math.ceil(t0 / 10.0) * 10)
        while tick <= t1:
            c.create_line(X(tick), top + ph, X(tick), top + ph + 4, fill="#666666")
            c.create_text(X(tick), top + ph + 6, text=f"{tick}s", anchor="n",
                          fill="#8f8f8f", font=("Helvetica", 8))
            tick += 10

        # alert thresholds
        c.create_line(left, Y(UPPER_TH), left + pw, Y(UPPER_TH),
                      fill="#e05555", dash=(5, 3))
        c.create_text(left + pw - 2, Y(UPPER_TH) - 2, text=f"alert on > {UPPER_TH:.2f}",
                      anchor="se", fill="#e05555", font=("Helvetica", 8))
        c.create_line(left, Y(LOWER_TH), left + pw, Y(LOWER_TH),
                      fill="#d9a441", dash=(5, 3))
        c.create_text(left + pw - 2, Y(LOWER_TH) + 2, text=f"clear on < {LOWER_TH:.2f}",
                      anchor="ne", fill="#d9a441", font=("Helvetica", 8))

        # series
        def series(key, color, width):
            pts = [(X(e["t"]), Y(e[key])) for e in hist if e["t"] >= t0]
            if len(pts) >= 2:
                flat = [v for p in pts for v in p]
                c.create_line(*flat, fill=color, width=width, smooth=False)
            elif len(pts) == 1:
                x, y = pts[0]
                c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)

        live = bool(hist) and hist[-1]["sys_live"]
        if live:
            series("theta_s", "#b48ef5", 1)
            series("theta_c", "#e0d36a", 1)
        series("theta_o", "#e8964a", 1)
        series("nhp", "#5cc8ff", 2)

        # legend
        legend = [("NHP", "#5cc8ff", 4), ("theta_o latency", "#e8964a", 60)]
        if live:
            legend += [("theta_s memory+disk", "#b48ef5", 200),
                       ("theta_c CPU", "#e0d36a", 370)]
        for text, color, dx in legend:
            c.create_text(left + dx, 6, text=text, anchor="nw", fill=color,
                          font=("Helvetica", 9, "bold"))

    def draw_sys(self):
        c = self.sysp
        c.delete("all")
        if not self.history:
            return
        ev = self.history[-1]
        w = max(c.winfo_width(), 320)

        c.create_text(12, 6, text="laptop readings (live)", anchor="nw",
                      fill="#cccccc", font=("Helvetica", 9, "bold"))
        if not ev["sys_live"]:
            c.create_text(12, 30, anchor="nw", fill="#9a9a9a", font=("Helvetica", 9),
                          text="off: theta_s and theta_c are fixed at 0.2.  "
                               "Install psutil (pip install psutil) for live readings, "
                               "or you started with --no-system.")
            return
        c.create_text(w - 12, 6, anchor="ne", fill="#cccccc", font=("Helvetica", 9, "bold"),
                      text=f"theta_s {ev['theta_s']:.2f}      theta_c {ev['theta_c']:.2f}")

        x0, x1 = 130, max(w - 230, 200)
        rows = [
            ("Memory", ev["mem_pct"], 100.0, "{:.0f} %", "-> theta_s", None),
            ("Disk activity", ev["disk_q"], 10.0, "{:.2f} queue", "-> theta_s", None),
            ("CPU load", ev["cpu_pct"], 100.0, "{:.0f} %", "-> theta_c", 80.0),
        ]
        y = 30
        for label, val, vmax, fmt, tag, knee in rows:
            c.create_text(12, y + 6, text=label, anchor="w", fill="#aaaaaa",
                          font=("Helvetica", 9))
            c.create_rectangle(x0, y, x1, y + 12, fill="#2a2a2a", outline="")
            if val is None:
                c.create_text(x1 + 10, y + 6, text="n/a (no disk counters)", anchor="w",
                              fill="#777777", font=("Helvetica", 9))
            else:
                frac = min(max(val / vmax, 0.0), 1.0)
                hot = (knee is not None and val >= knee) or (label == "Memory" and val >= 90.0)
                c.create_rectangle(x0, y, x0 + frac * (x1 - x0), y + 12,
                                   fill="#d9a441" if hot else "#5b8def", outline="")
                c.create_text(x1 + 10, y + 6, text=fmt.format(val), anchor="w",
                              fill="#dddddd", font=("Helvetica", 9))
            if knee is not None:
                kx = x0 + (knee / vmax) * (x1 - x0)
                c.create_line(kx, y - 2, kx, y + 14, fill="#999999")
            c.create_text(w - 12, y + 6, text=tag, anchor="e", fill="#8f8f8f",
                          font=("Helvetica", 9))
            y += 25

    def draw_bars(self):
        c = self.bars
        c.delete("all")
        if not self.history:
            return
        ev = self.history[-1]
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 120)
        left, right, top, bottom = 52, 16, 22, 34
        pw, ph = w - left - right, h - top - bottom

        lo, hi = 1.0, 2000.0
        span_log = math.log10(hi) - math.log10(lo)

        def Y(ms):
            v = (math.log10(min(max(ms, lo), hi)) - math.log10(lo)) / span_log
            return top + (1.0 - v) * ph

        c.create_text(left + 4, 4, text="per-instance median latency (log scale)",
                      anchor="nw", fill="#cccccc", font=("Helvetica", 9, "bold"))

        for ms in (1, 10, 100, 1000):
            c.create_line(left, Y(ms), left + pw, Y(ms), fill="#2a2a2a")
            c.create_text(left - 6, Y(ms), text=f"{ms}ms", anchor="e",
                          fill="#8f8f8f", font=("Helvetica", 8))

        base = ev["baseline_ms"]
        c.create_line(left, Y(base), left + pw, Y(base), fill="#777777", dash=(4, 3))
        c.create_text(left + pw - 2, Y(base) - 2, text=f"adapter baseline {base:.0f} ms",
                      anchor="se", fill="#777777", font=("Helvetica", 8))

        names = sorted(ev["medians"])
        slot = pw / max(len(names), 1)
        bw = slot * 0.5
        for i, name in enumerate(names):
            ms = ev["medians"][name]
            x0 = left + i * slot + (slot - bw) / 2
            flagged = name in ev["flagged"]
            color = "#d9453f" if flagged else "#3fb27f"
            c.create_rectangle(x0, Y(ms), x0 + bw, top + ph, fill=color, outline="")
            c.create_text(x0 + bw / 2, Y(ms) - 3, text=f"{ms:.0f} ms", anchor="s",
                          fill="#dddddd", font=("Helvetica", 8))
            label = name + ("  FLAGGED" if flagged else "")
            c.create_text(x0 + bw / 2, top + ph + 6, text=label, anchor="n",
                          fill="#ff8f8a" if flagged else "#aaaaaa", font=("Helvetica", 8))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-time service health test with a live window")
    p.add_argument("--headless", action="store_true", help="CLI only, no window")
    p.add_argument("--duration", type=float, default=0.0,
                   help="stop automatically after N seconds (0 = run until stopped)")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the throwaway warm-up round (reproduces the original "
                        "script's cold-start behavior)")
    p.add_argument("--no-system", action="store_true",
                   help="do not read laptop memory/CPU/disk; keep theta_s = theta_c = 0.2")
    p.add_argument("--calibration-steps", type=int, default=CALIBRATION_STEPS)
    p.add_argument("--bad-after", type=float, default=BAD_AFTER_SECONDS)
    p.add_argument("--recover-after", type=float, default=RECOVER_AFTER_SECONDS)
    return p.parse_args()


def main():
    args = parse_args()

    print("=== JOR 4.0 -- REAL-TIME Local Service Health Test (live window) ===")
    print(f"Starting {NUM_INSTANCES} local instances on ports "
          f"{BASE_PORT}-{BASE_PORT + NUM_INSTANCES - 1}...")

    servers = []
    for i in range(NUM_INSTANCES):
        server, port = start_instance(i)
        servers.append(server)
        print(f"  instance_{i} listening on 127.0.0.1:{port}")

    print(f"\nInstance {BAD_INSTANCE_INDEX} will go slow after "
          f"{args.bad_after:g}s, recover after {args.recover_after:g}s.")

    stop = threading.Event()
    root = None
    use_gui = tk is not None and not args.headless
    if use_gui:
        try:
            root = tk.Tk()
        except Exception as exc:  # no display available
            print(f"(No window available: {exc}. Running CLI only.)")
            use_gui = False
    elif not args.headless:
        print("(tkinter not available. Running CLI only.)")

    print("Close the window or press Ctrl+C to stop.\n" if use_gui
          else "Press Ctrl+C to stop.\n")

    try:
        if use_gui:
            events = queue.Queue()
            StatusWindow(root, events, stop, args)
            worker = threading.Thread(target=monitor_loop, args=(events, stop, args),
                                      daemon=True)
            worker.start()
            try:
                root.mainloop()
            finally:
                stop.set()
                worker.join(timeout=4.0)
        else:
            monitor_loop(None, stop, args)
    except KeyboardInterrupt:
        stop.set()
    finally:
        print("\n\nStopped. Shutting down local instances...")
        for server in servers:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    main()
