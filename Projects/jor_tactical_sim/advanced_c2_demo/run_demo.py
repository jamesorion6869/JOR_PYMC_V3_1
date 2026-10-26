#!/usr/bin/env python3
"""
JOR 4.0 — one-command synthetic demo launcher.

Canonical path (repeatable pitch):
  scenario=baseline  seed=42  units=6  skill=expert  policy=intervene

Modes:
  python3 run_demo.py              # start live stream server + print talk track
  python3 run_demo.py --offline    # headless run, write frames + AAR, exit
  python3 run_demo.py --playback   # static HTTP for JSONL dashboard only

Talk track is printed to the terminal so a presenter can follow without notes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Canonical demo constants (do not change lightly; tests / pitch depend on them) ---
DEMO = {
    "scenario": "baseline",
    "seed": 42,
    "n_units": 6,
    "commander": "expert",
    "policy": "intervene",
    "ticks": 120,
    "decision_timeout": 8.0,
    "feedback": 1.0,
}


TALK_TRACK = """
================================================================
  JOR 4.0  ·  CANONICAL SYNTHETIC DEMO (~5 min)
================================================================
  Config:  scenario={scenario}  seed={seed}  units={n_units}
           commander={commander}  policy={policy}

  1. Open the dashboard URL printed below.
  2. Click  Map focus  (or press M) so the map fills the screen.
  3. Click  Live stream  (live mode only).
  4. Watch early ticks: green readiness, low NHP, first decision.
  5. When a decision prompt appears, click [1] or [2] (intervene).
  6. Around t=30: local strain on a task force — note the red card.
  7. Pressure / NHP rise; ALERT may arm; risk rings expand.
  8. Near t=110: catastrophic stress — rings deep red, cues stack.
  9. At end: read the AFTER-ACTION SUMMARY (terminal + UI card).

  Keys:  [  matrix   ]  decision   M  map focus
================================================================
""".format(**DEMO)


def run_offline(out_frames: Path, open_browser: bool) -> None:
    from headless_runner import make_input_fn
    from maneuver_sim_interactive import run_simulation
    from aar_summary import summarize_frames, format_summary

    print(TALK_TRACK)
    print("Running offline headless mission…")
    with contextlib.redirect_stdout(io.StringIO()):
        *_, sink = run_simulation(
            scenario_name=DEMO["scenario"],
            commander_level=DEMO["commander"],
            feedback=DEMO["feedback"],
            seed=DEMO["seed"],
            input_fn=make_input_fn(DEMO["policy"]),
            delay=0.0,
            log_file=str(ROOT / "maneuver_log_demo.json"),
            n_ticks=DEMO["ticks"],
            n_units=DEMO["n_units"],
            map_log=str(out_frames),
            engine_profile="graded",
            prompt_guards="fixed",
            decision_timeout=DEMO["decision_timeout"],
        )
    from debrief import build_debrief, format_debrief
    from map_telemetry import append_end_record
    summary = summarize_frames(sink.frames)
    debrief = build_debrief(sink.frames)
    text = format_summary(summary)
    print(text)
    print(format_debrief(debrief))
    aar_path = ROOT / "aar_demo.txt"
    aar_path.write_text(text + format_debrief(debrief))
    append_end_record(out_frames, summary, debrief)
    print(f"Frames → {out_frames}")
    print(f"AAR    → {aar_path}")

    if open_browser:
        # Serve package root so dashboard can fetch the new frames
        import http.server
        import socketserver

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **k):
                super().__init__(*a, directory=str(ROOT), **k)

            def log_message(self, *args):
                pass

        port = 8765
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/dashboard.html#t=0"
        print(f"Dashboard → {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        print("Static server running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")


def run_playback() -> None:
    import http.server
    import socketserver

    print(TALK_TRACK)
    print("Playback mode: use existing map_frames_baseline.jsonl")
    print("  Play / scrub in the footer. Map focus with M.")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(ROOT), **k)

        def log_message(self, *args):
            pass

    port = 8765
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/dashboard.html"
    print(f"Dashboard → {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def run_live(port: int, open_browser: bool) -> None:
    # Configure stream_server globals then start its main
    import stream_server as ss

    ss._run_cfg.update({
        "scenario": DEMO["scenario"],
        "n_units": DEMO["n_units"],
        "seed": DEMO["seed"],
        "policy": DEMO["policy"],
        "ticks": DEMO["ticks"],
        "decision_timeout": DEMO["decision_timeout"],
        "map_log": str(ROOT / "map_frames_baseline.jsonl"),
    })

    print(TALK_TRACK)
    print(f"Live stream server → http://127.0.0.1:{port}/")
    print("  Click  Live stream  in the footer — the mission starts then (not before).")
    print("  Click decision options when the right rail prompts you.")
    print()

    t = threading.Thread(
        target=ss._run_mission_loop, args=(False,), daemon=True
    )
    t.start()

    httpd = ss.ThreadingHTTPServer(("127.0.0.1", port), ss.Handler)
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(f"Serving {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="JOR 4.0 synthetic demo launcher")
    ap.add_argument("--offline", action="store_true",
                    help="Headless run + AAR; optional static dashboard")
    ap.add_argument("--playback", action="store_true",
                    help="Static HTTP only (existing JSONL frames)")
    ap.add_argument("--port", type=int, default=8766,
                    help="Live stream port (default 8766)")
    ap.add_argument("--no-browser", action="store_true",
                    help="Do not open a browser window")
    ap.add_argument("--frames", type=str, default="map_frames_baseline.jsonl",
                    help="Offline frames output path")
    args = ap.parse_args(argv)

    open_browser = not args.no_browser
    if args.offline:
        run_offline(ROOT / args.frames, open_browser)
    elif args.playback:
        run_playback()
    else:
        run_live(args.port, open_browser)


if __name__ == "__main__":
    main()
