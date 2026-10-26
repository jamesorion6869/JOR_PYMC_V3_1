"""
JOR live telemetry stream (SSE) + optional decision injection.

  python3 stream_server.py
  # dashboard connects to /stream and /decision

Endpoints:
  GET  /                  → dashboard.html
  GET  /stream            → text/event-stream of map frames
  POST /decision         → JSON {selected: "1"|"2"|"3"} for next prompt
  GET  /map_frames_*.jsonl → static files for offline playback
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from headless_runner import make_input_fn
from maneuver_sim_interactive import run_simulation
from aar_summary import summarize_frames, format_summary
from debrief import build_debrief, format_debrief

ROOT = Path(__file__).resolve().parent

# Shared decision mailbox: dashboard POSTs a choice; sim input_fn reads it.
_decision_q: queue.Queue = queue.Queue()
_stream_clients: list = []
_stream_lock = threading.Lock()
_mission_gate = threading.Event()  # set when first Live stream client connects

_run_cfg = {
    "scenario": "baseline",
    "n_units": 6,
    "seed": 42,
    "policy": "intervene",  # fallback if no live choice arrives
    "ticks": 120,
    "decision_timeout": 3.0,
    "map_log": str(ROOT / "map_frames_baseline.jsonl"),
}


# CLI-viewer mode: the terminal run owns the simulation and the decisions; the
# browser is a read-only live map. Frames are kept so a browser opened (or
# refreshed) mid-run catches up instead of starting blank.
_cli_mode = {"on": False}
_history: list = []


def _broadcast(frame: dict) -> None:
    line = "data: " + json.dumps(frame) + "\n\n"
    dead = []
    with _stream_lock:
        if _cli_mode["on"]:
            _history.append(line)
        for q in _stream_clients:
            try:
                q.put_nowait(line)
            except Exception:
                dead.append(q)
        for q in dead:
            _stream_clients.remove(q)


def _live_input_fn(prompt: str = "") -> str:
    """Prefer a choice posted by the dashboard; else fall back to policy."""
    text = (prompt or "").lower()
    if text.startswith("scenario"):
        return "5"
    try:
        # Wait up to decision_timeout for a live click
        timeout = float(_run_cfg.get("decision_timeout") or 8.0)
        sel = _decision_q.get(timeout=timeout)
        if sel in ("1", "2", "3"):
            return sel
    except queue.Empty:
        pass
    # Fallback policy
    return make_input_fn(_run_cfg.get("policy") or "intervene")(prompt)


def _run_mission_loop(repeat: bool) -> None:
    while True:
        # Do not start (or restart) until a browser clicks Live stream (/stream)
        print("Waiting for Live stream client…", flush=True)
        _mission_gate.clear()
        _mission_gate.wait()
        while not _decision_q.empty():
            try:
                _decision_q.get_nowait()
            except queue.Empty:
                break

        import contextlib, io
        from map_telemetry import MapTelemetrySink as _Sink

        collected_frames = []
        original_emit = _Sink.emit

        def emit_and_broadcast(self, frame):
            original_emit(self, frame)
            collected_frames.append(frame)
            _broadcast(frame)

        _Sink.emit = emit_and_broadcast
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run_simulation(
                    scenario_name=_run_cfg["scenario"],
                    commander_level="expert",
                    feedback=1.0,
                    seed=_run_cfg["seed"],
                    input_fn=_live_input_fn,
                    delay=0.08,
                    log_file=os.devnull,
                    n_ticks=int(_run_cfg["ticks"]),
                    n_units=int(_run_cfg["n_units"]),
                    map_log=_run_cfg.get("map_log") or str(ROOT / "map_frames_baseline.jsonl"),
                    engine_profile="graded",
                    prompt_guards="fixed",
                    decision_timeout=float(_run_cfg["decision_timeout"]),
                )
        finally:
            _Sink.emit = original_emit

        summary = summarize_frames(collected_frames) if collected_frames else None
        debrief = build_debrief(collected_frames) if collected_frames else None
        if summary:
            print(format_summary(summary), flush=True)
        if debrief:
            print(format_debrief(debrief), flush=True)
        _broadcast({"type": "end", "t": -1, "summary": summary, "debrief": debrief})
        if not repeat:
            break
        time.sleep(1.0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if _cli_mode["on"]:
            return          # keep the terminal clean for the simulation
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/dashboard.html"):
            data = (ROOT / "dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            q: queue.Queue = queue.Queue()
            with _stream_lock:
                for past in _history:       # replay (CLI viewer mode only)
                    q.put_nowait(past)
                _stream_clients.append(q)
            _mission_gate.set()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        line = q.get(timeout=15.0)
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _stream_lock:
                    if q in _stream_clients:
                        _stream_clients.remove(q)
            return

        if path == "/config":
            body = json.dumps(dict(_run_cfg, mode="cli" if _cli_mode["on"] else "server")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # static files from ROOT
        rel = path.lstrip("/")
        fp = (ROOT / rel).resolve()
        if str(fp).startswith(str(ROOT)) and fp.is_file():
            data = fp.read_bytes()
            ctype = "application/octet-stream"
            if fp.suffix == ".jsonl":
                ctype = "application/x-ndjson"
            elif fp.suffix == ".json":
                ctype = "application/json"
            elif fp.suffix == ".js":
                ctype = "application/javascript"
            elif fp.suffix == ".css":
                ctype = "text/css"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            payload = {}

        if parsed.path == "/decision":
            sel = str(payload.get("selected") or payload.get("choice") or "").strip()
            if _cli_mode["on"]:
                body = json.dumps({"ok": False, "error": "decisions are typed in the terminal"}).encode()
            elif sel in ("1", "2", "3"):
                _decision_q.put(sel)
                body = json.dumps({"ok": True, "selected": sel}).encode()
            else:
                body = json.dumps({"ok": False, "error": "selected must be 1|2|3"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/start":
            for k in ("scenario", "n_units", "seed", "policy", "ticks", "decision_timeout"):
                if k in payload:
                    _run_cfg[k] = payload[k]
            body = json.dumps({"ok": True, "config": _run_cfg}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)


class _Viewer:
    def broadcast(self, frame: dict) -> None:
        _broadcast(frame)

    def finish(self, frames, feed=None) -> None:
        summary = debrief = None
        try:
            summary = summarize_frames(frames) if frames else None
            # the feed exists even when no map frames were built (large forces)
            debrief = build_debrief(feed if feed else frames) if (feed or frames) else None
        except Exception:
            pass
        _broadcast({"type": "end", "t": -1, "summary": summary, "debrief": debrief})


def start_viewer(port: int = 8766, host: str = "127.0.0.1", open_browser: bool = True) -> _Viewer:
    """Serve the dashboard in the background for a terminal-driven run.

    The terminal keeps its normal output and typed decisions; the browser shows
    the same mission live (map, units, decision menu + countdown, read-only).
    """
    _cli_mode["on"] = True
    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://{host}:{port}/"
    print(f"  Live map:              {url}")
    print("                         Snap this terminal to one half of the screen and the")
    print("                         browser to the other (Win+Left/Right, or drag to an edge).")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    return _Viewer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--n-units", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", action="store_true")
    args = ap.parse_args()

    _run_cfg["scenario"] = args.scenario
    _run_cfg["n_units"] = args.n_units
    _run_cfg["seed"] = args.seed

    t = threading.Thread(target=_run_mission_loop, args=(args.repeat,), daemon=True)
    t.start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"JOR stream server http://{args.host}:{args.port}/")
    print("  SSE  /stream")
    print("  POST /decision  {\"selected\":\"1\"}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
