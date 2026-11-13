#!/usr/bin/env python3
"""
viz.py - optional visualisation for the JOR C-UAS sensing-assurance demo.

Reads the JSONL log written by cuas_demo.py and renders:
  - a static timeline PNG   (README / report figure)
  - an animated replay      (.mp4 via ffmpeg, or .gif fallback)

Standalone: the core demo has no dependency on this file. matplotlib is
needed only here (ffmpeg only for mp4).

    python viz.py cuas_airspace_log.jsonl --png timeline.png --video replay.mp4

The log is appended to on every demo run. By default only the most recent
run is drawn; use --run ID to pick another or --all to draw every record.

Every figure is stamped SYNTHETIC DATA.
"""
import argparse
import json
import shutil

STATE_COLORS = {"CALIBRATING": "#9A9A9A", "NOMINAL": "#009E73",
                "DEGRADED": "#E69F00", "UNRELIABLE": "#D55E00"}
CHANNEL_COLORS = {"sensor_integrity": "#0072B2", "sensing_load": "#CC79A7",
                  "track_quality": "#56B4E9", None: "#E6E6E6"}
CHANNEL_LABELS = {"sensor_integrity": "sensor integrity", "sensing_load": "sensing load",
                  "track_quality": "track quality", None: "none"}
THETA_STYLE = (("th_s", "θ_s sensor integrity", "#0072B2", "-"),
               ("th_c", "θ_c sensing load", "#CC79A7", "--"),
               ("th_o", "θ_o track quality", "#56B4E9", "-"))
FOOTER = "SYNTHETIC DATA  ·  demonstration of a sensing-assurance meta-layer  ·  not validated on real sensor feeds"


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise SystemExit("viz.py needs matplotlib:  pip install matplotlib")


# ----------------------------------------------------------------------
# Log parsing
# ----------------------------------------------------------------------
def load_log(path, run_id="latest"):
    """run_id: "latest" (default), "all", or a specific run id string."""
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Log file not found: {path}. Run cuas_demo.py first.")
    recs, order = [], []
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("metadata", {}).get("status"):
                continue                      # older log format without status reports
            rid = rec.get("run_id")
            if rid not in order:
                order.append(rid)
            recs.append((rid, rec))
    if run_id == "latest":
        keep = order[-1] if order else None
        if len(order) > 1:
            print(f"Log holds {len(order)} runs; drawing the latest ({keep}). "
                  f"Use --run ID or --all to change.")
        recs = [r for r in recs if r[0] == keep]
    elif run_id != "all":
        recs = [r for r in recs if r[0] == run_id]
        if not recs:
            raise SystemExit(f"Run '{run_id}' not found. Runs in log: {[str(o) for o in order]}")

    rows = []
    for _, rec in recs:
        meta = rec["metadata"]
        st = meta["status"]
        rows.append({
            "t": len(rows), "phase": meta.get("phase", ""),
            "nhp": st["nhp"], "state": st["state"], "sop": st["sop"],
            "baseline": st.get("baseline_sop"),
            "th_s": meta.get("theta_s"), "th_c": meta.get("theta_c"), "th_o": meta.get("theta_o"),
            "driver": st.get("dominant_channel"), "details": st.get("details", {}),
            "nhp_range": st.get("nhp_range", [0.015, 0.715]),
        })
    if not rows:
        raise SystemExit(f"No status records in {path}. Re-run cuas_demo.py to regenerate the log.")
    return rows


def _runs(seq):
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            yield start, i - 1, seq[start]
            start = i


def _phases(rows):
    return [(a, b, rows[a]["phase"]) for a, b, _ in _runs([r["phase"] for r in rows])]


def _events(rows):
    """First time a source is named (degraded / stale) or quiet sky is reported.
    Each distinct message is shown once, at its earliest occurrence."""
    ev, seen = [], set()
    for r in rows:
        d = r["details"]
        msgs = []
        if d.get("degraded_sources"):
            msgs.append("degraded: " + ", ".join(d["degraded_sources"]))
        if d.get("stale_sources"):
            msgs.append("stale feeds: " + ", ".join(d["stale_sources"]))
        if d.get("no_targets"):
            msgs.append("no targets, feeds alive")
        for m in msgs:
            if m not in seen:
                seen.add(m)
                ev.append((r["t"], m))
    return ev


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def _draw(axes, rows, upto):
    from matplotlib.ticker import FuncFormatter
    ax_n, ax_t, ax_s = axes
    n = len(rows)
    vis = rows[:upto + 1]
    for ax in axes:
        ax.clear()

    # --- NHP panel
    for a, b, s in _runs([r["state"] for r in vis]):
        ax_n.axvspan(a - 0.5, b + 0.5, color=STATE_COLORS[s], alpha=0.13, lw=0)
    ax_n.plot([r["t"] for r in vis], [r["nhp"] for r in vis], color="black", lw=1.9)
    lo, hi = rows[0]["nhp_range"]
    for y in (hi, lo):
        ax_n.axhline(y, ls=":", color="#555", lw=0.8)
    ax_n.text(n - 0.7, hi + 0.012, f"reachable max {hi:.1%}", ha="right", va="bottom", fontsize=7, color="#444")
    ax_n.set_ylim(0, 0.92)
    ax_n.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_n.set_ylabel("NHP (non-healthy posterior)\nuncalibrated index", fontsize=8)

    # --- annotations for named sources / quiet sky
    for k, (t, text) in enumerate(_events(rows)):
        if t > upto:
            continue
        y_text = 0.84 if k % 2 == 0 else 0.76
        ax_n.annotate(text, xy=(t, rows[t]["nhp"]), xytext=(t, y_text), ha="center", fontsize=7,
                      arrowprops=dict(arrowstyle="-", color="#333", lw=0.7),
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", lw=0.6))

    # --- theta panel
    for key, label, color, ls in THETA_STYLE:
        ax_t.plot([r["t"] for r in vis], [r[key] for r in vis], color=color, ls=ls, lw=1.5, label=label)
    ax_t.set_ylim(0, 1.0)
    ax_t.set_ylabel("evidence\nchannels", fontsize=8)
    ax_t.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)

    # --- state / driver strip
    for y, seq, colors in ((1.05, [r["state"] for r in vis], STATE_COLORS),
                           (0.0, [r["driver"] if r["state"] in ("DEGRADED", "UNRELIABLE") else None
                                  for r in vis], CHANNEL_COLORS)):
        for a, b, v in _runs(seq):
            ax_s.broken_barh([(a - 0.5, b - a + 1)], (y, 0.95), facecolors=colors[v], lw=0)
            if y > 0.5 and b - a + 1 >= 7:
                ax_s.text((a + b) / 2, y + 0.47, v, ha="center", va="center", fontsize=6, color="white",
                          fontweight="bold")
    ax_s.set_ylim(0, 2.05)
    ax_s.set_yticks([0.47, 1.52])
    ax_s.set_yticklabels(["driver", "state"], fontsize=8)

    # --- shared x: phase boundaries and labels
    phases = _phases(rows)
    for ax in axes:
        ax.set_xlim(-0.5, n - 0.5)
        for a, _, _ in phases[1:]:
            ax.axvline(a - 0.5, color="#999", ls="--", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in (ax_n, ax_t):
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_s.set_xticks([(a + b) / 2 for a, b, _ in phases])
    ax_s.set_xticklabels([p for _, _, p in phases], rotation=22, ha="right", fontsize=7)


def _legend(fig, loc_y):
    from matplotlib.patches import Patch
    states = [Patch(color=c, label=s.title()) for s, c in STATE_COLORS.items()]
    chans = [Patch(color=CHANNEL_COLORS[k], label=CHANNEL_LABELS[k]) for k in
             ("sensor_integrity", "sensing_load", "track_quality")]
    fig.legend(handles=states + chans, loc="lower center", ncol=7, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, loc_y))


def _stamp(fig):
    fig.text(0.5, 0.008, FOOTER, ha="center", fontsize=7, color="#444")


def _title(fig, y=0.965):
    fig.text(0.07, y, "JOR C-UAS Sensing-Assurance Monitor", fontsize=13, fontweight="bold", ha="left")
    fig.text(0.07, y - 0.035, "Does not detect or defeat drones. It reports when the ability to detect "
             "and track is becoming unreliable.", fontsize=8.5, color="#333", ha="left")


# ----------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------
def render_png(rows, path, dpi=150):
    plt = _mpl()
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.8, 1.2], hspace=0.1,
                          left=0.08, right=0.98, top=0.875, bottom=0.17)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    _draw(axes, rows, len(rows) - 1)
    _title(fig)
    _legend(fig, 0.045)
    _stamp(fig)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _ffmpeg_path():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _readout(ax, r, n, total):
    ax.clear()
    ax.axis("off")
    d = r["details"]
    ax.text(0.0, 0.78, r["state"], fontsize=24, fontweight="bold", color=STATE_COLORS[r["state"]],
            va="center", transform=ax.transAxes)
    base = f"baseline {r['baseline']:.3f}" if r["baseline"] is not None else "baseline calibrating"
    ax.text(0.30, 0.82, f"NHP {r['nhp']:.1%}     SOP {r['sop']:.3f}  ({base})",
            fontsize=11, va="center", transform=ax.transAxes)
    parts = []
    if d.get("degraded_sources"):
        parts.append("degraded " + ", ".join(d["degraded_sources"]))
    if d.get("stale_sources"):
        parts.append("stale " + ", ".join(d["stale_sources"]))
    if d.get("malformed_sources"):
        parts.append("malformed " + ", ".join(d["malformed_sources"]))
    mode = "NO DATA" if d.get("data_missing") else ("no targets, feeds alive" if d.get("no_targets") else "tracking")
    if d.get("per_source") is False:
        src_txt = "n/a (aggregate input)"
    else:
        src_txt = "; ".join(parts) if parts else "none flagged"
    ax.text(0.30, 0.45, f"driver: {CHANNEL_LABELS[r['driver']]}     sources: {src_txt}",
            fontsize=9.5, va="center", transform=ax.transAxes)
    ax.text(0.30, 0.12, f"phase: {r['phase']}     mode: {mode}     step {r['t'] + 1}/{total}",
            fontsize=9, color="#444", va="center", transform=ax.transAxes)


def render_video(rows, path, fps=6, hold=14, dpi=110):
    plt = _mpl()
    import matplotlib
    import matplotlib.animation as animation
    plt_path = path
    ffmpeg = _ffmpeg_path()
    if path.lower().endswith(".mp4") and not ffmpeg:
        plt_path = path[:-4] + ".gif"
        print(f"ffmpeg not found - writing GIF instead: {plt_path}")
    if ffmpeg:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 3, 1.8, 1.2], hspace=0.12,
                          left=0.08, right=0.98, top=0.875, bottom=0.17)
    ax_r = fig.add_subplot(gs[0])
    axes = [fig.add_subplot(gs[i]) for i in (1, 2, 3)]
    _title(fig)
    _legend(fig, 0.045)
    _stamp(fig)

    frames = list(range(len(rows))) + [len(rows) - 1] * hold

    def update(k):
        i = frames[k]
        _draw(axes, rows, i)
        _readout(ax_r, rows[i], i, len(rows))

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    if plt_path.lower().endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2400,
                                        extra_args=["-pix_fmt", "yuv420p"])
    else:
        writer = animation.PillowWriter(fps=fps)
    ani.save(plt_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return plt_path


def render(log_path, png=None, video=None, fps=6, run_id="latest"):
    rows = load_log(log_path, run_id)
    out = []
    if png:
        out.append(render_png(rows, png))
    if video:
        out.append(render_video(rows, video, fps=fps))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", nargs="?", default="cuas_airspace_log.jsonl")
    p.add_argument("--png", default=None, help="static timeline (default cuas_timeline.png if no output given)")
    p.add_argument("--video", default=None, help="animated replay (.mp4, falls back to .gif without ffmpeg)")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--run", default=None, help="draw a specific run id (default: latest run in the log)")
    p.add_argument("--all", action="store_true", help="draw every record in the log")
    a = p.parse_args(argv)
    if not a.png and not a.video:
        a.png = "cuas_timeline.png"
    run_id = "all" if a.all else (a.run or "latest")
    for path in render(a.log, a.png, a.video, a.fps, run_id):
        print("wrote", path)


if __name__ == "__main__":
    main()
