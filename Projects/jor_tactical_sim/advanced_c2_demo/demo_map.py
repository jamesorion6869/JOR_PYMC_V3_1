"""
Headless demo: run a seeded mission, write map telemetry, render key frames + GIF.

    python3 demo_map.py
    python3 demo_map.py --scenario high_enemy_pressure --n-units 9 --animate
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from headless_runner import make_input_fn
from maneuver_sim_interactive import run_simulation
from map_view import render_animation, render_frame


def main():
    parser = argparse.ArgumentParser(description="JOR map telemetry demo")
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--n-units", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--policy", default="intervene",
                        choices=["intervene", "ignore", "alternate"])
    parser.add_argument("--out-dir", default="map_out")
    parser.add_argument("--animate", action="store_true",
                        help="also write a mission GIF")
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / f"map_frames_{args.scenario}.jsonl"
    log_path = out_dir / f"maneuver_log_{args.scenario}.json"

    print(f"Running {args.scenario}  n={args.n_units}  seed={args.seed}  policy={args.policy}")
    with contextlib.redirect_stdout(io.StringIO()):
        logger, commander, skill, scenario, map_sink = run_simulation(
            scenario_name=args.scenario,
            commander_level="expert",
            feedback=1.0,
            seed=args.seed,
            input_fn=make_input_fn(args.policy),
            delay=0.0,
            log_file=str(log_path),
            n_ticks=args.ticks,
            n_units=args.n_units,
            map_log=str(map_path),
            engine_profile="graded",
            prompt_guards="fixed",
        )

    frames = map_sink.frames
    print(f"Frames: {len(frames)}  → {map_path}")

    targets = sorted({0, 30, 60, 110, frames[-1]["t"]})
    for t in targets:
        frame = next((f for f in frames if f["t"] == t), None)
        if frame is None:
            continue
        out = out_dir / f"map_{args.scenario}_t{t:03d}.png"
        render_frame(frame, str(out))

    render_frame(frames[-1], str(out_dir / f"map_{args.scenario}_final.png"))

    if args.animate:
        gif = out_dir / f"map_{args.scenario}_mission.gif"
        render_animation(frames, str(gif), stride=args.stride, fps=6)

    print("Done.")


if __name__ == "__main__":
    main()
