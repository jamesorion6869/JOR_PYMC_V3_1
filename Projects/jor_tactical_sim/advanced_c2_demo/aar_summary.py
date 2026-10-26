"""
After-action summary from map telemetry frames or a mission log list.

Used by run_demo.py (offline) and stream_server (live end event).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from maneuver_sim_interactive import (
    COLLAPSE_SUSTAIN_TICKS, enemy_collapse_flags, friendly_collapse_flags,
    sustained_collapse,
)

COLLAPSE_LABELS = ("FRIENDLY_COLLAPSE", "MUTUAL_COLLAPSE", "ENEMY_COLLAPSE")
FRIENDLY_CONDITIONS = ("readiness", "pressure", "outcomes", "morale")


def _tick_frames(frames):
    """One resolved frame per tick: drop pending-decision frames and repeats."""
    seen, out = set(), []
    for f in frames:
        if (f.get("decision") or {}).get("pending"):
            continue
        t = f.get("t")
        if t is None or t < 0 or t in seen:
            continue
        seen.add(t)
        out.append(f)
    return out


def sustained_outcome(frames):
    """
    Collapse by the same rule as the terminal mission summary: every joint
    condition held for COLLAPSE_SUSTAIN_TICKS consecutive ticks (a one-tick
    breach that recovers is not a collapse). Also reports which friendly
    conditions came closest, so scenarios that break differently read differently.
    """
    fr = _tick_frames(frames)
    ticks = [f["t"] for f in fr]
    ff = [friendly_collapse_flags(f.get("theta_s", 1), f.get("theta_c", 0),
                                  f.get("theta_o", 0), f.get("morale", 1)) for f in fr]
    ef = []
    for f in fr:
        e = f.get("enemy") or {}
        ef.append(enemy_collapse_flags(e.get("readiness", 1), e.get("pressure", 0), e.get("morale", 1))
                  if e else (False, False, False))
    fr_run = sustained_collapse(ticks, [all(x) for x in ff])
    en_run = sustained_collapse(ticks, [all(x) for x in ef])

    # worst tick: most friendly conditions met at once (latest such tick)
    best_n, best_flags, best_t = 0, (False,) * 4, None
    for t, fl in zip(ticks, ff):
        n = sum(fl)
        if n >= best_n and n > 0:
            best_n, best_flags, best_t = n, fl, t
    met = [name for name, on in zip(FRIENDLY_CONDITIONS, best_flags) if on]

    if fr_run["collapsed"] and en_run["collapsed"]:
        label = "MUTUAL_COLLAPSE"
    elif fr_run["collapsed"]:
        label = "FRIENDLY_COLLAPSE"
    elif en_run["collapsed"]:
        label = "ENEMY_COLLAPSE"
    else:
        label = None

    if fr_run["collapsed"]:
        detail = (f"Collapse at t={fr_run['onset_t']}: all 4 conditions held "
                  f"{fr_run['held']} ticks (≥{COLLAPSE_SUSTAIN_TICKS} needed)")
    elif best_n == 4:
        detail = (f"Brief breach at t={fr_run['longest_t']}: all 4 conditions for "
                  f"{fr_run['longest']} tick(s); {COLLAPSE_SUSTAIN_TICKS} needed")
    elif best_n == 3:
        detail = f"Near-collapse: {', '.join(met)} aligned at t={best_t}"
    elif best_n > 0:
        detail = f"Contained: {', '.join(met)} at t={best_t}"
    else:
        detail = "Contained: collapse conditions never overlapped"
    return {
        "label": label, "detail": detail,
        "friendly_collapsed": fr_run["collapsed"], "collapse_onset_t": fr_run["onset_t"],
        "enemy_collapsed": en_run["collapsed"],
        "closest_conditions": met, "closest_count": best_n,
        "breach_t": fr_run["longest_t"], "breach_len": fr_run["longest"],
    }


def summarize_frames(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not frames:
        return {
            "ticks": 0,
            "final_state": "—",
            "peak_nhp": 0.0,
            "alert_ticks": 0,
            "decisions": [],
            "degraded_units": [],
            "scenario": None,
        }

    peak_nhp = max(float(f.get("nhp") or 0) for f in frames)
    alert_ticks = sum(1 for f in frames if f.get("alert"))
    final = frames[-1]
    decisions = []
    for f in frames:
        d = f.get("decision")
        if not d or d.get("pending"):
            continue        # the pending frame precedes the resolved one for the same decision
        decisions.append({
            "t": f.get("t"),
            "type": d.get("type"),
            "title": d.get("title"),
            "selected": d.get("selected"),
            "category": d.get("category"),
        })

    degraded = set()
    for f in frames:
        for u in f.get("units") or []:
            if u.get("degraded"):
                degraded.add(u.get("name") or f"U{u.get('id')}")

    # Final state follows the terminal summary's sustained-collapse rule. The
    # per-tick mission_state label is instantaneous (it reads COLLAPSE on the very
    # last tick of any run the catastrophe hits), so it is only used for the
    # non-collapse part: the most recent non-STABLE, non-collapse state.
    outcome = sustained_outcome(frames)
    final_state = outcome["label"]
    if final_state is None:
        final_state = "STABLE"
        for f in reversed(_tick_frames(frames)):
            st = f.get("mission_state")
            if st and st != "STABLE" and st not in COLLAPSE_LABELS:
                final_state = st
                break

    nhp_series = [float(f.get("nhp") or 0) for f in _tick_frames(frames)]
    first_alert = next((f["t"] for f in _tick_frames(frames) if f.get("alert")), None)

    return {
        "ticks": len(frames),
        "scenario": final.get("scenario"),
        "final_state": final_state,
        "collapse_detail": outcome["detail"],
        "collapse_onset_t": outcome["collapse_onset_t"],
        "closest_conditions": outcome["closest_conditions"],
        "first_alert_t": first_alert,
        "mean_nhp": round(sum(nhp_series) / len(nhp_series), 4) if nhp_series else 0.0,
        "peak_nhp": round(peak_nhp, 4),
        "final_nhp": round(float(final.get("nhp") or 0), 4),
        "alert_ticks": alert_ticks,
        "decisions": decisions,
        "n_decisions": len(decisions),
        "degraded_units": sorted(degraded),
        "final_theta_s": round(float(final.get("theta_s") or 0), 4),
        "final_theta_c": round(float(final.get("theta_c") or 0), 4),
        "final_theta_o": round(float(final.get("theta_o") or 0), 4),
        "final_morale": round(float(final.get("morale") or 0), 4),
    }


def format_summary(summary: Dict[str, Any], title: str = "AFTER-ACTION SUMMARY") -> str:
    lines = [
        "",
        "=" * 52,
        f"  {title}",
        "=" * 52,
        f"  Scenario:      {summary.get('scenario') or '—'}",
        f"  Ticks:         {summary.get('ticks', 0)}",
        f"  Final state:   {summary.get('final_state')}",
        f"  Collapse:      {summary.get('collapse_detail')}",
        f"  First alert:   t={summary.get('first_alert_t')}   Mean NHP: {summary.get('mean_nhp')}",
        f"  Peak NHP:      {summary.get('peak_nhp')}",
        f"  Final NHP:     {summary.get('final_nhp')}",
        f"  Alert ticks:   {summary.get('alert_ticks')}",
        f"  θs / θc / θo:  {summary.get('final_theta_s')} / "
        f"{summary.get('final_theta_c')} / {summary.get('final_theta_o')}",
        f"  Morale:        {summary.get('final_morale')}",
        f"  Decisions:     {summary.get('n_decisions', 0)}",
    ]
    for d in summary.get("decisions") or []:
        lines.append(
            f"    t={d.get('t'):>3}  {d.get('type') or '':<16}  "
            f"choice={d.get('selected')}  {d.get('title') or ''}"
        )
    if summary.get("degraded_units"):
        lines.append("  Degraded:      " + ", ".join(summary["degraded_units"]))
    lines.append("=" * 52)
    lines.append("")
    return "\n".join(lines)
