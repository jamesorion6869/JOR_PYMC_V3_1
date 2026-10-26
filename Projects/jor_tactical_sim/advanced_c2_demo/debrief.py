"""
Cause-and-effect debrief for a finished mission.

Works on any list of per-tick records: the sim's slim feed (logger.feed), the
map frames, or frames loaded from a saved file. It never re-simulates; every
statement is something that was observed in this run:

  * IMMEDIATE effect of a decision = the change in readiness / pressure /
    outcomes between the pre-choice (pending) record and the resolved record on
    the same tick. This is the direct effect of the choice.
  * FOLLOW-ON change = severity before vs the mean over the next WINDOW ticks.
    That is an observation over time, not a controlled comparison (the scenario
    keeps moving), so it is worded as "followed by", never "caused".

Used by the terminal (format_debrief), run_demo / stream_server (end event) and
the dashboard (JSON via the end record).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aar_summary import _tick_frames, sustained_outcome
from maneuver_sim_interactive import COLLAPSE_SUSTAIN_TICKS
from severity import severity_band

WINDOW = 5                 # ticks after a decision used for the follow-on reading
CHANGE = 0.03              # severity change that counts as a real move
MIN_AFTER = 2              # fewest follow-on ticks needed to say anything

# Menu option 3 is the "accept the strain / push on" answer in every menu except
# the catastrophe menu, where 1 = stabilize, 2 = abort, 3 = hold course.
CATEGORY_NAMES = {"FATIGUE": "fatigue", "PRESSURE": "coordination load",
                  "OUTCOMES": "unit friction", "CATASTROPHIC": "collapse"}


def _option_label(decision: Dict[str, Any]) -> str:
    sel = str(decision.get("selected") or "")
    for opt in decision.get("options") or []:
        if str(opt.get("id")) == sel:
            return opt.get("label") or sel
    return sel or "no answer"


def _posture(decision: Dict[str, Any]) -> str:
    sel = str(decision.get("selected") or "")
    if decision.get("timed_out"):
        return "timeout"
    if decision.get("type") == "CATASTROPHIC":
        return {"1": "act", "2": "abort", "3": "accept"}.get(sel, "accept")
    return "accept" if sel == "3" else ("act" if sel in ("1", "2") else "accept")


def _immediate(before: Optional[Dict[str, Any]], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Direct effect of the choice: pre-choice record vs the resolved record, same tick."""
    if not before:
        return []
    out = []
    spec = (("theta_s", "readiness", +1), ("theta_c", "pressure", -1), ("theta_o", "outcomes", -1))
    for key, name, good_sign in spec:
        if key not in before or key not in after:
            continue
        d = float(after[key]) - float(before[key])
        if abs(d) >= 0.005:
            out.append({"metric": name, "delta": round(d, 3), "good": (d * good_sign) > 0})
    return out


def _fmt_immediate(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "no immediate change"
    return ", ".join(f"{i['metric']} {i['delta']:+.2f}" for i in items)


def build_debrief(frames: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    fr = _tick_frames(frames)
    if not fr:
        return None
    by_t = {f["t"]: f for f in fr}
    pending = {}
    for f in frames:                                   # pre-choice record per decision tick
        if (f.get("decision") or {}).get("pending") and f.get("t") is not None:
            pending[f["t"]] = f
    has_sev = any("severity" in f for f in fr)
    key = "severity" if has_sev else "nhp"
    metric_name = "severity" if has_sev else "NHP"

    def val(f):
        return float(f.get(key) or 0.0)

    outcome = sustained_outcome(frames)
    final = fr[-1]
    peak_f = max(fr, key=val)

    # ---------------- decisions ----------------
    decisions = []
    for f in fr:
        d = f.get("decision")
        if not d:
            continue
        t = f["t"]
        before = pending.get(t) or by_t.get(t - 1)
        sev_before = val(before) if before else val(f)
        window = [by_t[x] for x in range(t + 1, t + 1 + WINDOW) if x in by_t]
        sev_after = (sum(val(w) for w in window) / len(window)) if len(window) >= MIN_AFTER else None
        delta = None if sev_after is None else sev_after - sev_before
        trend = ("too late to measure" if delta is None else
                 "improved" if delta <= -CHANGE else
                 "worsened" if delta >= CHANGE else "little change")
        imm = _immediate(pending.get(t), f)
        decisions.append({
            "t": t, "type": d.get("type"), "category": d.get("category"),
            "title": str(d.get("title") or "").replace(" Choose:", ""),
            "selected": d.get("selected"), "option": _option_label(d),
            "posture": _posture(d), "timed_out": bool(d.get("timed_out")),
            "immediate": imm,
            "before": round(sev_before, 3),
            "after": None if sev_after is None else round(sev_after, 3),
            "delta": None if delta is None else round(delta, 3),
            "trend": trend,
        })

    # ---------------- group repeated decisions ----------------
    cat_t = next((f["t"] for f in fr for ev in (f.get("events") or []) if ev.get("kind") == "catastrophe"), None)
    strain_t = next((f["t"] for f in fr for ev in (f.get("events") or []) if ev.get("kind") == "strain"), None)
    breaks = [x for x in (cat_t, strain_t) if x is not None]

    def in_shock(t):
        return cat_t is not None and cat_t <= t < cat_t + WINDOW

    for d in decisions:
        d["during_shock"] = in_shock(d["t"])

    groups: List[Dict[str, Any]] = []
    for d in decisions:
        g = groups[-1] if groups else None
        same = (g and g["type"] == d["type"] and g["selected"] == d["selected"]
                and g["posture"] == d["posture"] and g["during_shock"] == d["during_shock"]
                and not any(g["t_end"] < b <= d["t"] for b in breaks))
        if same:
            g["members"].append(d)
            g["t_end"] = d["t"]
        else:
            groups.append({**{k: d[k] for k in ("type", "category", "title", "selected", "option",
                                                 "posture", "during_shock")},
                           "t": d["t"], "t_end": d["t"], "members": [d]})
    for g in groups:
        m = g["members"]
        g["n"] = len(m)
        g["timed_out"] = any(x["timed_out"] for x in m)
        sums: Dict[str, List[float]] = {}
        good: Dict[str, bool] = {}
        for x in m:
            for i in x["immediate"]:
                sums.setdefault(i["metric"], []).append(i["delta"])
                good[i["metric"]] = i["good"]
        g["immediate"] = [{"metric": k, "delta": round(sum(v) / len(v), 3), "good": good[k]}
                          for k, v in sums.items()]
        g["before"] = m[0]["before"]
        last_meas = next((x for x in reversed(m) if x["after"] is not None), None)
        g["after"] = last_meas["after"] if last_meas else None
        if g["after"] is None:
            g["delta"], g["trend"] = None, "too late to measure"
        else:
            g["delta"] = round(g["after"] - g["before"], 3)
            g["trend"] = ("improved" if g["delta"] <= -CHANGE else
                          "worsened" if g["delta"] >= CHANGE else "little change")
        g["measured"] = g["delta"] is not None and not g["during_shock"]

    # ---------------- key moments ----------------
    timeline: List[Dict[str, Any]] = []

    def add(t, kind, title, detail="", tone="info", order=5):
        timeline.append({"t": t, "kind": kind, "title": title, "detail": detail,
                         "tone": tone, "order": order})

    for f in fr:
        for ev in f.get("events") or []:
            if ev.get("kind") == "catastrophe":
                add(f["t"], "event", ev.get("text") or "Catastrophic event", tone="bad", order=1)
            elif ev.get("kind") == "strain":
                add(f["t"], "event", ev.get("text") or "Local strain", tone="warn", order=1)

    alert_on = next((f["t"] for f in fr if f.get("alert")), None)
    if alert_on is not None:
        add(alert_on, "alert", "NHP alert armed", f"NHP {fr[[x['t'] for x in fr].index(alert_on)]['nhp']:.2f}",
            tone="warn", order=2)

    if has_sev:
        seen_band = set()
        for f in fr:
            b = f.get("severity_band") or severity_band(val(f))
            if b in ("STRAINED", "CRITICAL", "BREAKING") and b not in seen_band:
                seen_band.add(b)
                add(f["t"], "band", f"Severity reached {b}", f"severity {val(f):.2f}",
                    tone="bad" if b != "STRAINED" else "warn", order=3)

    for g in groups:
        post = g["posture"]
        if post == "act":
            tone = "good" if any(i["good"] for i in g["immediate"]) else "info"
        elif post == "abort":
            tone = "warn"
        else:   # accepted the risk or timed out
            tone = "bad" if (g["timed_out"] or g["during_shock"]
                             or (g["measured"] and g["trend"] == "worsened")) else "warn"
        label = f"[{g['selected']}] {g['option']}" if g["selected"] else g["option"]
        each = " each time" if g["n"] > 1 and g["immediate"] else ""
        pieces = [f"immediate{each}: {_fmt_immediate(g['immediate'])}"]
        if g["during_shock"]:
            pieces.append("during the shock: later changes are not attributable to this choice")
        elif g["delta"] is not None:
            span = "across the stretch" if g["n"] > 1 else f"over the next {WINDOW} ticks"
            verdict = f" ({g['trend']})" if g["n"] == 1 else ""
            pieces.append(f"{metric_name} {g['before']:.2f} → {g['after']:.2f} {span}{verdict}")
        else:
            pieces.append(f"{metric_name} follow-on: {g['trend']}")
        when = (f"t={g['t']}–{g['t_end']} ×{g['n']}" if g["n"] > 1 else "")
        title = (("TIMED OUT on " if g["timed_out"] else "Decision: ") + (g["title"] or g["type"] or "")
                 + f" → {label}" + (f"  ({when.split(' ', 1)[1]})" if when else ""))
        add(g["t"], "decision", title, "; ".join(pieces), tone=tone, order=4)
        timeline[-1]["t_end"] = g["t_end"]
        timeline[-1]["n"] = g["n"]

    if outcome["friendly_collapsed"]:
        add(outcome["collapse_onset_t"], "collapse", "Collapse conditions aligned",
            outcome["detail"], tone="bad", order=6)
    elif outcome["closest_count"] == 4:
        add(outcome["breach_t"], "breach", "Brief breach of all collapse conditions",
            f"{outcome['breach_len']} tick(s); {COLLAPSE_SUSTAIN_TICKS} needed", tone="warn", order=6)

    timeline.sort(key=lambda e: (e["t"], e["order"]))

    # ---------------- entering the shock ----------------
    entering = None
    if cat_t is not None and (cat_t - 1) in by_t:
        e = by_t[cat_t - 1]
        entering = {"t": cat_t - 1, "readiness": round(float(e["theta_s"]), 3),
                    "pressure": round(float(e["theta_c"]), 3), "morale": round(float(e["morale"]), 3),
                    "severity": round(val(e), 3)}

    # ---------------- verdict ----------------
    if outcome["friendly_collapsed"]:
        outcome_kind = "collapse"
        headline = f"FORCE BROKE at t={outcome['collapse_onset_t']}"
    elif outcome["closest_count"] == 4:
        outcome_kind = "close"
        headline = f"HELD — by a hair (all collapse conditions aligned at t={outcome['breach_t']} for {outcome['breach_len']} tick(s); {COLLAPSE_SUSTAIN_TICKS} needed)"
    else:
        outcome_kind = "held"
        headline = "FORCE HELD"
    final_state = None
    # reuse the AAR's final-state rule (sustained collapse, else last non-collapse state)
    from aar_summary import summarize_frames
    final_state = summarize_frames(frames).get("final_state")

    # ---------------- lessons (each tied to observed evidence) ----------------
    lessons: List[Dict[str, str]] = []
    timeouts = [d for d in decisions if d["timed_out"]]
    if timeouts:
        ts = ", ".join(f"t={d['t']}" for d in timeouts[:4])
        lessons.append({"title": "Decide inside the window",
                        "evidence": f"{len(timeouts)} decision(s) timed out ({ts}). A forfeited decision "
                                    "defaults to 'accept the risk', so hesitation was treated as choosing to push on."})

    # the earliest accepted prompt that was followed by a real worsening (before the shock)
    pre = [g for g in groups if g["type"] != "CATASTROPHIC" and not g["during_shock"]
           and (cat_t is None or g["t_end"] < cat_t)]
    costly = next((g for g in pre if g["posture"] in ("accept", "timeout") and g["measured"]
                   and g["delta"] >= CHANGE), None)
    if costly:
        n_acc = sum(g["n"] for g in pre if g["posture"] in ("accept", "timeout"))
        n_all = sum(g["n"] for g in pre)
        lessons.append({"title": "Accepting early strain was costly",
                        "evidence": f"The first one was t={costly['t']}: '{costly['option']}' on a "
                                    f"{CATEGORY_NAMES.get(costly['category'], 'strain')} prompt, and {metric_name} then went "
                                    f"{costly['before']:.2f} → {costly['after']:.2f}. You accepted the risk on {n_acc} of "
                                    f"{n_all} prompts before the shock. A light action at the first sign is "
                                    "cheaper than a heavy one later."})

    def good_gain(g):
        return sum(abs(i["delta"]) for i in g["immediate"] if i["good"])
    best = max((g for g in pre if g["posture"] == "act"), key=lambda g: (good_gain(g), g["n"]), default=None)
    if best and good_gain(best) > 0:
        times = f"{best['n']} times, t={best['t']}–{best['t_end']}" if best["n"] > 1 else f"at t={best['t']}"
        lessons.append({"title": "A call that paid off",
                        "evidence": f"'{best['option']}' ({times}): "
                                    f"{_fmt_immediate([i for i in best['immediate'] if i['good']])}"
                                    + (" each time." if best["n"] > 1 else ".")})

    if entering is not None:
        before_shock = [d for d in decisions if d["t"] < cat_t]
        n_pass = len([d for d in before_shock if d["posture"] != "act"])
        n_act = len(before_shock) - n_pass
        if outcome_kind == "collapse":
            lessons.append({"title": "The state you carry into the shock decides it",
                            "evidence": f"You entered the catastrophe (t={cat_t}) with readiness {entering['readiness']:.2f}, "
                                        f"morale {entering['morale']:.2f}, severity {entering['severity']:.2f}; "
                                        f"{n_pass} of your {len(before_shock)} earlier decisions were passive. "
                                        "The shock removes a large share of whatever readiness you have, so readiness "
                                        "banked beforehand is what survives it."})
        elif outcome_kind == "close":
            lessons.append({"title": "Close, and preparation made the difference",
                            "evidence": f"You entered the catastrophe (t={cat_t}) with readiness {entering['readiness']:.2f}, "
                                        f"morale {entering['morale']:.2f}, severity {entering['severity']:.2f} after "
                                        f"{n_act} active decisions. {outcome['detail']}."})
        else:
            lessons.append({"title": "Preparation carried you through",
                            "evidence": f"You entered the catastrophe (t={cat_t}) with readiness {entering['readiness']:.2f}, "
                                        f"morale {entering['morale']:.2f}, severity {entering['severity']:.2f}"
                                        + (f" after {n_act} active decisions." if n_act else ".")})
    cat_dec = next((d for d in decisions if d["type"] == "CATASTROPHIC"), None)
    if cat_dec:
        if cat_dec["posture"] == "act":
            lessons.append({"title": "Emergency response",
                            "evidence": f"Your emergency call at t={cat_dec['t']} was '{cat_dec['option']}' — "
                                        f"immediate {_fmt_immediate(cat_dec['immediate'])}."})
        elif cat_dec["posture"] == "abort":
            lessons.append({"title": "You aborted the maneuver",
                            "evidence": f"At t={cat_dec['t']} you ended the maneuver rather than ride out the collapse."})
        else:
            lessons.append({"title": "You held course through the collapse",
                            "evidence": f"At t={cat_dec['t']} you chose '{cat_dec['option']}'. "
                                        "Stabilizing or aborting are the other options on that menu."})
    if not decisions:
        lessons.append({"title": "No decisions were prompted",
                        "evidence": "The decision engine never asked for input in this run."})
    lessons = lessons[:5]

    # strain → first response lead time
    lead = None
    if strain_t is not None:
        first_resp = next((d for d in decisions if d["t"] >= strain_t and d["posture"] == "act"), None)
        lead = {"strain_t": strain_t, "first_active_t": first_resp["t"] if first_resp else None}

    return {
        "headline": headline, "outcome": outcome_kind, "final_state": final_state,
        "collapse_detail": outcome["detail"],
        "metric": metric_name,
        "severity": {"peak": round(val(peak_f), 3), "peak_t": peak_f["t"],
                     "peak_band": severity_band(val(peak_f)) if has_sev else None,
                     "final": round(val(final), 3),
                     "mean": round(sum(val(f) for f in fr) / len(fr), 3)},
        "entering_shock": entering,
        "strain_response": lead,
        "decisions": decisions,
        "decision_groups": [{k: v for k, v in g.items() if k != "members"} for g in groups],
        "timeline": [{k: v for k, v in e.items() if k != "order"} for e in timeline],
        "lessons": lessons,
        "series": {"t": [f["t"] for f in fr], "severity": [round(val(f), 3) for f in fr],
                   "nhp": [round(float(f.get("nhp") or 0), 3) for f in fr]},
    }


def format_debrief(d: Optional[Dict[str, Any]], title: str = "DEBRIEF — WHAT HAPPENED AND WHY") -> str:
    if not d:
        return "\n(No debrief: the run produced no per-tick record.)\n"
    lines = ["", "=" * 66, f"  {title}", "=" * 66,
             f"  {d['headline']}   [{d.get('final_state') or '—'}]",
             f"  {d['collapse_detail']}",
             f"  {d['metric'].capitalize()}: mean {d['severity']['mean']:.2f}, peak {d['severity']['peak']:.2f} "
             f"at t={d['severity']['peak_t']}" + (f" ({d['severity']['peak_band']})" if d['severity'].get('peak_band') else ""),
             "", "  TIMELINE"]
    mark = {"good": "+", "warn": "!", "bad": "x", "info": "·"}
    for e in d["timeline"]:
        lines.append(f"   t={e['t']:>3} {mark.get(e['tone'], '·')} {e['title']}")
        if e["detail"]:
            lines.append(f"            {e['detail']}")
    lines += ["", "  WHAT TO TAKE AWAY"]
    for i, l in enumerate(d["lessons"], 1):
        lines.append(f"   {i}. {l['title']}")
        lines.append(f"      {l['evidence']}")
    lines += ["", "  Notes: 'immediate' is the direct effect of the choice on that tick. The "
              f"{d['metric']} change over the next {WINDOW} ticks is what followed, not a controlled",
              "  comparison — the scenario keeps moving. Synthetic training scenario, not a prediction.",
              "=" * 66, ""]
    return "\n".join(lines)
