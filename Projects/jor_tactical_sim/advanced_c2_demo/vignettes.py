"""
Scripted training vignettes for JOR 4.0.

A free-play run ends with a debrief but no stated purpose. A vignette fixes one
objective, one scenario and one pass/fail rule, runs the normal simulation under
it, and grades the run against that objective.

  early_warning   Act on strain while it is still cheap.
  tempo           Push for the objective without borrowing more than the force can repay.
  shock           Survive a collapse event: stabilize, then follow through.

What a vignette changes (and what it does not)
  * The engine, scenarios and menus are the normal ones. A vignette only picks a
    scenario, adjusts the shock size and timing, and fixes the seed.
  * The base engine rebuilds readiness and pressure fresh every tick, so nothing
    done before a shock carries into it. That makes "early warning" meaningless in
    free play. Two vignettes therefore switch on an explicit, optional carry-over
    cost (run_simulation neglect_cost / push_cost; both off by default):
      early_warning : an ignored prompt leaves lasting strain
      tempo         : pushing on borrows readiness from the next ticks
    The brief of each says so. These are modeling choices made for teaching, not
    measurements.
  * Mission progress in the tempo vignette is a scoreboard kept here; the engine
    does not know about it.

Every vignette is repeatable (fixed seed by default; --vary picks a new one and
prints it) and comes with a reference line: one choice pattern known to pass.
The what-if replay re-runs the same seed with your choices up to the first point
where you differed from the reference line, then the reference line. Same dice,
different decisions.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import maneuver_sim_interactive as sim
from aar_summary import sustained_outcome
from debrief import build_debrief, format_debrief
from severity import severity_band

FRIENDLY_COLLAPSE = ("FRIENDLY_COLLAPSE", "MUTUAL_COLLAPSE")
ESCALATED = ("FATIGUE", "PRESSURE", "OUTCOMES")
CAT_KEYS = ("cat_readiness_scale", "cat_pressure_scale", "cat_morale_scale", "cat_outcome_scale")

Policy = Callable[[str, int, int], str]      # (decision_type, tick, decision_index) -> "1"|"2"|"3"


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

@dataclass
class Vignette:
    key: str
    title: str
    objective: str
    brief: List[str]
    scenario: str
    ticks: int
    cat_scale: float                      # multiplies the scenario's shock terms
    reference: Policy
    seed: int = 7
    shock_tick: Optional[int] = None      # None = the scenario's own
    push_cost: Optional[Dict[str, float]] = None
    neglect_cost: Optional[Dict[str, float]] = None
    tempo: Optional[Dict[str, float]] = None
    timer: Optional[float] = None         # default interactive decision window (s)
    params: Dict[str, Any] = field(default_factory=dict)

    def overrides(self) -> Dict[str, Any]:
        prof = sim.Scenario.EVENT_PROFILES[self.scenario]
        o = {k: prof[k] * self.cat_scale for k in CAT_KEYS}
        if self.shock_tick is not None:
            o["catastrophe_tick"] = self.shock_tick
        return o

    def shock_at(self) -> int:
        if self.shock_tick is not None:
            return self.shock_tick
        return sim.Scenario.EVENT_PROFILES[self.scenario]["catastrophe_tick"]

    def strain_at(self) -> int:
        return sim.Scenario.EVENT_PROFILES[self.scenario]["strain_tick"]


def _ref_early_warning(dtype: str, t: int, i: int) -> str:
    return "1"                                            # act on everything


TEMPO_PUSH_UNTIL = 65                                     # reference stops borrowing here: later debt reaches the shock


def _ref_tempo(dtype: str, t: int, i: int) -> str:
    if dtype in ("FATIGUE", "FATIGUE_EARLY", "PRESSURE", "PRESSURE_EARLY") and t < TEMPO_PUSH_UNTIL:
        return "3"                                        # borrow early, while there is time to repay
    return "1"


def _ref_shock(dtype: str, t: int, i: int) -> str:
    return "1"                                            # stabilize, then keep acting


VIGNETTES: Dict[str, Vignette] = {
    "early_warning": Vignette(
        key="early_warning",
        title="Early Warning",
        objective="Act on strain while it is still cheap, then hold through the shock.",
        brief=[
            "Dense terrain. Street fighting will wear on the lead elements from the first weeks;",
            "a heavy shock is coming late in the operation.",
            "Every time the staff flags strain (fatigue, coordination, unit friction) you can act",
            "or accept it. In this exercise strain you accept does not go away: it stays on the",
            "force and is still there when the shock arrives.",
            "Graded on: how many warnings you let pass, whether you answered the FIRST one,",
            "and whether the force held through the shock.",
        ],
        scenario="urban", ticks=120, cat_scale=0.8,
        neglect_cost={"readiness": 0.12, "pressure": 0.03, "steps": 100},
        reference=_ref_early_warning,
        params={"max_ignored": 2},
    ),
    "tempo": Vignette(
        key="tempo",
        title="Tempo vs Sustainment",
        objective="Reach the objective line by the deadline without breaking the force.",
        brief=[
            "Supply is thin. Command wants the objective line reached by the deadline.",
            "Slowing down is safe but costs ground; accepting higher tempo (option 3 on a",
            "fatigue or coordination prompt) buys ground now and borrows readiness from the",
            "next 12 ticks. Borrow too much, or too close to the shock, and the force breaks.",
            "Do nothing risky and you will arrive late.",
            "Graded on: objective reached by the deadline AND force held through the shock.",
        ],
        scenario="logistics_constrained", ticks=120, cat_scale=0.7,
        push_cost={"readiness": 0.10, "pressure": 0.03, "steps": 12},
        tempo={"deadline": 100, "target": 108.0, "push_mult": 1.6, "slow_mult": 0.5, "window": 6},
        reference=_ref_tempo,
    ),
    "shock": Vignette(
        key="shock",
        title="Surviving the Shock",
        objective="Hold the line through a coordinated enemy offensive.",
        brief=[
            "Intelligence expects a coordinated enemy offensive around tick 100.",
            "The decisive window is the shock and the ticks right after it: stabilizing is",
            "necessary but not enough, and the follow-on prompts keep coming.",
            "Graded on: stabilizing at the shock, answering every prompt in the 12 ticks after",
            "it with action, and the force still holding at the end.",
            "Decisions are timed (set --no-timer to turn that off).",
        ],
        scenario="high_enemy_pressure", ticks=112, cat_scale=0.9, shock_tick=100,
        timer=15.0, reference=_ref_shock,
        params={"follow_window": 12},
    ),
}


# ---------------------------------------------------------------------------
# Recording one run
# ---------------------------------------------------------------------------

class Recorder:
    """Read-only observer fed by the simulator's tick_hook."""

    def __init__(self, v: Vignette, announce: bool = False):
        self.v = v
        self.decisions: List[Dict[str, Any]] = []
        self.ticks: List[Dict[str, Any]] = []
        self.announce = announce
        self.progress = 0.0
        self.progress_by_t: Dict[int, float] = {}
        self._mult = 1.0
        self._until = -1
        self._last_t = -1

    def _tempo_effect(self, d: Dict[str, Any], t: int) -> None:
        cfg = self.v.tempo
        if not cfg:
            return
        dtype = d.get("type") or ""
        cat = dtype.replace("_EARLY", "")
        sel = "3" if d.get("timed_out") else str(d.get("selected") or "")
        if cat not in ("FATIGUE", "PRESSURE"):
            return
        slow_opt = "1" if cat == "FATIGUE" else "2"
        if sel == "3":
            self._mult, self._until = cfg["push_mult"], t + cfg["window"]
        elif sel == slow_opt:
            self._mult, self._until = cfg["slow_mult"], t + cfg["window"]

    def hook(self, info: Dict[str, Any]) -> None:
        t = info["t"]
        if t <= self._last_t:
            return
        self._last_t = t
        d = info.get("decision")
        if d:
            self.decisions.append({
                "i": len(self.decisions), "t": t, "type": d.get("type"),
                "selected": str(d.get("selected") or ""),
                "timed_out": bool(d.get("timed_out")),
            })
            self._tempo_effect(d, t)
        self.ticks.append({"t": t, "severity": info["severity"], "nhp": info["nhp"],
                           "mission_state": info["mission_state"]})
        if self.v.tempo:
            mult = self._mult if t < self._until else 1.0
            self.progress += mult
            self.progress_by_t[t] = self.progress
            if self.announce and (d or t % 10 == 0):
                cfg = self.v.tempo
                print(f"    [progress {self.progress:5.1f} / {cfg['target']:.0f}  "
                      f"deadline t={cfg['deadline']:.0f}  tempo x{mult:.1f}]")


def _ignored(d: Dict[str, Any]) -> bool:
    return d["timed_out"] or d["selected"] == "3"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _crit(text: str, met: bool, detail: str) -> Dict[str, Any]:
    return {"text": text, "met": bool(met), "detail": detail}


def grade_run(v: Vignette, rec: Recorder, feed: List[Dict[str, Any]]) -> Dict[str, Any]:
    outcome = sustained_outcome(feed)
    collapsed = outcome["label"] in FRIENDLY_COLLAPSE
    aborted = any(d["type"] == "CATASTROPHIC" and d["selected"] == "2" and not d["timed_out"]
                  for d in rec.decisions)
    held = not collapsed and not aborted
    shock_t = v.shock_at()
    crit: List[Dict[str, Any]] = []
    extra: Dict[str, Any] = {}

    if v.key == "early_warning":
        pre = [d for d in rec.decisions if d["type"] != "CATASTROPHIC" and d["t"] < shock_t]
        ign = [d for d in pre if _ignored(d)]
        limit = v.params["max_ignored"]
        crit.append(_crit("Let few warnings pass", len(ign) <= limit,
                          f"{len(ign)} of {len(pre)} prompts before the shock ignored (limit {limit})"))
        first = next((d for d in rec.decisions if d["t"] >= v.strain_at() and d["type"] != "CATASTROPHIC"), None)
        if first is None:
            crit.append(_crit("Answered the first warning", False, "no warning ever reached you"))
        else:
            crit.append(_crit("Answered the first warning", not _ignored(first),
                              f"first prompt at t={first['t']} ({first['type']}): "
                              + ("ignored" if _ignored(first) else "acted")))
        extra["ignored"] = len(ign)
    elif v.key == "tempo":
        cfg = v.tempo
        reached_t = next((t for t in sorted(rec.progress_by_t) if rec.progress_by_t[t] >= cfg["target"]), None)
        at_deadline = rec.progress_by_t.get(int(cfg["deadline"]),
                                            rec.progress_by_t[max(rec.progress_by_t)] if rec.progress_by_t else 0.0)
        reached = reached_t is not None and reached_t <= cfg["deadline"]
        pushes = sum(1 for d in rec.decisions if d["type"] in ("FATIGUE", "FATIGUE_EARLY", "PRESSURE", "PRESSURE_EARLY")
                     and _ignored(d))
        crit.append(_crit("Objective reached by the deadline", reached,
                          (f"reached at t={reached_t}" if reached_t is not None
                           else f"progress {at_deadline:.1f} of {cfg['target']:.0f} at t={int(cfg['deadline'])}")
                          + f"; pushes taken: {pushes}"))
        extra.update(progress=at_deadline, reached=reached, reached_t=reached_t, pushes=pushes)
    else:  # shock
        shock_d = next((d for d in rec.decisions if d["type"] == "CATASTROPHIC"), None)
        if shock_d is None:
            crit.append(_crit("Stabilized at the shock", False, "the shock prompt never appeared"))
        else:
            stab = shock_d["selected"] == "1" and not shock_d["timed_out"]
            how = ("timed out" if shock_d["timed_out"]
                   else {"1": "stabilized", "2": "aborted", "3": "held course"}.get(shock_d["selected"], "?"))
            crit.append(_crit("Stabilized at the shock", stab, f"t={shock_d['t']}: {how}"))
        win = v.params["follow_window"]
        after = [d for d in rec.decisions if d["type"] != "CATASTROPHIC"
                 and shock_t <= d["t"] <= shock_t + win]
        slipped = [d for d in after if _ignored(d)]
        crit.append(_crit("Followed through", not slipped and bool(after),
                          f"{len(after) - len(slipped)} of {len(after)} follow-on prompts answered with action"
                          if after else "no follow-on prompts came"))

    crit.append(_crit("Force held through the shock", held,
                      "collapsed" if collapsed else ("aborted the maneuver" if aborted else "no sustained collapse")))

    behaviour_ok = all(c["met"] for c in crit[:-1])
    if held and behaviour_ok:
        grade, headline = "PASS", "Objective met."
    elif held:
        grade = "PARTIAL"
        headline = ("Force held, but you withdrew." if aborted else
                    ("Force held, but you fell short of the objective." if v.key == "tempo"
                     else "Force held, but the margin was thin."))
    else:
        grade = "FAIL"
        headline = ("Objective reached, but the force broke." if (v.key == "tempo" and extra.get("reached"))
                    else "The force collapsed.")
    peak = max((x["severity"] for x in rec.ticks), default=0.0)
    return {"grade": grade, "headline": headline, "criteria": crit, "collapsed": collapsed,
            "aborted": aborted, "outcome": outcome["label"], "peak_severity": peak,
            "peak_band": severity_band(peak), "extra": extra}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _policy_input(policy: Policy, counter: Dict[str, int]) -> Callable[[str], str]:
    def fn(prompt: str = "") -> str:
        i = counter["i"]
        counter["i"] += 1
        return policy(sim.CURRENT_DECISION["type"] or "", sim.CURRENT_DECISION["t"] or 0, i)
    return fn


def run_vignette(key: str, policy: Optional[Policy] = None, seed: Optional[int] = None,
                 delay: float = 0.0, timer: Optional[float] = None, quiet: bool = True,
                 commander_level: str = "expert", n_units: int = 6) -> Dict[str, Any]:
    """Run one vignette. policy=None means a human answering at the terminal."""
    v = VIGNETTES[key]
    seed = v.seed if seed is None else int(seed)
    rec = Recorder(v, announce=(policy is None and not quiet))
    counter = {"i": 0}
    input_fn = input if policy is None else _policy_input(policy, counter)
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        logger, commander, skill, scenario, map_sink = sim.run_simulation(
            scenario_name=v.scenario, commander_level=commander_level, feedback=1.0, seed=seed,
            input_fn=input_fn, delay=delay, log_file="/dev/null" if quiet else None, n_ticks=v.ticks,
            n_units=n_units, map_log=None, engine_profile="graded", prompt_guards="fixed",
            decision_timeout=timer, scenario_overrides=v.overrides(), push_cost=v.push_cost,
            neglect_cost=v.neglect_cost, tick_hook=rec.hook, collect_map=False)
    feed = getattr(logger, "feed", None) or []
    res = grade_run(v, rec, feed)
    res.update(key=key, title=v.title, objective=v.objective, seed=seed,
               decisions=rec.decisions, feed=feed, recorder=rec)
    return res


def selections_policy(recorded: List[Dict[str, Any]], upto: int, then: Policy) -> Policy:
    """Replay recorded selections for decisions with index < upto, then follow `then`."""
    sel = [d["selected"] or "3" for d in recorded]

    def p(dtype: str, t: int, i: int) -> str:
        return sel[i] if i < upto and i < len(sel) else then(dtype, t, i)
    return p


GRADE_RANK = {"FAIL": 0, "PARTIAL": 1, "PASS": 2}
MAX_REPLAYS = 40


def differences(v: Vignette, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every decision where the player's choice differs from the reference line."""
    out = []
    for d in decisions:
        mine = "3" if d["timed_out"] else d["selected"]
        ref = v.reference(d["type"], d["t"], d["i"])
        if mine != ref:
            out.append({"i": d["i"], "t": d["t"], "type": d["type"], "mine": mine, "ref": ref,
                        "timed_out": d["timed_out"]})
    return out


def _single_change_policy(decisions: List[Dict[str, Any]], at: int, ref_choice: str) -> Policy:
    sel = [d["selected"] or "3" for d in decisions]

    def p(dtype: str, t: int, i: int) -> str:
        return ref_choice if i == at else (sel[i] if i < len(sel) else "1")
    return p


def what_if(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same seed (same dice), different decisions. Three questions, cheapest first:

      singles    change ONE decision to the reference choice, keep the rest as played:
                 which single changes would have improved the grade on their own?
      last_chance  return to the reference line from decision k onward (your choices
                 before k): the latest k that still improves the grade
      fallback   if neither helps, the whole reference line from your first difference

    Later prompts can shift when a decision changes, so "the rest as played" means
    your recorded answers in order, an approximation that is stated in the output.
    """
    v = VIGNETTES[res["key"]]
    diffs = differences(v, res["decisions"])
    rank = GRADE_RANK[res["grade"]]
    out: Dict[str, Any] = {"differences": len(diffs), "already_passed": res["grade"] == "PASS"}
    if not diffs or res["grade"] == "PASS":
        return out

    def better(r):
        return (GRADE_RANK[r["grade"]], -r["peak_severity"]) > (rank, -res["peak_severity"]) \
            and GRADE_RANK[r["grade"]] > rank

    singles = []
    for d in diffs[:MAX_REPLAYS]:
        r = run_vignette(res["key"], policy=_single_change_policy(res["decisions"], d["i"], d["ref"]),
                         seed=res["seed"])
        if better(r):
            singles.append({"diff": d, "replay": r})
    out["tested"] = min(len(diffs), MAX_REPLAYS)
    out["singles"] = singles

    last = None
    for d in reversed(diffs[:MAX_REPLAYS]):
        r = run_vignette(res["key"], policy=selections_policy(res["decisions"], d["i"], v.reference),
                         seed=res["seed"])
        if better(r):
            last = {"diff": d, "replay": r}
            break
    out["last_chance"] = last
    if not singles and last is None:
        d = diffs[0]
        out["fallback"] = {"diff": d, "replay": run_vignette(
            res["key"], policy=selections_policy(res["decisions"], d["i"], v.reference), seed=res["seed"])}
    return out


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def _label(dtype: str, sel: str) -> str:
    menu = sim_menu(dtype)
    for o in menu:
        if o["id"] == sel:
            return o["label"]
    return sel


def sim_menu(dtype: str) -> List[Dict[str, str]]:
    from decision_catalog import DECISION_MENUS
    return (DECISION_MENUS.get(dtype) or {}).get("options", [])


def format_brief(v: Vignette, seed: int) -> str:
    lines = ["", "=" * 70, f"VIGNETTE: {v.title}", "=" * 70,
             f"Objective: {v.objective}", ""] + ["  " + b for b in v.brief]
    lines += ["", f"  Scenario: {v.scenario}   Ticks: {v.ticks}   Seed: {seed}", "=" * 70, ""]
    return "\n".join(lines)


def format_result(res: Dict[str, Any], whatif: Optional[Dict[str, Any]] = None) -> str:
    v = VIGNETTES[res["key"]]
    out = ["", "=" * 70, f"VIGNETTE RESULT: {res['title']} — {res['grade']}", "=" * 70,
           res["headline"], f"Objective: {res['objective']}", ""]
    for c in res["criteria"]:
        out.append(f"  [{'met ' if c['met'] else 'MISS'}] {c['text']}: {c['detail']}")
    out.append(f"  Peak severity {res['peak_severity']:.2f} ({res['peak_band']})   seed {res['seed']}")
    if whatif is not None:
        out += ["", "WHAT WOULD HAVE HAPPENED (same seed, same dice):"]
        n = whatif.get("differences", 0)

        def pick(d, r):
            mine = "timed out" if d["timed_out"] else f"'{_label(d['type'], d['mine'])}'"
            return (f"t={d['t']} ({d['type']}): you chose {mine}; the reference chose "
                    f"'{_label(d['type'], d['ref'])}' -> {r['grade']}, peak severity {r['peak_severity']:.2f}"
                    + (f", progress {r['extra'].get('progress', 0):.0f}" if r['extra'].get('progress') else ""))

        if n == 0:
            out.append("  You followed the reference line on every decision; there is nothing to replay.")
        elif whatif.get("already_passed"):
            out.append(f"  You passed. You differed from the reference line on {n} decision(s), and it did not")
            out.append("  cost you: the reference line is one path known to pass, not the only one.")
        else:
            out.append(f"  You differed from the reference line on {n} decision(s). Testing "
                       f"{whatif.get('tested', n)} of them one at a time:")
            sg = whatif.get("singles") or []
            if sg:
                out.append(f"  - {len(sg)} single change(s) would each have improved the result on their own. Earliest:")
                out.append("      " + pick(sg[0]["diff"], sg[0]["replay"]))
            else:
                out.append("  - No single change would have been enough on its own.")
            lc = whatif.get("last_chance")
            if lc:
                d, r = lc["diff"], lc["replay"]
                out.append(f"  - Last chance: your choices up to t={d['t']}, then the reference line, gives {r['grade']} "
                           f"(peak severity {r['peak_severity']:.2f}). Turning it round later than that would not have helped.")
            fb = whatif.get("fallback")
            if fb:
                d, r = fb["diff"], fb["replay"]
                out.append(f"  - The whole reference line from your first difference (t={d['t']}) gives "
                           f"{r['grade']}, peak severity {r['peak_severity']:.2f}.")
            out.append(f"  You: {res['grade']}, peak severity {res['peak_severity']:.2f}"
                       + (f", progress {res['extra'].get('progress', 0):.0f}" if res['extra'].get('progress') else ""))
            out.append("  Notes: the reference line is one path known to pass, not the only one. After a changed")
            out.append("  decision, later prompts can shift, so 'the rest as played' means your answers in order.")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Policies used for demos, tests and tuning
# ---------------------------------------------------------------------------

NAMED_POLICIES: Dict[str, Policy] = {
    "reference": None,                                              # filled per vignette
    "ignore": lambda ty, t, i: "3",
    # always slow down: the slowing option on every fatigue / coordination prompt
    "cautious": lambda ty, t, i: "2" if ty in ("PRESSURE", "PRESSURE_EARLY") else "1",
    # let the early warnings pass, answer the rest
    "sloppy": lambda ty, t, i: "3" if ty.endswith("_EARLY") else "1",
    "hold": lambda ty, t, i: "3" if ty == "CATASTROPHIC" else "1",
    "push_all": lambda ty, t, i: ("3" if ty in ("FATIGUE", "FATIGUE_EARLY", "PRESSURE", "PRESSURE_EARLY") else "1"),
    "stabilize_then_relax": lambda ty, t, i: "1" if ty == "CATASTROPHIC" else "3",
}


def resolve_policy(name: str, key: str) -> Policy:
    if name == "reference":
        return VIGNETTES[key].reference
    if name not in NAMED_POLICIES:
        raise SystemExit(f"unknown policy {name!r}; choose from {', '.join(NAMED_POLICIES)}")
    return NAMED_POLICIES[name]


def check(keys: Optional[List[str]] = None, n: int = 20) -> str:
    """Grade distribution of the named policies over n seeds (used to tune and to test)."""
    rows = ["vignette         policy                 PASS PARTIAL FAIL   (seeds 0..%d)" % (n - 1)]
    for key in keys or list(VIGNETTES):
        for name in NAMED_POLICIES:
            pol = resolve_policy(name, key)
            tally = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
            for s in range(n):
                tally[run_vignette(key, policy=pol, seed=s)["grade"]] += 1
            rows.append(f"{key:16s} {name:22s} {tally['PASS']:4d} {tally['PARTIAL']:7d} {tally['FAIL']:4d}")
        rows.append("")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="JOR 4.0 training vignettes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the vignettes")
    r = sub.add_parser("run", help="run one vignette (you answer at the terminal)")
    r.add_argument("key", choices=sorted(VIGNETTES))
    r.add_argument("--seed", type=int, default=None, help="seed (default: the vignette's fixed seed)")
    r.add_argument("--vary", action="store_true", help="use a fresh random seed (printed)")
    r.add_argument("--policy", default=None,
                   help="auto-play instead of asking: " + ", ".join(NAMED_POLICIES))
    r.add_argument("--timer", type=float, default=None, help="seconds per decision (overrides the default)")
    r.add_argument("--no-timer", action="store_true", help="blocking input, no clock")
    r.add_argument("--no-whatif", action="store_true", help="skip the what-if replay")
    r.add_argument("--delay", type=float, default=0.1, help="seconds between ticks when you play")
    c = sub.add_parser("check", help="grade distribution of the named policies (tuning / self-test)")
    c.add_argument("keys", nargs="*", choices=sorted(VIGNETTES))
    c.add_argument("-n", type=int, default=20)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for v in VIGNETTES.values():
            print(f"  {v.key:14s} {v.title}: {v.objective}")
        return 0
    if args.cmd == "check":
        print(check(args.keys or None, args.n))
        return 0

    v = VIGNETTES[args.key]
    seed = random.randrange(1, 10**6) if args.vary else (args.seed if args.seed is not None else v.seed)
    print(format_brief(v, seed))
    if args.policy:
        res = run_vignette(args.key, policy=resolve_policy(args.policy, args.key), seed=seed)
        print(f"(auto-played with policy '{args.policy}')")
    else:
        timer = None if args.no_timer else (args.timer if args.timer is not None else v.timer)
        res = run_vignette(args.key, policy=None, seed=seed, delay=args.delay, timer=timer, quiet=False)
    try:
        print(format_debrief(build_debrief(res["feed"])))
    except Exception as exc:
        print(f"(debrief unavailable: {exc})")
    wi = None if args.no_whatif else what_if(res)
    print(format_result(res, wi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
