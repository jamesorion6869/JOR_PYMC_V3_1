"""
Regression check: run_simulation(feedback=0.0, engine_profile="legacy") must
reproduce the ORIGINAL interactive simulation tick for tick for the same seed
and decisions. Also checks the skill-independent collapse criteria.

Runs maneuver_sim_original.py's untouched main() with scripted input, reads
the JSON-lines log it writes, and compares it with the refactored
run_simulation(feedback=0.0) buffer.
"""

import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

import maneuver_sim_original as original
from headless_runner import make_input_fn
from maneuver_sim_interactive import run_simulation

SCENARIO_KEYS = {
    "urban": "1",
    "deep_maneuver": "2",
    "logistics_constrained": "3",
    "high_enemy_pressure": "4",
    "baseline": "5",
}
FIELDS_TOP = ["sop", "nhp", "alert"]
# mission_state is deliberately not compared: its collapse labels changed when
# the skill-scaled collapse thresholds were replaced by skill-independent ones.
# It is output-only (nothing feeds back into the dynamics), so every field that
# drives the simulation is still checked exactly.
FIELDS_META = ["t", "theta_s", "theta_c", "theta_o", "morale", "enemy_readiness",
               "enemy_pressure", "enemy_outcomes", "enemy_morale", "enemy_posture",
               "readiness", "outliers", "median_tempo", "worst_ema"]


REGRESSION_LEVEL = "experienced"   # skill multiplier 1.0, where the skill fixes are no-ops


def run_original(scenario, seed, policy):
    decide = make_input_fn(policy)

    # The original main() hardcodes an "expert" commander. Anchor the comparison
    # at skill 1.0 instead: the skill-direction fixes (catastrophe readiness
    # term, readiness prompt thresholds) change behavior only when skill != 1.
    original_skill_cls = original.CommanderSkill

    class Skill1(original_skill_cls):
        def __init__(self, level="expert"):
            super().__init__(REGRESSION_LEVEL)

    original.CommanderSkill = Skill1

    def scripted_input(prompt=""):
        return SCENARIO_KEYS[scenario] if prompt.startswith("Scenario") else decide(prompt)

    real_input = __builtins__["input"] if isinstance(__builtins__, dict) else __builtins__.input
    real_sleep = original.time.sleep
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            original.time.sleep = lambda s: None
            import builtins
            builtins.input = scripted_input
            np.random.seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                original.main()
            with open(f"maneuver_log_{scenario}.json", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        finally:
            builtins.input = real_input
            original.time.sleep = real_sleep
            original.CommanderSkill = original_skill_cls
            os.chdir(cwd)


def run_new(scenario, seed, policy):
    with contextlib.redirect_stdout(io.StringIO()):
        logger, *_ = run_simulation(
            scenario_name=scenario, commander_level=REGRESSION_LEVEL, feedback=0.0, seed=seed,
            input_fn=make_input_fn(policy), delay=0.0, log_file=os.devnull,
            engine_profile="legacy", prompt_guards="legacy",
            scenario_events="legacy",
        )
    return logger.buffer


def check_collapse_skill_independent():
    """Same force state must classify identically at every skill multiplier."""
    from maneuver_sim_interactive import (
        EnemyForce, classify_mission_state, friendly_collapse_flags
    )
    enemy = EnemyForce()  # healthy enemy defaults: not critical
    cases = {
        # (theta_s, theta_c, theta_o, morale): expected label
        (0.10, 0.95, 0.50, 0.10): "FRIENDLY_COLLAPSE",
        (0.80, 0.50, 0.10, 0.80): "STABLE",
        (0.30, 0.70, 0.30, 0.50): "HIGH_PRESSURE",   # near but not critical
    }
    ok = True
    for (ts, tc, to, mo), expected in cases.items():
        labels = {skill: classify_mission_state(ts, tc, to, None, mo, skill, enemy)
                  for skill in (0.6, 1.0, 1.4, 1.8)}
        if set(labels.values()) != {expected}:
            ok = False
            print(f"COLLAPSE MISMATCH state={(ts, tc, to, mo)} expected={expected} got={labels}")
    # the four summary conditions must all be reachable for a deeply collapsed state
    if not all(friendly_collapse_flags(0.10, 0.95, 0.50, 0.10)):
        ok = False
        print("COLLAPSE FLAGS not all true for a fully collapsed state")
    print("collapse criteria skill-independent:", "OK" if ok else "FAILED")
    return ok


def check_skill_direction():
    """Skill must help: more skill -> less readiness lost, earlier prompts."""
    from maneuver_sim_interactive import (
        DecisionState, catastrophe_readiness_retention, maybe_prompt_user,
        readiness_prompt_threshold,
    )
    skills = [0.6, 1.0, 1.4, 1.8]
    ok = True

    # 1. catastrophe readiness retention rises with skill; unchanged at 1.0
    ret = [catastrophe_readiness_retention(s) for s in skills]
    if not all(a < b for a, b in zip(ret, ret[1:])) or abs(ret[1] - 0.25) > 1e-12:
        ok = False
        print(f"CATASTROPHE retention wrong: {dict(zip(skills, ret))}")

    # 2. readiness prompt thresholds rise with skill (notice sooner); unchanged at 1.0
    for base in (0.70, 0.55):
        th = [readiness_prompt_threshold(base, s) for s in skills]
        if not all(a < b for a, b in zip(th, th[1:])) or abs(th[1] - base) > 1e-12:
            ok = False
            print(f"READINESS threshold wrong for base {base}: {dict(zip(skills, th))}")

    # 3. behavioral probe through maybe_prompt_user: sweep one metric until the
    #    first prompt and compare skills.
    def first_trigger(kind, skill):
        for step in range(0, 1001):
            x = step / 1000.0
            ds = DecisionState()
            if kind == "readiness":      # sweep readiness downward from 1.0
                val = 1.0 - x
                got = maybe_prompt_user(val, 0.0, 0.0, 0.5, False, ds, skill)
                want = "FATIGUE_EARLY"
            elif kind == "pressure":     # sweep pressure upward from 0.0
                val = x
                got = maybe_prompt_user(1.0, val, 0.0, 0.9, False, ds, skill)
                want = "PRESSURE_EARLY"
            else:                        # outcomes upward from 0.0
                val = x
                got = maybe_prompt_user(1.0, 0.0, val, 0.9, False, ds, skill)
                want = "OUTCOMES_EARLY"
            if got == want:
                return val
        return None

    r = [first_trigger("readiness", s) for s in skills]
    p = [first_trigger("pressure", s) for s in skills]
    o = [first_trigger("outcomes", s) for s in skills]
    # readiness: higher trigger level = noticed sooner as it falls -> must rise with skill
    if None in r or not all(a < b for a, b in zip(r, r[1:])):
        ok = False
        print(f"PROMPT readiness trigger levels not rising with skill: {dict(zip(skills, r))}")
    # pressure / outcomes: lower trigger level = noticed sooner as it rises -> must fall with skill
    for name, v in (("pressure", p), ("outcomes", o)):
        if None in v or not all(a > b for a, b in zip(v, v[1:])):
            ok = False
            print(f"PROMPT {name} trigger levels not falling with skill: {dict(zip(skills, v))}")

    print("skill-direction checks:", "OK" if ok else "FAILED")
    return ok


PROMPT_TITLES = {
    "Early signs of fatigue.": "FATIGUE_EARLY",
    "Force fatigue rising.": "FATIGUE",
    "Coordination strain emerging.": "PRESSURE_EARLY",
    "Coordination load heavy.": "PRESSURE",
    "Friction emerging.": "OUTCOMES_EARLY",
    "Unit friction detected.": "OUTCOMES",
    "Collapse imminent.": "CATASTROPHIC",
}


def prompts_in_run(scenario, level, policy, seed, prompt_guards):
    """Return [(prompt_kind, tick)] for every decision prompt in one headless run."""
    import re
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_simulation(
            scenario_name=scenario, commander_level=level, feedback=0.0, seed=seed,
            input_fn=make_input_fn(policy), delay=0.0, log_file=os.devnull,
            prompt_guards=prompt_guards,
        )
    found = []
    for chunk in buf.getvalue().split("--- DECISION POINT ---")[1:]:
        # The printed title line is "<title> Choose:" so match on its prefix,
        # and take the tick from the next "t=" line printed after the decision.
        title = chunk.strip().splitlines()[0].strip()
        tick = re.search(r"^t=\s*(\d+)", chunk, re.M)
        kind = next((k for t_, k in PROMPT_TITLES.items() if title.startswith(t_)), None)
        if kind and tick:
            found.append((kind, int(tick.group(1))))
    return found


def cooldown_violations(prompts):
    """Prompts of one category closer together than PROMPT_COOLDOWN, plus repeated one-shot prompts."""
    from maneuver_sim_interactive import PROMPT_COOLDOWN
    bad = 0
    for cat in ("FATIGUE", "PRESSURE", "OUTCOMES"):
        ticks = [t for kind, t in prompts if kind.replace("_EARLY", "") == cat]
        bad += sum(1 for a, b in zip(ticks, ticks[1:]) if b - a < PROMPT_COOLDOWN)
    for once in ("FATIGUE_EARLY", "PRESSURE_EARLY", "OUTCOMES_EARLY", "CATASTROPHIC"):
        bad += max(0, sum(1 for kind, _ in prompts if kind == once) - 1)
    return bad


def check_prompt_guards():
    ok = True
    runs = fixed_prompts = 0
    worst = 0
    for scenario in SCENARIO_KEYS:
        for level in ("novice", "experienced", "expert", "ai_assisted"):
            for policy in ("intervene", "alternate", "ignore"):
                for seed in (0, 1):
                    p = prompts_in_run(scenario, level, policy, seed, "fixed")
                    runs += 1
                    fixed_prompts += len(p)
                    v = cooldown_violations(p)
                    worst = max(worst, len(p))
                    if v:
                        ok = False
                        print(f"PROMPT GUARD VIOLATION ({v}) {scenario}/{level}/{policy}/seed {seed}: {p}")
    # the same check must catch the bug in legacy mode, or it proves nothing
    legacy = prompts_in_run("baseline", "expert", "intervene", 0, "legacy")
    if cooldown_violations(legacy) == 0:
        ok = False
        print("PROMPT TEST VACUOUS: legacy guards produced no violations")
    print(f"prompt guards: {'OK' if ok else 'FAILED'} "
          f"(fixed: {fixed_prompts / runs:.1f} prompts/run avg, {worst} max over {runs} runs; "
          f"legacy sample: {len(legacy)} prompts)")
    return ok


def main():
    failures = 0
    checked = 0
    collapse_ok = (check_collapse_skill_independent() and check_skill_direction()
                   and check_prompt_guards())
    for scenario in SCENARIO_KEYS:
        for policy in ["intervene", "ignore", "alternate"]:
            for seed in range(5):
                a = run_original(scenario, seed, policy)
                b = run_new(scenario, seed, policy)
                checked += 1
                ok = len(a) == len(b)
                if ok:
                    for ea, eb in zip(a, b):
                        for k in FIELDS_TOP:
                            ok &= ea[k] == eb[k]
                        for k in FIELDS_META:
                            ok &= ea["metadata"][k] == eb["metadata"][k]
                if not ok:
                    failures += 1
                    print(f"MISMATCH scenario={scenario} policy={policy} seed={seed} "
                          f"(len {len(a)} vs {len(b)})")
    print(f"{checked - failures}/{checked} runs identical to the original at feedback=0")
    sys.exit(1 if (failures or not collapse_ok) else 0)


if __name__ == "__main__":
    main()
