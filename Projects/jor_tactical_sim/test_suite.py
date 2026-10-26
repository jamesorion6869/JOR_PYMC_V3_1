"""
Self-contained test suite for the current maneuver sim work.

Covers:
  - intervention_factor (novice rotation fix)
  - matrix state scaling (n_units, redistribute, outcomes matrix)
  - timed decision input wiring (non-tty / scripted path)
  - skill-independent collapse criteria
  - skill-direction of catastrophe retention & readiness prompts
  - fixed prompt guards (no cooldown spam)
  - self-consistency: same seed + policy → identical buffers
  - adapter accepts both matrix and list-of-dicts

Does not require maneuver_sim_original (not present in this tree).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

from headless_runner import make_input_fn
from maneuver_adapter import ManeuverAdapter
from maneuver_sim_interactive import (
    COL_FAILURE,
    COL_STALL,
    COL_SUCCESS,
    DEFAULT_N_UNITS,
    DecisionState,
    EnemyForce,
    PROMPT_COOLDOWN,
    Scenario,
    catastrophe_readiness_retention,
    classify_mission_state,
    friendly_collapse_flags,
    generate_outcomes,
    generate_readiness,
    intervention_factor,
    make_timed_input_fn,
    maybe_prompt_user,
    readiness_prompt_threshold,
    redistribute_load,
    run_simulation,
    scale_outcome_column,
)


def _ok(name, passed, detail=""):
    status = "OK" if passed else "FAILED"
    extra = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{extra}")
    return passed


# ---------------------------------------------------------------------------
# intervention_factor
# ---------------------------------------------------------------------------

def test_intervention_factor():
    print("\n== intervention_factor ==")
    skills = [0.6, 1.0, 1.4, 1.8]
    all_pass = True
    for base in (0.9, 0.95, 0.7, 0.8, 0.6):
        vals = [intervention_factor(base, s) for s in skills]
        all_pass &= _ok(
            f"base={base} skill=1.0 identity",
            abs(vals[1] - base) < 1e-12,
            f"got {vals[1]}",
        )
        all_pass &= _ok(
            f"base={base} always <= 1",
            all(v <= 1.0 + 1e-12 for v in vals),
            str([round(v, 4) for v in vals]),
        )
        all_pass &= _ok(
            f"base={base} shrinks with skill",
            all(a > b - 1e-12 for a, b in zip(vals, vals[1:])),
            str([round(v, 4) for v in vals]),
        )
        # novice must not *increase* stalls (the original 0.9/0.6 = 1.5 bug)
        all_pass &= _ok(
            f"base={base} novice factor < 1",
            vals[0] < 1.0,
            f"novice={vals[0]:.4f}",
        )
    return all_pass


# ---------------------------------------------------------------------------
# matrix scaling
# ---------------------------------------------------------------------------

def test_matrix_scaling():
    print("\n== matrix state scaling ==")
    all_pass = True
    np.random.seed(42)
    sc = Scenario("baseline")

    for n in (1, 3, 9, 24):
        r = generate_readiness(5, None, sc, n_units=n)
        o = generate_outcomes(5, 0 if n > 1 else None, n_units=n)
        all_pass &= _ok(f"readiness shape n={n}", r.shape == (n,), str(r.shape))
        all_pass &= _ok(f"outcomes shape n={n}", o.shape == (n, 3), str(o.shape))
        all_pass &= _ok(f"readiness in [0,1] n={n}", float(r.min()) >= 0 and float(r.max()) <= 1)

    # n=3 redistribute matches original hardcoded deltas
    r = np.array([0.80, 0.80, 0.80])
    got = redistribute_load(r, 1.0, 0.06, 0.03)
    all_pass &= _ok("redistribute n=3 light", np.allclose(got, [0.86, 0.77, 0.77]))
    got = redistribute_load(r, 1.0, 0.12, 0.06)
    all_pass &= _ok("redistribute n=3 heavy", np.allclose(got, [0.92, 0.74, 0.74]))

    # scale_outcome_column
    mat = np.array([[10, 4, 2], [12, 6, 3]], dtype=int)
    scale_outcome_column(mat, COL_STALL, 0.5)
    all_pass &= _ok("scale stall column", list(mat[:, COL_STALL]) == [2, 3])

    # headless run at theater scale
    with contextlib.redirect_stdout(io.StringIO()):
        logger, *_ = run_simulation(
            scenario_name="baseline", commander_level="experienced",
            feedback=0.0, seed=1, input_fn=make_input_fn("ignore"), delay=0.0,
            log_file=os.devnull, n_ticks=20, n_units=12,
            engine_profile="graded", prompt_guards="fixed",
        )
    last = logger.buffer[-1]
    all_pass &= _ok("theater run n=12 completes", len(logger.buffer) == 20)
    all_pass &= _ok("log records n_units", last["metadata"]["n_units"] == 12)
    all_pass &= _ok("log readiness length", len(last["metadata"]["readiness"]) == 12)
    return all_pass


# ---------------------------------------------------------------------------
# adapter dual input
# ---------------------------------------------------------------------------

def test_adapter_dual_form():
    print("\n== adapter matrix / list-of-dicts ==")
    mat = np.array([[14, 3, 1], [12, 5, 2], [15, 2, 0]], dtype=int)
    dicts = [
        {"success": 14, "stall": 3, "failure": 1},
        {"success": 12, "stall": 5, "failure": 2},
        {"success": 15, "stall": 2, "failure": 0},
    ]
    a1 = ManeuverAdapter()
    a2 = ManeuverAdapter()
    t1, o1, m1, w1 = a1.compute_theta_o(mat)
    t2, o2, m2, w2 = a2.compute_theta_o(dicts)
    ok = (abs(t1 - t2) < 1e-12 and o1 == o2 and abs(m1 - m2) < 1e-12 and abs(w1 - w2) < 1e-12)
    return _ok("matrix ≡ list-of-dicts", ok, f"theta_o={t1:.4f}")


# ---------------------------------------------------------------------------
# timed input wiring
# ---------------------------------------------------------------------------

def test_timed_input_wiring():
    print("\n== timed decision input ==")
    all_pass = True
    # None / 0 → pass-through base fn
    fn = make_timed_input_fn(None, base_input_fn=lambda p: "2")
    all_pass &= _ok("timeout=None uses base", fn("x") == "2")
    fn = make_timed_input_fn(0, base_input_fn=lambda p: "1")
    all_pass &= _ok("timeout=0 uses base", fn("x") == "1")
    # scripted base always wins even when timeout > 0
    fn = make_timed_input_fn(8.0, base_input_fn=make_input_fn("intervene"))
    all_pass &= _ok("scripted base with timeout", fn("Decision (1/2/3): ") == "1")
    # headless run with decision_timeout set must still complete (scripted path)
    with contextlib.redirect_stdout(io.StringIO()):
        logger, *_ = run_simulation(
            scenario_name="baseline", commander_level="expert",
            feedback=0.0, seed=2, input_fn=make_input_fn("intervene"), delay=0.0,
            log_file=os.devnull, n_ticks=30, decision_timeout=5.0,
            engine_profile="graded", prompt_guards="fixed",
        )
    all_pass &= _ok("run with decision_timeout completes", len(logger.buffer) == 30)
    return all_pass


# ---------------------------------------------------------------------------
# collapse + skill direction (from test_regression, no original needed)
# ---------------------------------------------------------------------------

def test_collapse_skill_independent():
    print("\n== collapse criteria skill-independent ==")
    enemy = EnemyForce()
    cases = {
        (0.10, 0.95, 0.50, 0.10): "FRIENDLY_COLLAPSE",
        (0.80, 0.50, 0.10, 0.80): "STABLE",
        (0.30, 0.70, 0.30, 0.50): "HIGH_PRESSURE",
    }
    ok = True
    for (ts, tc, to, mo), expected in cases.items():
        labels = {
            skill: classify_mission_state(ts, tc, to, None, mo, skill, enemy)
            for skill in (0.6, 1.0, 1.4, 1.8)
        }
        if set(labels.values()) != {expected}:
            ok = False
            print(f"    MISMATCH state={(ts, tc, to, mo)} expected={expected} got={labels}")
    if not all(friendly_collapse_flags(0.10, 0.95, 0.50, 0.10)):
        ok = False
        print("    COLLAPSE FLAGS not all true for fully collapsed state")
    return _ok("collapse labels independent of skill", ok)


def test_skill_direction():
    print("\n== skill-direction checks ==")
    skills = [0.6, 1.0, 1.4, 1.8]
    ok = True

    ret = [catastrophe_readiness_retention(s) for s in skills]
    if not all(a < b for a, b in zip(ret, ret[1:])) or abs(ret[1] - 0.25) > 1e-12:
        ok = False
        print(f"    CATASTROPHE retention wrong: {dict(zip(skills, ret))}")

    for base in (0.70, 0.55):
        th = [readiness_prompt_threshold(base, s) for s in skills]
        if not all(a < b for a, b in zip(th, th[1:])) or abs(th[1] - base) > 1e-12:
            ok = False
            print(f"    READINESS threshold wrong for base {base}: {dict(zip(skills, th))}")

    def first_trigger(kind, skill):
        for step in range(0, 1001):
            x = step / 1000.0
            ds = DecisionState()
            if kind == "readiness":
                val = 1.0 - x
                got = maybe_prompt_user(val, 0.0, 0.0, 0.5, False, ds, skill)
                want = "FATIGUE_EARLY"
            elif kind == "pressure":
                val = x
                got = maybe_prompt_user(1.0, val, 0.0, 0.9, False, ds, skill)
                want = "PRESSURE_EARLY"
            else:
                val = x
                got = maybe_prompt_user(1.0, 0.0, val, 0.9, False, ds, skill)
                want = "OUTCOMES_EARLY"
            if got == want:
                return val
        return None

    r = [first_trigger("readiness", s) for s in skills]
    p = [first_trigger("pressure", s) for s in skills]
    o = [first_trigger("outcomes", s) for s in skills]
    if None in r or not all(a < b for a, b in zip(r, r[1:])):
        ok = False
        print(f"    PROMPT readiness not rising with skill: {dict(zip(skills, r))}")
    for name, v in (("pressure", p), ("outcomes", o)):
        if None in v or not all(a > b for a, b in zip(v, v[1:])):
            ok = False
            print(f"    PROMPT {name} not falling with skill: {dict(zip(skills, v))}")

    return _ok("skill helps (retention, thresholds, prompts)", ok)


# ---------------------------------------------------------------------------
# prompt guards
# ---------------------------------------------------------------------------

PROMPT_TITLES = {
    "Early signs of fatigue.": "FATIGUE_EARLY",
    "Force fatigue rising.": "FATIGUE",
    "Coordination strain emerging.": "PRESSURE_EARLY",
    "Coordination load heavy.": "PRESSURE",
    "Friction emerging.": "OUTCOMES_EARLY",
    "Unit friction detected.": "OUTCOMES",
    "Collapse imminent.": "CATASTROPHIC",
}


def _prompts_in_run(scenario, level, policy, seed, prompt_guards):
    import re
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_simulation(
            scenario_name=scenario, commander_level=level, feedback=0.0, seed=seed,
            input_fn=make_input_fn(policy), delay=0.0, log_file=os.devnull,
            prompt_guards=prompt_guards, n_ticks=120,
            engine_profile="legacy",  # steeper NHP; more decision pressure in short runs
        )
    found = []
    for chunk in buf.getvalue().split("--- DECISION POINT ---")[1:]:
        raw = chunk.strip().splitlines()[0].strip()
        # Menus print "Title. Choose:" — strip the trailing " Choose:" for lookup.
        title = raw[:-8].rstrip() if raw.endswith(" Choose:") else raw
        tick = re.search(r"^t=\s*(\d+)", chunk, re.M)
        if title in PROMPT_TITLES and tick:
            found.append((PROMPT_TITLES[title], int(tick.group(1))))
    return found


def _cooldown_violations(prompts):
    bad = 0
    for cat in ("FATIGUE", "PRESSURE", "OUTCOMES"):
        ticks = [t for kind, t in prompts if kind.replace("_EARLY", "") == cat]
        bad += sum(1 for a, b in zip(ticks, ticks[1:]) if b - a < PROMPT_COOLDOWN)
    for once in ("FATIGUE_EARLY", "PRESSURE_EARLY", "OUTCOMES_EARLY", "CATASTROPHIC"):
        bad += max(0, sum(1 for kind, _ in prompts if kind == once) - 1)
    return bad


def test_prompt_guards():
    print("\n== prompt guards ==")
    ok = True
    runs = fixed_prompts = 0
    worst = 0
    for scenario in ("baseline", "urban"):
        for level in ("novice", "experienced", "expert"):
            for policy in ("intervene", "ignore"):
                for seed in (0, 1):
                    p = _prompts_in_run(scenario, level, policy, seed, "fixed")
                    runs += 1
                    fixed_prompts += len(p)
                    worst = max(worst, len(p))
                    v = _cooldown_violations(p)
                    if v:
                        ok = False
                        print(f"    VIOLATION ({v}) {scenario}/{level}/{policy}/seed{seed}: {p}")
    legacy = _prompts_in_run("baseline", "expert", "intervene", 0, "legacy")
    if _cooldown_violations(legacy) == 0:
        ok = False
        print("    PROMPT TEST VACUOUS: legacy produced no violations")
    detail = (
        f"fixed avg {fixed_prompts / max(runs, 1):.1f} prompts/run, "
        f"max {worst} over {runs} runs; legacy sample {len(legacy)} prompts"
    )
    return _ok("fixed guards clean; legacy still spammy", ok, detail)


# ---------------------------------------------------------------------------
# self-consistency (stand-in for original regression when original is absent)
# ---------------------------------------------------------------------------

def _snapshot(buffer):
    """Extract comparable fields from a logger buffer."""
    out = []
    for e in buffer:
        m = e["metadata"]
        out.append({
            "sop": e["sop"], "nhp": e["nhp"], "alert": e["alert"],
            "t": m["t"], "theta_s": m["theta_s"], "theta_c": m["theta_c"],
            "theta_o": m["theta_o"], "morale": m["morale"],
            "enemy_readiness": m["enemy_readiness"],
            "enemy_pressure": m["enemy_pressure"],
            "enemy_outcomes": m["enemy_outcomes"],
            "enemy_morale": m["enemy_morale"],
            "enemy_posture": m["enemy_posture"],
            "readiness": m["readiness"],
            "outliers": m["outliers"],
            "median_tempo": m["median_tempo"],
            "worst_ema": m["worst_ema"],
        })
    return out


def test_self_consistency():
    print("\n== self-consistency (same seed → identical) ==")
    all_pass = True
    for scenario in ("baseline", "high_enemy_pressure"):
        for policy in ("intervene", "ignore", "alternate"):
            for seed in (0, 3):
                runs = []
                for _ in range(2):
                    with contextlib.redirect_stdout(io.StringIO()):
                        logger, *_ = run_simulation(
                            scenario_name=scenario, commander_level="experienced",
                            feedback=0.0, seed=seed, input_fn=make_input_fn(policy),
                            delay=0.0, log_file=os.devnull, n_ticks=60,
                            engine_profile="legacy", prompt_guards="legacy",
                            n_units=DEFAULT_N_UNITS, decision_timeout=None,
                        )
                    runs.append(_snapshot(logger.buffer))
                match = runs[0] == runs[1]
                all_pass &= _ok(
                    f"{scenario}/{policy}/seed{seed}",
                    match,
                    f"len={len(runs[0])}" if match else "buffers differ",
                )
    return all_pass


def test_feedback_zero_noop():
    """feedback=0 must not change dynamics vs a second feedback=0 run (trivial),
    and readiness_debt / pressure_creep must stay 0."""
    print("\n== feedback=0 is a no-op on debt/creep ==")
    with contextlib.redirect_stdout(io.StringIO()):
        logger, *_ = run_simulation(
            scenario_name="baseline", commander_level="experienced",
            feedback=0.0, seed=7, input_fn=make_input_fn("ignore"),
            delay=0.0, log_file=os.devnull, n_ticks=40,
            engine_profile="legacy", prompt_guards="legacy",
        )
    debts = [s["metadata"]["readiness_debt"] for s in logger.buffer]
    creeps = [s["metadata"]["pressure_creep"] for s in logger.buffer]
    return _ok(
        "debt and creep stay 0 at feedback=0",
        all(d == 0.0 for d in debts) and all(c == 0.0 for c in creeps),
        f"max debt={max(debts)} max creep={max(creeps)}",
    )


def _reference_theta_o(adapter_params, ema, mat):
    """The original per-unit-loop theta_o computation, kept as the test oracle."""
    alpha, thr, floor = adapter_params
    totals = np.maximum(mat.sum(axis=1), 1.0)
    scores = (0.5 * mat[:, 1] + 1.0 * mat[:, 2]) / totals
    median = float(np.median(scores))
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = max(q3 - q1, floor)
    outliers = 0
    for idx, score in enumerate(scores):
        z = (score - median) / iqr
        ema[idx] = alpha * max(z, 0) + (1 - alpha) * ema.get(idx, 0.0)
        if ema[idx] > thr:
            outliers += 1
    worst = max(ema.values(), default=0.0)
    sev = np.clip((worst - thr) / max(thr, 1e-6), 0.0, 1.0)
    frac = np.clip(outliers / len(scores), 0.0, 1.0)
    comb = np.clip(0.7 * max(sev, frac) + 0.3 * ((sev + frac) / 2.0), 0.0, 1.0)
    theta_o = float(np.clip(float(np.clip(median, 0.0, 1.0)) + 0.7 * comb, 0.0, 1.0))
    return theta_o, outliers, median, worst


def test_adapter_vectorized_equivalence():
    """Vectorized EMA/outlier update must match the original per-unit loop exactly."""
    rng = np.random.default_rng(7)
    ok, calls, with_outliers = True, 0, 0
    for n in (1, 3, 7, 50, 257):
        for trial in range(12):
            adapter = ManeuverAdapter()
            ema, params = {}, (adapter.ema_alpha, adapter.outlier_threshold, adapter.iqr_floor)
            for _ in range(50):
                m = np.column_stack([rng.integers(8, 20, n), rng.integers(0, 9, n), rng.integers(0, 6, n)])
                if trial % 3 == 0:                      # sustained outliers push EMA over threshold
                    m[: max(1, n // 5), 2] += 12
                got = adapter.compute_theta_o(m)
                want = _reference_theta_o(params, ema, m.astype(float))
                calls += 1
                with_outliers += want[1] > 0
                if tuple(map(float, got)) != tuple(map(float, want)):
                    ok = False
    covered = with_outliers > 100                       # the comparison must exercise the outlier branch
    # growth: a force that gets bigger keeps old state, new units start at zero
    a = ManeuverAdapter()
    a.compute_theta_o(np.array([[10, 3, 4]] * 3))
    old = a.unit_ema
    a.compute_theta_o(np.array([[10, 3, 4]] * 5))
    grew = len(a.unit_ema) == 5 and all(a.unit_ema[i] >= 0 for i in range(5))
    return _ok("adapter vectorized ≡ per-unit loop (bit-identical)", ok and covered and grew,
               f"{calls} calls, {with_outliers} with outliers")


def test_vectorized_sampler_distribution():
    """Vectorized outcomes sampler must match the scalar sampler's distributions."""
    from maneuver_sim_interactive import generate_outcomes

    def sample(sampler, n, ticks, degraded):
        np.random.seed(11)
        return np.concatenate([generate_outcomes(t, degraded, n, sampler=sampler) for t in range(ticks)])

    ok = True
    for degraded in (None, 1):
        a = sample("scalar", 120, 300, degraded)        # 36k unit-draws each
        b = sample("vectorized", 120, 300, degraded)
        for col, name in ((0, "success"), (1, "stall"), (2, "failure")):
            ma, mb = a[:, col].mean(), b[:, col].mean()
            sa, sb = a[:, col].std(), b[:, col].std()
            se = np.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b))
            if abs(ma - mb) > 5 * se or abs(sa - sb) > 0.05 * max(sa, 1e-9):
                ok = False
                print(f"    {name} (degraded={degraded}): mean {ma:.3f} vs {mb:.3f}, std {sa:.3f} vs {sb:.3f}")
            # whole distribution, not just two moments
            hi = int(max(a[:, col].max(), b[:, col].max())) + 1
            ha = np.bincount(a[:, col], minlength=hi) / len(a)
            hb = np.bincount(b[:, col], minlength=hi) / len(b)
            if np.abs(ha - hb).max() > 0.02:
                ok = False
                print(f"    {name} (degraded={degraded}): histogram gap {np.abs(ha - hb).max():.3f}")
    # the scalar stream is unchanged at normal sizes: auto == scalar at n <= 64
    np.random.seed(5); x = generate_outcomes(3, 1, 12)
    np.random.seed(5); y = generate_outcomes(3, 1, 12, sampler="scalar")
    ok &= bool(np.array_equal(x, y))
    return _ok("vectorized sampler ≈ scalar distributions; auto keeps scalar stream ≤ 64 units", ok)


def test_large_force_logging():
    """Compact logs and no unrequested map frames above LARGE_FORCE_UNITS; unchanged below."""
    from maneuver_sim_interactive import LARGE_FORCE_UNITS

    def run(n, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            logger, *_, sink = run_simulation(
                "baseline", "experienced", 0.0, seed=1, input_fn=make_input_fn("intervene"),
                delay=0.0, log_file=os.devnull, n_ticks=30, n_units=n, **kw)
        return logger.buffer, sink

    ok = True

    def tick_frames(sink):
        # decision ticks also emit a PENDING frame before the choice is made
        return [f for f in sink.frames if not (f.get("decision") or {}).get("pending")]

    small, sink_small = run(LARGE_FORCE_UNITS)
    big, sink_big = run(LARGE_FORCE_UNITS + 1)
    m_small, m_big = small[5]["metadata"], big[5]["metadata"]
    ok &= "readiness" in m_small and "positions" in m_small and "log_mode" not in m_small
    ok &= len(tick_frames(sink_small)) == len(small)                     # frames still collected
    ok &= "readiness" not in m_big and m_big["log_mode"] == "compact"
    summary = m_big["readiness_summary"]
    ok &= summary["min"] <= summary["mean"] <= summary["max"] and len(summary["lowest_units"]) == 5
    ok &= "positions" in big[0]["metadata"] and all("positions" not in s["metadata"] for s in big[1:])
    ok &= len(sink_big.frames) == 0                                  # not requested, not built
    _, sink_forced = run(LARGE_FORCE_UNITS + 1, collect_map=True)
    ok &= len(tick_frames(sink_forced)) == 30                             # explicit request honored
    return _ok("large-force logging (compact log, positions once, map frames on request)", ok)


def test_strain_fraction():
    """strain_fraction scales the strain event with N; default and N=3 equivalence preserved."""
    from maneuver_sim_interactive import (
        degraded_indices, generate_outcomes, generate_readiness, is_degraded, strained_units,
    )
    ok = True

    # default (None) is the original single-unit event; fraction 1/3 at N=3 is the same unit
    ok &= strained_units(3, None) == 1 and strained_units(3, 1 / 3) == [1]
    # block size follows the fraction at every N, stays inside the force, and is never empty
    for n in (1, 2, 3, 4, 6, 12, 50, 300, 3072):
        for f in (1 / 3, 0.1, 0.5, 1.0):
            block = strained_units(n, f)
            k = min(n, max(1, int(round(f * n))))
            ok &= block == list(range(block[0], block[0] + k)) and block[0] >= 0 and block[-1] < n
    # fraction-mode at N=3 reproduces the original generators draw for draw (same RNG stream)
    sc = Scenario("baseline")
    for seed in range(5):
        np.random.seed(seed); r1 = generate_readiness(40, 1, sc, 3); o1 = generate_outcomes(40, 1, 3)
        np.random.seed(seed); r2 = generate_readiness(40, strained_units(3, 1 / 3), sc, 3)
        o2 = generate_outcomes(40, strained_units(3, 1 / 3), 3)
        ok &= bool(np.array_equal(r1, r2) and np.array_equal(o1, o2))
    # helper semantics
    ok &= not is_degraded(None) and is_degraded(1) and is_degraded([2, 3]) and not is_degraded([])
    ok &= degraded_indices([1, 2, 99], 3).tolist() == [1, 2]

    # the point of the feature: the readiness signal no longer vanishes as N grows
    def readiness_drop(n, degraded):
        np.random.seed(0)
        base = np.mean([generate_readiness(40, None, sc, n).mean() for _ in range(200)])
        np.random.seed(0)
        hit = np.mean([generate_readiness(40, degraded, sc, n).mean() for _ in range(200)])
        return base - hit
    single = [readiness_drop(n, strained_units(n, None)) for n in (3, 48, 768)]
    frac = [readiness_drop(n, strained_units(n, 1 / 3)) for n in (3, 48, 768)]
    ok &= single[2] < single[0] / 100                         # single unit: diluted ~1/N
    ok &= max(frac) - min(frac) < 0.005 and frac[0] > 0.03     # fraction: ~0.04 at any N

    # end to end: a strained force is flagged, in the log and on the map
    with contextlib.redirect_stdout(io.StringIO()):
        logger, *_, sink = run_simulation(
            "baseline", "experienced", 0.0, seed=2, input_fn=make_input_fn("intervene"),
            delay=0.0, log_file=os.devnull, n_ticks=40, n_units=12, strain_fraction=1 / 3,
            collect_map=True)
    last_meta = logger.buffer[-1]["metadata"]
    flagged = [u["id"] for u in sink.frames[-1]["units"] if u["degraded"]]
    ok &= last_meta["degraded_unit"] == flagged == [1, 2, 3, 4]
    ok &= "strain" in last_meta["narrative"] and last_meta["mission_state"] in (
        "DEGRADED", "HIGH_PRESSURE", "FRIENDLY_COLLAPSE", "MUTUAL_COLLAPSE", "ENEMY_COLLAPSE")
    return _ok("strain fraction (scales with N; N=3 identical; log + map agree)", ok,
               f"single-unit signal {single[0]:.4f}→{single[2]:.5f}, fraction {frac[0]:.4f}→{frac[2]:.4f}")


def _fake_state(t, collapsed_friendly=False, collapsed_enemy=False):
    """A logger-buffer entry; all four friendly (and/or all three enemy) collapse conditions on or off."""
    return {
        "sop": 0.5, "nhp": 0.9, "alert": True,
        "metadata": {
            "t": t,
            "theta_s": 0.10 if collapsed_friendly else 0.60,
            "theta_c": 0.95 if collapsed_friendly else 0.50,
            "theta_o": 0.50 if collapsed_friendly else 0.10,
            "morale": 0.10 if collapsed_friendly else 0.80,
            "enemy_readiness": 0.10 if collapsed_enemy else 0.70,
            "enemy_pressure": 0.95 if collapsed_enemy else 0.40,
            "enemy_outcomes": 0.5, "enemy_morale": 0.10 if collapsed_enemy else 0.70,
            "enemy_posture": "probing",
        },
    }


def _summary_text(friendly_pattern, enemy_pattern=None):
    """Run print_mission_summary on a synthetic mission and return what it printed."""
    from maneuver_sim_interactive import CommanderSkill, print_mission_summary

    enemy_pattern = enemy_pattern or [False] * len(friendly_pattern)
    states = [_fake_state(t, f, e) for t, (f, e) in enumerate(zip(friendly_pattern, enemy_pattern))]

    class FakeLogger:
        buffer = states

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print_mission_summary(FakeLogger(), CommanderSkill("expert"), 1.4, Scenario("baseline"))
    return out.getvalue()


def test_sustained_collapse():
    """Collapse needs all conditions held for COLLAPSE_SUSTAIN_TICKS consecutive ticks."""
    from maneuver_sim_interactive import (
        COLLAPSE_SUSTAIN_TICKS, collapse_runs, enemy_collapse_flags, sustained_collapse,
    )
    ok = True
    K = COLLAPSE_SUSTAIN_TICKS
    ok &= K == 5

    def result(pattern, start_t=0):
        ticks = list(range(start_t, start_t + len(pattern)))
        return sustained_collapse(ticks, pattern)

    # a one-tick dip, a run one short of the requirement, a run exactly long enough
    ok &= not result([0, 0, 1, 0, 0])["collapsed"]
    ok &= not result([1] * (K - 1))["collapsed"]
    r = result([0, 0] + [1] * K + [0])
    ok &= r["collapsed"] and r["onset_t"] == 2 and r["held"] == K
    # a long run reports its whole length, and onset is the first sustained run
    r = result([1, 1, 0] + [1] * 8)
    ok &= r["collapsed"] and r["onset_t"] == 3 and r["held"] == 8
    # interruption resets the count: 3 + 3 does not make 5 (or 6)
    ok &= not result([1, 1, 1, 0, 1, 1, 1])["collapsed"]
    # a gap in the tick numbers also ends a run
    ok &= not sustained_collapse([0, 1, 2, 5, 6], [1, 1, 1, 1, 1])["collapsed"]
    ok &= collapse_runs([0, 1, 2, 5, 6], [1, 1, 1, 1, 1]) == [(0, 3), (5, 2)]
    # too-brief breaches are still reported (longest run), and never-met reports nothing
    r = result([0, 1, 1, 0, 0])
    ok &= (not r["collapsed"]) and r["longest"] == 2 and r["longest_t"] == 1
    r = result([0, 0, 0])
    ok &= (not r["collapsed"]) and r["longest"] == 0 and r["longest_t"] is None
    ok &= collapse_runs([], []) == [] and not sustained_collapse([], [])["collapsed"]
    ok &= enemy_collapse_flags(0.1, 0.95, 0.1) == (True, True, True) and not all(enemy_collapse_flags(0.5, 0.95, 0.1))

    # through the mission summary: friendly, enemy, brief breach, near miss
    text = _summary_text([False] * 10 + [True] * K)
    ok &= "Collapse Signature:      FRIENDLY COLLAPSE" in text and f"onset t=10, held {K} ticks" in text
    text = _summary_text([False] * 10 + [True] * (K - 1) + [False] * 3)
    ok &= "NOT DETECTED" in text and "BRIEF BREACH" in text and f"{K - 1} tick(s)" in text
    text = _summary_text([False] * 12, [False] * 5 + [True] * K + [False] * 2)
    ok &= "Collapse Signature:      ENEMY COLLAPSE" in text and "Enemy Collapse:          onset t=5" in text
    text = _summary_text([True] * K, [True] * K)
    ok &= "MUTUAL COLLAPSE" in text
    return _ok("sustained collapse (5 ticks; resets on a break or gap; summary reports onset/hold/brief breach)", ok)


def _collapsed(scen, level, policy, seed, **kw):
    from maneuver_sim_interactive import sustained_collapse
    with contextlib.redirect_stdout(io.StringIO()):
        logger, *_ = run_simulation(
            scen, level, 0.0, seed=seed, input_fn=make_input_fn(policy),
            delay=0.0, log_file=os.devnull, **kw)
    ticks = [s["metadata"]["t"] for s in logger.buffer]
    met = [all(friendly_collapse_flags(m["theta_s"], m["theta_c"], m["theta_o"], m["morale"]))
           for m in (s["metadata"] for s in logger.buffer)]
    return sustained_collapse(ticks, met)["collapsed"]


def test_collapse_discriminates_policy():
    """At the default horizon, 5-tick collapse separates good handling from poor, and skill matters.

    Judged on baseline, whose events are unchanged. (The other scenarios have their
    own difficulty; see test_scenarios_distinct.)"""
    seeds = range(20)
    expert_intervene = sum(_collapsed("baseline", "expert", "intervene", sd) for sd in seeds)
    novice_intervene = sum(_collapsed("baseline", "novice", "intervene", sd) for sd in seeds)
    expert_ignore = sum(_collapsed("baseline", "expert", "ignore", sd) for sd in seeds)
    novice_ignore = sum(_collapsed("baseline", "novice", "ignore", sd) for sd in seeds)
    ok = (expert_intervene == 0 and novice_intervene == 20
          and expert_ignore == 20 and novice_ignore == 20)
    return _ok("collapse discriminates (baseline, 20 runs each: expert+intervene / novice+intervene / expert+ignore / novice+ignore)",
               ok, f"{expert_intervene} / {novice_intervene} / {expert_ignore} / {novice_ignore}")


def test_scenarios_distinct():
    """Scenarios differ in event timing/composition and in how hard they are on a good commander;
    legacy mode restores the identical original events."""
    from maneuver_sim_interactive import Scenario
    names = ("deep_maneuver", "baseline", "logistics_constrained", "urban", "high_enemy_pressure")
    ok = True

    # distinct profiles (no two scenarios share a full event signature)
    sigs = {n: tuple(sorted((k, v) for k, v in Scenario(n).EVENT_PROFILES[n].items()
                            if not k.endswith("_cause"))) for n in names}
    ok &= len(set(sigs.values())) == len(names)
    # baseline is the original event; legacy mode makes every scenario match it
    base = Scenario("baseline")
    ok &= (base.strain_tick, base.catastrophe_tick) == (30, 110)
    for n in names:
        sc = Scenario(n)
        sc.use_legacy_events()
        ok &= (sc.strain_tick, sc.catastrophe_tick, sc.cat_readiness_scale, sc.cat_pressure_scale,
               sc.cat_morale_scale, sc.cat_outcome_scale) == (30, 110, 1.0, 1.0, 1.0, 1.0)

    # difficulty for a well-handled expert: forgiving < standard < contested < hardest
    rate = {n: sum(_collapsed(n, "expert", "intervene", sd) for sd in range(20)) for n in names}
    ok &= rate["deep_maneuver"] == 0 and rate["baseline"] == 0
    ok &= rate["logistics_constrained"] > 0 and rate["urban"] > 0
    ok &= rate["high_enemy_pressure"] > max(rate["logistics_constrained"], rate["urban"])
    # the forgiving scenario forgives a novice too; poor handling still loses everywhere
    ok &= sum(_collapsed("deep_maneuver", "novice", "intervene", sd) for sd in range(10)) == 0
    for n in names:
        ok &= all(_collapsed(n, "novice", "ignore", sd) for sd in range(5))
    return _ok("scenarios distinct (profiles, legacy mode, difficulty ladder)", ok,
               " / ".join(f"{n.split('_')[0]} {rate[n]}/20" for n in names))


def test_aar_matches_terminal_rule():
    """The After-Action final state uses the terminal summary's 5-tick sustained rule,
    ignores pending-decision frames, and a one-tick breach is not a collapse."""
    from aar_summary import summarize_frames
    ok = True
    runs = 0
    for scen in ("baseline", "urban", "high_enemy_pressure"):
        for level, policy in (("expert", "intervene"), ("novice", "ignore"), ("experienced", "alternate")):
            for seed in (0, 1):
                with contextlib.redirect_stdout(io.StringIO()):
                    logger, *_, sink = run_simulation(
                        scen, level, 1.0, seed=seed, input_fn=make_input_fn(policy),
                        delay=0.0, log_file=os.devnull, n_units=6, decision_timeout=8.0)
                from maneuver_sim_interactive import sustained_collapse
                ticks = [s["metadata"]["t"] for s in logger.buffer]
                met = [all(friendly_collapse_flags(m["theta_s"], m["theta_c"], m["theta_o"], m["morale"]))
                       for m in (s["metadata"] for s in logger.buffer)]
                terminal = sustained_collapse(ticks, met)["collapsed"]
                summary = summarize_frames(sink.frames)
                aar = summary["final_state"] in ("FRIENDLY_COLLAPSE", "MUTUAL_COLLAPSE")
                ok &= (aar == terminal)
                n_prompts = sum(1 for f in sink.frames
                                if f.get("decision") and not f["decision"].get("pending"))
                ok &= summary["n_decisions"] == n_prompts
                runs += 1
    # the instantaneous last-tick label alone must not be what decides it
    with contextlib.redirect_stdout(io.StringIO()):
        _, *_, sink = run_simulation("baseline", "expert", 1.0, seed=3, input_fn=make_input_fn("intervene"),
                                     delay=0.0, log_file=os.devnull, n_units=6, decision_timeout=8.0)
    ok &= sink.frames[-1]["mission_state"].endswith("COLLAPSE")
    ok &= summarize_frames(sink.frames)["final_state"] == "DEGRADED"
    return _ok(f"AAR final state == terminal sustained rule ({runs} runs; pending frames not counted)", ok)


def _mission(scen="baseline", level="expert", policy="intervene", seed=3, **kw):
    kw.setdefault("n_units", 6)
    with contextlib.redirect_stdout(io.StringIO()):
        logger, *_, sink = run_simulation(
            scen, level, 1.0, seed=seed, input_fn=make_input_fn(policy) if isinstance(policy, str) else policy,
            delay=0.0, log_file=os.devnull, **kw)
    return logger, sink


def test_severity_gauge():
    """Severity is graded (separates handling quality, does not saturate), stateless-safe, and
    recorded in the log, feed and frames. NHP / the engine are untouched (see regression)."""
    from severity import SeverityGauge, raw_severity, severity_band, SEVERITY_BANDS
    ok = True
    # math: clipped, zero before calibration, monotone in SOP, bands partition [0, 1]
    ok &= raw_severity(0.9, None) == 0.0 and raw_severity(0.2, 0.3) == 0.0 and raw_severity(5.0, 0.3) == 1.0
    vals = [raw_severity(x / 100, 0.30) for x in range(30, 91)]
    ok &= all(a <= b for a, b in zip(vals, vals[1:])) and vals[0] == 0.0 and vals[-1] == 1.0
    ok &= [severity_band(v) for v in (0.0, 0.19, 0.2, 0.5, 0.79, 0.8, 1.0)] == \
          ["NOMINAL", "NOMINAL", "ELEVATED", "STRAINED", "CRITICAL", "BREAKING", "BREAKING"]
    g = SeverityGauge()
    before = g.value
    peek = g.peek(0.8, 0.3)
    ok &= g.value == before and peek > 0                      # peek never advances the gauge
    ok &= abs(g.update(0.8, 0.3) - peek) < 1e-12 and g.value == peek
    # records everywhere, and the strain / catastrophe events land on the scenario's ticks
    logger, sink = _mission()
    ok &= all("severity" in r["metadata"] and "severity_band" in r["metadata"] for r in logger.buffer)
    ok &= all("severity" in f for f in sink.frames)
    ev = {e["kind"]: f["t"] for f in logger.feed for e in f["events"]}
    ok &= ev == {"strain": 30, "catastrophe": 110}
    # graded: over the post-calibration window (calibration ticks are 0 by construction) a
    # well-handled force sits far below a neglected one, and does not saturate
    def mean_sev(level, policy):
        out = []
        for sd in range(4):
            lg, _ = _mission(level=level, policy=policy, seed=sd)
            out.append(np.mean([r["metadata"]["severity"] for r in lg.buffer if 40 <= r["metadata"]["t"] < 108]))
        return float(np.mean(out))
    good, bad = mean_sev("expert", "intervene"), mean_sev("novice", "ignore")
    ok &= bad - good > 0.20
    lg, _ = _mission()
    ok &= max(r["metadata"]["severity"] for r in lg.buffer if r["metadata"]["t"] < 108) < 0.5
    # NHP saturates where severity still discriminates between two catastrophes
    peaks = {}
    for lvl, pol in (("expert", "intervene"), ("novice", "ignore")):
        lg, _ = _mission(level=lvl, policy=pol)
        peaks[(lvl, pol)] = (max(r["nhp"] for r in lg.buffer), max(r["metadata"]["severity"] for r in lg.buffer))
    ok &= peaks[("novice", "ignore")][1] - peaks[("expert", "intervene")][1] > 0.08
    return _ok("severity gauge (graded, peek-safe, recorded, events on schedule)", ok,
               f"mean sev expert/intervene {good:.2f} vs novice/ignore {bad:.2f}")


def test_debrief():
    """Debrief statements are traceable to what was observed: immediate effects are the
    pending→resolved change, decisions during the shock are not blamed, timeouts are surfaced."""
    import json as _json
    import maneuver_sim_interactive as M
    from aar_summary import summarize_frames
    from debrief import WINDOW, build_debrief, format_debrief
    from map_telemetry import append_end_record
    ok = True

    logger, sink = _mission()
    d_feed, d_frames = build_debrief(logger.feed), build_debrief(sink.frames)
    # same story from the slim feed and from full map frames
    ok &= all(d_feed[k] == d_frames[k] for k in ("headline", "outcome", "final_state", "decisions"))
    n_prompts = sum(1 for f in logger.feed if f["decision"] and not f["decision"].get("pending"))
    ok &= len(d_feed["decisions"]) == n_prompts == sum(g["n"] for g in d_feed["decision_groups"])
    # immediate effect == resolved - pending on the same tick, recomputed independently
    pend = {f["t"]: f for f in logger.feed if (f["decision"] or {}).get("pending")}
    for d in d_feed["decisions"]:
        res = next(f for f in logger.feed if f["t"] == d["t"] and not (f["decision"] or {}).get("pending"))
        want = {m: round(res[k] - pend[d["t"]][k], 3) for k, m in
                (("theta_s", "readiness"), ("theta_c", "pressure"), ("theta_o", "outcomes"))
                if abs(res[k] - pend[d["t"]][k]) >= 0.005}
        ok &= {i["metric"]: i["delta"] for i in d["immediate"]} == want
    # a reinforcing choice lowers pressure immediately; "push through" costs readiness
    lo, _ = _mission(level="novice", policy="ignore")
    imm = [i for d in build_debrief(lo.feed)["decisions"] for i in d["immediate"]]
    ok &= any(i["metric"] == "readiness" and i["delta"] < 0 for i in imm)
    ok &= any(i["metric"] == "pressure" and i["delta"] < 0 for d in d_feed["decisions"] for i in d["immediate"])
    # nothing at or just after the catastrophe is blamed on the choice made then
    cat = 110
    shock = [d for d in d_feed["decisions"] if cat <= d["t"] < cat + WINDOW]
    ok &= bool(shock) and all(d["during_shock"] for d in shock)
    for dbf in (d_feed, build_debrief(lo.feed)):
        for l in dbf["lessons"]:
            if l["title"] == "Accepting early strain was costly":
                first_t = int(l["evidence"].split("t=")[1].split(":")[0])
                ok &= first_t < cat
    # verdict agrees with the AAR / terminal rule, and the neglected run is a collapse
    ok &= (build_debrief(lo.feed)["outcome"] == "collapse") == (summarize_frames(lo.feed)["final_state"] in
          ("FRIENDLY_COLLAPSE", "MUTUAL_COLLAPSE"))
    ok &= build_debrief(lo.feed)["outcome"] == "collapse"

    # a timed-out decision is surfaced end to end (flag -> feed -> posture -> lesson)
    def expiring(prompt=""):
        M.timed_input.expired = True
        return "3"
    lg, _ = _mission(level="novice", policy=expiring, decision_timeout=5.0, n_ticks=60)
    dt = build_debrief(lg.feed)
    ok &= any(d["timed_out"] and d["posture"] == "timeout" for d in dt["decisions"])
    ok &= any(l["title"] == "Decide inside the window" for l in dt["lessons"])
    # edge inputs: nothing, no decisions, and an old file with no severity or events
    ok &= build_debrief([]) is None and "No debrief" in format_debrief(None)
    bare = [{"t": t, "theta_s": 0.8, "theta_c": 0.4, "theta_o": 0.1, "morale": 0.8, "nhp": 0.1,
             "alert": False, "mission_state": "STABLE"} for t in range(30)]
    db = build_debrief(bare)
    ok &= db["metric"] == "NHP" and any(l["title"] == "No decisions were prompted" for l in db["lessons"])
    # large force: no map frames are built, the feed still carries the debrief
    lg, sk = _mission(n_units=M.LARGE_FORCE_UNITS + 6, n_ticks=40)
    ok &= len(sk.frames) == 0 and build_debrief(lg.feed) is not None
    # serialisable, printable, and writable as the playback trailer
    text = format_debrief(d_feed)
    ok &= d_feed["headline"] in text and "WHAT TO TAKE AWAY" in text
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "frames.jsonl")
        sink.path = None
        with open(path, "w") as fh:
            for f in sink.frames:
                fh.write(_json.dumps(f) + "\n")
        append_end_record(path, summarize_frames(sink.frames), d_feed)
        last = _json.loads(open(path).read().strip().splitlines()[-1])
        ok &= last["type"] == "end" and last["debrief"]["headline"] == d_feed["headline"]
    return _ok("debrief (traceable effects, no blame during the shock, timeouts, edge inputs, trailer)", ok)


def test_scale_invariance():
    """Decisions must not change the force's TOTAL readiness just because N grew,
    and per-tick narrative text must stay bounded for large forces."""
    from maneuver_sim_interactive import describe_troop_state, redistribute_load

    ok = True
    for primary, peer in ((0.06, 0.03), (0.12, 0.06)):
        for skill in (0.6, 1.0, 1.8):
            totals = {}
            for n in (3, 6, 12, 48, 300):
                r = np.full(n, 0.7)
                totals[n] = float(redistribute_load(r, skill, primary, peer).sum() - r.sum())
            if max(abs(v) for v in totals.values()) > 1e-9:
                ok = False
                print(f"    redistribute not conservative (gain {primary}, skill {skill}): {totals}")

    # narrative length at large N must not scale with N
    enemy = EnemyForce()
    n_big = 768
    outcomes = np.tile(np.array([[10, 8, 6]]), (n_big, 1))  # every unit flagged
    narrative = describe_troop_state(
        50, np.full(n_big, 0.6), 0.6, outcomes, 1, 0.6, 1.0, "experienced", enemy
    )
    bounded = len(narrative) < 1500
    if not bounded:
        ok = False
        print(f"    narrative too long at N={n_big}: {len(narrative)} chars")
    return _ok("scale invariance (redistribute conserves total; narrative bounded)", ok)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("JOR 4.0 test suite")
    results = [
        test_intervention_factor(),
        test_matrix_scaling(),
        test_scale_invariance(),
        test_adapter_vectorized_equivalence(),
        test_vectorized_sampler_distribution(),
        test_large_force_logging(),
        test_strain_fraction(),
        test_sustained_collapse(),
        test_collapse_discriminates_policy(),
        test_scenarios_distinct(),
        test_aar_matches_terminal_rule(),
        test_severity_gauge(),
        test_debrief(),
        test_adapter_dual_form(),
        test_timed_input_wiring(),
        test_collapse_skill_independent(),
        test_skill_direction(),
        test_feedback_zero_noop(),
        test_self_consistency(),
        test_prompt_guards(),  # slowest — last
    ]
    print("\n" + "=" * 50)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"{passed}/{total} test groups passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
