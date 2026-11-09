"""JOR vs CUSUM study v2.1 (pre-registered; conservative threshold rule)

Why v2.1
    In v2.0 (TEST 70000-70999) JOR's TEST FPR overshot the 0.12 budget
    (TRAIN 0.115 -> TEST 0.179). Picking the best of many weight/threshold
    settings by a TRAIN FPR point estimate favours settings that got lucky on
    TRAIN. v2.1 changes ONE thing: a setting is eligible only if the upper
    95% Wilson bound of its TRAIN FPR is at or below 0.12. The rule is applied
    identically to every method (JOR, C/E/P avg, SNR, both CUSUMs).
    Everything else is v2.0. TEST is a fresh block (80000-80999).
    Descriptive row (not a claim): plain CUSUM on C/E/P weighted with JOR's own
    tuned weights, grid-searched like CUSUM on C/E/P. It checks whether CUSUM was
    understated by running on weights tuned for averaging.
    Descriptive row (not a claim): C/E/P + memory -- the C/E/P average passed through
    the same exponential smoothing JOR applies to its posterior (alpha = 0.30), with
    its own weight search. It checks whether memory alone, without the Bayes step,
    explains JOR's lead. It has no analogue of JOR's prior retention.
    Housekeeping (no effect on results): a final threshold pick that falls back
    to the minimum-FPR setting is now flagged and listed with the claims; the
    script hash, like the library hash, is taken with LF line endings. An error
    traceback, if any, is written into the results file as well as the terminal.

What it reuses (unchanged) from jor_tuned_study_v1.1.py, imported as a library
    scenario generator, DSP, evidence mapping, JOR update, C/E/P average,
    detection rule, evaluation, weight search, threshold pick, Wilson interval,
    paired bootstrap. The locked study file is not modified.

Selection rule (TRAIN only)
    During weight search, threshold pick and CUSUM grid search, the FPR each
    candidate is judged by is the upper 95% Wilson bound of its TRAIN FPR.
    Reported FPRs (TRAIN, VAL, TEST) are ordinary point estimates.

Methods (all declared before the TEST run; all tuned on TRAIN only, FPR budget 0.12)
    JOR (tuned)            locked study's weight search + threshold pick
    C/E/P avg (tuned)      locked study's weight search + threshold pick
    SNR (CFAR excess)      locked study's native threshold sweep
    CUSUM on C/E/P         one-sided CUSUM on the tuned C/E/P stream;
                           grid over reference nu, barrier h, sustain, rise path
    CUSUM-detrended        one-sided CUSUM on (C/E/P minus a slow running mean);
                           grid over EWMA rate, reference nu, barrier h, sustain, rise path
    CUSUM on JOR weights*  plain CUSUM on JOR's tuned weight mix (no Bayes, no memory);
                           same grid as CUSUM on C/E/P. Descriptive, not a claim.
    C/E/P + memory*        C/E/P average smoothed by an EMA with JOR's posterior alpha (0.30);
                           own weight search (same 150 candidates, coarse grid) and threshold
                           pick, like C/E/P avg. Descriptive, not a claim.
    Rise path (all CUSUMs): level only, or rise >= {0.05, 0.10, 0.20, 0.40} over 4 steps
                           with margin {0.5, 1.0} x h -- the same detector form JOR uses.

Claims: see PREDECLARED_CLAIMS_CUSUM_v2.1.md (must match CLAIMS below).
Commit that file and this script (hashes printed at start) BEFORE the full run.

Flags
    --quick     development seeds + 20 candidates; NOT a result
    --dev       TEST = 20000-20999 (development seeds); NOT a result
    --jobs N    parallel weight search (fork platforms only; Windows runs serially)

Output
    Everything printed is also saved next to this script as
    jor_cusum_v2.1_results_<quick|dev|TEST80000-80999>_<date-time>.txt
    (a new file each run; nothing is overwritten).
"""

import hashlib
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load the locked study as a library (its own __main__ does not run)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "jor_tuned_study_v1.1.py"
if not ORIGINAL.exists():
    raise SystemExit("Cannot find jor_tuned_study_v1.1.py. Place it in the same folder as this script.")

_saved_argv = sys.argv[:]
# Pass only the mode flags through, so seeds and candidate counts come from the locked study
sys.argv = [str(ORIGINAL)] + [f for f in ("--quick", "--dev") if f in _saved_argv]
if "--jobs" in _saved_argv:
    sys.argv += ["--jobs", _saved_argv[_saved_argv.index("--jobs") + 1]]
spec = importlib.util.spec_from_file_location("jor_original", ORIGINAL)
jor = importlib.util.module_from_spec(spec)
sys.modules["jor_original"] = jor  # lets --jobs worker processes find the study's functions
spec.loader.exec_module(jor)
sys.argv = _saved_argv

QUICK = jor.QUICK
DEV_RUN = jor.DEV_RUN
QUICK_TAG = jor.QUICK_TAG
FPR_BUDGET = jor.FPR_BUDGET
SUSTAIN_STEPS = jor.SUSTAIN_STEPS

# ---------------------------------------------------------------------------
# Seeds: TRAIN / VAL as in the locked study; TEST is a fresh block
# ---------------------------------------------------------------------------

NEW_TEST_BLOCK = (80_000, 80_999)
USED_SEED_RANGES = list(jor.DEVELOPMENT_SEED_RANGES) + [(60_000, 60_999), (70_000, 70_999)]
# 60000s = locked study TEST; 70000s = v2.0 TEST

TRAIN_SEEDS = list(jor.TRAIN_SEEDS)
VAL_SEEDS = list(jor.VAL_SEEDS)
TEST_SEEDS = list(jor.TEST_SEEDS) if DEV_RUN else list(range(NEW_TEST_BLOCK[0], NEW_TEST_BLOCK[1] + 1))
ALL_SEEDS = TRAIN_SEEDS + VAL_SEEDS + TEST_SEEDS

# The locked study's helpers (weight search, evaluation) read these module globals.
jor.TRAIN_SEEDS, jor.VAL_SEEDS, jor.TEST_SEEDS, jor.ALL_SEEDS = TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS, ALL_SEEDS


def seed_checks():
    for name, seeds in (("TRAIN", TRAIN_SEEDS), ("VAL", VAL_SEEDS), ("TEST", TEST_SEEDS)):
        assert len(seeds) == len(set(seeds)), f"{name} contains duplicate seeds"
    assert not set(TRAIN_SEEDS) & set(VAL_SEEDS)
    assert not set(TRAIN_SEEDS) & set(TEST_SEEDS)
    assert not set(VAL_SEEDS) & set(TEST_SEEDS)
    print(f"Seed partition check: PASS (TRAIN {len(TRAIN_SEEDS)}, VAL {len(VAL_SEEDS)}, TEST {len(TEST_SEEDS)})")
    if DEV_RUN:
        print("Seed freeze check: " + ("QUICK MODE" if QUICK else "DEV RUN")
              + ". TEST uses development seeds. This is a code check, not a result.")
        return
    lo, hi = min(TEST_SEEDS), max(TEST_SEEDS)
    for a, b in USED_SEED_RANGES:
        if not (hi < a or lo > b):
            raise SystemExit(f"TEST block {lo}-{hi} overlaps a used seed range {a}-{b}")
    print(f"Seed freeze check: PASS (TEST block {lo}-{hi} is new; it overlaps no development seeds "
          f"and not the spent TEST blocks 60000-60999 (locked study) or 70000-70999 (v2.0))")


# ---------------------------------------------------------------------------
# CUSUM methods
# ---------------------------------------------------------------------------

def cusum_track(scores, nu):
    """One-sided CUSUM statistic S_t = max(0, S_{t-1} + x_t - nu)."""
    s = 0.0
    out = np.empty(len(scores))
    for i, x in enumerate(scores):
        s = max(0.0, s + (x - nu))
        out[i] = s
    return out


def detrended(scores, rate):
    """x_t minus an EWMA of earlier scores (removes slow drift in the score's level)."""
    scores = np.asarray(scores, dtype=float)
    out = np.empty(len(scores))
    m = scores[0]
    for i, x in enumerate(scores):
        out[i] = x - m
        m = (1.0 - rate) * m + rate * x
    return out


NU_GRID = [round(x, 2) for x in np.arange(0.30, 0.951, 0.05)]
NU_GRID_DETRENDED = [round(x, 3) for x in np.arange(0.0, 0.301, 0.025)]
H_GRID = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00]
SUSTAIN_GRID = (1, 2, 3, 4)
RATE_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
# v2.1: CUSUM may also fire on the same "fast rise near the barrier" path JOR and the averages use
# (detect_temporal's rise rule). Options: level only, or rise >= r over 4 steps while the window stays
# above h - f*h. Chosen on dev seeds after a fairness check (CUSUM had been level-only).
RISE_GRID = (None, 0.05, 0.10, 0.20, 0.40)
RISE_MARGIN_FRAC = (0.5, 1.0)
RISE_OPTIONS = [(None, 0.0)] + [(r, f) for r in RISE_GRID if r is not None for f in RISE_MARGIN_FRAC]


def cusum_config(h, sustain, rise=None, frac=0.0):
    # Same detector as JOR: level path, plus an optional rise path (margin scales with h).
    return dict(level_thr=h, rise_thr=rise, sustain=sustain, rise_window=4, rise_margin=frac * h)


class _FastCusumEval:
    """Vectorised twin of the locked detect_temporal + evaluate_tracks, used ONLY to search the
    CUSUM grid on TRAIN (the pure-Python loop is too slow for the rise-path grid). Every chosen
    setting is re-scored with the locked evaluator and must match exactly (asserted), and all
    reported numbers come from the locked evaluator."""

    def __init__(self, tracks, seeds, cache):
        self.seeds = list(seeds)
        self.T = np.vstack([np.asarray(tracks[s], dtype=float) for s in self.seeds])
        n = self.T.shape[1]
        self.has_event = np.array([jor.scenario_has_event(cache[s]["ground_truth"]) for s in self.seeds])
        self.events = [(cache[s]["ground_truth"], cache[s]["events"]) for s in self.seeds]
        self.memo = {}
        self.W = {}
        for k in SUSTAIN_GRID:
            w = np.full(self.T.shape, -np.inf)
            w[:, k - 1:] = np.lib.stride_tricks.sliding_window_view(self.T, k, axis=1).min(axis=2)
            self.W[k] = w
        self.R = np.full(self.T.shape, -np.inf)
        self.R[:, 3:] = self.T[:, 3:] - self.T[:, :-3]   # rise over rise_window = 4 steps
        self.steps = np.arange(n)

    def _correct(self, i, flag_step):
        key = (i, flag_step)
        if key not in self.memo:
            gt, ev = self.events[i]
            self.memo[key] = jor.flag_matches_truth_and_get_onset(flag_step, gt, ev)[0]
        return self.memo[key]

    def evaluate(self, config):
        k, h = config["sustain"], config["level_thr"]
        W = self.W[k]
        cond = W > h
        if config["rise_thr"] is not None:
            cond = cond | ((self.R >= config["rise_thr"]) & (W > (h - config["rise_margin"])))
        cond[:, :k - 1] = False
        flagged = cond.any(axis=1)
        first = cond.argmax(axis=1) - k + 1
        tp = fp = fn = tn = 0
        for i in range(len(self.seeds)):
            he = self.has_event[i]
            if flagged[i]:
                ok = self._correct(i, int(first[i]))
                if he and ok:
                    tp += 1
                elif he:
                    fn += 1
                else:
                    fp += 1
            elif he:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        if jor.evaluate_tracks is _wilson_selection_evaluate:
            fpr = jor.wilson_interval(fp, fp + tn)[1]
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=recall, f1=f1, fpr=fpr)


def tune_cusum(base_scores, nus, rates=(None,)):
    """Grid search on TRAIN; best TRAIN recall under the FPR budget (ties: F1, then smaller h)."""
    cache = _STATE["cache"]
    best = None
    for rate in rates:
        stream = base_scores if rate is None else {s: detrended(base_scores[s], rate) for s in TRAIN_SEEDS}
        for nu in nus:
            tracks = {s: cusum_track(stream[s], nu) for s in TRAIN_SEEDS}
            fast = _FastCusumEval(tracks, TRAIN_SEEDS, cache)
            for h in H_GRID:
                for sustain in SUSTAIN_GRID:
                    for rise, frac in RISE_OPTIONS:
                        res = fast.evaluate(cusum_config(h, sustain, rise, frac))
                        if res["fpr"] > FPR_BUDGET:
                            continue
                        key = (res["recall"], res["f1"], -h)
                        if best is None or key > best[0]:
                            best = (key, dict(rate=rate, nu=nu, h=h, sustain=sustain, rise=rise, frac=frac))
    if best is not None:
        # Exactness check: the chosen setting re-scored with the locked evaluator must match.
        p = best[1]
        stream = base_scores if p["rate"] is None else {s: detrended(base_scores[s], p["rate"]) for s in TRAIN_SEEDS}
        tracks = {s: cusum_track(stream[s], p["nu"]) for s in TRAIN_SEEDS}
        cfg = cusum_config(p["h"], p["sustain"], p["rise"], p["frac"])
        a = jor.evaluate_tracks(tracks, cfg, TRAIN_SEEDS, cache)
        b = _FastCusumEval(tracks, TRAIN_SEEDS, cache).evaluate(cfg)
        assert all(a[x] == b[x] for x in ("tp", "fp", "fn", "tn", "fpr")), "fast CUSUM evaluator mismatch"
    if best is None:
        raise SystemExit("No CUSUM setting met the FPR budget on TRAIN; widen the grid.")
    p = best[1]
    edges = []
    if p["nu"] in (min(nus), max(nus)):
        edges.append("nu")
    if p["h"] in (min(H_GRID), max(H_GRID)):
        edges.append("h")
    if p["rise"] is not None and p["rise"] == max(r for r in RISE_GRID if r is not None):
        edges.append("rise")
    if p["rate"] is not None and p["rate"] in (min(rates), max(rates)):
        edges.append("rate")
    p["edge"] = edges
    return p


_STATE = {}
FALLBACKS = []   # methods whose TRAIN selection fell back (reported with the claims)

# ---------------------------------------------------------------------------
# v2.1 selection rule: upper 95% Wilson bound of TRAIN FPR <= FPR_BUDGET
# ---------------------------------------------------------------------------

_LOCKED_EVALUATE = jor.evaluate_tracks


def _wilson_selection_evaluate(tracks, config, seeds, cache):
    """Same as the locked evaluate_tracks, except 'fpr' is the upper 95% Wilson bound.
    Used ONLY while choosing settings on TRAIN; never for reported numbers."""
    r = dict(_LOCKED_EVALUATE(tracks, config, seeds, cache))
    r["fpr_point"] = r["fpr"]
    r["fpr"] = jor.wilson_interval(r["fp"], r["fp"] + r["tn"])[1]
    return r


def selection_rule(on):
    jor.evaluate_tracks = _wilson_selection_evaluate if on else _LOCKED_EVALUATE

# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------

METHOD_NAMES = ("JOR (tuned)", "C/E/P avg (tuned)", "SNR (CFAR excess)", "CUSUM on C/E/P", "CUSUM-detrended",
                "CUSUM on JOR weights", "C/E/P + memory")
DESCRIPTIVE = ("CUSUM on JOR weights", "C/E/P + memory")   # reported, not part of any claim
MEMORY_ALPHA = jor.DEFAULT_PARAMS["POSTERIOR_ALPHA"]       # 0.30, the smoothing JOR applies to its posterior


def ema_track(x, alpha=MEMORY_ALPHA):
    """Exponential smoothing with the same recursion JOR uses for its posterior."""
    out = np.empty(len(x))
    s = None
    for i, v in enumerate(x):
        s = v if s is None else alpha * v + (1 - alpha) * s
        out[i] = s
    return out


def memory_weight_search(cache):
    """Weight search for C/E/P + memory, mirroring the locked library's search for C/E/P avg:
    same candidates, coarse threshold grid, same ranking (budget met, recall, F1, earliest)."""
    candidates = jor.make_weight_candidates()
    ranked = []
    for i, c in enumerate(candidates):
        tr = {s: ema_track(jor.cep_track(cache[s], c)) for s in TRAIN_SEEDS}
        pick = jor.pick_config(jor.sweep_configs(tr, TRAIN_SEEDS, cache, jor.COARSE_CONFIGS),
                               jor.COARSE_CONFIGS, FPR_BUDGET)
        ranked.append((pick["fpr"] <= FPR_BUDGET, pick["recall"], pick["f1"], -i))
    best = max(range(len(ranked)), key=lambda i: ranked[i])
    c = candidates[best]
    return {"W_C": c["W_C"], "W_E": c["W_E"], "W_P": c["W_P"]}


def run_condition(name, drifting):
    print(f"\n[{name}] generating scenarios ...", flush=True)
    t0 = time.time()
    cache = {s: jor.generate_scenario(s, drifting_floor=drifting) for s in ALL_SEEDS}
    _STATE["cache"] = cache

    selection_rule(True)   # all choices below, up to the results, are made on TRAIN with the Wilson rule
    tuned, _ = jor.run_weight_search(cache, name)
    jor.ACTIVE.update(jor=tuned["jor"], cep=tuned["cep"], ep_w=tuned["ep_w"])

    tracks, configs = {}, {}
    for m in ("JOR (tuned)", "C/E/P avg (tuned)", "SNR (CFAR excess)"):
        spec_m = jor.ALL_METHODS[m]
        tracks[m] = jor.build_tracks(spec_m["track"], ALL_SEEDS, cache)
        sweep = jor.sweep_configs(tracks[m], TRAIN_SEEDS, cache, spec_m["configs"])
        pick = jor.pick_config(sweep, spec_m["configs"], FPR_BUDGET)
        if pick["fpr"] > FPR_BUDGET:
            # The locked library silently falls back to the minimum-FPR setting; v2.1 flags it.
            FALLBACKS.append(f"{name}: {m}")
            print(f"    WARNING: {m} has no threshold meeting the selection rule on TRAIN; "
                  f"fell back to the minimum-FPR setting (Wilson upper bound {pick['fpr']:.3f}).", flush=True)
        configs[m] = dict(pick["config"], sustain=SUSTAIN_STEPS)

    # Descriptive: C/E/P + memory (EMA), own weight search and threshold pick on TRAIN.
    print("  C/E/P + memory weight search on TRAIN ...", flush=True)
    mem_w = memory_weight_search(cache)
    m = "C/E/P + memory"
    tracks[m] = {s: ema_track(jor.cep_track(cache[s], mem_w)) for s in ALL_SEEDS}
    unit = jor.unit_configs()
    pick = jor.pick_config(jor.sweep_configs(tracks[m], TRAIN_SEEDS, cache, unit), unit, FPR_BUDGET)
    if pick["fpr"] > FPR_BUDGET:
        FALLBACKS.append(f"{name}: {m}")
        print(f"    WARNING: {m} has no threshold meeting the selection rule on TRAIN; "
              f"fell back to the minimum-FPR setting (Wilson upper bound {pick['fpr']:.3f}).", flush=True)
    configs[m] = dict(pick["config"], sustain=SUSTAIN_STEPS)

    cep = tracks["C/E/P avg (tuned)"]
    print("  CUSUM search on TRAIN ...", flush=True)
    p_plain = tune_cusum({s: cep[s] for s in TRAIN_SEEDS}, NU_GRID)
    p_detr = tune_cusum({s: cep[s] for s in TRAIN_SEEDS}, NU_GRID_DETRENDED, RATE_GRID)
    # Descriptive: the same plain CUSUM, but on C/E/P weighted with JOR's own tuned weights
    # (answers "CUSUM was given weights tuned for averaging"). Not part of any claim.
    jw = {s: jor.cep_track(cache[s], tuned["jor"]) for s in ALL_SEEDS}
    p_jw = tune_cusum({s: jw[s] for s in TRAIN_SEEDS}, NU_GRID)
    for label, p in (("CUSUM on C/E/P", p_plain), ("CUSUM-detrended", p_detr), ("CUSUM on JOR weights", p_jw)):
        rate = "" if p["rate"] is None else f"rate={p['rate']:.2f} "
        print(f"    {label}: {rate}nu={p['nu']:.3f} h={p['h']:.2f} sustain={p['sustain']} {_rise_txt(p)}", flush=True)
        if p["edge"]:
            print(f"    WARNING: {label} optimum on grid edge ({', '.join(p['edge'])}). "
                  "Widen the grid in a dev run before the TEST run.", flush=True)

    tracks["CUSUM on C/E/P"] = {s: cusum_track(cep[s], p_plain["nu"]) for s in ALL_SEEDS}
    tracks["CUSUM-detrended"] = {
        s: cusum_track(detrended(cep[s], p_detr["rate"]), p_detr["nu"]) for s in ALL_SEEDS
    }
    tracks["CUSUM on JOR weights"] = {s: cusum_track(jw[s], p_jw["nu"]) for s in ALL_SEEDS}
    configs["CUSUM on JOR weights"] = cusum_config(p_jw["h"], p_jw["sustain"], p_jw["rise"], p_jw["frac"])
    configs["CUSUM on C/E/P"] = cusum_config(p_plain["h"], p_plain["sustain"], p_plain["rise"], p_plain["frac"])
    configs["CUSUM-detrended"] = cusum_config(p_detr["h"], p_detr["sustain"], p_detr["rise"], p_detr["frac"])
    selection_rule(False)  # reported numbers use ordinary point estimates
    assert jor.evaluate_tracks is _LOCKED_EVALUATE

    results, rows = {}, {}
    for m in METHOD_NAMES:
        results[m] = {
            split: jor.evaluate_tracks(tracks[m], configs[m], seeds, cache)
            for split, seeds in (("TRAIN", TRAIN_SEEDS), ("VAL", VAL_SEEDS), ("TEST", TEST_SEEDS))
        }
        rows[m] = jor.scenario_outcomes(tracks[m], configs[m], TEST_SEEDS, cache)

    pairs = [
        ("JOR (tuned)", "CUSUM on C/E/P"),
        ("JOR (tuned)", "CUSUM-detrended"),
        ("JOR (tuned)", "C/E/P avg (tuned)"),
        ("JOR (tuned)", "SNR (CFAR excess)"),
        ("CUSUM on C/E/P", "C/E/P avg (tuned)"),
        ("CUSUM-detrended", "C/E/P avg (tuned)"),
        ("JOR (tuned)", "CUSUM on JOR weights"),          # descriptive
        ("JOR (tuned)", "C/E/P + memory"),                # descriptive
    ]
    boots = {pair: jor.paired_bootstrap(rows[pair[0]], rows[pair[1]]) for pair in pairs}

    print(f"  condition done in {time.time() - t0:.0f}s", flush=True)
    return dict(name=name, tuned=tuned, cusum=dict(plain=p_plain, detr=p_detr, jw=p_jw), mem_w=mem_w,
                configs=configs, results=results, boots=boots)


def _rise_txt(p):
    return "level only" if p["rise"] is None else f"rise>={p['rise']:.2f} margin {p['frac']:.1f}h"


def report(c):
    print("\n" + "=" * 104)
    print(f"CONDITION: {c['name']}   (all settings chosen on TRAIN; FPR budget {FPR_BUDGET}){QUICK_TAG}")
    print("=" * 104)
    tj, tc = c["tuned"]["jor"], c["tuned"]["cep"]
    print(f"  JOR   W_C {tj['W_C']:.2f} W_E {tj['W_E']:.2f} W_P {tj['W_P']:.2f} K {tj['K']:.2f}   "
          f"C/E/P W_C {tc['W_C']:.2f} W_E {tc['W_E']:.2f} W_P {tc['W_P']:.2f}")
    pp, pd = c["cusum"]["plain"], c["cusum"]["detr"]
    print(f"  CUSUM nu {pp['nu']:.3f} h {pp['h']:.2f} sustain {pp['sustain']} {_rise_txt(pp)}")
    print(f"  CUSUM-detrended rate {pd['rate']:.2f} nu {pd['nu']:.3f} h {pd['h']:.2f} sustain {pd['sustain']} {_rise_txt(pd)}")
    pj = c["cusum"]["jw"]
    mw = c["mem_w"]
    print(f"  C/E/P + memory (descriptive) W_C {mw['W_C']:.2f} W_E {mw['W_E']:.2f} W_P {mw['W_P']:.2f} EMA alpha {MEMORY_ALPHA:.2f}")
    print(f"  CUSUM on JOR weights (descriptive) nu {pj['nu']:.3f} h {pj['h']:.2f} sustain {pj['sustain']} {_rise_txt(pj)}")
    print(f"\n{'Method':<22}{'TRAIN R / FPR':<16}{'VAL R / FPR':<16}{'TEST recall':>12}{'FPR':>8}{'F1':>8}{'Latency':>9}")
    for m in METHOD_NAMES:
        r = c["results"][m]
        lat = r["TEST"]["mean_latency"]
        print(f"{m + ('*' if m in DESCRIPTIVE else ''):<22}{r['TRAIN']['recall']:.3f} / {r['TRAIN']['fpr']:.3f}   "
              f"{r['VAL']['recall']:.3f} / {r['VAL']['fpr']:.3f}   "
              f"{r['TEST']['recall']:>12.3f}{r['TEST']['fpr']:>8.3f}{r['TEST']['f1']:>8.3f}"
              f"{('%.2f' % lat) if lat is not None else 'n/a':>9}")
    print("  * descriptive row: reported, not part of any pre-declared claim")
    print("\n  Paired bootstrap on TEST (2,000 resamples of scenarios), A minus B, 95% interval:")
    for (a, b), d in c["boots"].items():
        print(f"    {a} - {b}:")
        print(f"      Recall {d['recall'][0]:+.3f} [{d['recall'][1]:+.3f}, {d['recall'][2]:+.3f}]   "
              f"FPR {d['fpr'][0]:+.3f} [{d['fpr'][1]:+.3f}, {d['fpr'][2]:+.3f}]   "
              f"F1 {d['f1'][0]:+.3f} [{d['f1'][1]:+.3f}, {d['f1'][2]:+.3f}]")


# ---------------------------------------------------------------------------
# Pre-declared claims (must match PREDECLARED_CLAIMS_CUSUM_v2.1.md)
# ---------------------------------------------------------------------------

def check_claims(stat, drift):
    if DEV_RUN:
        print("\nPRE-DECLARED CLAIMS: not evaluated on development seeds." + QUICK_TAG)
        print("  Selection fallbacks: " + ("; ".join(FALLBACKS) if FALLBACKS else "none"))
        return

    def lo(c, a, b):
        return c["boots"][(a, b)]["recall"]

    claims = []
    for tag, c in (("Stationary", stat), ("Drifting", drift)):
        claims += [
            (f"{tag}: JOR recall exceeds CUSUM on C/E/P (CI lower bound > 0)",
             lo(c, "JOR (tuned)", "CUSUM on C/E/P")),
            (f"{tag}: JOR recall exceeds CUSUM-detrended (CI lower bound > 0)",
             lo(c, "JOR (tuned)", "CUSUM-detrended")),
            (f"{tag}: JOR recall exceeds C/E/P avg (CI lower bound > 0; replication on fresh seeds)",
             lo(c, "JOR (tuned)", "C/E/P avg (tuned)")),
        ]
    print("\n" + "=" * 104)
    print("PRE-DECLARED CLAIM CHECKS")
    print("=" * 104)
    held = 0
    for i, (text, d) in enumerate(claims, 1):
        ok = d[1] > 0
        held += ok
        print(f"  V{i}  {'HELD    ' if ok else 'NOT HELD'}  {text}")
        print(f"        point {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    fprs = (stat["results"]["JOR (tuned)"]["TEST"]["fpr"], drift["results"]["JOR (tuned)"]["TEST"]["fpr"])
    ok = max(fprs) <= 0.15
    held += ok
    print(f"  V7  {'HELD    ' if ok else 'NOT HELD'}  JOR TEST FPR at or below 0.15 in both conditions")
    print(f"        {fprs[0]:.3f}, {fprs[1]:.3f}")
    print(f"\n  {held} of 7 pre-declared claims held.")
    print("  Selection fallbacks: " + ("; ".join(FALLBACKS) if FALLBACKS else "none"))


class _Tee:
    """Copies everything printed to the terminal into the results file as it happens."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def open_results_file():
    mode = "quick" if QUICK else ("dev" if DEV_RUN else f"TEST{NEW_TEST_BLOCK[0]}-{NEW_TEST_BLOCK[1]}")
    path = HERE / f"jor_cusum_v2.1_results_{mode}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    handle = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, handle)
    return path, handle


def main():
    results_path, results_handle = open_results_file()
    try:
        _run()
    except SystemExit as e:
        print(f"\nSTOPPED: {e}")               # seed-overlap / CUSUM-grid stops land in the results file too
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)   # lands in the results file via the Tee
        raise
    finally:
        print(f"\nResults saved to {results_path.name}")
        sys.stdout = sys.__stdout__
        results_handle.close()


def _run():
    with open(__file__, "rb") as f:
        own_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    with open(ORIGINAL, "rb") as f:
        lib_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    print("=" * 104)
    print("JOR vs CUSUM study v2.1 (pre-registered; Wilson-bound threshold rule; fresh TEST block)")
    print(f"Selection rule: upper 95% Wilson bound of TRAIN FPR <= {FPR_BUDGET} (all methods)")
    print(f"Weight search: {jor.N_CANDIDATES} candidates, seed {jor.SEARCH_SEED}; jobs {jor.N_JOBS}")
    if DEV_RUN:
        print("QUICK MODE - not a result" if QUICK else "DEV RUN - not a result")
    print(f"Script SHA-256 (LF line endings): {own_hash}")
    print(f"Library jor_tuned_study_v1.1.py SHA-256 (LF line endings): {lib_hash}")
    print("=" * 104)
    seed_checks()

    stat = run_condition("STATIONARY", False)
    drift = run_condition("DRIFTING", True)
    report(stat)
    report(drift)
    check_claims(stat, drift)
    print("\n" + "=" * 104)
    print("Run complete." + QUICK_TAG)


if __name__ == "__main__":
    main()
