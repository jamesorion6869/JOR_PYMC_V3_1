"""JOR vs CUSUM study v2.0 (new pre-registered study)

Why a new study
    CUSUM was not part of the locked tuned-weights study
    (https://doi.org/10.5281/zenodo.23980949) and was designed after that
    study's TEST block (60000-60999) had been seen. To keep the comparison
    fair, every method here is declared up front and scored on a fresh TEST
    block (70000-70999) that no earlier work has touched.

What it reuses (unchanged) from jor_tuned_study_v1.1.py, imported as a library
    scenario generator, DSP, evidence mapping, JOR update, C/E/P average,
    detection rule, evaluation, weight search, threshold pick, paired bootstrap.
    The locked study file is not modified.

Methods (all declared before the TEST run; all tuned on TRAIN only, FPR budget 0.12)
    JOR (tuned)            locked study's weight search + threshold pick
    C/E/P avg (tuned)      locked study's weight search + threshold pick
    SNR (CFAR excess)      locked study's native threshold sweep
    CUSUM on C/E/P         one-sided CUSUM on the tuned C/E/P stream;
                           grid over reference nu, barrier h, sustain
    CUSUM-detrended        one-sided CUSUM on (C/E/P minus a slow running mean);
                           grid over EWMA rate, reference nu, barrier h, sustain

Claims: see PREDECLARED_CLAIMS_CUSUM_v2.md (must match CLAIMS below).
Commit that file and this script (hashes printed at start) BEFORE the full run.

Flags
    --quick     development seeds + 20 candidates; NOT a result
    --dev       TEST = 20000-20999 (development seeds); NOT a result
    --jobs N    parallel weight search (fork platforms)

Output
    Everything printed is also saved next to this script as
    jor_cusum_v2_results_<quick|dev|TEST70000-70999>_<date-time>.txt
    (a new file each run; nothing is overwritten).
"""

import hashlib
import importlib.util
import sys
import time
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

NEW_TEST_BLOCK = (70_000, 70_999)
USED_SEED_RANGES = list(jor.DEVELOPMENT_SEED_RANGES) + [(60_000, 60_999)]  # 60000s = locked study TEST

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
          f"and not the locked study's TEST block 60000-60999)")


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


def cusum_config(h, sustain):
    # No rise rule: the CUSUM statistic is already cumulative.
    return dict(level_thr=h, rise_thr=None, sustain=sustain, rise_window=4, rise_margin=0.0)


def tune_cusum(base_scores, nus, rates=(None,)):
    """Grid search on TRAIN; best TRAIN recall under the FPR budget (ties: F1, then smaller h)."""
    cache = _STATE["cache"]
    best = None
    for rate in rates:
        stream = base_scores if rate is None else {s: detrended(base_scores[s], rate) for s in TRAIN_SEEDS}
        for nu in nus:
            tracks = {s: cusum_track(stream[s], nu) for s in TRAIN_SEEDS}
            for h in H_GRID:
                for sustain in SUSTAIN_GRID:
                    res = jor.evaluate_tracks(tracks, cusum_config(h, sustain), TRAIN_SEEDS, cache)
                    if res["fpr"] > FPR_BUDGET:
                        continue
                    key = (res["recall"], res["f1"], -h)
                    if best is None or key > best[0]:
                        best = (key, dict(rate=rate, nu=nu, h=h, sustain=sustain))
    if best is None:
        raise SystemExit("No CUSUM setting met the FPR budget on TRAIN; widen the grid.")
    p = best[1]
    edges = []
    if p["nu"] in (min(nus), max(nus)):
        edges.append("nu")
    if p["h"] in (min(H_GRID), max(H_GRID)):
        edges.append("h")
    if p["rate"] is not None and p["rate"] in (min(rates), max(rates)):
        edges.append("rate")
    p["edge"] = edges
    return p


_STATE = {}

# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------

METHOD_NAMES = ("JOR (tuned)", "C/E/P avg (tuned)", "SNR (CFAR excess)", "CUSUM on C/E/P", "CUSUM-detrended")


def run_condition(name, drifting):
    print(f"\n[{name}] generating scenarios ...", flush=True)
    t0 = time.time()
    cache = {s: jor.generate_scenario(s, drifting_floor=drifting) for s in ALL_SEEDS}
    _STATE["cache"] = cache

    tuned, _ = jor.run_weight_search(cache, name)
    jor.ACTIVE.update(jor=tuned["jor"], cep=tuned["cep"], ep_w=tuned["ep_w"])

    tracks, configs = {}, {}
    for m in ("JOR (tuned)", "C/E/P avg (tuned)", "SNR (CFAR excess)"):
        spec_m = jor.ALL_METHODS[m]
        tracks[m] = jor.build_tracks(spec_m["track"], ALL_SEEDS, cache)
        sweep = jor.sweep_configs(tracks[m], TRAIN_SEEDS, cache, spec_m["configs"])
        configs[m] = dict(jor.pick_config(sweep, spec_m["configs"], FPR_BUDGET)["config"], sustain=SUSTAIN_STEPS)

    cep = tracks["C/E/P avg (tuned)"]
    print("  CUSUM search on TRAIN ...", flush=True)
    p_plain = tune_cusum({s: cep[s] for s in TRAIN_SEEDS}, NU_GRID)
    p_detr = tune_cusum({s: cep[s] for s in TRAIN_SEEDS}, NU_GRID_DETRENDED, RATE_GRID)
    for label, p in (("CUSUM on C/E/P", p_plain), ("CUSUM-detrended", p_detr)):
        rate = "" if p["rate"] is None else f"rate={p['rate']:.2f} "
        print(f"    {label}: {rate}nu={p['nu']:.3f} h={p['h']:.2f} sustain={p['sustain']}", flush=True)
        if p["edge"]:
            print(f"    WARNING: {label} optimum on grid edge ({', '.join(p['edge'])}). "
                  "Widen the grid in a dev run before the TEST run.", flush=True)

    tracks["CUSUM on C/E/P"] = {s: cusum_track(cep[s], p_plain["nu"]) for s in ALL_SEEDS}
    tracks["CUSUM-detrended"] = {
        s: cusum_track(detrended(cep[s], p_detr["rate"]), p_detr["nu"]) for s in ALL_SEEDS
    }
    configs["CUSUM on C/E/P"] = cusum_config(p_plain["h"], p_plain["sustain"])
    configs["CUSUM-detrended"] = cusum_config(p_detr["h"], p_detr["sustain"])

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
    ]
    boots = {pair: jor.paired_bootstrap(rows[pair[0]], rows[pair[1]]) for pair in pairs}

    print(f"  condition done in {time.time() - t0:.0f}s", flush=True)
    return dict(name=name, tuned=tuned, cusum=dict(plain=p_plain, detr=p_detr),
                configs=configs, results=results, boots=boots)


def report(c):
    print("\n" + "=" * 104)
    print(f"CONDITION: {c['name']}   (all settings chosen on TRAIN; FPR budget {FPR_BUDGET}){QUICK_TAG}")
    print("=" * 104)
    tj, tc = c["tuned"]["jor"], c["tuned"]["cep"]
    print(f"  JOR   W_C {tj['W_C']:.2f} W_E {tj['W_E']:.2f} W_P {tj['W_P']:.2f} K {tj['K']:.2f}   "
          f"C/E/P W_C {tc['W_C']:.2f} W_E {tc['W_E']:.2f} W_P {tc['W_P']:.2f}")
    pp, pd = c["cusum"]["plain"], c["cusum"]["detr"]
    print(f"  CUSUM nu {pp['nu']:.3f} h {pp['h']:.2f} sustain {pp['sustain']}   "
          f"CUSUM-detrended rate {pd['rate']:.2f} nu {pd['nu']:.3f} h {pd['h']:.2f} sustain {pd['sustain']}")
    print(f"\n{'Method':<22}{'TRAIN R / FPR':<16}{'VAL R / FPR':<16}{'TEST recall':>12}{'FPR':>8}{'F1':>8}{'Latency':>9}")
    for m in METHOD_NAMES:
        r = c["results"][m]
        lat = r["TEST"]["mean_latency"]
        print(f"{m:<22}{r['TRAIN']['recall']:.3f} / {r['TRAIN']['fpr']:.3f}   "
              f"{r['VAL']['recall']:.3f} / {r['VAL']['fpr']:.3f}   "
              f"{r['TEST']['recall']:>12.3f}{r['TEST']['fpr']:>8.3f}{r['TEST']['f1']:>8.3f}"
              f"{('%.2f' % lat) if lat is not None else 'n/a':>9}")
    print("\n  Paired bootstrap on TEST (2,000 resamples of scenarios), A minus B, 95% interval:")
    for (a, b), d in c["boots"].items():
        print(f"    {a} - {b}:")
        print(f"      Recall {d['recall'][0]:+.3f} [{d['recall'][1]:+.3f}, {d['recall'][2]:+.3f}]   "
              f"FPR {d['fpr'][0]:+.3f} [{d['fpr'][1]:+.3f}, {d['fpr'][2]:+.3f}]   "
              f"F1 {d['f1'][0]:+.3f} [{d['f1'][1]:+.3f}, {d['f1'][2]:+.3f}]")


# ---------------------------------------------------------------------------
# Pre-declared claims (must match PREDECLARED_CLAIMS_CUSUM_v2.md)
# ---------------------------------------------------------------------------

def check_claims(stat, drift):
    if DEV_RUN:
        print("\nPRE-DECLARED CLAIMS: not evaluated on development seeds." + QUICK_TAG)
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
    path = HERE / f"jor_cusum_v2_results_{mode}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    handle = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, handle)
    return path, handle


def main():
    results_path, results_handle = open_results_file()
    try:
        _run()
    finally:
        print(f"\nResults saved to {results_path.name}")
        sys.stdout = sys.__stdout__
        results_handle.close()


def _run():
    with open(__file__, "rb") as f:
        own_hash = hashlib.sha256(f.read()).hexdigest()
    with open(ORIGINAL, "rb") as f:
        lib_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    print("=" * 104)
    print("JOR vs CUSUM study v2.0 (new pre-registered study; fresh TEST block)")
    print(f"Weight search: {jor.N_CANDIDATES} candidates, seed {jor.SEARCH_SEED}; jobs {jor.N_JOBS}")
    if DEV_RUN:
        print("QUICK MODE - not a result" if QUICK else "DEV RUN - not a result")
    print(f"Script SHA-256: {own_hash}")
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
