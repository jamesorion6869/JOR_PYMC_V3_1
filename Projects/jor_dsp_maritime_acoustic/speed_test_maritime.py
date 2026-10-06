"""
JOR Maritime/Acoustic Concept Demo — Offline Speed Test
=======================================================
Measures pure fusion cost of the JOR Bayesian posterior recursion
(and optionally the full offline pipeline: scenario generation + fusion
+ detection).  No streaming wrapper, no plots, no hyperparameter search.

Designed so the numbers are easy to cite:

  - Per-scenario latency (ms)
  - Throughput (scenarios/s and steps/s)

CONCEPT DEMO only — sensor model is illustrative.
"""

import time
import platform
import sys
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Core constants (identical to the main demo)
# ---------------------------------------------------------------------------
TIME_STEPS = 60
P_PRIOR_NH_INITIAL = 0.20
P_PRIOR_H_INITIAL = 0.80
ASSUMED_OPERATOR_CONFIDENCE_C = 0.65
TS_BASELINE_DB = 3.0

DEFAULT_PARAMS = dict(
    W_C=0.40, W_E=0.30, W_P=0.30, K=0.20,
    PRIOR_RETENTION=0.70, POSTERIOR_ALPHA=0.30,
)

# Frozen operating point from the VAL=400 run (TUNED_PARAMS, not DEFAULT_PARAMS)
FROZEN_PARAMS = dict(
    W_C=0.30, W_E=0.10, W_P=0.60, K=0.10,
    PRIOR_RETENTION=0.55, POSTERIOR_ALPHA=0.40,
)
FROZEN_CONFIRM_THR = 0.53
FROZEN_DETECT = dict(sustain=3, rise_thr=0.04, rise_window=4)

TRANSITIONS = {
    "CALM_WATER":    {"CALM_WATER": 0.92, "ELEVATED_NOISE": 0.08},
    "ELEVATED_NOISE": {"ELEVATED_NOISE": 0.55, "HIGH_SEA_STATE": 0.15, "SETTLING": 0.30},
    "HIGH_SEA_STATE": {"HIGH_SEA_STATE": 0.55, "SETTLING": 0.45},
    "SETTLING":      {"SETTLING": 0.35, "CALM_WATER": 0.65},
}

OPERATING_FREQ_HZ = 5_000.0
SEA_STATE_BY_CONDITION = {
    "CALM_WATER": 0.5, "SETTLING": 1.5,
    "ELEVATED_NOISE": 2.5, "HIGH_SEA_STATE": 5.5,
}

def ambient_noise_level_db(freq_hz, sea_state):
    return 10.0 * np.log10(freq_hz ** (-5.0 / 3.0)) + 94.5 + 30.0 * np.log10(sea_state + 1.0)

_NL_BASELINE = ambient_noise_level_db(OPERATING_FREQ_HZ, SEA_STATE_BY_CONDITION["CALM_WATER"])

CONDITION_EFFECTS = {
    name: dict(
        snr_shift=round(_NL_BASELINE - ambient_noise_level_db(OPERATING_FREQ_HZ, ss), 2),
        var_mult=v, dropout_p=d, detp_shift=dp,
    )
    for name, ss, v, d, dp in [
        ("CALM_WATER",     SEA_STATE_BY_CONDITION["CALM_WATER"],     0.55, 0.008,  0.0),
        ("ELEVATED_NOISE", SEA_STATE_BY_CONDITION["ELEVATED_NOISE"], 1.5,  0.07,  -0.06),
        ("HIGH_SEA_STATE", SEA_STATE_BY_CONDITION["HIGH_SEA_STATE"], 3.0,  0.22,  -0.20),
        ("SETTLING",       SEA_STATE_BY_CONDITION["SETTLING"],       0.85, 0.02,  -0.02),
    ]
}

PHENOTYPES = ["A_STRONG_MULTI", "C_INTERMITTENT", "G_GRADUAL_ONSET", "E_CONFLICTING", "F_SPARSE"]
FALSE_CATEGORIES = ["BIOLOGICS", "SURFACE_BOTTOM_BOUNCE", "TRANSIENT_NOISE", "THERMOCLINE_INSTABILITY"]

# ---------------------------------------------------------------------------
# Minimal supporting functions
# ---------------------------------------------------------------------------
def step_condition(rng, state):
    r = rng.random()
    c = 0.0
    for k, p in TRANSITIONS[state].items():
        c += p
        if r < c:
            return k
    return state

class AR1:
    def __init__(self, phi, sigma):
        self.phi = phi
        self.sigma = sigma
        self.state = 0.0
    def step(self, rng, var_mult=1.0):
        innov = rng.normal(0, self.sigma * np.sqrt(var_mult))
        self.state = self.phi * self.state + innov
        return self.state

def make_ar_bank():
    return {
        "snr": AR1(0.60, 1.0),
        "ts": AR1(0.70, 0.30),
        "rl": AR1(0.60, 35.0),
        "range_res": AR1(0.50, 0.35),
        "doppler": AR1(0.50, 0.8),
        "kin_consistency": AR1(0.65, 0.06),
    }

def target_strength_fluctuation(rng, mean_ts):
    if rng.random() < 0.65:
        return max(0.45 * mean_ts, rng.exponential(mean_ts))
    return max(0.55 * mean_ts, mean_ts + rng.normal(0, 0.30 * mean_ts))

def sample_event_schedule(rng):
    events, cursor = [], 5
    for _ in range(rng.integers(0, 3)):
        onset = cursor + int(rng.integers(0, 6))
        duration = int(rng.integers(5, 12))
        if onset + duration > TIME_STEPS - 3:
            break
        phenotype = PHENOTYPES[rng.integers(0, len(PHENOTYPES))]
        events.append(dict(onset=onset, duration=duration, phenotype=phenotype))
        cursor = onset + duration + int(rng.integers(3, 10))
    return events

def sample_false_event_schedule(rng):
    events, cursor = [], 3
    for _ in range(rng.integers(0, 3)):
        onset = cursor + int(rng.integers(0, 8))
        duration = int(rng.integers(2, 6))
        if onset + duration > TIME_STEPS - 1:
            break
        category = FALSE_CATEGORIES[rng.integers(0, len(FALSE_CATEGORIES))]
        events.append(dict(onset=onset, duration=duration, category=category))
        cursor = onset + duration + int(rng.integers(2, 8))
    return events

def in_window(step, ev):
    return ev["onset"] <= step < ev["onset"] + ev["duration"]

def phenotype_effect(phenotype, progress):
    base = dict(ts=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0,
                extra_dropout=0.0, kin_consistency=0.0)
    if phenotype == "A_STRONG_MULTI":
        base.update(ts=2.5, snr=4.0, detp=0.15, trackc=0.05, kin_consistency=0.28)
    elif phenotype == "C_INTERMITTENT":
        on = (int(progress * 6) % 2 == 0)
        amp = 1.0 if on else 0.0
        base.update(ts=2.0*amp, snr=3.0*amp, detp=0.12*amp, kin_consistency=0.24*amp)
    elif phenotype == "G_GRADUAL_ONSET":
        amp = min(1.0, progress * 1.6)
        base.update(ts=2.2*amp, snr=3.5*amp, detp=0.13*amp, trackc=0.04*amp, kin_consistency=0.26*amp)
    elif phenotype == "E_CONFLICTING":
        base.update(ts=3.0, snr=-2.5, detp=-0.10, trackc=0.03, kin_consistency=0.30)
    elif phenotype == "F_SPARSE":
        base.update(ts=1.2, snr=0.8, detp=0.05, extra_dropout=0.40, trackc=0.04, kin_consistency=0.34)
    return base

def false_event_effect(category):
    if category == "BIOLOGICS":
        return dict(ts=1.8, snr=1.0, detp=0.05, trackc=-0.05, doppler=0.0, multipath=0.0, kin_consistency=-0.40)
    if category == "SURFACE_BOTTOM_BOUNCE":
        return dict(ts=0.3, snr=-1.0, detp=-0.03, trackc=-0.08, doppler=0.5, multipath=0.10, kin_consistency=-0.42)
    if category == "TRANSIENT_NOISE":
        return dict(ts=2.5, snr=2.0, detp=0.02, trackc=0.0, doppler=1.0, multipath=0.0, kin_consistency=-0.38)
    if category == "THERMOCLINE_INSTABILITY":
        return dict(ts=0.2, snr=-0.5, detp=-0.02, trackc=-0.15, doppler=0.3, multipath=0.0, kin_consistency=-0.44)
    return dict(ts=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0, multipath=0.0, kin_consistency=0.0)

class EnrichedDSP:
    def __init__(self, window=7, hist=5, persist_hist=12):
        self.buffers = {k: deque(maxlen=window) for k in ["snr", "rl", "doppler", "range_res"]}
        self.smooth_ts = None
        self.prev_innovation = 0.0
        self.innov_var = 0.15
        self.snr_hist = deque(maxlen=hist)
        self.innov_hist = deque(maxlen=hist)
        self.ts_hist = deque(maxlen=hist)
        self.detp_hist = deque(maxlen=hist)
        self.valid_hist = deque(maxlen=hist)
        self.persist_hist = deque(maxlen=persist_hist)

    def process(self, raw, ts_available, det_prob_obs):
        filtered = {}
        n_valid = n_possible = 0
        for k, buf in self.buffers.items():
            n_possible += 1
            if raw[k] is not None:
                buf.append(raw[k])
                n_valid += 1
            filtered[k] = float(np.mean(buf)) if buf else None
        if ts_available:
            n_valid += 1
        n_possible += 1
        if det_prob_obs is not None:
            n_valid += 1
        n_possible += 1
        valid_frac = n_valid / max(n_possible, 1)
        self.valid_hist.append(valid_frac)
        self.persist_hist.append(1.0 if n_valid > 0 else 0.0)

        innovation = 0.0
        if ts_available:
            if self.smooth_ts is None:
                self.smooth_ts = raw["ts"]
            else:
                innovation = raw["ts"] - self.smooth_ts
                gated = np.clip(innovation, -2.5 * np.sqrt(self.innov_var), 2.5 * np.sqrt(self.innov_var))
                a = 0.18
                self.smooth_ts = a * (self.smooth_ts + gated) + (1 - a) * self.smooth_ts
                self.innov_var = 0.85 * self.innov_var + 0.15 * (innovation ** 2)
            self.prev_innovation = 0.82 * self.prev_innovation + 0.18 * abs(innovation)
        filtered["ts"] = self.smooth_ts

        if self.smooth_ts:
            innov_norm = abs(self.prev_innovation) / (0.30 + 0.06 * abs(self.smooth_ts) + 1e-6)
        else:
            innov_norm = 0.0
        filtered["track_consistency"] = float(np.clip(0.95 - 0.25 * innov_norm, 0.55, 0.985))
        filtered["maneuver_index"] = float(np.clip(0.04 + 0.09 * self.prev_innovation, 0.0, 0.45))

        snr_val = filtered["snr"]
        if snr_val is not None:
            self.snr_hist.append(snr_val)
        self.innov_hist.append(abs(innovation))
        if filtered["ts"] is not None:
            self.ts_hist.append(filtered["ts"])
        if det_prob_obs is not None:
            self.detp_hist.append(det_prob_obs)

        snr_trend_n = float(np.clip(((self.snr_hist[-1] - self.snr_hist[0]) / max(len(self.snr_hist) - 1, 1)) / 3.0, -1.0, 1.0)) if len(self.snr_hist) >= 3 else 0.0
        innov_trend_n = float(np.clip(((self.innov_hist[-1] - self.innov_hist[0]) / max(len(self.innov_hist) - 1, 1)) / 0.5, -1.0, 1.0)) if len(self.innov_hist) >= 3 else 0.0

        consistency = 0.5
        if len(self.ts_hist) >= 3 and len(self.snr_hist) >= 3 and len(self.detp_hist) >= 3:
            d_ts = self.ts_hist[-1] - self.ts_hist[0]
            d_snr = self.snr_hist[-1] - self.snr_hist[0]
            d_det = self.detp_hist[-1] - self.detp_hist[0]
            signs = [np.sign(d_ts), np.sign(d_snr), np.sign(d_det)]
            consistency = max(sum(1 for s in signs if s > 0), sum(1 for s in signs if s < 0)) / 3.0

        filtered["snr_trend"] = snr_trend_n
        filtered["innov_trend"] = innov_trend_n
        filtered["consistency"] = float(consistency)
        filtered["valid_frac"] = float(np.mean(self.valid_hist)) if self.valid_hist else 1.0
        filtered["innov_mag"] = float(np.clip(self.prev_innovation / 0.8, 0.0, 1.0))
        filtered["persist_score"] = float(np.mean(self.persist_hist)) if self.persist_hist else 0.0
        return filtered

def normalize(value, v_min, v_max):
    if value is None:
        return None
    return (max(min(value, v_max), v_min) - v_min) / (v_max - v_min)

def wavg(components):
    num = sum(v * w for v, w in components if v is not None)
    den = sum(w for v, w in components if v is not None)
    return num / den if den > 0 else None

def generate_scenario(seed):
    ss = np.random.SeedSequence(seed)
    child_events, child_false, child_condition, child_sensor, child_dropout, child_kin = ss.spawn(6)
    rng_events = np.random.default_rng(child_events)
    rng_false = np.random.default_rng(child_false)
    rng_condition = np.random.default_rng(child_condition)
    rng_sensor = np.random.default_rng(child_sensor)
    rng_dropout = np.random.default_rng(child_dropout)
    rng_kin = np.random.default_rng(child_kin)

    events = sample_event_schedule(rng_events)
    false_events = sample_false_event_schedule(rng_false)
    ground_truth = np.zeros(TIME_STEPS, dtype=bool)
    for ev in events:
        for t in range(ev["onset"], min(ev["onset"] + ev["duration"], TIME_STEPS)):
            ground_truth[t] = True

    condition = "CALM_WATER"
    ar = make_ar_bank()
    dsp = EnrichedDSP()
    C_track, E_track, P_track = [], [], []

    for step in range(TIME_STEPS):
        condition = step_condition(rng_condition, condition)
        eff = CONDITION_EFFECTS[condition]

        active_true = next((e for e in events if in_window(step, e)), None)
        active_false = next((e for e in false_events if in_window(step, e)), None)

        shift = dict(ts=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0,
                     multipath=0.0, extra_dropout=0.0, kin_consistency=0.0)
        if active_true:
            progress = (step - active_true["onset"]) / max(active_true["duration"], 1)
            for k, v in phenotype_effect(active_true["phenotype"], progress).items():
                shift[k] = shift.get(k, 0.0) + v
        if active_false:
            for k, v in false_event_effect(active_false["category"]).items():
                shift[k] = shift.get(k, 0.0) + v

        p_drop = min(0.95, eff["dropout_p"] + shift.get("extra_dropout", 0.0))
        avail = {ch: (rng_dropout.random() > p_drop) for ch in ["snr", "ts", "rl", "doppler", "range_res", "detp"]}

        snr_db = 18.0 + eff["snr_shift"] + shift["snr"] + ar["snr"].step(rng_sensor, eff["var_mult"]) + rng_sensor.normal(0, 0.4)
        true_ts = target_strength_fluctuation(rng_sensor, TS_BASELINE_DB + shift["ts"])
        ts_val = max(0.15, true_ts + ar["ts"].step(rng_sensor, eff["var_mult"]))
        received_level = max(150.0, 1400.0 * (ts_val / TS_BASELINE_DB) * (10 ** ((snr_db - 18.0) / 10.0)) + ar["rl"].step(rng_sensor, eff["var_mult"]))
        doppler = max(1.0, 10.0 + shift["doppler"] + ar["doppler"].step(rng_sensor, eff["var_mult"]))
        range_res = max(5.0, 15.0 + ar["range_res"].step(rng_sensor, eff["var_mult"]))
        det_prob = float(np.clip(0.75 + eff["detp_shift"] + shift["detp"] + rng_sensor.normal(0, 0.02), 0.05, 0.99))
        multipath = float(np.clip(0.03 + shift.get("multipath", 0.0) + rng_sensor.normal(0, 0.01), 0.0, 1.0))
        kin_consistency_val = float(np.clip(0.55 + shift["kin_consistency"] + ar["kin_consistency"].step(rng_kin, 1.0), 0.0, 1.0))

        raw = {
            "snr": snr_db if avail["snr"] else None,
            "rl": received_level if avail["rl"] else None,
            "doppler": doppler if avail["doppler"] else None,
            "range_res": range_res if avail["range_res"] else None,
            "ts": ts_val if avail["ts"] else None,
        }
        rd = dsp.process(raw, avail["ts"], det_prob if avail["detp"] else None)
        track_consistency = float(np.clip(rd["track_consistency"] + shift["trackc"], 0.0, 1.0))

        C = float(np.clip(ASSUMED_OPERATOR_CONFIDENCE_C + rng_sensor.normal(0, 0.02), 0.30, 0.85))
        E = wavg([
            (normalize(rd["snr"], 10.0, 30.0), 0.30),
            (normalize(rd["rl"], 800.0, 2000.0), 0.20),
            (normalize(rd["ts"], 0.5, 10.0), 0.20),
            (1.0 - normalize(rd["range_res"], 10.0, 25.0) if rd["range_res"] is not None else None, 0.10),
            ((rd["snr_trend"] + 1.0) / 2.0, 0.10),
            (rd["valid_frac"], 0.10),
        ])
        P_raw = wavg([
            (det_prob if avail["detp"] else None, 0.09),
            (normalize(rd["ts"], 0.5, 10.0), 0.07),
            (normalize(rd["rl"], 800.0, 2000.0), 0.07),
            (track_consistency, 0.09),
            (1.0 - rd["maneuver_index"], 0.04),
            (1.0 - multipath, 0.04),
            (1.0 - normalize(rd["doppler"], 5.0, 25.0) if rd["doppler"] is not None else None, 0.03),
            (rd["consistency"], 0.07),
            (1.0 - rd["innov_mag"], 0.03),
            (rd["valid_frac"], 0.07),
            (rd["persist_score"], 0.08),
            (kin_consistency_val, 0.32),
        ])

        C_track.append(float(np.clip(C, 0.30, 0.85)))
        E_track.append(float(np.clip(E if E is not None else (E_track[-1] if E_track else 0.5), 0.30, 0.85)))
        P_track.append(float(np.clip(P_raw if P_raw is not None else (P_track[-1] if P_track else 0.5), 0.30, 0.95)))

    return {
        "C": np.array(C_track), "E": np.array(E_track), "P": np.array(P_track),
        "ground_truth": ground_truth, "events": events, "false_events": false_events,
    }

# ---------------------------------------------------------------------------
# JOR fusion core (unchanged math)
# ---------------------------------------------------------------------------
def new_jor_state():
    return dict(prior_NH=P_PRIOR_NH_INITIAL, prior_H=P_PRIOR_H_INITIAL, smooth_post=None)

def jor_step(state, C_t, E_t, P_t, params, mod_t=0.0):
    SOP = params["W_C"] * C_t + params["W_E"] * E_t + params["W_P"] * P_t
    P_for_NHP = min(max(P_t + mod_t, 0.0), 0.95)
    NHP = params["W_C"] * C_t + params["W_E"] * E_t + params["W_P"] * P_for_NHP
    P_E_given_NH = NHP
    P_E_given_H = min(max(1.0 - NHP + params["K"] * SOP, 0.0), 1.0)
    numerator = P_E_given_NH * state["prior_NH"]
    denominator = numerator + P_E_given_H * state["prior_H"]
    post = numerator / denominator if denominator > 0 else 0.0
    prev_smooth = state["smooth_post"]
    smooth_post = post if prev_smooth is None else params["POSTERIOR_ALPHA"] * post + (1 - params["POSTERIOR_ALPHA"]) * prev_smooth
    new_prior_NH = params["PRIOR_RETENTION"] * post + (1 - params["PRIOR_RETENTION"]) * P_PRIOR_NH_INITIAL
    new_state = dict(prior_NH=new_prior_NH, prior_H=1.0 - new_prior_NH, smooth_post=smooth_post)
    return new_state, smooth_post

def run_jor(scenario, params):
    C, E, P = scenario["C"], scenario["E"], scenario["P"]
    n = len(C)
    state = new_jor_state()
    posterior_track = np.empty(n)
    for t in range(n):
        state, post = jor_step(state, C[t], E[t], P[t], params)
        posterior_track[t] = post
    return posterior_track

def new_detector_state(sustain=3, rise_window=4):
    return dict(buf=deque(maxlen=max(sustain, rise_window)), i=-1, fired=False, fired_step=None)

def detector_step(state, post_t, level_thr, rise_thr=0.04, sustain=3, rise_window=4):
    state["buf"].append(post_t)
    state["i"] += 1
    i = state["i"]
    if state["fired"]:
        return state, True, state["fired_step"]
    if i < sustain - 1:
        return state, False, None
    buf = state["buf"]
    window_sustain = list(buf)[-sustain:]
    fire = min(window_sustain) > level_thr
    if not fire and i >= rise_window - 1 and len(buf) >= rise_window:
        window_rise = list(buf)
        fire = (window_rise[-1] - window_rise[-rise_window]) >= rise_thr and min(window_sustain) > (level_thr - 0.06)
    if fire:
        state["fired"] = True
        state["fired_step"] = i - sustain + 1
        return state, True, state["fired_step"]
    return state, False, None

def detect_temporal(posterior_track, level_thr, rise_thr=0.04, sustain=3, rise_window=4):
    state = new_detector_state(sustain, rise_window)
    for post_t in posterior_track:
        state, flagged, flag_step = detector_step(state, post_t, level_thr, rise_thr, sustain, rise_window)
        if flagged:
            return True, flag_step
    return False, None

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_pure_fusion(n_scenarios=500, n_repeats=3, seeds_start=30_000):
    """Time only the Bayesian recursion (scenarios pre-generated)."""
    seeds = list(range(seeds_start, seeds_start + n_scenarios))
    print(f"Pre-generating {n_scenarios} scenarios ...", end=" ", flush=True)
    t_gen0 = time.perf_counter()
    scenarios = [generate_scenario(s) for s in seeds]
    t_gen1 = time.perf_counter()
    print(f"done ({t_gen1 - t_gen0:.2f} s)")

    # Warm-up
    _ = run_jor(scenarios[0], FROZEN_PARAMS)

    times = []
    for rep in range(n_repeats):
        t0 = time.perf_counter()
        for sc in scenarios:
            _ = run_jor(sc, FROZEN_PARAMS)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    best = min(times)
    median = float(np.median(times))
    per_scen_ms = (best / n_scenarios) * 1000.0
    scen_per_s = n_scenarios / best
    steps_per_s = (n_scenarios * TIME_STEPS) / best

    print("\n--- Pure fusion (Bayesian posterior recursion only) ---")
    print(f"  Scenarios          : {n_scenarios}  ×  {TIME_STEPS} steps")
    print(f"  Best of {n_repeats} runs      : {best:.3f} s")
    print(f"  Median of {n_repeats} runs    : {median:.3f} s")
    print(f"  Per scenario       : {per_scen_ms:.2f} ms")
    print(f"  Throughput         : {scen_per_s:.0f} scenarios/s")
    print(f"  Step rate          : {steps_per_s:,.0f} steps/s")
    return dict(best=best, per_scen_ms=per_scen_ms, scen_per_s=scen_per_s, steps_per_s=steps_per_s)

def bench_full_offline(n_scenarios=300, seeds_start=40_000):
    """Generation + fusion + detection (closest to a batch offline job)."""
    seeds = list(range(seeds_start, seeds_start + n_scenarios))
    params = FROZEN_PARAMS
    thr = FROZEN_CONFIRM_THR
    detect_kw = FROZEN_DETECT

    # Warm-up
    sc0 = generate_scenario(seeds[0])
    _ = detect_temporal(run_jor(sc0, params), thr, **detect_kw)

    t0 = time.perf_counter()
    for s in seeds:
        sc = generate_scenario(s)
        post = run_jor(sc, params)
        _ = detect_temporal(post, thr, **detect_kw)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    per_scen_ms = (elapsed / n_scenarios) * 1000.0
    scen_per_s = n_scenarios / elapsed

    print("\n--- Full offline pipeline (generate + fuse + detect) ---")
    print(f"  Scenarios          : {n_scenarios}")
    print(f"  Wall time          : {elapsed:.3f} s")
    print(f"  Per scenario       : {per_scen_ms:.2f} ms")
    print(f"  Throughput         : {scen_per_s:.0f} scenarios/s")
    return dict(elapsed=elapsed, per_scen_ms=per_scen_ms, scen_per_s=scen_per_s)

def bench_single_step(n_calls=200_000):
    """Micro-benchmark of the innermost jor_step call."""
    state = new_jor_state()
    params = FROZEN_PARAMS
    # Warm-up
    for _ in range(1000):
        state, _ = jor_step(state, 0.65, 0.55, 0.60, params)

    state = new_jor_state()
    t0 = time.perf_counter()
    for i in range(n_calls):
        state, _ = jor_step(state, 0.65, 0.55, 0.60, params)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    us_per_call = (elapsed / n_calls) * 1e6
    calls_per_s = n_calls / elapsed

    print("\n--- Micro: single jor_step call ---")
    print(f"  Calls              : {n_calls:,}")
    print(f"  Wall time          : {elapsed:.3f} s")
    print(f"  Per call           : {us_per_call:.2f} µs")
    print(f"  Throughput         : {calls_per_s:,.0f} steps/s")
    return dict(us_per_call=us_per_call, calls_per_s=calls_per_s)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 62)
    print("JOR Maritime/Acoustic Concept Demo — Offline Speed Test")
    print("=" * 62)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"NumPy           : {np.__version__}")
    print(f"Platform        : {platform.platform()}")
    print(f"Processor       : {platform.processor() or 'n/a'}")
    print()
    print("Note: pure fusion cost only. No streaming wrapper, no plots,")
    print("      no hyperparameter search. CONCEPT DEMO numbers.")
    print()

    pure = bench_pure_fusion(n_scenarios=500, n_repeats=3)
    full = bench_full_offline(n_scenarios=300)
    micro = bench_single_step(n_calls=200_000)

    print("\n" + "=" * 62)
    print("Summary")
    print("=" * 62)
    print(f"  Pure fusion latency     : {pure['per_scen_ms']:.1f} ms / scenario")
    print(f"  Pure fusion throughput  : {pure['scen_per_s']:.0f} scenarios/s")
    print(f"  Step rate (fusion only) : {pure['steps_per_s']:,.0f} steps/s")
    print(f"  Full offline pipeline   : {full['per_scen_ms']:.1f} ms / scenario")
    print(f"  Single jor_step         : {micro['us_per_call']:.1f} µs")
    print()
    print("These numbers measure the offline Bayesian fusion core only.")
    print("They do not claim real-time sonar performance.")
    print("=" * 62)
