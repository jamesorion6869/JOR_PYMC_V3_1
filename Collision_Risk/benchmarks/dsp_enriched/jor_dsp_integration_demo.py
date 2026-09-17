"""
JOR V3.1-enriched -- Full-Scale Baseline & Robustness Run (Final Version with Foxes)
---------------------------------------------------------------------------
Compliant with full development freeze and evaluation protocol:
  - TRAIN: 1,000 scenarios (Seeds 0-999)
  - VAL:     500 scenarios (Seeds 10000-10499)
  - TEST:  1,000 scenarios (Seeds 20000-20999)
  - Includes strict seed integrity checks and precise distance-based latency matching.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import time

TIME_STEPS = 60
P_PRIOR_NH_INITIAL = 0.20
P_PRIOR_H_INITIAL = 0.80
ASSUMED_WITNESS_C = 0.65

# --- Scaled & Isolated Partitions ---
TRAIN_SEEDS = list(range(0, 1000))
VAL_SEEDS = list(range(10_000, 10_500))
TEST_SEEDS = list(range(20_000, 21_000))

SUSTAIN_STEPS = 3
LAG_ALLOWANCE = 3
FPR_BUDGET = 0.12

DEFAULT_PARAMS = dict(W_C=0.40, W_E=0.30, W_P=0.30, K=0.20,
                       PRIOR_RETENTION=0.70, POSTERIOR_ALPHA=0.30)

TRANSITIONS = {
    "NOMINAL":           {"NOMINAL": 0.92, "DEGRADED": 0.08},
    "DEGRADED":          {"DEGRADED": 0.55, "SEVERELY_DEGRADED": 0.15, "RECOVERING": 0.30},
    "SEVERELY_DEGRADED": {"SEVERELY_DEGRADED": 0.55, "RECOVERING": 0.45},
    "RECOVERING":        {"RECOVERING": 0.35, "NOMINAL": 0.65},
}
CONDITION_EFFECTS = {
    "NOMINAL":           dict(snr_shift=0.0, var_mult=0.55, dropout_p=0.008, detp_shift=0.0),
    "DEGRADED":          dict(snr_shift=-2.5, var_mult=1.5, dropout_p=0.07, detp_shift=-0.06),
    "SEVERELY_DEGRADED": dict(snr_shift=-7.0, var_mult=3.0, dropout_p=0.22, detp_shift=-0.20),
    "RECOVERING":        dict(snr_shift=-1.0, var_mult=0.85, dropout_p=0.02, detp_shift=-0.02),
}

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
        "snr": AR1(phi=0.60, sigma=1.0),
        "rcs": AR1(phi=0.70, sigma=0.30),
        "power": AR1(phi=0.60, sigma=35.0),
        "range_res": AR1(phi=0.50, sigma=0.35),
        "doppler": AR1(phi=0.50, sigma=0.8),
    }

def swerling_rcs(rng, mean_rcs):
    if rng.random() < 0.65:
        return max(0.45 * mean_rcs, rng.exponential(mean_rcs))
    return max(0.55 * mean_rcs, mean_rcs + rng.normal(0, 0.30 * mean_rcs))

PHENOTYPES = ["A_STRONG_MULTI", "C_INTERMITTENT", "G_GRADUAL_ONSET", "E_CONFLICTING", "F_SPARSE"]

def sample_event_schedule(rng):
    events = []
    cursor = 5
    n_events = rng.integers(0, 3)
    for _ in range(n_events):
        onset = cursor + int(rng.integers(0, 6))
        duration = int(rng.integers(5, 12))
        if onset + duration > TIME_STEPS - 3:
            break
        phenotype = PHENOTYPES[rng.integers(0, len(PHENOTYPES))]
        events.append(dict(onset=onset, duration=duration, phenotype=phenotype))
        cursor = onset + duration + int(rng.integers(3, 10))
    return events

FALSE_CATEGORIES = ["CLUTTER", "MULTIPATH", "TRANSIENT_SPIKE", "TRACK_INSTABILITY"]

def sample_false_event_schedule(rng):
    events = []
    cursor = 3
    n = rng.integers(0, 3)
    for _ in range(n):
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
    base = dict(rcs=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0, extra_dropout=0.0)
    if phenotype == "A_STRONG_MULTI":
        base.update(rcs=2.5, snr=4.0, detp=0.15, trackc=0.05)
    elif phenotype == "C_INTERMITTENT":
        on = (int(progress * 6) % 2 == 0)
        amp = 1.0 if on else 0.0
        base.update(rcs=2.0 * amp, snr=3.0 * amp, detp=0.12 * amp)
    elif phenotype == "G_GRADUAL_ONSET":
        amp = min(1.0, progress * 1.6)
        base.update(rcs=2.2 * amp, snr=3.5 * amp, detp=0.13 * amp, trackc=0.04 * amp)
    elif phenotype == "E_CONFLICTING":
        base.update(rcs=3.0, snr=-2.5, detp=-0.10, trackc=0.03)
    elif phenotype == "F_SPARSE":
        base.update(rcs=1.0, snr=0.5, detp=0.03, extra_dropout=0.55)
    return base

def false_event_effect(category):
    if category == "CLUTTER":
        return dict(rcs=1.8, snr=1.0, detp=0.05, trackc=-0.05, doppler=0.0, multipath=0.0)
    if category == "MULTIPATH":
        return dict(rcs=0.3, snr=-1.0, detp=-0.03, trackc=-0.08, doppler=0.5, multipath=0.10)
    if category == "TRANSIENT_SPIKE":
        return dict(rcs=2.5, snr=2.0, detp=0.02, trackc=0.0, doppler=1.0, multipath=0.0)
    if category == "TRACK_INSTABILITY":
        return dict(rcs=0.2, snr=-0.5, detp=-0.02, trackc=-0.15, doppler=0.3, multipath=0.0)
    return dict(rcs=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0, multipath=0.0)

class EnrichedDSP:
    def __init__(self, window=7, hist=5):
        self.buffers = {k: deque(maxlen=window) for k in
                         ["snr", "power", "doppler", "range_res"]}
        self.smooth_rcs = None
        self.prev_innovation = 0.0
        self.innov_var = 0.15
        self.snr_hist = deque(maxlen=hist)
        self.innov_hist = deque(maxlen=hist)
        self.rcs_hist = deque(maxlen=hist)
        self.detp_hist = deque(maxlen=hist)
        self.valid_hist = deque(maxlen=hist)

    def process(self, raw, rcs_available, det_prob_obs):
        filtered = {}
        n_valid = 0
        n_possible = 0
        for k, buf in self.buffers.items():
            n_possible += 1
            if raw[k] is not None:
                buf.append(raw[k])
                n_valid += 1
            filtered[k] = float(np.mean(buf)) if buf else None
        if rcs_available:
            n_valid += 1
        n_possible += 1
        if det_prob_obs is not None:
            n_valid += 1
        n_possible += 1
        valid_frac = n_valid / max(n_possible, 1)
        self.valid_hist.append(valid_frac)

        innovation = 0.0
        if rcs_available:
            if self.smooth_rcs is None:
                self.smooth_rcs = raw["rcs"]
            else:
                innovation = raw["rcs"] - self.smooth_rcs
                gated = np.clip(innovation, -2.5 * np.sqrt(self.innov_var), 2.5 * np.sqrt(self.innov_var))
                a = 0.18
                self.smooth_rcs = a * (self.smooth_rcs + gated) + (1 - a) * self.smooth_rcs
                self.innov_var = 0.85 * self.innov_var + 0.15 * (innovation ** 2)
            self.prev_innovation = 0.82 * self.prev_innovation + 0.18 * abs(innovation)
        filtered["rcs"] = self.smooth_rcs

        if self.smooth_rcs:
            innov_norm = abs(self.prev_innovation) / (0.30 + 0.06 * abs(self.smooth_rcs) + 1e-6)
        else:
            innov_norm = 0.0
        filtered["track_consistency"] = float(np.clip(0.95 - 0.25 * innov_norm, 0.55, 0.985))
        filtered["maneuver_index"] = float(np.clip(0.04 + 0.09 * self.prev_innovation, 0.0, 0.45))

        snr_val = filtered["snr"]
        if snr_val is not None:
            self.snr_hist.append(snr_val)
        self.innov_hist.append(abs(innovation))
        if filtered["rcs"] is not None:
            self.rcs_hist.append(filtered["rcs"])
        if det_prob_obs is not None:
            self.detp_hist.append(det_prob_obs)

        snr_trend_n = float(np.clip(((self.snr_hist[-1] - self.snr_hist[0]) / max(len(self.snr_hist) - 1, 1)) / 3.0, -1.0, 1.0)) if len(self.snr_hist) >= 3 else 0.0
        innov_trend_n = float(np.clip(((self.innov_hist[-1] - self.innov_hist[0]) / max(len(self.innov_hist) - 1, 1)) / 0.5, -1.0, 1.0)) if len(self.innov_hist) >= 3 else 0.0

        consistency = 0.5
        if len(self.rcs_hist) >= 3 and len(self.snr_hist) >= 3 and len(self.detp_hist) >= 3:
            d_rcs = self.rcs_hist[-1] - self.rcs_hist[0]
            d_snr = self.snr_hist[-1] - self.snr_hist[0]
            d_det = self.detp_hist[-1] - self.detp_hist[0]
            signs = [np.sign(d_rcs), np.sign(d_snr), np.sign(d_det)]
            consistency = max(sum(1 for s in signs if s > 0), sum(1 for s in signs if s < 0)) / 3.0

        filtered["snr_trend"] = snr_trend_n
        filtered["innov_trend"] = innov_trend_n
        filtered["consistency"] = float(consistency)
        filtered["valid_frac"] = float(np.mean(self.valid_hist)) if self.valid_hist else 1.0
        filtered["innov_mag"] = float(np.clip(self.prev_innovation / 0.8, 0.0, 1.0))
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
    child_events, child_false, child_condition, child_sensor, child_dropout = ss.spawn(5)
    rng_events = np.random.default_rng(child_events)
    rng_false = np.random.default_rng(child_false)
    rng_condition = np.random.default_rng(child_condition)
    rng_sensor = np.random.default_rng(child_sensor)
    rng_dropout = np.random.default_rng(child_dropout)

    events = sample_event_schedule(rng_events)
    false_events = sample_false_event_schedule(rng_false)
    ground_truth = np.zeros(TIME_STEPS, dtype=bool)
    for ev in events:
        for t in range(ev["onset"], min(ev["onset"] + ev["duration"], TIME_STEPS)):
            ground_truth[t] = True

    condition = "NOMINAL"
    ar = make_ar_bank()
    dsp = EnrichedDSP()

    C_track, E_track, P_track = [], [], []
    phenotype_track = [None] * TIME_STEPS

    for step in range(TIME_STEPS):
        condition = step_condition(rng_condition, condition)
        eff = CONDITION_EFFECTS[condition]

        active_true = next((e for e in events if in_window(step, e)), None)
        active_false = next((e for e in false_events if in_window(step, e)), None)
        if active_true:
            phenotype_track[step] = active_true["phenotype"]

        shift = dict(rcs=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0, multipath=0.0, extra_dropout=0.0)
        if active_true:
            progress = (step - active_true["onset"]) / max(active_true["duration"], 1)
            for k, v in phenotype_effect(active_true["phenotype"], progress).items():
                shift[k] = shift.get(k, 0.0) + v
        if active_false:
            for k, v in false_event_effect(active_false["category"]).items():
                shift[k] = shift.get(k, 0.0) + v

        p_drop = min(0.95, eff["dropout_p"] + shift.get("extra_dropout", 0.0))
        avail = {ch: (rng_dropout.random() > p_drop) for ch in ["snr", "rcs", "power", "doppler", "range_res", "detp"]}

        snr_db = 18.0 + eff["snr_shift"] + shift["snr"] + ar["snr"].step(rng_sensor, eff["var_mult"]) + rng_sensor.normal(0, 0.4)
        true_rcs = swerling_rcs(rng_sensor, 3.0 + shift["rcs"])
        rcs_val = max(0.15, true_rcs + ar["rcs"].step(rng_sensor, eff["var_mult"]))
        peak_power = max(150.0, 1400.0 * (rcs_val / 3.0) * (10 ** ((snr_db - 18.0) / 10.0)) + ar["power"].step(rng_sensor, eff["var_mult"]))
        doppler = max(1.0, 10.0 + shift["doppler"] + ar["doppler"].step(rng_sensor, eff["var_mult"]))
        range_res = max(5.0, 15.0 + ar["range_res"].step(rng_sensor, eff["var_mult"]))
        det_prob = float(np.clip(0.75 + eff["detp_shift"] + shift["detp"] + rng_sensor.normal(0, 0.02), 0.05, 0.99))
        multipath = float(np.clip(0.03 + shift.get("multipath", 0.0) + rng_sensor.normal(0, 0.01), 0.0, 1.0))

        raw = {
            "snr": snr_db if avail["snr"] else None,
            "power": peak_power if avail["power"] else None,
            "doppler": doppler if avail["doppler"] else None,
            "range_res": range_res if avail["range_res"] else None,
            "rcs": rcs_val if avail["rcs"] else None,
        }
        rd = dsp.process(raw, avail["rcs"], det_prob if avail["detp"] else None)
        track_consistency = float(np.clip(rd["track_consistency"] + shift["trackc"], 0.0, 1.0))
        C = float(np.clip(ASSUMED_WITNESS_C + rng_sensor.normal(0, 0.02), 0.30, 0.85))

        E = wavg([
            (normalize(rd["snr"], 10.0, 30.0), 0.30),
            (normalize(rd["power"], 800.0, 2000.0), 0.20),
            (normalize(rd["rcs"], 0.5, 10.0), 0.20),
            (1.0 - normalize(rd["range_res"], 10.0, 25.0) if rd["range_res"] is not None else None, 0.10),
            ((rd["snr_trend"] + 1.0) / 2.0, 0.10),
            (rd["valid_frac"], 0.10),
        ])

        P_raw = wavg([
            (det_prob if avail["detp"] else None, 0.18),
            (normalize(rd["rcs"], 0.5, 10.0), 0.12),
            (normalize(rd["power"], 800.0, 2000.0), 0.12),
            (track_consistency, 0.15),
            (1.0 - rd["maneuver_index"], 0.08),
            (1.0 - multipath, 0.08),
            (1.0 - normalize(rd["doppler"], 5.0, 25.0) if rd["doppler"] is not None else None, 0.05),
            (rd["consistency"], 0.12),
            (1.0 - rd["innov_mag"], 0.05),
            (rd["valid_frac"], 0.05),
        ])

        C_track.append(float(np.clip(C, 0.30, 0.85)))
        E_track.append(float(np.clip(E if E is not None else (E_track[-1] if E_track else 0.5), 0.30, 0.85)))
        P_track.append(float(np.clip(P_raw if P_raw is not None else (P_track[-1] if P_track else 0.5), 0.30, 0.95)))

    return {
        "C": np.array(C_track), "E": np.array(E_track), "P": np.array(P_track),
        "ground_truth": ground_truth, "events": events, "false_events": false_events,
        "phenotype_track": phenotype_track,
    }

def run_jor(scenario, params, modifier_track=None):
    # Restored per the original JOR paper's design: NHP is a distinct
    # quantity from SOP, conditionally boosted by a per-step modifier
    # (the Flight Characteristics Modifier in the paper's worked
    # examples -- e.g. Aguadilla uses P=0.91 for SOP but P=0.95 for
    # NHP). modifier_track defaults to all-zero, which makes NHP == SOP
    # exactly -- identical to this function's previous behavior for
    # every experiment run so far (all of which use modifier=0 by
    # design, per the JOR-blind evaluation protocol). This change has
    # NO effect on any existing result; it only re-enables a code path
    # that wasn't being used.
    prior_NH, prior_H = P_PRIOR_NH_INITIAL, P_PRIOR_H_INITIAL
    smooth_post = None
    posterior_track = np.empty(TIME_STEPS)
    C, E, P = scenario["C"], scenario["E"], scenario["P"]
    mod = modifier_track if modifier_track is not None else np.zeros(TIME_STEPS)
    for t in range(TIME_STEPS):
        SOP = params["W_C"] * C[t] + params["W_E"] * E[t] + params["W_P"] * P[t]
        P_for_NHP = min(max(P[t] + mod[t], 0.0), 0.95)
        NHP = params["W_C"] * C[t] + params["W_E"] * E[t] + params["W_P"] * P_for_NHP
        P_E_given_NH = NHP
        P_E_given_H = min(max(1.0 - NHP + params["K"] * SOP, 0.0), 1.0)
        numerator = P_E_given_NH * prior_NH
        denominator = numerator + P_E_given_H * prior_H
        post = numerator / denominator if denominator > 0 else 0.0
        smooth_post = post if smooth_post is None else params["POSTERIOR_ALPHA"] * post + (1 - params["POSTERIOR_ALPHA"]) * smooth_post
        posterior_track[t] = smooth_post
        prior_NH = params["PRIOR_RETENTION"] * post + (1 - params["PRIOR_RETENTION"]) * P_PRIOR_NH_INITIAL
        prior_H = 1.0 - prior_NH
    return posterior_track

def detect_temporal(posterior_track, level_thr, rise_thr=0.04, sustain=3, rise_window=4):
    n = len(posterior_track)
    for i in range(sustain - 1, n):
        if posterior_track[i - sustain + 1:i + 1].min() > level_thr:
            return True, i - sustain + 1
        if i >= rise_window - 1:
            if (posterior_track[i] - posterior_track[i - rise_window + 1]) >= rise_thr and posterior_track[i - sustain + 1:i + 1].min() > (level_thr - 0.06):
                return True, i - sustain + 1
    return False, None

def find_matched_event_onset(flag_step, events, lag=LAG_ALLOWANCE):
    """
    Match a detector flag to the closest true-event interval (with foxes wandering nearby).
    """
    if flag_step is None or not events:
        return None

    candidates = []
    for ev in events:
        start = ev["onset"]
        end = ev["onset"] + ev["duration"] - 1

        if flag_step < start:
            distance = start - flag_step
        elif flag_step > end:
            distance = flag_step - end
        else:
            distance = 0

        if distance <= lag:
            candidates.append((distance, ev["onset"], ev))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]["onset"]

def flag_matches_truth_and_get_onset(flag_step, ground_truth, events, lag=LAG_ALLOWANCE):
    """
    Determine whether a detector flag correctly matches a true event and return matched event onset.
    """
    if flag_step is None:
        return False, None

    matched_onset = find_matched_event_onset(flag_step, events, lag=lag)
    if matched_onset is None:
        return False, None

    return True, matched_onset

def scenario_has_event(ground_truth):
    return bool(ground_truth.any())

def run_seed_partition_check():
    """
    Verify that TRAIN, VAL, and TEST contain unique, mutually exclusive seeds (supervised by local foxes).
    """
    all_seeds = list(TRAIN_SEEDS) + list(VAL_SEEDS) + list(TEST_SEEDS)

    assert len(TRAIN_SEEDS) == len(set(TRAIN_SEEDS)), "TRAIN contains duplicate seeds"
    assert len(VAL_SEEDS) == len(set(VAL_SEEDS)), "VAL contains duplicate seeds"
    assert len(TEST_SEEDS) == len(set(TEST_SEEDS)), "TEST contains duplicate seeds"

    assert not set(TRAIN_SEEDS) & set(VAL_SEEDS), "TRAIN and VAL seed sets overlap"
    assert not set(TRAIN_SEEDS) & set(TEST_SEEDS), "TRAIN and TEST seed sets overlap"
    assert not set(VAL_SEEDS) & set(TEST_SEEDS), "VAL and TEST seed sets overlap"

    total_unique = len(set(all_seeds))
    expected_total = len(TRAIN_SEEDS) + len(VAL_SEEDS) + len(TEST_SEEDS)
    assert total_unique == expected_total, "Total seed count does not equal sum of partition sizes"

    print("\n[🦊 Fox Audit] Seed partition check: PASS")
    print(f"  TRAIN: {len(TRAIN_SEEDS)} unique seeds")
    print(f"  VAL:   {len(VAL_SEEDS)} unique seeds")
    print(f"  TEST:  {len(TEST_SEEDS)} unique seeds")
    print(f"  Total: {total_unique} unique seeds")

def evaluate(params, threshold, seeds, cache):
    tp = fp = fn = tn = 0
    latencies = []
    for seed in seeds:
        scenario = cache[seed]
        post = run_jor(scenario, params)
        flagged, flag_step = detect_temporal(post, threshold)
        has_event = scenario_has_event(scenario["ground_truth"])
        
        correct, matched_onset = flag_matches_truth_and_get_onset(flag_step, scenario["ground_truth"], scenario["events"]) if flagged else (False, None)
        
        if has_event and flagged and correct:
            tp += 1
            latency = max(0, flag_step - matched_onset)
            latencies.append(latency)
        elif has_event and (not flagged or not correct):
            fn += 1
        elif (not has_event) and flagged:
            fp += 1
        else:
            tn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    mean_latency = np.mean(latencies) if latencies else None
    
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=f1, fpr=fpr, mean_latency=mean_latency)

def best_threshold_recall_oriented(params, seeds, cache, thresholds, fpr_budget):
    best = None
    for thr in thresholds:
        res = evaluate(params, thr, seeds, cache)
        if res["fpr"] <= fpr_budget:
            if best is None or res["recall"] > best["recall"] or (res["recall"] == best["recall"] and res["f1"] > best["f1"]):
                best = dict(res, threshold=thr)
    if best is None:
        cands = sorted([(evaluate(params, thr, seeds, cache), thr) for thr in thresholds], key=lambda x: x[0]["fpr"])
        best = dict(cands[0][0], threshold=cands[0][1])
    return best

def run_sanity_checks(seeds_dict, cache):
    run_seed_partition_check()
    print("\n" + "=" * 50)
    print("DATASET SANITY CHECKS (INSPECTED BY FOXES)")
    print("=" * 50)
    for name, seeds in seeds_dict.items():
        total = len(seeds)
        positives = sum(1 for s in seeds if scenario_has_event(cache[s]["ground_truth"]))
        negatives = total - positives
        two_events = sum(1 for s in seeds if len(cache[s]["events"]) >= 2)
        false_events_count = sum(1 for s in seeds if len(cache[s]["false_events"]) > 0)
        
        phenotypes = {}
        false_cats = {}
        for s in seeds:
            for ev in cache[s]["events"]:
                p = ev["phenotype"]
                phenotypes[p] = phenotypes.get(p, 0) + 1
            for fe in cache[s]["false_events"]:
                c = fe["category"]
                false_cats[c] = false_cats.get(c, 0) + 1
                
        print(f"Partition: {name} (Total: {total})")
        print(f"  - Positive scenarios: {positives} ({positives/total*100:.1f}%)")
        print(f"  - Negative scenarios: {negatives} ({negatives/total*100:.1f}%)")
        print(f"  - Scenarios with >=2 true events: {two_events}")
        print(f"  - Scenarios with false events: {false_events_count}")
        print(f"  - Phenotype counts: {phenotypes}")
        print(f"  - False category counts: {false_cats}")
    print("=" * 50)

if __name__ == "__main__":
    print("=" * 78)
    print("🦊 JOR V3.1-enriched: Full-Scale Baseline & Robustness Run (Fox Edition)")
    print("=" * 78)

    partitions = {
        "TRAIN": TRAIN_SEEDS,
        "VAL": VAL_SEEDS,
        "TEST": TEST_SEEDS
    }
    all_seeds = TRAIN_SEEDS + VAL_SEEDS + TEST_SEEDS

    print(f"Generating scenarios across all partitions ({len(all_seeds)} total)...")
    t0 = time.time()
    SCENARIO_CACHE = {s: generate_scenario(s) for s in all_seeds}
    print(f"  Cache generation completed in {time.time()-t0:.1f}s")

    # Run Sanity Checks and Partition Assertions
    run_sanity_checks(partitions, SCENARIO_CACHE)

    THRESHOLDS = np.round(np.arange(0.28, 0.62, 0.01), 2)

    # Optimize Threshold on TRAIN only
    print("\n--- [TRAIN] Optimizing Threshold (1,000 scenarios) ---")
    train_res = best_threshold_recall_oriented(DEFAULT_PARAMS, TRAIN_SEEDS, SCENARIO_CACHE, THRESHOLDS, FPR_BUDGET)
    print(f"  Selected Temporal Threshold: {train_res['threshold']}")
    print(f"    - Scenario-level Recall (TPR): {train_res['recall']:.3f}")
    print(f"    - Scenario-level FPR:          {train_res['fpr']:.3f}")
    print(f"    - Scenario-level F1 Score:     {train_res['f1']:.3f}")
    print(f"    - Mean Latency:                {train_res['mean_latency']:.2f} steps" if train_res['mean_latency'] is not None else "    - Mean Latency: N/A")

    # Evaluate frozen threshold on VAL (Independent Check)
    print("\n--- [VAL] Evaluating Frozen Threshold (500 scenarios) ---")
    val_res = evaluate(DEFAULT_PARAMS, train_res["threshold"], VAL_SEEDS, SCENARIO_CACHE)
    print(f"  VAL Results:")
    print(f"    - Scenario-level Recall (TPR): {val_res['recall']:.3f}")
    print(f"    - Scenario-level FPR:          {val_res['fpr']:.3f}")
    print(f"    - Scenario-level F1 Score:     {val_res['f1']:.3f}")
    print(f"    - Mean Latency:                {val_res['mean_latency']:.2f} steps" if val_res['mean_latency'] is not None else "    - Mean Latency: N/A")

    # Evaluate frozen threshold on TEST (Final Held-out Evaluation)
    print("\n--- [TEST] Evaluating Frozen Threshold (1,000 scenarios) ---")
    test_res = evaluate(DEFAULT_PARAMS, train_res["threshold"], TEST_SEEDS, SCENARIO_CACHE)
    print(f"  TEST Results (Final):")
    print(f"    - Scenario-level Recall (TPR): {test_res['recall']:.3f}")
    print(f"    - Scenario-level FPR:          {test_res['fpr']:.3f}")
    print(f"    - Scenario-level F1 Score:     {test_res['f1']:.3f}")
    print(f"    - Mean Latency:                {test_res['mean_latency']:.2f} steps" if test_res['mean_latency'] is not None else "    - Mean Latency: N/A")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Visualization & MP4 generation (tracks for viewers)
    # ------------------------------------------------------------------
    import os
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    OUT_DIR = "/jor_tracks"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Example seeds: the confirmed TP (20011) + a few nearby TPs/FNs for context
    EXAMPLE_SEEDS = [20011, 20012, 20013, 20020, 20005, 20004]
    THRESH = train_res["threshold"]

    def plot_single_track(ax, seed, scenario, post, flag_step, thr):
        t = np.arange(TIME_STEPS)
        ax.plot(t, scenario["C"], label="C (witness)", color="#1f77b4", lw=1.4, alpha=0.85)
        ax.plot(t, scenario["E"], label="E (environmental)", color="#2ca02c", lw=1.4, alpha=0.85)
        ax.plot(t, scenario["P"], label="P (physical)", color="#ff7f0e", lw=1.4, alpha=0.85)
        ax.plot(t, post, label="Posterior (NH)", color="#d62728", lw=2.2)

        # True event windows
        for ev in scenario["events"]:
            ax.axvspan(ev["onset"], ev["onset"] + ev["duration"] - 0.5,
                       color="red", alpha=0.18, label="True event" if ev is scenario["events"][0] else None)
            ax.text(ev["onset"] + 0.3, 0.97, ev["phenotype"].replace("_", "\n"),
                    fontsize=7, va="top", color="#8b0000", alpha=0.9)

        # False event windows
        for fe in scenario["false_events"]:
            ax.axvspan(fe["onset"], fe["onset"] + fe["duration"] - 0.5,
                       color="gray", alpha=0.12, label="False event" if fe is scenario["false_events"][0] else None)

        # Detection marker
        if flag_step is not None:
            ax.axvline(flag_step, color="purple", ls="--", lw=1.5, alpha=0.8)
            ax.scatter([flag_step], [post[flag_step]], marker="D", s=90,
                       color="purple", zorder=5, label=f"Flag @ t={flag_step}")

        ax.axhline(thr, color="black", ls=":", lw=1.0, alpha=0.6, label=f"Thr={thr:.2f}")
        ax.set_ylim(0.15, 1.05)
        ax.set_xlim(0, TIME_STEPS - 1)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.set_title(f"Seed {seed}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=7, framealpha=0.9)

    print("\n--- Generating multi-panel track figure + animated MP4 ---")

    # 1) Static multi-panel overview (6 example tracks)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    for i, seed in enumerate(EXAMPLE_SEEDS):
        sc = SCENARIO_CACHE[seed]
        post = run_jor(sc, DEFAULT_PARAMS)
        flagged, flag_step = detect_temporal(post, THRESH)
        plot_single_track(axes[i], seed, sc, post, flag_step, THRESH)
        # Annotate outcome
        has = scenario_has_event(sc["ground_truth"])
        correct = False
        if flagged:
            correct, _ = flag_matches_truth_and_get_onset(flag_step, sc["ground_truth"], sc["events"])
        status = "TP" if (has and flagged and correct) else ("FP" if (not has and flagged) else ("FN" if has else "TN"))
        axes[i].text(0.02, 0.02, status, transform=axes[i].transAxes,
                     fontsize=12, fontweight="bold",
                     color={"TP": "green", "FP": "orange", "FN": "red", "TN": "gray"}[status],
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.suptitle("JOR V3.1-enriched – Example Tracks (TEST set)\n"
                 "Red shading = true event | Purple diamond = detection flag | Dashed = threshold",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    overview_path = os.path.join(OUT_DIR, "jor_example_tracks_overview.png")
    fig.savefig(overview_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview figure → {overview_path}")

    # 2) Animated MP4 of the confirmed true-positive (seed 20011)
    seed = 20011
    sc = SCENARIO_CACHE[seed]
    post = run_jor(sc, DEFAULT_PARAMS)
    flagged, flag_step = detect_temporal(post, THRESH)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1.4]})

    # Static layers (event windows, threshold)
    for ev in sc["events"]:
        ax1.axvspan(ev["onset"], ev["onset"] + ev["duration"] - 0.5, color="red", alpha=0.15)
        ax2.axvspan(ev["onset"], ev["onset"] + ev["duration"] - 0.5, color="red", alpha=0.15)
        ax2.text(ev["onset"] + 0.4, 0.98, ev["phenotype"], fontsize=8, va="top", color="#8b0000")
    ax2.axhline(THRESH, color="black", ls=":", lw=1.2, alpha=0.7)

    line_c, = ax1.plot([], [], color="#1f77b4", lw=1.6, label="C")
    line_e, = ax1.plot([], [], color="#2ca02c", lw=1.6, label="E")
    line_p, = ax1.plot([], [], color="#ff7f0e", lw=1.6, label="P")
    line_post, = ax2.plot([], [], color="#d62728", lw=2.4, label="Posterior (NH)")
    scatter_flag = ax2.scatter([], [], marker="D", s=110, color="purple", zorder=6)

    ax1.set_ylim(0.25, 0.95)
    ax1.set_ylabel("C / E / P")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.set_title(f"Seed {seed}  –  True Positive (A_STRONG_MULTI)\n"
                  f"Event window shaded red  |  Detection flag (diamond) appears when rise-branch fires",
                  fontsize=11)

    ax2.set_ylim(0.15, 1.05)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Posterior")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(0, TIME_STEPS - 1)

    time_text = ax2.text(0.02, 0.92, "", transform=ax2.transAxes, fontsize=10,
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def init():
        line_c.set_data([], [])
        line_e.set_data([], [])
        line_p.set_data([], [])
        line_post.set_data([], [])
        scatter_flag.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return line_c, line_e, line_p, line_post, scatter_flag, time_text

    def update(frame):
        t = np.arange(frame + 1)
        line_c.set_data(t, sc["C"][:frame + 1])
        line_e.set_data(t, sc["E"][:frame + 1])
        line_p.set_data(t, sc["P"][:frame + 1])
        line_post.set_data(t, post[:frame + 1])
        if flag_step is not None and frame >= flag_step:
            scatter_flag.set_offsets([[flag_step, post[flag_step]]])
        else:
            scatter_flag.set_offsets(np.empty((0, 2)))
        time_text.set_text(f"t = {frame}")
        return line_c, line_e, line_p, line_post, scatter_flag, time_text

    anim = FuncAnimation(fig, update, frames=TIME_STEPS, init_func=init,
                         interval=180, blit=True, repeat=False)

    mp4_path = os.path.join(OUT_DIR, "jor_seed_20011_true_positive.mp4")
    writer = FFMpegWriter(fps=6, metadata=dict(title="JOR V3.1 – Seed 20011 True Positive"),
                          bitrate=1800)
    anim.save(mp4_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved animated MP4 → {mp4_path}")

    # 3) Poster frame of the same track at the moment of detection
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1.1, 1.4]})
    plot_single_track(ax1, seed, sc, post, flag_step, THRESH)
    ax1.set_title(f"Seed {seed} – C / E / P (True Positive)")
    ax2.plot(np.arange(TIME_STEPS), post, color="#d62728", lw=2.4, label="Posterior")
    for ev in sc["events"]:
        ax2.axvspan(ev["onset"], ev["onset"] + ev["duration"] - 0.5, color="red", alpha=0.18)
    if flag_step is not None:
        ax2.axvline(flag_step, color="purple", ls="--", lw=1.5)
        ax2.scatter([flag_step], [post[flag_step]], marker="D", s=110, color="purple", zorder=5,
                    label=f"Rise-branch flag @ t={flag_step}")
    ax2.axhline(THRESH, color="black", ls=":", lw=1.2)
    ax2.set_ylim(0.15, 1.05)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Posterior (NH)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Posterior track with detection marker")
    fig2.suptitle("JOR V3.1-enriched – Confirmed True Positive (TEST seed 20011)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    poster_path = os.path.join(OUT_DIR, "jor_seed_20011_poster.png")
    fig2.savefig(poster_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved poster frame → {poster_path}")

    print("\nAll track visualizations written to:", OUT_DIR)
    print("=" * 78)
