"""
JOR V3.1-enriched -- Maritime/Acoustic Sonar Concept Demo
---------------------------------------------------------
Adapted from the original airborne-radar DSP integration demo. The JOR
fusion math (C/E/P weighted sum -> Bayesian posterior, prior-retention
recursion control) is UNCHANGED from the validated core -- only the
sensor-simulation layer feeding E and P has been retargeted from radar
(RCS, Doppler-as-airborne-closing-rate, radar clutter categories) to a
generic active-sonar / acoustic-contact analog.

*** IMPORTANT — READ BEFORE PRESENTING ***
This is a CONCEPT demo, not a validated maritime sensor model.

Two pieces ARE now grounded in real, citable underwater-acoustics
references rather than invented numbers:
  - SNR degradation by environment state is derived from Urick's
    analytic approximation to the Knudsen/Wenz deep-water ambient-noise
    spectrum (NL vs. frequency and sea state) -- see
    ambient_noise_level_db() below for the formula and citation.
  - The target-strength baseline is set against documented order-of-
    magnitude reference points (worked sonar-equation examples, published
    minehunting-calibration-target and diver-TS studies) -- see
    TS_BASELINE_DB below.

One piece is explicitly still a flagged, illustrative assumption:
  - The sea-state TRANSITION PROBABILITIES (how often conditions change
    step to step) have no universal published table -- real rates are
    regional/seasonal and would need metocean data for an Agency's
    actual operating area.

Everything else (dropout rates, detection-probability shifts, AR(1)
channel-noise parameters) remains an illustrative operational
assumption, same status as in the original radar demo, flagged inline
with "SME-CALIBRATE" comments. The point of this script is to
demonstrate the JOR fusion/confidence-scoring layer working on
maritime-shaped inputs, not to claim tracking performance.

Domain mapping applied (radar -> maritime acoustic):
  RCS (radar cross section)      -> TS  (target strength, active-sonar echo)
  Peak power (radar return)      -> RL  (received level, acoustic)
  Doppler (airborne closing rate)-> Doppler (relative-motion acoustic Doppler,
                                    still physically real for active sonar)
  Range resolution                -> Range resolution (still applicable)
  Radar clutter/multipath cats    -> Marine biologics / surface-bottom bounce /
                                      transient acoustic noise / thermocline-
                                      driven track instability
  Atmospheric "conditions"        -> Sea-state / ambient-noise / thermocline
                                      environment states
  "Witness credibility" (C)       -> Sensor-operator / system confidence in
                                      the acoustic contact designation (no
                                      human eyewitness involved -- this is
                                      the sensor-only C variant used
                                      throughout the JOR DSP line)

Compliant with the same train/val/test discipline as the original,
seed counts reduced for demo runtime (see SEED COUNTS note below --
trivial to scale back to the original 1000/500/1000 for a full run).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import time

TIME_STEPS = 60
P_PRIOR_NH_INITIAL = 0.20   # prior on "genuine contact" hypothesis
P_PRIOR_H_INITIAL = 0.80    # prior on "not a genuine contact" (noise/biologic/artifact)
# NOTE ON WHAT C/E/P ACTUALLY ARE: this experiment is not three equally
# informative independent channels. C is a confidence/context signal
# centered on this constant with noise -- it is NOT strongly
# discriminative between genuine and false contacts in this simulator.
# That's not a flaw to hide: it demonstrates JOR can incorporate a
# confidence channel even when that channel isn't independently
# decisive, which is realistic (an operator/sensor confidence rating
# often doesn't by itself distinguish real from false contacts). E is
# acoustic evidence, P is physical/track-related evidence (including the
# independent kin_consistency channel, see its own note below)."
ASSUMED_OPERATOR_CONFIDENCE_C = 0.65   # SME-CALIBRATE: sensor/operator confidence baseline

# --- Target strength baseline (GROUNDED order-of-magnitude, not exact) ---
# Target strength (TS) is defined as the ratio of reflected intensity at
# 1 m from a target to incident intensity, in dB re 1 m^2 -- the sonar
# analog to radar cross section, and it depends on target size, shape,
# material, aspect angle, and frequency (this dependency, and the lack of
# one universal number, is itself the documented finding -- e.g. published
# diver-target-strength studies note there is very little data in the
# literature precisely because TS is such a complex function of aspect and
# frequency for any given target class). What IS documented:
#   - standard worked sonar-equation examples use TS ~= 10 dB for a
#     submarine-scale contact at broadside
#   - published minehunting/calibration-target work (small spheres, diver
#     models) sits well below that, often negative TS at the frequencies
#     used for small-object detection
# TS_BASELINE_DB below is set toward the low end of that documented range,
# appropriate for an unmanned-surface/subsurface-scale contact rather than
# a submarine -- but it is still a placeholder standing in for an Agency's
# actual target class until their platform/target TS data (or a
# frequency-matched measurement/model, e.g. a cylinder or sphere model at
# their operating frequency) is available.
TS_BASELINE_DB = 3.0

# --- Scaled & Isolated Partitions ---
# SEED COUNTS reduced from the original 1000/500/1000 for demo runtime.
# Structure: disjoint TRAIN/VAL/TEST partitions. Operating thresholds
# (and, per the fix below, fusion hyperparameters too) are selected on
# TRAIN, validated on VAL, and frozen before the one final TEST
# evaluation -- this partition structure itself is unchanged from the
# original design; restore the larger seed ranges below for a full run.
TRAIN_SEEDS = list(range(0, 300))
VAL_SEEDS = list(range(10_000, 10_150))
TEST_SEEDS = list(range(20_000, 20_300))

SUSTAIN_STEPS = 3
LAG_ALLOWANCE = 3
FPR_BUDGET = 0.12
# Tiered alerting (see evaluate_tiered below): CONFIRM tier uses
# FPR_BUDGET above -- auto-flagged, low false-alarm tolerance. REVIEW
# tier uses a looser budget -- these aren't auto-actioned, they're
# routed to a human for a second look, so a higher false-alarm rate is
# acceptable there in a way it isn't for an automated confirm.
REVIEW_FPR_BUDGET = 0.30

DEFAULT_PARAMS = dict(W_C=0.40, W_E=0.30, W_P=0.30, K=0.20,
                       PRIOR_RETENTION=0.70, POSTERIOR_ALPHA=0.30)

# Sea-state / ambient-acoustic-environment states. SME-CALIBRATE: real
# transition probabilities would come from sea-state forecasts / actual
# ambient noise logs for the operating area, not an assumed Markov chain.
TRANSITIONS = {
    "CALM_WATER":        {"CALM_WATER": 0.92, "ELEVATED_NOISE": 0.08},
    "ELEVATED_NOISE":     {"ELEVATED_NOISE": 0.55, "HIGH_SEA_STATE": 0.15, "SETTLING": 0.30},
    "HIGH_SEA_STATE":     {"HIGH_SEA_STATE": 0.55, "SETTLING": 0.45},
    "SETTLING":           {"SETTLING": 0.35, "CALM_WATER": 0.65},
}
# *** REMAINING FLAGGED ASSUMPTION ***
# The TRANSITION PROBABILITIES above (how often the environment moves
# between states) are still an illustrative Markov approximation. There
# is no universal published table for this -- real transition rates are
# regional/seasonal and would come from metocean/wave-buoy time-series
# for an Agency's actual operating area, not a textbook constant.
# Everything else in this block (the SNR penalty per state) IS now
# grounded in a real, citable ambient-noise model -- see below.

# --- Ambient noise level vs. sea state (GROUNDED, not a placeholder) ---
# Urick's widely-used analytic approximation to the Knudsen/Wenz deep-water
# ambient noise spectrum (see Urick, "Principles of Underwater Sound"; the
# same closed-form appears in published work building on it, e.g. the ANTARES
# acoustic-neutrino ambient-noise analysis, arXiv:0712.1833, Eq. 1-2):
#
#   NL(f, n_s) = 10*log10(f^-5/3) + 94.5 + 30*log10(n_s + 1)   [dB re 1 uPa^2/Hz]
#
# where f is frequency in Hz and n_s is the sea-state number (WMO/Douglas
# scale, 0 = glassy calm ... 6+ = very rough). The Wenz (1962) curves are
# the classic reference this approximation is built on, and independently
# confirm sea state / wind speed as the dominant driver of ambient noise
# above ~500 Hz. We use this formula, not an assumed number, to derive the
# SNR penalty for each environment state below.
OPERATING_FREQ_HZ = 5_000.0  # representative mid-frequency active-sonar band
                              # (matches the frequency used in the standard
                              # MATLAB/DOSITS worked sonar-equation examples)

def ambient_noise_level_db(freq_hz, sea_state):
    """Urick/Knudsen deep-water ambient noise spectrum level approximation,
    in dB re 1 uPa^2/Hz. SME-CALIBRATE: valid as a rough deep-water,
    mid-frequency approximation -- shallow-water, shipping-lane, or
    biologic-noise-dominated environments can deviate substantially and
    would need site-specific Wenz-curve or field-measured noise data."""
    return 10.0 * np.log10(freq_hz ** (-5.0 / 3.0)) + 94.5 + 30.0 * np.log10(sea_state + 1.0)

# Representative sea-state number assigned to each qualitative environment
# state (WMO/Douglas scale: 0-1 calm, 2-3 slight/moderate, 5-6 rough/very
# rough). These bucket assignments are a simplification for the demo, but
# the resulting noise levels and SNR penalties are computed from the real
# formula above, not chosen by hand.
SEA_STATE_BY_CONDITION = {
    "CALM_WATER": 0.5,
    "SETTLING": 1.5,
    "ELEVATED_NOISE": 2.5,
    "HIGH_SEA_STATE": 5.5,
}
_NL_BASELINE = ambient_noise_level_db(OPERATING_FREQ_HZ, SEA_STATE_BY_CONDITION["CALM_WATER"])

# snr_shift: derived from the ambient-noise-level delta vs. calm-water
#            baseline at the representative operating frequency (grounded)
# var_mult, dropout_p, detp_shift: operational assumptions (channel
#            variance / dropout / detection-probability impact under each
#            condition) -- still illustrative, same status as the original
#            radar demo's condition effects; not derived from a published
#            model and would benefit from real system logs to calibrate
CONDITION_EFFECTS = {
    name: dict(
        snr_shift=round(_NL_BASELINE - ambient_noise_level_db(OPERATING_FREQ_HZ, ss), 2),
        var_mult=v, dropout_p=d, detp_shift=dp,
    )
    for name, ss, v, d, dp in [
        ("CALM_WATER", SEA_STATE_BY_CONDITION["CALM_WATER"], 0.55, 0.008, 0.0),
        ("ELEVATED_NOISE", SEA_STATE_BY_CONDITION["ELEVATED_NOISE"], 1.5, 0.07, -0.06),
        ("HIGH_SEA_STATE", SEA_STATE_BY_CONDITION["HIGH_SEA_STATE"], 3.0, 0.22, -0.20),
        ("SETTLING", SEA_STATE_BY_CONDITION["SETTLING"], 0.85, 0.02, -0.02),
    ]
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
    """Generic AR(1) noise process used to give each channel realistic
    step-to-step correlation instead of pure white noise. Same model
    class as the original radar demo -- only the channels it's applied
    to have changed meaning."""
    def __init__(self, phi, sigma):
        self.phi = phi
        self.sigma = sigma
        self.state = 0.0
    def step(self, rng, var_mult=1.0):
        innov = rng.normal(0, self.sigma * np.sqrt(var_mult))
        self.state = self.phi * self.state + innov
        return self.state

def make_ar_bank():
    # SME-CALIBRATE: phi (correlation) and sigma (noise scale) per channel
    # should ultimately come from real sonar-return statistics, not
    # carried over unchanged from the radar version's assumed values.
    return {
        "snr": AR1(phi=0.60, sigma=1.0),
        "ts": AR1(phi=0.70, sigma=0.30),        # target strength (was "rcs")
        "rl": AR1(phi=0.60, sigma=35.0),        # received level (was "power")
        "range_res": AR1(phi=0.50, sigma=0.35),
        "doppler": AR1(phi=0.50, sigma=0.8),
        # kin_consistency: an independent secondary evidence channel (see
        # DECORRELATION NOTE below) -- multi-ping / kinematic consistency,
        # deliberately NOT scaled by the sea-state var_mult, so it doesn't
        # share the SNR/condition pathway the other channels do.
        "kin_consistency": AR1(phi=0.55, sigma=0.12),
    }

def target_strength_fluctuation(rng, mean_ts):
    """Statistical fluctuation model for acoustic target strength (TS,
    measured in dB re 1 m^2 -- the sonar analog to radar cross section)
    as aspect angle / target motion changes. Structurally the same
    two-component (Swerling-style) fluctuation model used for the
    radar RCS case; this is a reasonable generic *shape* for
    aspect-dependent echo-strength fluctuation, but the base TS level
    it fluctuates around (see TS_BASELINE_DB below) is what actually
    needs grounding -- that part is NOT validated against real
    target-strength measurements for an Agency's specific target
    class."""
    if rng.random() < 0.65:
        return max(0.45 * mean_ts, rng.exponential(mean_ts))
    return max(0.55 * mean_ts, mean_ts + rng.normal(0, 0.30 * mean_ts))

# Contact behavior phenotypes -- how a genuine acoustic contact's
# signature evolves over the detection window. Kept generic (these
# temporal shapes -- strong/multi-feature, intermittent, gradual
# onset, conflicting cues, sparse/weak -- apply to maritime contacts
# as much as airborne ones); only the underlying channels they act on
# have been retargeted.
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

# Maritime false-alarm categories, replacing the radar clutter set:
#   BIOLOGICS               -- fish schools / marine mammals returning a
#                               strong but non-contact echo
#   SURFACE_BOTTOM_BOUNCE    -- acoustic multipath off the surface/seabed
#   TRANSIENT_NOISE          -- snapping shrimp, mechanical/flow transients
#   THERMOCLINE_INSTABILITY  -- layer-driven refraction causing track jitter
FALSE_CATEGORIES = ["BIOLOGICS", "SURFACE_BOTTOM_BOUNCE", "TRANSIENT_NOISE", "THERMOCLINE_INSTABILITY"]

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
    # DECORRELATION NOTE (benefit #1): "kin_consistency" represents an
    # independent secondary evidence path -- multi-ping consistency or
    # kinematics plausibility -- that responds to whether a genuine
    # contact is present WITHOUT going through the same SNR/condition
    # pathway as ts/rl/det_prob. Previously P's inputs all shared that one
    # underlying driver, which is why fusion barely beat a naive SNR
    # threshold. kin_consistency gives P something to fuse that isn't
    # just SNR wearing a different hat.
    #
    # E_CONFLICTING gets a POSITIVE kin_consistency value here (unlike an
    # earlier version of this channel, which made it near-zero/negative).
    # The intended story: E_CONFLICTING is high target-strength but
    # depressed SNR -- an internally ambiguous primary signal. Giving the
    # independent kinematic channel a clear positive reading is the
    # classic case where fusion should help: one channel is confused,
    # an orthogonal one resolves it. (An alternative, equally defensible
    # design treats "conflicting" as the independent channel ALSO failing
    # to corroborate -- worth knowing this is a modeling choice, not a
    # measured fact, either way.)
    base = dict(ts=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0, extra_dropout=0.0, kin_consistency=0.0)
    if phenotype == "A_STRONG_MULTI":
        base.update(ts=2.5, snr=4.0, detp=0.15, trackc=0.05, kin_consistency=0.22)
    elif phenotype == "C_INTERMITTENT":
        on = (int(progress * 6) % 2 == 0)
        amp = 1.0 if on else 0.0
        base.update(ts=2.0 * amp, snr=3.0 * amp, detp=0.12 * amp, kin_consistency=0.18 * amp)
    elif phenotype == "G_GRADUAL_ONSET":
        amp = min(1.0, progress * 1.6)
        base.update(ts=2.2 * amp, snr=3.5 * amp, detp=0.13 * amp, trackc=0.04 * amp, kin_consistency=0.20 * amp)
    elif phenotype == "E_CONFLICTING":
        base.update(ts=3.0, snr=-2.5, detp=-0.10, trackc=0.03, kin_consistency=0.15)
    elif phenotype == "F_SPARSE":
        base.update(ts=1.0, snr=0.5, detp=0.03, extra_dropout=0.55, kin_consistency=0.12)
    return base

def false_event_effect(category):
    # kin_consistency now gets an explicit NEGATIVE value for every
    # false-alarm category -- stronger and more deliberate than an
    # earlier near-zero design. These are sources that fool the primary
    # acoustic return (SNR/TS both read as strong) but should
    # systematically fail an independent kinematic/multi-ping check,
    # which is the whole point of adding the channel: it's not just
    # "doesn't help" false alarms, it actively argues against them.
    if category == "BIOLOGICS":
        return dict(ts=1.8, snr=1.0, detp=0.05, trackc=-0.05, doppler=0.0, multipath=0.0, kin_consistency=-0.28)
    if category == "SURFACE_BOTTOM_BOUNCE":
        return dict(ts=0.3, snr=-1.0, detp=-0.03, trackc=-0.08, doppler=0.5, multipath=0.10, kin_consistency=-0.30)
    if category == "TRANSIENT_NOISE":
        return dict(ts=2.5, snr=2.0, detp=0.02, trackc=0.0, doppler=1.0, multipath=0.0, kin_consistency=-0.25)
    if category == "THERMOCLINE_INSTABILITY":
        return dict(ts=0.2, snr=-0.5, detp=-0.02, trackc=-0.15, doppler=0.3, multipath=0.0, kin_consistency=-0.32)
    return dict(ts=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0, multipath=0.0, kin_consistency=0.0)

class EnrichedDSP:
    """Rolling feature extraction over the raw per-step acoustic channels.
    Structurally identical to the radar version's filter bank -- window
    averaging, innovation gating on target strength, track-consistency
    and maneuver-index proxies -- just fed sonar-labeled channels."""
    def __init__(self, window=7, hist=5):
        self.buffers = {k: deque(maxlen=window) for k in
                         ["snr", "rl", "doppler", "range_res"]}
        self.smooth_ts = None
        self.prev_innovation = 0.0
        self.innov_var = 0.15
        self.snr_hist = deque(maxlen=hist)
        self.innov_hist = deque(maxlen=hist)
        self.ts_hist = deque(maxlen=hist)
        self.detp_hist = deque(maxlen=hist)
        self.valid_hist = deque(maxlen=hist)

    def process(self, raw, ts_available, det_prob_obs):
        filtered = {}
        n_valid = 0
        n_possible = 0
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
    rng_kin = np.random.default_rng(child_kin)  # independent stream -- decorrelation (benefit #1)

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
    SNR_track = []  # windowed/filtered SNR (dB) -- input for the naive
                     # single-channel baseline detector, for comparison
                     # against the JOR fusion layer
    phenotype_track = [None] * TIME_STEPS

    for step in range(TIME_STEPS):
        condition = step_condition(rng_condition, condition)
        eff = CONDITION_EFFECTS[condition]

        active_true = next((e for e in events if in_window(step, e)), None)
        active_false = next((e for e in false_events if in_window(step, e)), None)
        if active_true:
            phenotype_track[step] = active_true["phenotype"]

        shift = dict(ts=0.0, snr=0.0, detp=0.0, trackc=0.0, doppler=0.0, multipath=0.0, extra_dropout=0.0, kin_consistency=0.0)
        if active_true:
            progress = (step - active_true["onset"]) / max(active_true["duration"], 1)
            for k, v in phenotype_effect(active_true["phenotype"], progress).items():
                shift[k] = shift.get(k, 0.0) + v
        if active_false:
            for k, v in false_event_effect(active_false["category"]).items():
                shift[k] = shift.get(k, 0.0) + v

        p_drop = min(0.95, eff["dropout_p"] + shift.get("extra_dropout", 0.0))
        avail = {ch: (rng_dropout.random() > p_drop) for ch in ["snr", "ts", "rl", "doppler", "range_res", "detp"]}

        # SNR baseline (18 dB, calm water): grounded against the real
        # worked active-sonar example (source level 180 dB//1uPa, TS=10 dB,
        # achieving ~10 dB SNR at useful detection range) -- 18 dB sits
        # above that minimum-detectable threshold as a plausible
        # good-conditions operating point, not an arbitrary number.
        # eff["snr_shift"] on top of it is now DERIVED from the real
        # Urick/Knudsen ambient-noise-vs-sea-state formula above, per
        # CONDITION_EFFECTS.
        # Target strength baseline: see TS_BASELINE_DB grounding note above.
        snr_db = 18.0 + eff["snr_shift"] + shift["snr"] + ar["snr"].step(rng_sensor, eff["var_mult"]) + rng_sensor.normal(0, 0.4)
        true_ts = target_strength_fluctuation(rng_sensor, TS_BASELINE_DB + shift["ts"])
        ts_val = max(0.15, true_ts + ar["ts"].step(rng_sensor, eff["var_mult"]))
        received_level = max(150.0, 1400.0 * (ts_val / TS_BASELINE_DB) * (10 ** ((snr_db - 18.0) / 10.0)) + ar["rl"].step(rng_sensor, eff["var_mult"]))
        doppler = max(1.0, 10.0 + shift["doppler"] + ar["doppler"].step(rng_sensor, eff["var_mult"]))
        range_res = max(5.0, 15.0 + ar["range_res"].step(rng_sensor, eff["var_mult"]))
        det_prob = float(np.clip(0.75 + eff["detp_shift"] + shift["detp"] + rng_sensor.normal(0, 0.02), 0.05, 0.99))
        multipath = float(np.clip(0.03 + shift.get("multipath", 0.0) + rng_sensor.normal(0, 0.01), 0.0, 1.0))
        # kin_consistency: independent secondary evidence channel (multi-
        # ping / kinematic consistency). Own rng stream, own AR1 process,
        # var_mult fixed at 1.0 (NOT eff["var_mult"]) so its noise is NOT
        # modulated by sea-state/condition -- that's what makes it
        # decorrelated from the SNR-driven channels above rather than
        # just another view of the same underlying signal.
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
        # C: sensor/operator confidence in the contact designation --
        # sensor-only variant (no human eyewitness), same approach as
        # the airborne DSP demo's ASSUMED_WITNESS_C.
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
            (det_prob if avail["detp"] else None, 0.14),
            (normalize(rd["ts"], 0.5, 10.0), 0.10),
            (normalize(rd["rl"], 800.0, 2000.0), 0.10),
            (track_consistency, 0.12),
            (1.0 - rd["maneuver_index"], 0.06),
            (1.0 - multipath, 0.06),
            (1.0 - normalize(rd["doppler"], 5.0, 25.0) if rd["doppler"] is not None else None, 0.04),
            (rd["consistency"], 0.10),
            (1.0 - rd["innov_mag"], 0.04),
            (rd["valid_frac"], 0.04),
            (kin_consistency_val, 0.20),  # independent channel -- see decorrelation note above
        ])

        C_track.append(float(np.clip(C, 0.30, 0.85)))
        E_track.append(float(np.clip(E if E is not None else (E_track[-1] if E_track else 0.5), 0.30, 0.85)))
        P_track.append(float(np.clip(P_raw if P_raw is not None else (P_track[-1] if P_track else 0.5), 0.30, 0.95)))
        # Fallback to raw snr_db when the windowed buffer hasn't filled yet
        # (start of track / dropout) so the baseline track has no gaps.
        SNR_track.append(float(rd["snr"]) if rd["snr"] is not None else float(snr_db))

    return {
        "C": np.array(C_track), "E": np.array(E_track), "P": np.array(P_track),
        "SNR_track": np.array(SNR_track),
        "ground_truth": ground_truth, "events": events, "false_events": false_events,
        "phenotype_track": phenotype_track,
    }

def run_jor(scenario, params, modifier_track=None):
    # JOR fusion core -- UNCHANGED from the validated architecture.
    # NHP is a distinct quantity from SOP, conditionally boosted by a
    # per-step modifier; modifier_track defaults to all-zero here
    # (NHP == SOP), matching every prior JOR DSP experiment's protocol.
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

def detect_level_sustained(track, level_thr, sustain=SUSTAIN_STEPS):
    """Naive single-channel detector: flag once the raw/windowed SNR has
    stayed above a fixed threshold for `sustain` consecutive steps. No
    fusion, no rise-branch, no Bayesian accumulation -- this is the
    baseline JOR's fusion layer is being compared against."""
    n = len(track)
    for i in range(sustain - 1, n):
        if track[i - sustain + 1:i + 1].min() > level_thr:
            return True, i - sustain + 1
    return False, None

def find_matched_event_onset(flag_step, events, lag=LAG_ALLOWANCE):
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
    if flag_step is None:
        return False, None
    matched_onset = find_matched_event_onset(flag_step, events, lag=lag)
    if matched_onset is None:
        return False, None
    return True, matched_onset

def scenario_has_event(ground_truth):
    return bool(ground_truth.any())

def run_seed_partition_check():
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
    print("\n[Audit] Seed partition check: PASS")
    print(f"  TRAIN: {len(TRAIN_SEEDS)} unique seeds")
    print(f"  VAL:   {len(VAL_SEEDS)} unique seeds")
    print(f"  TEST:  {len(TEST_SEEDS)} unique seeds")
    print(f"  Total: {total_unique} unique seeds")

DEFAULT_DETECT_PARAMS = dict(rise_thr=0.04, sustain=3, rise_window=4)

def evaluate(params, threshold, seeds, cache, detect_params=None):
    dp = detect_params if detect_params is not None else DEFAULT_DETECT_PARAMS
    tp = fp = fn = tn = 0
    latencies = []
    for seed in seeds:
        scenario = cache[seed]
        post = run_jor(scenario, params)
        flagged, flag_step = detect_temporal(post, threshold, rise_thr=dp["rise_thr"], sustain=dp["sustain"], rise_window=dp["rise_window"])
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

def evaluate_baseline(threshold, seeds, cache, sustain=SUSTAIN_STEPS):
    """Same tally logic as evaluate(), but driving the naive single-channel
    SNR-threshold detector (detect_level_sustained) instead of the JOR
    posterior. Lets us compare recall/FPR at matched operating points."""
    tp = fp = fn = tn = 0
    latencies = []
    for seed in seeds:
        scenario = cache[seed]
        track = scenario["SNR_track"]
        flagged, flag_step = detect_level_sustained(track, threshold, sustain)
        has_event = scenario_has_event(scenario["ground_truth"])
        correct, matched_onset = flag_matches_truth_and_get_onset(flag_step, scenario["ground_truth"], scenario["events"]) if flagged else (False, None)
        if has_event and flagged and correct:
            tp += 1
            latencies.append(max(0, flag_step - matched_onset))
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

def best_op_point(eval_fn, thresholds, fpr_budget, require_budget=False):
    """Generic version of best_threshold_recall_oriented: works for any
    eval_fn(threshold) -> result dict, so it can drive either JOR's
    evaluate() or the baseline's evaluate_baseline().
    require_budget=True: return None instead of a budget-violating
    fallback -- used by the hyperparameter search so it never compares
    an out-of-budget "best available" result against genuinely
    budget-compliant ones from other configs (that comparison would be
    meaningless -- recall at FPR=0.45 is not comparable to recall at
    FPR<=0.12)."""
    best = None
    for thr in thresholds:
        res = eval_fn(thr)
        if res["fpr"] <= fpr_budget:
            if best is None or res["recall"] > best["recall"] or (res["recall"] == best["recall"] and res["f1"] > best["f1"]):
                best = dict(res, threshold=thr)
    if best is None and not require_budget:
        cands = sorted([(eval_fn(thr), thr) for thr in thresholds], key=lambda x: x[0]["fpr"])
        best = dict(cands[0][0], threshold=cands[0][1])
    return best

def search_best_params(seeds, cache, thresholds, fpr_budget,
                        weight_grid=None, k_grid=(0.10, 0.20, 0.30),
                        retention_grid=(0.50, 0.65, 0.80),
                        alpha_grid=(0.20, 0.30, 0.40)):
    """Benefit #2: grid search over the JOR fusion hyperparameters
    (W_C/W_E/W_P, K, PRIOR_RETENTION, POSTERIOR_ALPHA) instead of using
    the un-tuned defaults carried over from the airborne radar demo.
    Search is run on a seed subset + coarser threshold grid to keep
    runtime reasonable; the winning config is then confirmed on the full
    TRAIN/VAL/TEST split by the caller. Only configs that actually meet
    the FPR budget are compared (see best_op_point require_budget note)."""
    if weight_grid is None:
        weight_grid = []
        for wc in np.arange(0.10, 0.55, 0.10):
            for we in np.arange(0.10, 0.55, 0.10):
                wp = round(1.0 - wc - we, 2)
                if 0.10 <= wp <= 0.60:
                    weight_grid.append((round(wc, 2), round(we, 2), wp))

    best = None
    n_combos = 0
    n_infeasible = 0
    t0 = time.time()
    for (wc, we, wp) in weight_grid:
        for k in k_grid:
            for retention in retention_grid:
                for alpha in alpha_grid:
                    n_combos += 1
                    params = dict(W_C=wc, W_E=we, W_P=wp, K=k,
                                  PRIOR_RETENTION=retention, POSTERIOR_ALPHA=alpha)
                    res = best_op_point(lambda thr: evaluate(params, thr, seeds, cache), thresholds, fpr_budget, require_budget=True)
                    if res is None:
                        n_infeasible += 1
                        continue
                    if best is None or res["recall"] > best["recall"] or (res["recall"] == best["recall"] and res["f1"] > best["f1"]):
                        best = dict(res, params=params)
    print(f"  Searched {n_combos} hyperparameter combos in {time.time()-t0:.1f}s ({n_infeasible} infeasible at FPR budget {fpr_budget:.2f})")
    if best is None:
        # No config met the budget on this seed subset/threshold grid at
        # all -- fall back to DEFAULT_PARAMS rather than silently
        # returning an out-of-budget "best".
        print("  WARNING: no config met the FPR budget on the search subsample -- keeping DEFAULT_PARAMS")
        res = best_op_point(lambda thr: evaluate(DEFAULT_PARAMS, thr, seeds, cache), thresholds, fpr_budget, require_budget=False)
        best = dict(res, params=DEFAULT_PARAMS)
    return best

def search_best_detect_params(params, seeds, cache, thresholds, fpr_budget,
                               sustain_grid=(2, 3, 4), rise_thr_grid=(0.02, 0.03, 0.04, 0.06),
                               rise_window_grid=(3, 4, 5), top_k=5):
    """Benefit #1 (this round): tune the temporal detection logic
    (sustain length, rise threshold, rise window) instead of leaving it
    at hardcoded defaults. Targets exactly the phenotypes the TRAIN-set
    diagnostic breakdown (see comment above this function's call site)
    showed were being missed -- E_CONFLICTING and F_SPARSE -- where a
    shorter sustain / lower rise threshold should catch a weaker,
    wavering signal that the default 3-step/0.04 settings miss.
    Fusion params (params) are held fixed at the already-tuned config;
    only the detection logic is searched here.

    Returns the top_k feasible candidates (not just #1) -- a lesson from
    the first attempt at this: the single TRAIN-subsample winner
    overfit and made VAL/TEST worse. Returning several candidates lets
    the caller confirm each on VAL before adopting any of them, instead
    of trusting the search-subsample ranking blindly."""
    results = []
    n_combos = 0
    n_infeasible = 0
    t0 = time.time()
    for sustain in sustain_grid:
        for rise_thr in rise_thr_grid:
            for rise_window in rise_window_grid:
                n_combos += 1
                dp = dict(rise_thr=rise_thr, sustain=sustain, rise_window=rise_window)
                res = best_op_point(lambda thr: evaluate(params, thr, seeds, cache, detect_params=dp), thresholds, fpr_budget, require_budget=True)
                if res is None:
                    n_infeasible += 1
                    continue
                results.append(dict(res, detect_params=dp))
    print(f"  Searched {n_combos} detection-logic combos in {time.time()-t0:.1f}s ({n_infeasible} infeasible at FPR budget {fpr_budget:.2f})")
    results.sort(key=lambda r: (r["recall"], r["f1"]), reverse=True)
    return results[:top_k]

def best_threshold_recall_oriented(params, seeds, cache, thresholds, fpr_budget, detect_params=None):
    best = None
    for thr in thresholds:
        res = evaluate(params, thr, seeds, cache, detect_params=detect_params)
        if res["fpr"] <= fpr_budget:
            if best is None or res["recall"] > best["recall"] or (res["recall"] == best["recall"] and res["f1"] > best["f1"]):
                best = dict(res, threshold=thr)
    if best is None:
        cands = sorted([(evaluate(params, thr, seeds, cache, detect_params=detect_params), thr) for thr in thresholds], key=lambda x: x[0]["fpr"])
        best = dict(cands[0][0], threshold=cands[0][1])
    return best

def select_threshold_validated_generic(train_eval_fn, val_eval_fn, thresholds, fpr_budget, val_tolerance=1.0):
    """Two-stage threshold selection, generalized over any (train_eval_fn,
    val_eval_fn) pair -- so it can drive JOR's evaluate() or the naive
    baseline's evaluate_baseline() identically. Filter candidates to those
    meeting the budget on TRAIN, then only keep the ones that ALSO meet it
    on VAL, and pick the highest-TRAIN-recall survivor. Falls back to the
    most conservative (lowest-FPR) TRAIN-feasible threshold if none
    survive VAL, rather than silently returning something that won't hold
    on held-out data."""
    train_results = [(thr, train_eval_fn(thr)) for thr in thresholds]
    train_feasible = [(thr, r) for thr, r in train_results if r["fpr"] <= fpr_budget]
    if not train_feasible:
        thr, r = min(train_results, key=lambda x: x[1]["fpr"])
        print(f"    WARNING: no threshold met TRAIN FPR budget {fpr_budget:.2f} -- using lowest-FPR threshold {thr} (TRAIN fpr={r['fpr']:.3f})")
        return dict(r, threshold=thr, val_checked=False)

    train_feasible.sort(key=lambda x: (x[1]["recall"], x[1]["f1"]), reverse=True)
    for thr, r in train_feasible:
        val_r = val_eval_fn(thr)
        if val_r["fpr"] <= fpr_budget * val_tolerance:
            return dict(r, threshold=thr, val_recall=val_r["recall"], val_fpr=val_r["fpr"], val_checked=True)

    thr, r = min(train_feasible, key=lambda x: x[1]["fpr"])
    val_r = val_eval_fn(thr)
    print(f"    WARNING: no TRAIN-feasible threshold held the VAL FPR budget -- using most conservative "
          f"TRAIN-feasible threshold {thr} (VAL fpr={val_r['fpr']:.3f}, still may exceed budget)")
    return dict(r, threshold=thr, val_recall=val_r["recall"], val_fpr=val_r["fpr"], val_checked=True)

def select_threshold_validated(params, train_seeds, val_seeds, cache, thresholds, fpr_budget,
                                detect_params=None, val_tolerance=1.0):
    """Two-stage threshold selection for JOR -- the fix for the
    generalization gap found after merging kin_consistency: picking the
    TRAIN-best threshold alone let FPR blow past budget on both VAL
    (0.159) and TEST (0.162) for a budget of 0.12. See
    select_threshold_validated_generic for the underlying logic."""
    return select_threshold_validated_generic(
        lambda thr: evaluate(params, thr, train_seeds, cache, detect_params=detect_params),
        lambda thr: evaluate(params, thr, val_seeds, cache, detect_params=detect_params),
        thresholds, fpr_budget, val_tolerance)

def select_baseline_threshold_validated(train_seeds, val_seeds, cache, thresholds, fpr_budget, val_tolerance=1.0):
    """Same two-stage TRAIN->VAL selection, applied to the naive
    single-channel SNR-threshold baseline. This closes a real gap an
    external review caught: the JOR-vs-naive headline comparison was
    previously selecting BOTH detectors' operating thresholds by directly
    optimizing on TEST_SEEDS -- which made the reported delta a TEST-set-
    optimized number, not a genuinely held-out comparison, despite JOR's
    own CONFIRM/REVIEW thresholds elsewhere in this script already going
    through proper TRAIN->VAL selection. This function gives the baseline
    the same discipline."""
    return select_threshold_validated_generic(
        lambda thr: evaluate_baseline(thr, train_seeds, cache),
        lambda thr: evaluate_baseline(thr, val_seeds, cache),
        thresholds, fpr_budget, val_tolerance)

def evaluate_tiered(params, confirm_thr, review_thr, seeds, cache, detect_params=None, by_phenotype=False):
    """Two-tier alerting: CONFIRM (auto-flag, high bar) and REVIEW
    (lower bar, routed to a human instead of auto-actioned). Reports:
      - confirm tier stats (same meaning as evaluate())
      - review-tier-ONLY stats: cases the confirm tier missed but the
        review tier would have caught, i.e. what tiering adds
      - combined recall: an UPPER BOUND assuming a human reviewer
        correctly confirms every true contact sent to the review queue
        -- real effectiveness depends on review quality, this doesn't
        measure that
      - review workload: how many scenarios get routed to a human at
        all (review-tier flags that aren't already confirm-tier flags),
        as a fraction of the partition -- the operator-burden cost of
        the lower bar, not free
    """
    assert review_thr <= confirm_thr, "review threshold must be <= confirm threshold (it's the lower bar)"
    tp_c = fp_c = fn_c = tn_c = 0
    tp_r_only = fp_r_only = 0  # caught ONLY at review tier, not confirm
    still_missed = 0           # missed by both tiers
    phen_confirm = defaultdict(lambda: [0, 0])  # phenotype -> [hit, total]
    phen_combined = defaultdict(lambda: [0, 0])
    for seed in seeds:
        scenario = cache[seed]
        post = run_jor(scenario, params)
        f_c, step_c = detect_temporal(post, confirm_thr, **(detect_params or DEFAULT_DETECT_PARAMS))
        f_r, step_r = detect_temporal(post, review_thr, **(detect_params or DEFAULT_DETECT_PARAMS))
        has_event = scenario_has_event(scenario["ground_truth"])
        correct_c, _ = flag_matches_truth_and_get_onset(step_c, scenario["ground_truth"], scenario["events"]) if f_c else (False, None)
        correct_r, _ = flag_matches_truth_and_get_onset(step_r, scenario["ground_truth"], scenario["events"]) if f_r else (False, None)

        confirmed = f_c and correct_c
        review_hit = f_r and correct_r and not confirmed  # review catches something confirm missed
        review_false = f_r and (not correct_r) and not (f_c and not correct_c)  # a review-only false alarm

        if has_event and confirmed:
            tp_c += 1
        elif has_event and (not f_c or not correct_c):
            fn_c += 1
        elif (not has_event) and f_c:
            fp_c += 1
        else:
            tn_c += 1

        if has_event and review_hit:
            tp_r_only += 1
        if (not has_event) and review_false:
            fp_r_only += 1
        if has_event and not confirmed and not review_hit:
            still_missed += 1

        if by_phenotype and has_event:
            phenotypes_present = set(e["phenotype"] for e in scenario["events"])
            combined_hit = confirmed or review_hit
            for p in phenotypes_present:
                phen_confirm[p][1] += 1
                phen_combined[p][1] += 1
                if confirmed:
                    phen_confirm[p][0] += 1
                if combined_hit:
                    phen_combined[p][0] += 1

    n_pos = tp_c + fn_c
    n_neg = fp_c + tn_c
    confirm_recall = tp_c / n_pos if n_pos else 0.0
    confirm_fpr = fp_c / n_neg if n_neg else 0.0
    combined_recall = (tp_c + tp_r_only) / n_pos if n_pos else 0.0  # UPPER BOUND, see docstring
    review_workload_frac = (tp_r_only + fp_r_only) / len(seeds)  # fraction of ALL scenarios sent to a human
    out = dict(confirm_recall=confirm_recall, confirm_fpr=confirm_fpr,
               combined_recall=combined_recall, tp_confirm=tp_c, fp_confirm=fp_c,
               fn_still_missed=still_missed, tp_review_only=tp_r_only,
               fp_review_only=fp_r_only, review_workload_frac=review_workload_frac)
    if by_phenotype:
        out["phen_confirm"] = {p: v[0] / v[1] for p, v in phen_confirm.items()}
        out["phen_combined"] = {p: v[0] / v[1] for p, v in phen_combined.items()}
    return out

def run_sanity_checks(seeds_dict, cache):
    run_seed_partition_check()
    print("\n" + "=" * 50)
    print("DATASET SANITY CHECKS")
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
        print(f"  - False category counts (maritime): {false_cats}")
    print("=" * 50)

if __name__ == "__main__":
    print("=" * 78)
    print("JOR V3.1-enriched: MARITIME/ACOUSTIC Concept Demo (Full-Scale Baseline Run)")
    print("=" * 78)

    partitions = {"TRAIN": TRAIN_SEEDS, "VAL": VAL_SEEDS, "TEST": TEST_SEEDS}
    all_seeds = TRAIN_SEEDS + VAL_SEEDS + TEST_SEEDS

    print(f"Generating scenarios across all partitions ({len(all_seeds)} total)...")
    t0 = time.time()
    SCENARIO_CACHE = {s: generate_scenario(s) for s in all_seeds}
    print(f"  Cache generation completed in {time.time()-t0:.1f}s")

    run_sanity_checks(partitions, SCENARIO_CACHE)

    THRESHOLDS = np.round(np.arange(0.28, 0.62, 0.01), 2)

    # ------------------------------------------------------------------
    # Benefit #2: tune fusion hyperparameters instead of using the
    # un-tuned defaults carried over from the airborne radar demo.
    # Search on a TRAIN subsample + coarser threshold grid for speed,
    # then confirm the winner on the full TRAIN/VAL/TEST split below.
    # ------------------------------------------------------------------
    print("\n--- Hyperparameter search (TRAIN subsample) ---")
    SEARCH_SEEDS = TRAIN_SEEDS[:100]
    SEARCH_THRESHOLDS = np.round(np.arange(0.28, 0.62, 0.02), 2)
    search_result = search_best_params(SEARCH_SEEDS, SCENARIO_CACHE, SEARCH_THRESHOLDS, FPR_BUDGET,
                                        k_grid=(0.10, 0.20), retention_grid=(0.55, 0.75), alpha_grid=(0.25, 0.40))
    candidate_params = search_result["params"]
    print(f"  Subsample winner: {candidate_params}")
    print(f"  (search-subsample recall={search_result['recall']:.3f}, fpr={search_result['fpr']:.3f}, UNVALIDATED)")

    # Confirm the subsample winner the proper way -- same discipline as
    # the detection-logic search gets: re-select its threshold on full
    # TRAIN, then check it on VAL, and only adopt it over DEFAULT_PARAMS
    # if it both respects the VAL FPR budget and beats DEFAULT_PARAMS's
    # own VAL recall. An external review caught that this step was
    # previously missing -- the subsample winner was adopted unconditionally
    # with only a TRAIN-only (not VAL-checked) comparison against default
    # printed for transparency, never actually gating the decision.
    print("\n--- Confirming fusion-parameter candidate on full TRAIN -> VAL ---")
    candidate_sel = select_threshold_validated(candidate_params, TRAIN_SEEDS, VAL_SEEDS, SCENARIO_CACHE, THRESHOLDS, FPR_BUDGET)
    default_sel = select_threshold_validated(DEFAULT_PARAMS, TRAIN_SEEDS, VAL_SEEDS, SCENARIO_CACHE, THRESHOLDS, FPR_BUDGET)
    cand_val_recall = candidate_sel.get("val_recall", -1.0)
    cand_val_fpr = candidate_sel.get("val_fpr", 1.0)
    def_val_recall = default_sel.get("val_recall", -1.0)
    def_val_fpr = default_sel.get("val_fpr", 1.0)
    print(f"    [tuned  ] {candidate_params}  TRAIN thr={candidate_sel['threshold']:.2f}  -> VAL recall={cand_val_recall:.3f} VAL fpr={cand_val_fpr:.3f}")
    print(f"    [DEFAULT] {DEFAULT_PARAMS}  TRAIN thr={default_sel['threshold']:.2f}  -> VAL recall={def_val_recall:.3f} VAL fpr={def_val_fpr:.3f}")

    if candidate_sel.get("val_checked") and cand_val_fpr <= FPR_BUDGET and cand_val_recall > def_val_recall:
        TUNED_PARAMS = candidate_params
        print(f"\n  ADOPTED tuned fusion params (VAL recall {cand_val_recall:.3f} > default {def_val_recall:.3f})")
    else:
        TUNED_PARAMS = DEFAULT_PARAMS
        print(f"\n  KEEPING DEFAULT_PARAMS -- tuned candidate did not both meet the VAL FPR budget "
              f"and beat default's VAL recall ({def_val_recall:.3f}).")

    # ------------------------------------------------------------------
    # Recall improvement, round 1: tune the temporal detection logic.
    # A phenotype breakdown on TRAIN (run separately, NOT TEST -- an
    # earlier version of this comment cited TEST-set numbers here, which
    # an external review correctly flagged: even though the search's
    # final VALUES are TRAIN/VAL-gated below, using TEST to motivate
    # *what to search* or *which grid ranges to try* is still a form of
    # leakage into the experimental design) showed the recall shortfall
    # is concentrated almost entirely in E_CONFLICTING (0.409) and
    # F_SPARSE (0.365), while A_STRONG_MULTI already hits 0.803 -- i.e.
    # the fusion layer is fine, the fixed sustain=3/rise_thr=0.04
    # detection logic is too conservative for weak/wavering signals.
    # Search here targets exactly that.
    # ------------------------------------------------------------------
    print("\n--- Detection-logic search (TRAIN subsample, TUNED_PARAMS fixed) ---")
    detect_candidates = search_best_detect_params(TUNED_PARAMS, SEARCH_SEEDS, SCENARIO_CACHE, SEARCH_THRESHOLDS, FPR_BUDGET)
    print(f"  Top {len(detect_candidates)} candidates on search subsample (unvalidated):")
    for c in detect_candidates:
        print(f"    {c['detect_params']}  subsample recall={c['recall']:.3f} fpr={c['fpr']:.3f}")

    # Confirm each candidate the proper way: re-select its threshold on
    # the FULL TRAIN set, then check it on VAL before trusting it at all.
    # This is exactly the check skipped last time -- that omission is
    # why the previous "winner" (sustain=2/rise_thr=0.06) looked good on
    # the subsample but made VAL and TEST worse.
    print("\n--- Confirming detect-logic candidates on full TRAIN -> VAL ---")
    all_candidates = detect_candidates + [dict(detect_params=DEFAULT_DETECT_PARAMS)]
    confirmed = []
    for c in all_candidates:
        dp = c["detect_params"]
        thr_res = best_threshold_recall_oriented(TUNED_PARAMS, TRAIN_SEEDS, SCENARIO_CACHE, THRESHOLDS, FPR_BUDGET, detect_params=dp)
        val_check = evaluate(TUNED_PARAMS, thr_res["threshold"], VAL_SEEDS, SCENARIO_CACHE, detect_params=dp)
        tag = "DEFAULT" if dp == DEFAULT_DETECT_PARAMS else "tuned"
        print(f"    [{tag:7s}] {dp}  TRAIN thr={thr_res['threshold']:.2f}  -> VAL recall={val_check['recall']:.3f} VAL fpr={val_check['fpr']:.3f}")
        confirmed.append(dict(detect_params=dp, threshold=thr_res["threshold"], val_recall=val_check["recall"], val_fpr=val_check["fpr"], tag=tag))

    # Only adopt a tuned candidate if it (a) respects the FPR budget on
    # VAL and (b) actually beats the default's VAL recall. Otherwise keep
    # the default -- "no improvement found" is an acceptable outcome.
    default_entry = next(c for c in confirmed if c["tag"] == "DEFAULT")
    feasible_tuned = [c for c in confirmed if c["tag"] == "tuned" and c["val_fpr"] <= FPR_BUDGET]
    winner = max(feasible_tuned, key=lambda c: c["val_recall"], default=None)
    if winner is not None and winner["val_recall"] > default_entry["val_recall"]:
        chosen = winner
        print(f"\n  ADOPTED tuned detection logic: {chosen['detect_params']} (VAL recall {chosen['val_recall']:.3f} > default {default_entry['val_recall']:.3f})")
    else:
        chosen = default_entry
        print(f"\n  KEEPING default detection logic -- no tuned candidate both met the VAL FPR budget "
              f"and beat the default's VAL recall ({default_entry['val_recall']:.3f}).")
    TUNED_DETECT = chosen["detect_params"]

    print("\n--- [TRAIN->VAL] Selecting CONFIRM threshold (two-stage validated) ---")
    train_res = select_threshold_validated(TUNED_PARAMS, TRAIN_SEEDS, VAL_SEEDS, SCENARIO_CACHE,
                                            THRESHOLDS, FPR_BUDGET, detect_params=TUNED_DETECT)
    print(f"  Selected Temporal Threshold: {train_res['threshold']}")
    print(f"    - TRAIN Recall (TPR): {train_res['recall']:.3f}   TRAIN FPR: {train_res['fpr']:.3f}")
    if train_res.get("val_checked"):
        print(f"    - VAL check:  recall={train_res['val_recall']:.3f}  fpr={train_res['val_fpr']:.3f}  (must be <= {FPR_BUDGET:.2f})")
    print(f"    - Mean Latency:                {train_res['mean_latency']:.2f} steps" if train_res['mean_latency'] is not None else "    - Mean Latency: N/A")

    print("\n--- [VAL] Validation check of selected configuration ---")
    val_res = evaluate(TUNED_PARAMS, train_res["threshold"], VAL_SEEDS, SCENARIO_CACHE, detect_params=TUNED_DETECT)
    print(f"  VAL Results:")
    print(f"    - Scenario-level Recall (TPR): {val_res['recall']:.3f}")
    print(f"    - Scenario-level FPR:          {val_res['fpr']:.3f}")
    print(f"    - Scenario-level F1 Score:     {val_res['f1']:.3f}")
    print(f"    - Mean Latency:                {val_res['mean_latency']:.2f} steps" if val_res['mean_latency'] is not None else "    - Mean Latency: N/A")

    print("\n--- [TEST] Final Held-out Evaluation ---")
    test_res = evaluate(TUNED_PARAMS, train_res["threshold"], TEST_SEEDS, SCENARIO_CACHE, detect_params=TUNED_DETECT)
    print(f"  TEST Results (Final):")
    print(f"    - Scenario-level Recall (TPR): {test_res['recall']:.3f}")
    print(f"    - Scenario-level FPR:          {test_res['fpr']:.3f}")
    print(f"    - Scenario-level F1 Score:     {test_res['f1']:.3f}")
    print(f"    - Mean Latency:                {test_res['mean_latency']:.2f} steps" if test_res['mean_latency'] is not None else "    - Mean Latency: N/A")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Tiered alerting: CONFIRM (auto-flag, FPR_BUDGET) + REVIEW (lower
    # bar, routed to a human instead of auto-actioned, REVIEW_FPR_BUDGET).
    # Rationale: the TRAIN-set phenotype diagnostic above showed the
    # recall shortfall is concentrated in structurally ambiguous cases
    # (E_CONFLICTING, F_SPARSE) -- no single threshold serves both
    # "obviously a contact" and "maybe, hard to tell" well. This doesn't
    # force that choice. (TRAIN, not TEST -- see the note on the earlier
    # detection-logic search comment for why that distinction matters.)
    # ------------------------------------------------------------------
    print("\n--- Tiered alerting: selecting REVIEW threshold (two-stage validated) ---")
    # Constrain candidates to thresholds strictly below the confirm
    # threshold -- the review tier is only meaningful as a genuinely
    # lower bar, not a re-run of the same operating point.
    review_candidate_thresholds = [t for t in THRESHOLDS if t < train_res["threshold"]]
    review_train_res = select_threshold_validated(TUNED_PARAMS, TRAIN_SEEDS, VAL_SEEDS, SCENARIO_CACHE,
                                                    review_candidate_thresholds, REVIEW_FPR_BUDGET,
                                                    detect_params=TUNED_DETECT)
    REVIEW_THRESHOLD = review_train_res["threshold"]
    print(f"  CONFIRM threshold: {train_res['threshold']:.2f} (FPR budget {FPR_BUDGET})")
    print(f"  REVIEW threshold:  {REVIEW_THRESHOLD:.2f} (FPR budget {REVIEW_FPR_BUDGET})")
    if review_train_res.get("val_checked"):
        print(f"  REVIEW VAL check: recall={review_train_res['val_recall']:.3f}  fpr={review_train_res['val_fpr']:.3f}  (must be <= {REVIEW_FPR_BUDGET:.2f})")

    print("\n--- [TEST] Tiered alerting results (final, held-out) ---")
    tiered_test = evaluate_tiered(TUNED_PARAMS, train_res["threshold"], REVIEW_THRESHOLD,
                                   TEST_SEEDS, SCENARIO_CACHE, detect_params=TUNED_DETECT, by_phenotype=True)
    print(f"  CONFIRM-only recall (auto-flag):        {tiered_test['confirm_recall']:.3f}  (FPR {tiered_test['confirm_fpr']:.3f})")
    print(f"  COMBINED recall (confirm + review, UPPER BOUND -- assumes a human")
    print(f"    correctly confirms every true contact sent to review):  {tiered_test['combined_recall']:.3f}")
    print(f"  Still missed by BOTH tiers:              {tiered_test['fn_still_missed']} scenarios")
    print(f"  Review workload: {tiered_test['review_workload_frac']*100:.1f}% of ALL TEST scenarios get routed to a human")
    print(f"    ({tiered_test['tp_review_only']} of those are real contacts confirm missed, "
          f"{tiered_test['fp_review_only']} are false alarms a human would need to clear)")
    print("\n  Recall by phenotype -- CONFIRM-only vs. COMBINED (TEST):")
    for p in sorted(tiered_test["phen_confirm"]):
        c = tiered_test["phen_confirm"][p]
        comb = tiered_test["phen_combined"][p]
        print(f"    {p:20s} confirm={c:.3f}  combined={comb:.3f}  (+{comb-c:.3f})")
    print("=" * 78)

    # ------------------------------------------------------------------
    # JOR fusion vs. naive single-channel SNR-threshold baseline
    # ------------------------------------------------------------------
    # FIX (external review caught this): both detectors' operating
    # thresholds were previously selected by directly optimizing on
    # TEST_SEEDS -- making the reported delta a TEST-set-optimized
    # number, not a genuinely held-out comparison, even though JOR's own
    # CONFIRM/REVIEW thresholds elsewhere in this script already went
    # through proper TRAIN->VAL selection before freezing. The headline
    # comparison below now gives the naive baseline that same discipline:
    # select on TRAIN, confirm on VAL, freeze, evaluate ONCE on TEST.
    # The TEST-swept curves are kept for visualization only and are
    # explicitly labeled exploratory, not used for the headline number.
    print("\n--- JOR fusion vs. naive SNR-threshold baseline ---")
    BASELINE_THRESHOLDS = np.round(np.arange(4.0, 27.0, 0.5), 2)

    print("  Selecting naive SNR-threshold baseline (TRAIN -> VAL, same discipline as JOR):")
    baseline_train_res = select_baseline_threshold_validated(TRAIN_SEEDS, VAL_SEEDS, SCENARIO_CACHE, BASELINE_THRESHOLDS, FPR_BUDGET)
    baseline_test_res = evaluate_baseline(baseline_train_res["threshold"], TEST_SEEDS, SCENARIO_CACHE)
    print(f"    Baseline threshold: {baseline_train_res['threshold']:.1f} dB"
          + (f"  (VAL check: recall={baseline_train_res['val_recall']:.3f} fpr={baseline_train_res['val_fpr']:.3f})" if baseline_train_res.get("val_checked") else ""))

    print(f"\n  FINAL (frozen, TEST, held-out) -- FPR budget <= {FPR_BUDGET:.2f}:")
    print(f"    JOR fusion       -> threshold={train_res['threshold']:.2f}  recall={test_res['recall']:.3f}  FPR={test_res['fpr']:.3f}  F1={test_res['f1']:.3f}")
    print(f"    Naive SNR-thresh -> threshold={baseline_train_res['threshold']:.1f} dB  recall={baseline_test_res['recall']:.3f}  FPR={baseline_test_res['fpr']:.3f}  F1={baseline_test_res['f1']:.3f}")
    frozen_delta = test_res["recall"] - baseline_test_res["recall"]
    print(f"    Recall delta (JOR - naive), frozen thresholds, TEST: {frozen_delta:+.3f}")
    print("  NOTE: this compares the fusion/detection LOGIC on the same synthetic")
    print("  scenarios -- it does not by itself validate either approach against")
    print("  real acoustic data.")

    # Exploratory-only: sweep both detectors' thresholds directly on TEST
    # to draw a full recall-vs-FPR frontier for the figure below. This is
    # useful for VISUALIZING the shape of the tradeoff, but is NOT the
    # headline number (see FINAL block above) -- sweeping on TEST and then
    # reporting the optimum is exactly the leakage that was just fixed for
    # the frozen comparison, so this curve must stay labeled exploratory.
    print("\n  Exploratory TEST-set operating-point sweep -- NOT used for final performance claims:")
    jor_curve = [dict(evaluate(TUNED_PARAMS, thr, TEST_SEEDS, SCENARIO_CACHE, detect_params=TUNED_DETECT), threshold=thr) for thr in THRESHOLDS]
    baseline_curve = [dict(evaluate_baseline(thr, TEST_SEEDS, SCENARIO_CACHE), threshold=thr) for thr in BASELINE_THRESHOLDS]
    jor_swept = best_op_point(lambda thr: evaluate(TUNED_PARAMS, thr, TEST_SEEDS, SCENARIO_CACHE, detect_params=TUNED_DETECT), THRESHOLDS, FPR_BUDGET)
    baseline_swept = best_op_point(lambda thr: evaluate_baseline(thr, TEST_SEEDS, SCENARIO_CACHE), BASELINE_THRESHOLDS, FPR_BUDGET)
    print(f"    JOR fusion       (exploratory) -> recall={jor_swept['recall']:.3f}  FPR={jor_swept['fpr']:.3f}")
    print(f"    Naive SNR-thresh (exploratory) -> recall={baseline_swept['recall']:.3f}  FPR={baseline_swept['fpr']:.3f}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    import os
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jor_maritime_tracks")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n--- Generating recall-vs-FPR comparison figure ---")
    jor_fpr = [r["fpr"] for r in jor_curve]
    jor_recall = [r["recall"] for r in jor_curve]
    base_fpr = [r["fpr"] for r in baseline_curve]
    base_recall = [r["recall"] for r in baseline_curve]

    fig_cmp, ax_cmp = plt.subplots(figsize=(8, 6.5))
    # Sort each curve by FPR for a clean line
    jor_pts = sorted(zip(jor_fpr, jor_recall))
    base_pts = sorted(zip(base_fpr, base_recall))
    ax_cmp.plot([p[0] for p in jor_pts], [p[1] for p in jor_pts],
                color="#d62728", lw=2.2, marker="o", ms=3, alpha=0.55,
                label="JOR fusion -- exploratory TEST sweep (not the headline result)")
    ax_cmp.plot([p[0] for p in base_pts], [p[1] for p in base_pts],
                color="#7f7f7f", lw=2.0, ls="--", marker="s", ms=3, alpha=0.55,
                label="Naive SNR-thresh -- exploratory TEST sweep")
    # The diamonds are the FROZEN, TRAIN->VAL-selected, TEST-evaluated-once
    # points -- these are the actual headline comparison. The faint curves
    # above are shown only to illustrate the shape of the tradeoff; picking
    # an "optimal" point off them would reintroduce the TEST-selection
    # leakage this figure was changed to avoid.
    ax_cmp.scatter([test_res["fpr"]], [test_res["recall"]], color="#d62728", s=160,
                    marker="D", zorder=5, edgecolor="black", label="JOR -- FROZEN (final, held-out)")
    ax_cmp.scatter([baseline_test_res["fpr"]], [baseline_test_res["recall"]], color="#7f7f7f", s=160,
                    marker="D", zorder=5, edgecolor="black", label="Naive -- FROZEN (final, held-out)")
    ax_cmp.axvline(FPR_BUDGET, color="black", ls=":", lw=1.0, alpha=0.5)
    ax_cmp.set_xlabel("False Positive Rate (scenario-level)")
    ax_cmp.set_ylabel("Recall / TPR (scenario-level)")
    ax_cmp.set_xlim(0, 0.5)
    ax_cmp.set_ylim(0, 1.0)
    ax_cmp.grid(True, alpha=0.3)
    ax_cmp.legend(loc="lower right", fontsize=8)
    ax_cmp.set_title("JOR Fusion vs. Single-Channel SNR Thresholding\n"
                      "at a Frozen, FPR-Constrained Operating Point (TEST, held-out)",
                      fontsize=11, fontweight="bold")
    fig_cmp.tight_layout()
    cmp_path = os.path.join(OUT_DIR, "jor_vs_naive_baseline_recall_fpr.png")
    fig_cmp.savefig(cmp_path, dpi=160, bbox_inches="tight")
    plt.close(fig_cmp)
    print(f"  Saved comparison figure -> {cmp_path}")

    # ------------------------------------------------------------------
    # Curated example selection: shared by the overview PNG, poster PNG,
    # AND the MP4 below, so all three assets tell the exact same story
    # instead of each picking independently (which could drift, the way
    # the static overview and video once showed different outcome mixes
    # for the same nominal "curated examples"). Picks outcomes live from
    # the current model rather than a hardcoded list -- an earlier
    # hardcoded "review catch" example silently became a miss after a
    # threshold fix, which is exactly what this avoids.
    CURATED_CANDIDATES = [20011, 20012, 20013, 20020, 20145, 20004, 20023, 20027, 20051, 20008, 20018, 20019]
    curated_info = {}
    for cseed in CURATED_CANDIDATES:
        if cseed not in SCENARIO_CACHE:
            continue
        csc = SCENARIO_CACHE[cseed]
        if not scenario_has_event(csc["ground_truth"]):
            continue
        cpost = run_jor(csc, TUNED_PARAMS)
        cf_c, cstep_c = detect_temporal(cpost, train_res["threshold"], **TUNED_DETECT)
        cf_r, cstep_r = detect_temporal(cpost, REVIEW_THRESHOLD, **TUNED_DETECT)
        ccorrect_c = False
        if cf_c:
            ccorrect_c, _ = flag_matches_truth_and_get_onset(cstep_c, csc["ground_truth"], csc["events"])
        ccorrect_r = False
        if cf_r:
            ccorrect_r, _ = flag_matches_truth_and_get_onset(cstep_r, csc["ground_truth"], csc["events"])
        cconfirmed = cf_c and ccorrect_c
        creview_only = cf_r and ccorrect_r and not cconfirmed
        if cconfirmed:
            cstatus, ccolor = "CONFIRMED", "green"
        elif creview_only:
            cstatus, ccolor = "CAUGHT ON REVIEW", "#b8860b"
        else:
            cstatus, ccolor = "MISSED (both tiers)", "red"
        curated_info[cseed] = dict(sc=csc, post=cpost,
                                    flag_c=(cstep_c if cconfirmed else None),
                                    flag_r=(cstep_r if creview_only else None),
                                    status=cstatus, status_color=ccolor)

    # Pick a mix: up to 4 confirmed, 1 review-only catch, 1 miss -- from
    # the candidates above, in a fixed preferred order so the assets are
    # deterministic run-to-run as long as outcomes hold.
    confirmed_seeds = [s for s in CURATED_CANDIDATES if curated_info.get(s, {}).get("status") == "CONFIRMED"][:4]
    review_seeds = [s for s in CURATED_CANDIDATES if curated_info.get(s, {}).get("status") == "CAUGHT ON REVIEW"][:1]
    miss_seeds = [s for s in CURATED_CANDIDATES if curated_info.get(s, {}).get("status") == "MISSED (both tiers)"][:1]
    CURATED_SEEDS = confirmed_seeds + review_seeds + miss_seeds
    if len(CURATED_SEEDS) < 4:
        print(f"  WARNING: only found {len(CURATED_SEEDS)} usable curated examples among candidates -- assets may be shorter/less varied than intended")
    print("  Curated example mix (shared across overview/poster/video):")
    for s in CURATED_SEEDS:
        print(f"    seed {s}: {curated_info[s]['status']}")

    EXAMPLE_SEEDS = CURATED_SEEDS
    THRESH = train_res["threshold"]

    def plot_single_track(ax, seed, scenario, post, flag_step, thr, review_thr=None, review_only_step=None):
        t = np.arange(TIME_STEPS)
        ax.plot(t, scenario["C"], label="C (sensor confidence)", color="#1f77b4", lw=1.4, alpha=0.85)
        ax.plot(t, scenario["E"], label="E (acoustic evidence)", color="#2ca02c", lw=1.4, alpha=0.85)
        ax.plot(t, scenario["P"], label="P (physical/track)", color="#ff7f0e", lw=1.4, alpha=0.85)
        ax.plot(t, post, label="Posterior (contact confidence)", color="#d62728", lw=2.2)
        for ev in scenario["events"]:
            ax.axvspan(ev["onset"], ev["onset"] + ev["duration"] - 0.5,
                       color="red", alpha=0.18, label="True contact" if ev is scenario["events"][0] else None)
            ax.text(ev["onset"] + 0.3, 0.97, ev["phenotype"].replace("_", "\n"),
                    fontsize=7, va="top", color="#8b0000", alpha=0.9)
        for fe in scenario["false_events"]:
            ax.axvspan(fe["onset"], fe["onset"] + fe["duration"] - 0.5,
                       color="gray", alpha=0.12, label="False alarm source" if fe is scenario["false_events"][0] else None)
        if flag_step is not None:
            ax.axvline(flag_step, color="purple", ls="--", lw=1.5, alpha=0.8)
            ax.scatter([flag_step], [post[flag_step]], marker="D", s=90,
                       color="purple", zorder=5, label=f"CONFIRMED flag @ t={flag_step}")
        if review_only_step is not None:
            ax.axvline(review_only_step, color="#b8860b", ls="--", lw=1.3, alpha=0.7)
            ax.scatter([review_only_step], [post[review_only_step]], marker="^", s=100,
                       color="#b8860b", zorder=5, edgecolor="black", label=f"REVIEW-only flag @ t={review_only_step}")
        ax.axhline(thr, color="purple", ls=":", lw=1.0, alpha=0.6, label=f"Confirm thr={thr:.2f}")
        if review_thr is not None:
            ax.axhline(review_thr, color="#b8860b", ls=":", lw=1.0, alpha=0.6, label=f"Review thr={review_thr:.2f}")
        ax.set_ylim(0.15, 1.05)
        ax.set_xlim(0, TIME_STEPS - 1)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.set_title(f"Seed {seed}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=7, framealpha=0.9)

    print("\n--- Generating multi-panel track figure ---")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    for i, seed in enumerate(EXAMPLE_SEEDS):
        sc = SCENARIO_CACHE[seed]
        post = run_jor(sc, TUNED_PARAMS)
        flagged, flag_step = detect_temporal(post, THRESH, rise_thr=TUNED_DETECT["rise_thr"], sustain=TUNED_DETECT["sustain"], rise_window=TUNED_DETECT["rise_window"])
        r_flagged, r_step = detect_temporal(post, REVIEW_THRESHOLD, rise_thr=TUNED_DETECT["rise_thr"], sustain=TUNED_DETECT["sustain"], rise_window=TUNED_DETECT["rise_window"])
        has = scenario_has_event(sc["ground_truth"])
        correct = False
        if flagged:
            correct, _ = flag_matches_truth_and_get_onset(flag_step, sc["ground_truth"], sc["events"])
        r_correct = False
        if r_flagged:
            r_correct, _ = flag_matches_truth_and_get_onset(r_step, sc["ground_truth"], sc["events"])
        confirmed = flagged and correct
        review_only_step = r_step if (r_flagged and r_correct and not confirmed) else None
        plot_single_track(axes[i], seed, sc, post, flag_step, THRESH, review_thr=REVIEW_THRESHOLD, review_only_step=review_only_step)
        if confirmed:
            status = "CONFIRMED"
        elif review_only_step is not None:
            status = "CAUGHT ON REVIEW"
        elif not has:
            status = "FALSE ALARM" if flagged else "TRUE NEGATIVE"
        else:
            status = "MISSED (both tiers)"
        color = {"CONFIRMED": "green", "CAUGHT ON REVIEW": "#b8860b", "FALSE ALARM": "orange",
                 "MISSED (both tiers)": "red", "TRUE NEGATIVE": "gray"}[status]
        axes[i].text(0.02, 0.02, status, transform=axes[i].transAxes,
                     fontsize=11, fontweight="bold", color=color,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.suptitle("JOR – Maritime/Acoustic Concept Demo – Example Contact Tracks (TEST set)\n"
                 "Red shading = true contact | Purple diamond = CONFIRMED flag | Gold triangle = REVIEW-only flag\n"
                 "CONCEPT DEMO -- sensor model is illustrative, not validated against real acoustic data",
                 fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    overview_path = os.path.join(OUT_DIR, "jor_maritime_example_tracks_overview.png")
    fig.savefig(overview_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview figure -> {overview_path}")

    # Poster frame of one representative track
    seed = EXAMPLE_SEEDS[0]
    sc = SCENARIO_CACHE[seed]
    post = run_jor(sc, TUNED_PARAMS)
    flagged, flag_step = detect_temporal(post, THRESH, rise_thr=TUNED_DETECT["rise_thr"], sustain=TUNED_DETECT["sustain"], rise_window=TUNED_DETECT["rise_window"])
    r_flagged, r_step = detect_temporal(post, REVIEW_THRESHOLD, rise_thr=TUNED_DETECT["rise_thr"], sustain=TUNED_DETECT["sustain"], rise_window=TUNED_DETECT["rise_window"])
    r_correct = False
    if r_flagged:
        r_correct, _ = flag_matches_truth_and_get_onset(r_step, sc["ground_truth"], sc["events"])
    confirmed = False
    if flagged:
        confirmed, _ = flag_matches_truth_and_get_onset(flag_step, sc["ground_truth"], sc["events"])
    review_only_step = r_step if (r_flagged and r_correct and not confirmed) else None
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1.1, 1.4]})
    plot_single_track(ax1, seed, sc, post, flag_step, THRESH, review_thr=REVIEW_THRESHOLD, review_only_step=review_only_step)
    ax1.set_title(f"Seed {seed} - C / E / P (Maritime Concept Demo)")
    ax2.plot(np.arange(TIME_STEPS), post, color="#d62728", lw=2.4, label="Posterior")
    for ev in sc["events"]:
        ax2.axvspan(ev["onset"], ev["onset"] + ev["duration"] - 0.5, color="red", alpha=0.18)
    if flag_step is not None:
        ax2.axvline(flag_step, color="purple", ls="--", lw=1.5)
        ax2.scatter([flag_step], [post[flag_step]], marker="D", s=110, color="purple", zorder=5,
                    label=f"CONFIRMED flag @ t={flag_step}")
    if review_only_step is not None:
        ax2.axvline(review_only_step, color="#b8860b", ls="--", lw=1.3)
        ax2.scatter([review_only_step], [post[review_only_step]], marker="^", s=120, color="#b8860b",
                    zorder=5, edgecolor="black", label=f"REVIEW-only flag @ t={review_only_step}")
    ax2.axhline(THRESH, color="purple", ls=":", lw=1.2, label=f"Confirm thr={THRESH:.2f}")
    ax2.axhline(REVIEW_THRESHOLD, color="#b8860b", ls=":", lw=1.0, alpha=0.7, label=f"Review thr={REVIEW_THRESHOLD:.2f}")
    ax2.set_ylim(0.15, 1.05)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Posterior (contact confidence)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Posterior track with detection marker")
    fig2.suptitle(f"JOR Maritime Concept Demo - Example Track (seed {seed})", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    poster_path = os.path.join(OUT_DIR, f"jor_maritime_seed_{seed}_poster.png")
    fig2.savefig(poster_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved poster frame -> {poster_path}")

    # ------------------------------------------------------------------
    # Curated MP4: 6 scenarios back-to-back (4 confirmed, 1 caught only
    # on review, 1 honest miss), 10s each at 6fps = 60s total. Uses the
    # SAME live TUNED_PARAMS/TUNED_DETECT/thresholds computed above, not
    # hardcoded values, so this stays in sync with the rest of the run
    # automatically. Picks the curated seeds' actual outcomes at runtime
    # rather than assuming -- outcomes can shift when the model changes
    # (this happened once already: an earlier hardcoded "review catch"
    # example silently became a miss after a threshold fix).
    # ------------------------------------------------------------------
    print("\n--- Generating curated MP4 (6 scenarios, confirmed/review/miss mix) ---")
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        # CURATED_SEEDS / curated_info already computed above (shared with
        # the overview + poster PNGs) -- reused here, not recomputed, so
        # all three assets are guaranteed to show the exact same mix.
        for s in CURATED_SEEDS:
            print(f"  seed {s}: {curated_info[s]['status']}")

        FPS = 6
        TOTAL_FRAMES = TIME_STEPS * len(CURATED_SEEDS)
        fig3, (axm1, axm2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1.1, 1.4]})
        mp4_artists = {}

        def draw_static_mp4(seed_idx):
            mseed = CURATED_SEEDS[seed_idx]
            md = curated_info[mseed]
            msc, mpost = md["sc"], md["post"]
            axm1.clear(); axm2.clear()
            for ev in msc["events"]:
                axm1.axvspan(ev["onset"], ev["onset"] + ev["duration"] - 0.5, color="red", alpha=0.15)
                axm2.axvspan(ev["onset"], ev["onset"] + ev["duration"] - 0.5, color="red", alpha=0.15)
                axm2.text(ev["onset"] + 0.4, 0.99, ev["phenotype"].replace("_", " "), fontsize=8, va="top", color="#8b0000")
            for fe in msc["false_events"]:
                axm1.axvspan(fe["onset"], fe["onset"] + fe["duration"] - 0.5, color="gray", alpha=0.10)
                axm2.axvspan(fe["onset"], fe["onset"] + fe["duration"] - 0.5, color="gray", alpha=0.10)
            axm2.axhline(train_res["threshold"], color="purple", ls=":", lw=1.2, alpha=0.7, label=f"Confirm thr={train_res['threshold']:.2f}")
            axm2.axhline(REVIEW_THRESHOLD, color="#b8860b", ls=":", lw=1.0, alpha=0.6, label=f"Review thr={REVIEW_THRESHOLD:.2f}")
            axm1.set_ylim(0.15, 1.0); axm1.set_xlim(0, TIME_STEPS - 1)
            axm1.set_ylabel("C / E / P"); axm1.grid(True, alpha=0.25)
            axm1.set_title(f"Seed {mseed}  ({seed_idx+1}/{len(CURATED_SEEDS)})  --  JOR Maritime Concept Demo", fontsize=12, fontweight="bold")
            axm2.set_ylim(0.15, 1.05); axm2.set_xlim(0, TIME_STEPS - 1)
            axm2.set_xlabel("Time step"); axm2.set_ylabel("Posterior (contact confidence)")
            axm2.grid(True, alpha=0.25)
            l_c, = axm1.plot([], [], color="#1f77b4", lw=1.6, label="C (sensor confidence)")
            l_e, = axm1.plot([], [], color="#2ca02c", lw=1.6, label="E (acoustic evidence)")
            l_p, = axm1.plot([], [], color="#ff7f0e", lw=1.6, label="P (physical/track)")
            l_post, = axm2.plot([], [], color="#d62728", lw=2.4, label="Posterior")
            s_c = axm2.scatter([], [], marker="D", s=120, color="purple", zorder=6, label="CONFIRMED flag")
            s_r = axm2.scatter([], [], marker="^", s=110, color="#b8860b", edgecolor="black", linewidths=1.2, zorder=6, label="REVIEW-only flag")
            axm1.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
            axm2.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
            st_text = axm2.text(0.015, 0.93, "", transform=axm2.transAxes, fontsize=11, fontweight="bold",
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            t_text = axm2.text(0.985, 0.93, "", transform=axm2.transAxes, fontsize=9, ha="right",
                                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            return l_c, l_e, l_p, l_post, s_c, s_r, st_text, t_text

        def update_mp4(frame):
            seed_idx = frame // TIME_STEPS
            local_t = frame % TIME_STEPS
            mseed = CURATED_SEEDS[seed_idx]
            md = curated_info[mseed]
            if local_t == 0:
                lc, le, lp, lpost, sc_, sr_, stt, tt = draw_static_mp4(seed_idx)
                mp4_artists.update(line_c=lc, line_e=le, line_p=lp, line_post=lpost,
                                    scat_c=sc_, scat_r=sr_, status_text=stt, time_text=tt)
            t = np.arange(local_t + 1)
            msc, mpost = md["sc"], md["post"]
            mp4_artists["line_c"].set_data(t, msc["C"][:local_t + 1])
            mp4_artists["line_e"].set_data(t, msc["E"][:local_t + 1])
            mp4_artists["line_p"].set_data(t, msc["P"][:local_t + 1])
            mp4_artists["line_post"].set_data(t, mpost[:local_t + 1])
            if md["flag_c"] is not None and local_t >= md["flag_c"]:
                mp4_artists["scat_c"].set_offsets([[md["flag_c"], mpost[md["flag_c"]]]])
            else:
                mp4_artists["scat_c"].set_offsets(np.empty((0, 2)))
            if md["flag_r"] is not None and local_t >= md["flag_r"]:
                mp4_artists["scat_r"].set_offsets([[md["flag_r"], mpost[md["flag_r"]]]])
            else:
                mp4_artists["scat_r"].set_offsets(np.empty((0, 2)))
            reveal_at = md["flag_c"] if md["flag_c"] is not None else (md["flag_r"] if md["flag_r"] is not None else TIME_STEPS - 1)
            if local_t >= reveal_at:
                mp4_artists["status_text"].set_text(md["status"])
                mp4_artists["status_text"].set_color(md["status_color"])
            else:
                mp4_artists["status_text"].set_text("")
            mp4_artists["time_text"].set_text(f"t = {local_t}")
            return tuple(mp4_artists.values())

        fig3.suptitle("JOR Maritime Concept Demo -- Curated Example Tracks\n"
                       "Purple diamond = CONFIRMED flag | Gold triangle = REVIEW-only flag | Red shading = true contact",
                       fontsize=11, fontweight="bold")
        anim = FuncAnimation(fig3, update_mp4, frames=TOTAL_FRAMES, blit=False, repeat=False)
        mp4_path = os.path.join(OUT_DIR, "jor_maritime_curated_examples.mp4")
        writer = FFMpegWriter(fps=FPS, metadata=dict(title="JOR Maritime Concept Demo - Curated Examples"), bitrate=2200)
        anim.save(mp4_path, writer=writer, dpi=120)
        plt.close(fig3)
        print(f"  Saved curated MP4 -> {mp4_path}  ({TOTAL_FRAMES/FPS:.0f}s)")
    except Exception as e:
        print(f"  WARNING: curated MP4 generation failed ({e!r}) -- skipping. "
              f"PNGs above are unaffected.")

    print("\nAll track visualizations written to:", OUT_DIR)
    print("=" * 78)
