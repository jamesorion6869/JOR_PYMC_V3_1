"""JOR V3.1-enriched: tuned-weights study (stationary and drifting noise floor).

Question: when JOR's evidence weights (W_C, W_E, W_P) and K are tuned to the data, as JOR is
designed to be, and the baselines are tuned the same way, how do they compare?

Protocol
    TRAIN:  1,000 scenarios (seeds 0-999)           weights, K and thresholds are selected here only
    VAL:      500 scenarios (seeds 10,000-10,499)   held-out check; never used for selection
    TEST:   1,000 scenarios (seeds 60,000-60,999)   evaluated once, after the design was frozen

Tuning (per condition, TRAIN only)
    JOR:           150 candidates of (W_C, W_E, W_P, K): the original defaults, the author's
                   hand-picked set (0.51, 0.03, 0.46, 0.18), and 148 draws from a fixed-seed
                   random search (weights ~ Dirichlet(1,1,1), K ~ U(0.05, 0.40)). Memory settings
                   (prior retention 0.70, posterior alpha 0.30) are held at their defaults.
    C/E/P average: the same 150 weight sets, averaged directly (no Bayes step, no memory).
    E+P average:   E weight w in {0.0, 0.1, ..., 1.0}, P weight 1 - w.
    Each candidate gets its detection threshold chosen on TRAIN under the FPR budget (0.12) on a
    coarse threshold grid; the candidate with the highest TRAIN recall (ties: F1, then earlier)
    is kept and its threshold is re-selected on the full grid. VAL and TEST play no part.

Contextualization
    A causal CA-CFAR runs inside EnrichedDSP on the SNR channel (guard 3, training 12 cells).
    Absolute SNR becomes "excess SNR" above the local floor and feeds the SNR term / trend /
    consistency inside E and P. The single-channel SNR tracker runs in its native dB domain.

Conditions
    STATIONARY: the original generator.
    DRIFTING:   each scenario's SNR baseline ramps linearly from 0 dB to A dB over the 60
                steps, A = +/- U(3, 6) dB, on its own random stream.

Flags
    --dev            TEST = seeds 20,000-20,999 (development range); code check, not a result.
    --candidates N   size of the weight search (default 150; other values only with --dev).
    --jobs N         worker processes for the search (fork platforms only; results identical).
"""

import hashlib
import multiprocessing as mp
import sys
import time
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIME_STEPS = 60
P_PRIOR_NH_INITIAL = 0.20
P_PRIOR_H_INITIAL = 0.80
ASSUMED_WITNESS_C = 0.65

TRAIN_SEEDS = list(range(0, 1_000))
VAL_SEEDS = list(range(10_000, 10_500))
DEV_RUN = "--dev" in sys.argv
TEST_SEEDS = list(range(20_000, 21_000)) if DEV_RUN else list(range(60_000, 61_000))
# Seed ranges used while developing the benchmark; the final TEST block must not touch them.
DEVELOPMENT_SEED_RANGES = [
    (0, 999), (10_000, 10_499), (20_000, 20_999), (30_000, 30_999), (40_000, 40_999), (50_000, 50_999),
]


def _flag_value(name, default):
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


N_CANDIDATES = _flag_value("--candidates", 150)
N_JOBS = _flag_value("--jobs", 1)
SEARCH_SEED = 2026
K_RANGE = (0.05, 0.40)
AUTHOR_WEIGHTS = dict(W_C=0.51, W_E=0.03, W_P=0.46, K=0.18)
if not DEV_RUN and N_CANDIDATES != 150:
    raise SystemExit("The final run must use the default 150 candidates.")
DRIFT_AMP_RANGE_DB = (3.0, 6.0)

SUSTAIN_STEPS = 3
LAG_ALLOWANCE = 3
FPR_BUDGET = 0.12

DEFAULT_PARAMS = {
    "W_C": 0.40,
    "W_E": 0.30,
    "W_P": 0.30,
    "K": 0.20,
    "PRIOR_RETENTION": 0.70,
    "POSTERIOR_ALPHA": 0.30,
}

CFAR_GUARD = 3
CFAR_TRAIN = 12
CFAR_MIN_TRAIN = 4

ABS_SNR_RANGE = (10.0, 30.0)
REL_SNR_RANGE = (-5.0, 15.0)

NATIVE_STEP_DB = 0.1
NATIVE_RISE_DB = (None, 0.5, 1.0, 2.0)
NATIVE_RISE_MARGIN_DB = 1.0
EXCESS_THR_RANGE_DB = (1.0, 15.0)
ABS_THR_RANGE_DB = (10.0, 30.0)


# ---------------------------------------------------------------------------
# Environment and synthetic-scenario definitions
# ---------------------------------------------------------------------------

TRANSITIONS = {
    "NOMINAL": {"NOMINAL": 0.92, "DEGRADED": 0.08},
    "DEGRADED": {
        "DEGRADED": 0.55,
        "SEVERELY_DEGRADED": 0.15,
        "RECOVERING": 0.30,
    },
    "SEVERELY_DEGRADED": {
        "SEVERELY_DEGRADED": 0.55,
        "RECOVERING": 0.45,
    },
    "RECOVERING": {"RECOVERING": 0.35, "NOMINAL": 0.65},
}

CONDITION_EFFECTS = {
    "NOMINAL": dict(snr_shift=0.0, var_mult=0.55, dropout_p=0.008, detp_shift=0.0),
    "DEGRADED": dict(snr_shift=-2.5, var_mult=1.5, dropout_p=0.07, detp_shift=-0.06),
    "SEVERELY_DEGRADED": dict(
        snr_shift=-7.0, var_mult=3.0, dropout_p=0.22, detp_shift=-0.20
    ),
    "RECOVERING": dict(snr_shift=-1.0, var_mult=0.85, dropout_p=0.02, detp_shift=-0.02),
}

PHENOTYPES = [
    "A_STRONG_MULTI",
    "C_INTERMITTENT",
    "G_GRADUAL_ONSET",
    "E_CONFLICTING",
    "F_SPARSE",
]

FALSE_CATEGORIES = [
    "CLUTTER",
    "MULTIPATH",
    "TRANSIENT_SPIKE",
    "TRACK_INSTABILITY",
]


def step_condition(rng, state):
    """Sample the next environmental state from the transition table."""
    draw = rng.random()
    cumulative_probability = 0.0

    for next_state, probability in TRANSITIONS[state].items():
        cumulative_probability += probability
        if draw < cumulative_probability:
            return next_state

    return state


class AR1:
    """First-order autoregressive noise process."""

    def __init__(self, phi, sigma):
        self.phi = phi
        self.sigma = sigma
        self.state = 0.0

    def step(self, rng, var_mult=1.0):
        innovation = rng.normal(0, self.sigma * np.sqrt(var_mult))
        self.state = self.phi * self.state + innovation
        return self.state


def make_ar_bank():
    """Create independent AR(1) processes for the simulated sensor channels."""
    return {
        "snr": AR1(phi=0.60, sigma=1.0),
        "rcs": AR1(phi=0.70, sigma=0.30),
        "power": AR1(phi=0.60, sigma=35.0),
        "range_res": AR1(phi=0.50, sigma=0.35),
        "doppler": AR1(phi=0.50, sigma=0.8),
    }


def swerling_rcs(rng, mean_rcs):
    """Generate a simplified Swerling-like RCS observation."""
    if rng.random() < 0.65:
        return max(0.45 * mean_rcs, rng.exponential(mean_rcs))
    return max(
        0.55 * mean_rcs,
        mean_rcs + rng.normal(0, 0.30 * mean_rcs),
    )


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
        events.append(
            dict(onset=onset, duration=duration, phenotype=phenotype)
        )
        cursor = onset + duration + int(rng.integers(3, 10))

    return events


def sample_false_event_schedule(rng):
    events = []
    cursor = 3
    n_events = rng.integers(0, 3)

    for _ in range(n_events):
        onset = cursor + int(rng.integers(0, 8))
        duration = int(rng.integers(2, 6))
        if onset + duration > TIME_STEPS - 1:
            break

        category = FALSE_CATEGORIES[rng.integers(0, len(FALSE_CATEGORIES))]
        events.append(dict(onset=onset, duration=duration, category=category))
        cursor = onset + duration + int(rng.integers(2, 8))

    return events


def in_window(step, event):
    return event["onset"] <= step < event["onset"] + event["duration"]


def phenotype_effect(phenotype, progress):
    effects = dict(
        rcs=0.0,
        snr=0.0,
        detp=0.0,
        trackc=0.0,
        doppler=0.0,
        extra_dropout=0.0,
    )

    if phenotype == "A_STRONG_MULTI":
        effects.update(rcs=2.5, snr=4.0, detp=0.15, trackc=0.05)
    elif phenotype == "C_INTERMITTENT":
        is_on = int(progress * 6) % 2 == 0
        amplitude = 1.0 if is_on else 0.0
        effects.update(
            rcs=2.0 * amplitude,
            snr=3.0 * amplitude,
            detp=0.12 * amplitude,
        )
    elif phenotype == "G_GRADUAL_ONSET":
        amplitude = min(1.0, progress * 1.6)
        effects.update(
            rcs=2.2 * amplitude,
            snr=3.5 * amplitude,
            detp=0.13 * amplitude,
            trackc=0.04 * amplitude,
        )
    elif phenotype == "E_CONFLICTING":
        effects.update(rcs=3.0, snr=-2.5, detp=-0.10, trackc=0.03)
    elif phenotype == "F_SPARSE":
        effects.update(rcs=1.0, snr=0.5, detp=0.03, extra_dropout=0.55)

    return effects


def false_event_effect(category):
    effects = dict(
        rcs=0.0,
        snr=0.0,
        detp=0.0,
        trackc=0.0,
        doppler=0.0,
        multipath=0.0,
    )

    if category == "CLUTTER":
        effects.update(rcs=1.8, snr=1.0, detp=0.05, trackc=-0.05)
    elif category == "MULTIPATH":
        effects.update(
            rcs=0.3, snr=-1.0, detp=-0.03, trackc=-0.08, doppler=0.5, multipath=0.10
        )
    elif category == "TRANSIENT_SPIKE":
        effects.update(rcs=2.5, snr=2.0, detp=0.02, doppler=1.0)
    elif category == "TRACK_INSTABILITY":
        effects.update(rcs=0.2, snr=-0.5, detp=-0.02, trackc=-0.15, doppler=0.3)

    return effects


# ---------------------------------------------------------------------------
# DSP preprocessing and evidence mapping
# ---------------------------------------------------------------------------

class EnrichedDSP:
    """Rolling sensor preprocessing with optional causal CA-CFAR on SNR."""

    def __init__(
        self,
        window=7,
        hist=5,
        use_cfar=True,
        cfar_guard=CFAR_GUARD,
        cfar_train=CFAR_TRAIN,
    ):
        self.use_cfar = use_cfar
        self.cfar_guard = cfar_guard
        self.cfar_train = cfar_train

        self.buffers = {
            key: deque(maxlen=window)
            for key in ("snr", "power", "doppler", "range_res")
        }
        self.cfar_cells = deque(maxlen=cfar_guard + cfar_train)
        self.cfar_floor = None

        self.smooth_rcs = None
        self.prev_innovation = 0.0
        self.innov_var = 0.15

        self.snr_hist = deque(maxlen=hist)
        self.innov_hist = deque(maxlen=hist)
        self.rcs_hist = deque(maxlen=hist)
        self.detp_hist = deque(maxlen=hist)
        self.valid_hist = deque(maxlen=hist)

    def cfar_excess(self, snr_obs):
        """Return SNR above the causal noise estimate, excluding guard cells."""
        cells = np.asarray(self.cfar_cells, dtype=float)
        training_cells = (
            cells[:-self.cfar_guard]
            if len(cells) > self.cfar_guard
            else cells[:0]
        )

        if len(training_cells) >= CFAR_MIN_TRAIN:
            self.cfar_floor = float(training_cells.mean())
            excess = float(snr_obs - self.cfar_floor)
        else:
            self.cfar_floor = None
            excess = 0.0

        self.cfar_cells.append(float(snr_obs))
        return excess

    def process(self, raw, rcs_available, det_prob_obs):
        filtered = {}
        n_valid = 0
        n_possible = 0

        for key, buffer in self.buffers.items():
            n_possible += 1
            if raw[key] is not None:
                value = raw[key]
                if key == "snr" and self.use_cfar:
                    value = self.cfar_excess(raw[key])
                buffer.append(value)
                n_valid += 1

            filtered[key] = float(np.mean(buffer)) if buffer else None

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
                gated = np.clip(
                    innovation,
                    -2.5 * np.sqrt(self.innov_var),
                    2.5 * np.sqrt(self.innov_var),
                )
                alpha = 0.18
                self.smooth_rcs = (
                    alpha * (self.smooth_rcs + gated)
                    + (1 - alpha) * self.smooth_rcs
                )
                self.innov_var = (
                    0.85 * self.innov_var + 0.15 * (innovation ** 2)
                )

            self.prev_innovation = (
                0.82 * self.prev_innovation + 0.18 * abs(innovation)
            )

        filtered["rcs"] = self.smooth_rcs

        if self.smooth_rcs:
            innov_norm = abs(self.prev_innovation) / (
                0.30 + 0.06 * abs(self.smooth_rcs) + 1e-6
            )
        else:
            innov_norm = 0.0

        filtered["track_consistency"] = float(
            np.clip(0.95 - 0.25 * innov_norm, 0.55, 0.985)
        )
        filtered["maneuver_index"] = float(
            np.clip(0.04 + 0.09 * self.prev_innovation, 0.0, 0.45)
        )

        snr_value = filtered["snr"]
        if snr_value is not None:
            self.snr_hist.append(snr_value)
        self.innov_hist.append(abs(innovation))
        if filtered["rcs"] is not None:
            self.rcs_hist.append(filtered["rcs"])
        if det_prob_obs is not None:
            self.detp_hist.append(det_prob_obs)

        snr_history = np.asarray(self.snr_hist, dtype=float)
        innov_history = np.asarray(self.innov_hist, dtype=float)
        rcs_history = np.asarray(self.rcs_hist, dtype=float)
        det_history = np.asarray(self.detp_hist, dtype=float)

        snr_trend_n = (
            float(
                np.clip(
                    (
                        (snr_history[-1] - snr_history[0])
                        / max(len(snr_history) - 1, 1)
                    )
                    / 3.0,
                    -1.0,
                    1.0,
                )
            )
            if len(snr_history) >= 3
            else 0.0
        )
        innov_trend_n = (
            float(
                np.clip(
                    (
                        (innov_history[-1] - innov_history[0])
                        / max(len(innov_history) - 1, 1)
                    )
                    / 0.5,
                    -1.0,
                    1.0,
                )
            )
            if len(innov_history) >= 3
            else 0.0
        )

        consistency = 0.5
        if (
            len(rcs_history) >= 3
            and len(snr_history) >= 3
            and len(det_history) >= 3
        ):
            delta_rcs = rcs_history[-1] - rcs_history[0]
            delta_snr = snr_history[-1] - snr_history[0]
            delta_det = det_history[-1] - det_history[0]
            signs = [
                np.sign(delta_rcs),
                np.sign(delta_snr),
                np.sign(delta_det),
            ]
            consistency = max(
                sum(1 for sign in signs if sign > 0),
                sum(1 for sign in signs if sign < 0),
            ) / 3.0

        filtered["snr_trend"] = snr_trend_n
        filtered["innov_trend"] = innov_trend_n
        filtered["consistency"] = float(consistency)
        filtered["valid_frac"] = (
            float(np.mean(self.valid_hist)) if self.valid_hist else 1.0
        )
        filtered["innov_mag"] = float(
            np.clip(self.prev_innovation / 0.8, 0.0, 1.0)
        )
        filtered["cfar_floor"] = self.cfar_floor

        return filtered


def normalize(value, v_min, v_max):
    if value is None:
        return None
    return (max(min(value, v_max), v_min) - v_min) / (v_max - v_min)


def wavg(components):
    numerator = sum(value * weight for value, weight in components if value is not None)
    denominator = sum(weight for value, weight in components if value is not None)
    return numerator / denominator if denominator > 0 else None


def map_evidence(rd, shift, det_obs, multipath, snr_range):
    track_consistency = float(
        np.clip(rd["track_consistency"] + shift["trackc"], 0.0, 1.0)
    )

    evidence = wavg(
        [
            (normalize(rd["snr"], snr_range[0], snr_range[1]), 0.30),
            (normalize(rd["power"], 800.0, 2000.0), 0.20),
            (normalize(rd["rcs"], 0.5, 10.0), 0.20),
            (
                1.0 - normalize(rd["range_res"], 10.0, 25.0)
                if rd["range_res"] is not None
                else None,
                0.10,
            ),
            ((rd["snr_trend"] + 1.0) / 2.0, 0.10),
            (rd["valid_frac"], 0.10),
        ]
    )

    probability_raw = wavg(
        [
            (det_obs, 0.18),
            (normalize(rd["rcs"], 0.5, 10.0), 0.12),
            (normalize(rd["power"], 800.0, 2000.0), 0.12),
            (track_consistency, 0.15),
            (1.0 - rd["maneuver_index"], 0.08),
            (1.0 - multipath, 0.08),
            (
                1.0 - normalize(rd["doppler"], 5.0, 25.0)
                if rd["doppler"] is not None
                else None,
                0.05,
            ),
            (rd["consistency"], 0.12),
            (1.0 - rd["innov_mag"], 0.05),
            (rd["valid_frac"], 0.05),
        ]
    )

    return evidence, probability_raw


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_scenario(
    seed,
    cfar_guard=CFAR_GUARD,
    cfar_train=CFAR_TRAIN,
    with_abs=True,
    drifting_floor=False,
):
    seed_sequence = np.random.SeedSequence(seed)
    (
        child_events,
        child_false,
        child_condition,
        child_sensor,
        child_dropout,
        child_drift,
    ) = seed_sequence.spawn(6)

    rng_events = np.random.default_rng(child_events)
    rng_false = np.random.default_rng(child_false)
    rng_condition = np.random.default_rng(child_condition)
    rng_sensor = np.random.default_rng(child_sensor)
    rng_dropout = np.random.default_rng(child_dropout)
    rng_drift = np.random.default_rng(child_drift)

    events = sample_event_schedule(rng_events)
    false_events = sample_false_event_schedule(rng_false)

    ground_truth = np.zeros(TIME_STEPS, dtype=bool)
    for event in events:
        event_end = min(event["onset"] + event["duration"], TIME_STEPS)
        for step in range(event["onset"], event_end):
            ground_truth[step] = True

    condition = "NOMINAL"
    ar = make_ar_bank()
    dsp_rel = EnrichedDSP(
        use_cfar=True,
        cfar_guard=cfar_guard,
        cfar_train=cfar_train,
    )
    dsp_abs = EnrichedDSP(use_cfar=False) if with_abs else None

    C_track, E_track, P_track = [], [], []
    E_abs_track, P_abs_track = [], []
    exc_track, snr_abs_track, consistency_track = [], [], []
    phenotype_track = [None] * TIME_STEPS
    availability_track = {
        channel: [] for channel in ("snr", "rcs", "power", "doppler", "range_res", "detp")
    }

    # Slow noise-floor drift: 0 dB at step 0 to drift_amp dB at the last step, random sign.
    # Drawn from its own stream so no other generated quantity depends on the condition.
    drift_amp = (
        float(rng_drift.choice([-1.0, 1.0]) * rng_drift.uniform(*DRIFT_AMP_RANGE_DB))
        if drifting_floor
        else 0.0
    )

    for step in range(TIME_STEPS):
        condition = step_condition(rng_condition, condition)
        condition_effect = CONDITION_EFFECTS[condition]

        active_true = next(
            (event for event in events if in_window(step, event)), None
        )
        active_false = next(
            (event for event in false_events if in_window(step, event)), None
        )

        if active_true:
            phenotype_track[step] = active_true["phenotype"]

        shift = dict(
            rcs=0.0,
            snr=0.0,
            detp=0.0,
            trackc=0.0,
            doppler=0.0,
            multipath=0.0,
            extra_dropout=0.0,
        )

        if active_true:
            progress = (step - active_true["onset"]) / max(
                active_true["duration"], 1
            )
            for key, value in phenotype_effect(
                active_true["phenotype"], progress
            ).items():
                shift[key] = shift.get(key, 0.0) + value

        if active_false:
            for key, value in false_event_effect(active_false["category"]).items():
                shift[key] = shift.get(key, 0.0) + value

        dropout_probability = min(
            0.95,
            condition_effect["dropout_p"] + shift.get("extra_dropout", 0.0),
        )
        available = {
            channel: rng_dropout.random() > dropout_probability
            for channel in ("snr", "rcs", "power", "doppler", "range_res", "detp")
        }
        for channel, is_available in available.items():
            availability_track[channel].append(bool(is_available))

        # Underlying SNR, plus the noise-floor drift when that condition is active.
        snr_base = (
            18.0
            + condition_effect["snr_shift"]
            + shift["snr"]
            + ar["snr"].step(rng_sensor, condition_effect["var_mult"])
            + rng_sensor.normal(0, 0.4)
        )
        if drifting_floor:
            snr_base += drift_amp * (step / (TIME_STEPS - 1))

        true_rcs = swerling_rcs(rng_sensor, 3.0 + shift["rcs"])
        rcs_value = max(
            0.15,
            true_rcs + ar["rcs"].step(rng_sensor, condition_effect["var_mult"]),
        )
        peak_power = max(
            150.0,
            1400.0
            * (rcs_value / 3.0)
            * (10 ** ((snr_base - 18.0) / 10.0))
            + ar["power"].step(rng_sensor, condition_effect["var_mult"]),
        )
        doppler = max(
            1.0,
            10.0
            + shift["doppler"]
            + ar["doppler"].step(rng_sensor, condition_effect["var_mult"]),
        )
        range_res = max(
            5.0,
            15.0 + ar["range_res"].step(rng_sensor, condition_effect["var_mult"]),
        )
        det_prob = float(
            np.clip(
                0.75
                + condition_effect["detp_shift"]
                + shift["detp"]
                + rng_sensor.normal(0, 0.02),
                0.05,
                0.99,
            )
        )
        multipath = float(
            np.clip(
                0.03
                + shift.get("multipath", 0.0)
                + rng_sensor.normal(0, 0.01),
                0.0,
                1.0,
            )
        )

        raw = {
            "snr": snr_base if available["snr"] else None,
            "power": peak_power if available["power"] else None,
            "doppler": doppler if available["doppler"] else None,
            "range_res": range_res if available["range_res"] else None,
            "rcs": rcs_value if available["rcs"] else None,
        }
        det_obs = det_prob if available["detp"] else None

        rd_rel = dsp_rel.process(raw, available["rcs"], det_obs)
        rd_abs = (
            dsp_abs.process(raw, available["rcs"], det_obs)
            if with_abs
            else None
        )
        C = float(
            np.clip(ASSUMED_WITNESS_C + rng_sensor.normal(0, 0.02), 0.30, 0.85)
        )

        E_r, P_r = map_evidence(
            rd_rel, shift, det_obs, multipath, REL_SNR_RANGE
        )
        E_a, P_a = (
            map_evidence(rd_abs, shift, det_obs, multipath, ABS_SNR_RANGE)
            if with_abs
            else (None, None)
        )

        C_track.append(float(np.clip(C, 0.30, 0.85)))
        E_track.append(
            float(
                np.clip(
                    E_r if E_r is not None else (E_track[-1] if E_track else 0.5),
                    0.30,
                    0.85,
                )
            )
        )
        P_track.append(
            float(
                np.clip(
                    P_r if P_r is not None else (P_track[-1] if P_track else 0.5),
                    0.30,
                    0.95,
                )
            )
        )

        if with_abs:
            E_abs_track.append(
                float(
                    np.clip(
                        E_a
                        if E_a is not None
                        else (E_abs_track[-1] if E_abs_track else 0.5),
                        0.30,
                        0.85,
                    )
                )
            )
            P_abs_track.append(
                float(
                    np.clip(
                        P_a
                        if P_a is not None
                        else (P_abs_track[-1] if P_abs_track else 0.5),
                        0.30,
                        0.95,
                    )
                )
            )

        exc_track.append(
            rd_rel["snr"]
            if rd_rel["snr"] is not None
            else (exc_track[-1] if exc_track else 0.0)
        )
        if with_abs:
            snr_abs_track.append(
                rd_abs["snr"]
                if rd_abs["snr"] is not None
                else (snr_abs_track[-1] if snr_abs_track else 18.0)
            )

        # Retain cross-feature consistency for adaptive JOR memory decay.
        consistency_track.append(rd_rel["consistency"])

    return {
        "C": np.array(C_track),
        "E": np.array(E_track),
        "P": np.array(P_track),
        "E_abs": np.array(E_abs_track),
        "P_abs": np.array(P_abs_track),
        "EXC": np.array(exc_track),
        "SNR_ABS": np.array(snr_abs_track),
        "consistency_track": np.array(consistency_track),
        "drift_amp": drift_amp,
        "ground_truth": ground_truth,
        "events": events,
        "false_events": false_events,
        "phenotype_track": phenotype_track,
        "availability_track": {
            channel: np.asarray(values, dtype=bool)
            for channel, values in availability_track.items()
        },
    }


# ---------------------------------------------------------------------------
# JOR temporal update and event detection
# ---------------------------------------------------------------------------

def run_jor(scenario, params, modifier_track=None, evidence="rel", adaptive=False):
    """JOR update loop. adaptive=True lowers prior retention and posterior smoothing
    when cross-feature consistency falls (variant); adaptive=False is the original JOR."""
    prior_NH = P_PRIOR_NH_INITIAL
    prior_H = P_PRIOR_H_INITIAL
    smooth_post = None
    posterior_track = np.empty(TIME_STEPS)

    suffix = "" if evidence == "rel" else "_abs"
    C = scenario["C"]
    E = scenario["E" + suffix]
    P = scenario["P" + suffix]
    consistency_track = scenario["consistency_track"]

    modifier = (
        modifier_track
        if modifier_track is not None
        else np.zeros(TIME_STEPS)
    )

    for step in range(TIME_STEPS):
        SOP = (
            params["W_C"] * C[step]
            + params["W_E"] * E[step]
            + params["W_P"] * P[step]
        )
        P_for_NHP = min(max(P[step] + modifier[step], 0.0), 0.95)
        NHP = (
            params["W_C"] * C[step]
            + params["W_E"] * E[step]
            + params["W_P"] * P_for_NHP
        )
        P_E_given_NH = NHP
        P_E_given_H = min(
            max(1.0 - NHP + params["K"] * SOP, 0.0),
            1.0,
        )

        numerator = P_E_given_NH * prior_NH
        denominator = numerator + P_E_given_H * prior_H
        post = numerator / denominator if denominator > 0 else 0.0

        if adaptive:
            consistency = consistency_track[step]
            dynamic_retention = max(
                params["PRIOR_RETENTION"] * (consistency ** 2),
                0.15,
            )
            dynamic_alpha = max(params["POSTERIOR_ALPHA"] * consistency, 0.10)
        else:
            dynamic_retention = params["PRIOR_RETENTION"]
            dynamic_alpha = params["POSTERIOR_ALPHA"]

        smooth_post = (
            post
            if smooth_post is None
            else dynamic_alpha * post + (1 - dynamic_alpha) * smooth_post
        )
        posterior_track[step] = smooth_post

        prior_NH = (
            dynamic_retention * post
            + (1 - dynamic_retention) * P_PRIOR_NH_INITIAL
        )
        prior_H = 1.0 - prior_NH

    return posterior_track


def detect_temporal(
    track,
    level_thr,
    rise_thr=0.04,
    sustain=SUSTAIN_STEPS,
    rise_window=4,
    rise_margin=0.06,
):
    n_steps = len(track)

    for step in range(sustain - 1, n_steps):
        sustained_window = track[step - sustain + 1 : step + 1]
        if sustained_window.min() > level_thr:
            return True, step - sustain + 1

        if rise_thr is not None and step >= rise_window - 1:
            rise = track[step] - track[step - rise_window + 1]
            if rise >= rise_thr and sustained_window.min() > (level_thr - rise_margin):
                return True, step - sustain + 1

    return False, None


def find_matched_event_onset(flag_step, events, lag=LAG_ALLOWANCE):
    if flag_step is None or not events:
        return None

    candidates = []
    for event in events:
        start = event["onset"]
        end = event["onset"] + event["duration"] - 1

        if flag_step < start:
            distance = start - flag_step
        elif flag_step > end:
            distance = flag_step - end
        else:
            distance = 0

        if distance <= lag:
            candidates.append((distance, event["onset"], event))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    return candidates[0][2]["onset"]


def flag_matches_truth_and_get_onset(
    flag_step,
    ground_truth,
    events,
    lag=LAG_ALLOWANCE,
):
    if flag_step is None:
        return False, None

    matched_onset = find_matched_event_onset(flag_step, events, lag=lag)
    if matched_onset is None:
        return False, None

    return True, matched_onset


def scenario_has_event(ground_truth):
    return bool(ground_truth.any())


# ---------------------------------------------------------------------------
# Benchmark configurations and evaluation
# ---------------------------------------------------------------------------

def unit_configs():
    return [
        dict(level_thr=float(threshold), rise_thr=0.04, rise_margin=0.06)
        for threshold in np.round(np.arange(0.200, 0.900, 0.002), 3)
    ]


def native_configs(lo, hi):
    levels = np.round(np.arange(lo, hi + 1e-9, NATIVE_STEP_DB), 1)
    return [
        dict(
            level_thr=float(threshold),
            rise_thr=rise_threshold,
            rise_margin=NATIVE_RISE_MARGIN_DB,
        )
        for rise_threshold in NATIVE_RISE_DB
        for threshold in levels
    ]


def ep_average_track(scenario, evidence="rel"):
    suffix = "" if evidence == "rel" else "_abs"
    return 0.5 * scenario["E" + suffix] + 0.5 * scenario["P" + suffix]


# Parameters chosen by the TRAIN search for the condition being processed. The track functions
# read them when called; run_condition sets them before any track is built.
ACTIVE = {"jor": DEFAULT_PARAMS, "cep": DEFAULT_PARAMS, "ep_w": 0.5}


def cep_track(scenario, params):
    """Weighted average of the C, E and P evidence (no Bayes step, no memory)."""
    return params["W_C"] * scenario["C"] + params["W_E"] * scenario["E"] + params["W_P"] * scenario["P"]


def ep_weighted_track(scenario, w):
    return w * scenario["E"] + (1.0 - w) * scenario["P"]


METHODS = {
    "JOR (tuned)": dict(
        track=lambda scenario: run_jor(scenario, ACTIVE["jor"], evidence="rel"),
        configs=unit_configs(),
    ),
    "SNR (CFAR excess)": dict(
        track=lambda scenario: scenario["EXC"],
        configs=native_configs(*EXCESS_THR_RANGE_DB),
    ),
    "C/E/P avg (tuned)": dict(
        track=lambda scenario: cep_track(scenario, ACTIVE["cep"]),
        configs=unit_configs(),
    ),
    "E+P avg (tuned)": dict(
        track=lambda scenario: ep_weighted_track(scenario, ACTIVE["ep_w"]),
        configs=unit_configs(),
    ),
}

UNTUNED_METHODS = {
    "JOR (default wts)": dict(
        track=lambda scenario: run_jor(scenario, DEFAULT_PARAMS, evidence="rel"),
        configs=unit_configs(),
    ),
    "E+P avg (0.5/0.5)": dict(
        track=lambda scenario: ep_average_track(scenario, "rel"),
        configs=unit_configs(),
    ),
}

REFERENCE_METHODS = {
    "ref: JOR abs-SNR (tuned)": dict(
        track=lambda scenario: run_jor(scenario, ACTIVE["jor"], evidence="abs"),
        configs=unit_configs(),
    ),
    "ref: SNR abs dB": dict(
        track=lambda scenario: scenario["SNR_ABS"],
        configs=native_configs(*ABS_THR_RANGE_DB),
    ),
}

# Variant of the headline method, reported next to it (not a replacement).
VARIANT_METHODS = {
    "JOR+adaptive (tuned)": dict(
        track=lambda scenario: run_jor(scenario, ACTIVE["jor"], evidence="rel", adaptive=True),
        configs=unit_configs(),
    ),
}

ALL_METHODS = {**METHODS, **UNTUNED_METHODS, **VARIANT_METHODS, **REFERENCE_METHODS}
# Methods whose tracks depend on the tuned parameters.
TUNED_METHODS = ("JOR (tuned)", "C/E/P avg (tuned)", "E+P avg (tuned)", "JOR+adaptive (tuned)", "ref: JOR abs-SNR (tuned)")


def build_tracks(track_fn, seeds, cache):
    return {
        seed: np.asarray(track_fn(cache[seed]), dtype=float)
        for seed in seeds
    }


def evaluate_tracks(tracks, config, seeds, cache):
    tp = fp = fn = tn = 0
    latencies = []

    for seed in seeds:
        scenario = cache[seed]
        flagged, flag_step = detect_temporal(tracks[seed], **config)
        has_event = scenario_has_event(scenario["ground_truth"])

        correct, matched_onset = (
            flag_matches_truth_and_get_onset(
                flag_step,
                scenario["ground_truth"],
                scenario["events"],
            )
            if flagged
            else (False, None)
        )

        if has_event and flagged and correct:
            tp += 1
            latencies.append(max(0, flag_step - matched_onset))
        elif has_event:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        mean_latency=float(np.mean(latencies)) if latencies else None,
    )


def evaluate_f_sparse_recall(tracks, config, seeds, cache):
    """Evaluate scenario-level recall for scenarios containing F_SPARSE.

    The detector and thresholds are unchanged. A positive scenario must contain
    at least one F_SPARSE event, and its first detection must match an
    F_SPARSE event onset within LAG_ALLOWANCE. Scenarios without F_SPARSE are
    excluded from this recall denominator. This is a phenotype-specific recall
    diagnostic, not a separate threshold-selection or FPR calculation.
    """
    tp = 0
    fn = 0
    sparse_scenario_count = 0

    for seed in seeds:
        scenario = cache[seed]
        sparse_events = [
            event
            for event in scenario["events"]
            if event["phenotype"] == "F_SPARSE"
        ]

        if not sparse_events:
            continue

        sparse_scenario_count += 1
        flagged, flag_step = detect_temporal(tracks[seed], **config)

        matched_sparse_event = (
            find_matched_event_onset(flag_step, sparse_events)
            if flagged
            else None
        )

        if matched_sparse_event is not None:
            tp += 1
        else:
            fn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return dict(
        tp=tp,
        fn=fn,
        recall=recall,
        scenarios=sparse_scenario_count,
    )


def detect_temporal_candidates(track, level_thr, rise_thr=0.04,
                               sustain=SUSTAIN_STEPS, rise_window=4,
                               rise_margin=0.06):
    """Return all distinct detection episodes using detect_temporal's rules.

    Unlike detect_temporal(), this diagnostic does not stop at the first
    qualifying window. Consecutive qualifying windows are collapsed into one
    episode so a sustained run is not counted as many separate detections.
    The detector thresholds and trigger conditions are otherwise unchanged.
    """
    candidates = []
    previous_trigger_step = None

    for step in range(sustain - 1, len(track)):
        sustained_window = track[step - sustain + 1:step + 1]
        triggered = sustained_window.min() > level_thr

        if not triggered and rise_thr is not None and step >= rise_window - 1:
            rise = track[step] - track[step - rise_window + 1]
            triggered = (
                rise >= rise_thr
                and sustained_window.min() > (level_thr - rise_margin)
            )

        if triggered:
            if previous_trigger_step is None or step != previous_trigger_step + 1:
                candidates.append(step - sustain + 1)
            previous_trigger_step = step
        else:
            previous_trigger_step = None

    return candidates


def match_sparse_events_one_to_one(detection_steps, sparse_events, lag=LAG_ALLOWANCE):
    """Greedily match distinct detection episodes to distinct sparse events.

    Candidate pairs are ordered by temporal distance, then event onset, then
    detection step. Each detection episode and each ground-truth event can be
    credited at most once.
    """
    possible_pairs = []
    for event_index, event in enumerate(sparse_events):
        start = event["onset"]
        end = start + event["duration"] - 1
        for detection_step in detection_steps:
            if detection_step < start:
                distance = start - detection_step
            elif detection_step > end:
                distance = detection_step - end
            else:
                distance = 0

            if distance <= lag:
                possible_pairs.append(
                    (distance, start, detection_step, event_index)
                )

    possible_pairs.sort()
    used_events = set()
    used_detections = set()
    matches = []

    for distance, onset, detection_step, event_index in possible_pairs:
        if event_index in used_events or detection_step in used_detections:
            continue
        used_events.add(event_index)
        used_detections.add(detection_step)
        matches.append({
            "event": sparse_events[event_index],
            "detection_step": detection_step,
            "offset": detection_step - onset,
            "distance": distance,
        })

    return matches


def evaluate_f_sparse_diagnostics(tracks, config, seeds, cache):
    """Evaluate later detections and event-level matches for F_SPARSE.

    This is a diagnostic only: it uses the already-selected configuration,
    does not tune thresholds, and does not replace the first-detection metric
    reported in Table 3. Event-level matching is one-to-one.
    """
    sparse_scenarios = 0
    any_sparse_scenario_hits = 0
    total_sparse_events = 0
    matched_sparse_events = 0
    offsets = []
    first_detection_sparse_true = 0
    first_detection_other_true = 0
    first_detection_unmatched = 0
    no_detection = 0

    for seed in seeds:
        scenario = cache[seed]
        sparse_events = [
            event for event in scenario["events"]
            if event["phenotype"] == "F_SPARSE"
        ]
        if not sparse_events:
            continue

        sparse_scenarios += 1
        total_sparse_events += len(sparse_events)

        detection_steps = detect_temporal_candidates(tracks[seed], **config)
        matches = match_sparse_events_one_to_one(detection_steps, sparse_events)

        if matches:
            any_sparse_scenario_hits += 1
            matched_sparse_events += len(matches)
            offsets.extend(match["offset"] for match in matches)

        if not detection_steps:
            no_detection += 1
        else:
            first_step = detection_steps[0]
            first_sparse = find_matched_event_onset(first_step, sparse_events)
            if first_sparse is not None:
                first_detection_sparse_true += 1
            else:
                first_any = find_matched_event_onset(first_step, scenario["events"])
                if first_any is not None:
                    matched_event = next(
                        event for event in scenario["events"]
                        if event["onset"] == first_any
                    )
                    if matched_event["phenotype"] != "F_SPARSE":
                        first_detection_other_true += 1
                    else:
                        # A sparse event can be matched by the first-detection
                        # helper but missed by one-to-one matching only in an
                        # unusual multi-event tie; classify it as unmatched here.
                        first_detection_unmatched += 1
                else:
                    first_detection_unmatched += 1

    return {
        "scenarios": sparse_scenarios,
        "any_scenario_recall": (
            any_sparse_scenario_hits / sparse_scenarios
            if sparse_scenarios else 0.0
        ),
        "total_events": total_sparse_events,
        "matched_events": matched_sparse_events,
        "event_recall": (
            matched_sparse_events / total_sparse_events
            if total_sparse_events else 0.0
        ),
        "mean_offset": float(np.mean(offsets)) if offsets else None,
        "first_sparse_true": first_detection_sparse_true,
        "first_other_true": first_detection_other_true,
        "first_unmatched": first_detection_unmatched,
        "no_detection": no_detection,
    }



def evaluate_f_sparse_observability(seeds, cache):
    """Summarize raw sensor observability during each F_SPARSE event.

    This is upstream of all scoring methods: it measures generated channel
    availability, not detection success or inferred evidence. Event-time
    samples are pooled, while max-run statistics are averaged per event.
    """
    channel_names = ("snr", "rcs", "power", "doppler", "range_res", "detp")
    total_steps = 0
    channel_valid = {name: 0 for name in channel_names}
    steps_with_any = 0
    steps_with_three_plus = 0
    event_max_runs = []
    event_lengths = []

    for seed in seeds:
        scenario = cache[seed]
        availability = scenario["availability_track"]
        for event in scenario["events"]:
            if event["phenotype"] != "F_SPARSE":
                continue
            start = event["onset"]
            end = min(start + event["duration"], TIME_STEPS)
            if end <= start:
                continue
            mask = np.column_stack([
                availability[name][start:end] for name in channel_names
            ])
            total_steps += mask.shape[0]
            event_lengths.append(mask.shape[0])
            for idx, name in enumerate(channel_names):
                channel_valid[name] += int(mask[:, idx].sum())
            counts = mask.sum(axis=1)
            steps_with_any += int((counts >= 1).sum())
            steps_with_three_plus += int((counts >= 3).sum())

            run = best_run = 0
            for observable in (counts >= 3):
                if observable:
                    run += 1
                    best_run = max(best_run, run)
                else:
                    run = 0
            event_max_runs.append(best_run)

    if total_steps == 0:
        return {"events": 0, "event_steps": 0}

    return {
        "events": len(event_lengths),
        "event_steps": total_steps,
        "mean_event_length": float(np.mean(event_lengths)),
        "channel_pct": {
            name: 100.0 * channel_valid[name] / total_steps
            for name in channel_names
        },
        "any_channel_pct": 100.0 * steps_with_any / total_steps,
        "three_plus_pct": 100.0 * steps_with_three_plus / total_steps,
        "mean_max_three_plus_run": float(np.mean(event_max_runs)),
    }


def sweep_configs(tracks, seeds, cache, configs):
    return [
        evaluate_tracks(tracks, config, seeds, cache)
        for config in configs
    ]


def pick_config(sweep, configs, fpr_budget):
    best = None

    for config, result in zip(configs, sweep):
        if result["fpr"] <= fpr_budget:
            if (
                best is None
                or result["recall"] > best["recall"]
                or (
                    result["recall"] == best["recall"]
                    and result["f1"] > best["f1"]
                )
            ):
                best = dict(result, config=config)

    if best is None:
        config, result = min(
            zip(configs, sweep),
            key=lambda pair: pair[1]["fpr"],
        )
        best = dict(result, config=config)

    return best


# ---------------------------------------------------------------------------
# Statistics and reporting helpers
# ---------------------------------------------------------------------------

def wilson_interval(k, n, z=1.96):
    """Wilson score interval (default 95%) for a binomial proportion k / n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_ci(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson_interval(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def classify_outcome(scenario, flagged, flag_step):
    """Scenario-level outcome (TP / FN / FP / TN), same rules as evaluate_tracks()."""
    if scenario_has_event(scenario["ground_truth"]):
        if flagged:
            matched, _ = flag_matches_truth_and_get_onset(
                flag_step, scenario["ground_truth"], scenario["events"]
            )
            if matched:
                return "TP"
        return "FN"
    return "FP" if flagged else "TN"


def scenario_outcomes(tracks, config, seeds, cache):
    """Per-scenario rows (seed, outcome, latency); latency is set for true positives only."""
    rows = []
    for seed in seeds:
        scenario = cache[seed]
        flagged, flag_step = detect_temporal(tracks[seed], **config)
        outcome = classify_outcome(scenario, flagged, flag_step)
        latency = None
        if outcome == "TP":
            _, onset = flag_matches_truth_and_get_onset(
                flag_step, scenario["ground_truth"], scenario["events"]
            )
            latency = max(0, flag_step - onset)
        rows.append((seed, outcome, latency))
    return rows


def stratum_of(scenario):
    """Breakdown stratum: single-event phenotype / TWO_EVENTS, or false-event category."""
    if scenario_has_event(scenario["ground_truth"]):
        if len(scenario["events"]) == 1:
            return ("POS", scenario["events"][0]["phenotype"])
        return ("POS", "TWO_EVENTS")
    false_events = scenario["false_events"]
    if not false_events:
        return ("NEG", "NO_FALSE_EVENT")
    if len(false_events) == 1:
        return ("NEG", false_events[0]["category"])
    return ("NEG", "TWO_FALSE_EVENTS")


def print_breakdown(label, rows_by_method, cache, lat_method="JOR fusion"):
    names = list(rows_by_method)
    groups = {}
    for method, rows in rows_by_method.items():
        for seed, outcome, latency in rows:
            groups.setdefault(stratum_of(cache[seed]), {}).setdefault(method, []).append(
                (outcome, latency)
            )

    head = "".join(f"{method:<26}" for method in names)
    print(f"\n--- Breakdown on {label} (each method at its own frozen TRAIN setting) ---")
    print("Recall by event phenotype (scenarios with a true event):")
    print(f"  {'Stratum':<18}{'n':>5}   {head}{lat_method + ' latency':>20}")
    for key in sorted(groups):
        kind, name = key
        if kind != "POS":
            continue
        n = len(groups[key][names[0]])
        cells = ""
        for method in names:
            tp = sum(1 for outcome, _ in groups[key][method] if outcome == "TP")
            cells += f"{fmt_ci(tp, n):<26}"
        latencies = [lat for outcome, lat in groups[key][lat_method] if outcome == "TP"]
        lat_text = f"{np.mean(latencies):.2f}" if latencies else "n/a"
        print(f"  {name:<18}{n:>5}   {cells}{lat_text:>20}")
    print("False-positive rate by false-event category (scenarios without a true event):")
    print(f"  {'Stratum':<18}{'n':>5}   {head}")
    for key in sorted(groups):
        kind, name = key
        if kind != "NEG":
            continue
        n = len(groups[key][names[0]])
        cells = ""
        for method in names:
            fp = sum(1 for outcome, _ in groups[key][method] if outcome == "FP")
            cells += f"{fmt_ci(fp, n):<26}"
        print(f"  {name:<18}{n:>5}   {cells}")


def paired_bootstrap(rows_a, rows_b, n_boot=2000, seed=12345):
    """Paired bootstrap over scenarios (same resample for both methods).
    Returns {metric: (point, lo, hi)} for the difference A - B in recall, FPR and F1."""

    def parts(rows):
        outcomes = np.array([row[1] for row in rows])
        return (outcomes == "TP", outcomes == "FP", outcomes == "FN", outcomes == "TN")

    def metrics(p, ix=None):
        tp, fp, fn, tn = (x.sum() if ix is None else x[ix].sum(axis=1) for x in p)
        tp, fp, fn, tn = (np.asarray(v, dtype=float) for v in (tp, fp, fn, tn))
        return dict(
            recall=tp / np.maximum(tp + fn, 1),
            fpr=fp / np.maximum(fp + tn, 1),
            f1=2 * tp / np.maximum(2 * tp + fp + fn, 1),
        )

    pa, pb = parts(rows_a), parts(rows_b)
    n = len(rows_a)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    point_a, point_b = metrics(pa), metrics(pb)
    boot_a, boot_b = metrics(pa, idx), metrics(pb, idx)
    out = {}
    for key in ("recall", "fpr", "f1"):
        diff = boot_a[key] - boot_b[key]
        out[key] = (
            float(point_a[key] - point_b[key]),
            float(np.percentile(diff, 2.5)),
            float(np.percentile(diff, 97.5)),
        )
    return out


def fmt_cfg(config):
    if config["rise_margin"] != NATIVE_RISE_MARGIN_DB:
        return f"thr={config['level_thr']:.3f}"
    rise = "no rise" if config["rise_thr"] is None else f"rise>={config['rise_thr']:.1f}dB"
    return f"thr={config['level_thr']:.1f}dB {rise}"


def run_seed_partition_check():
    all_seeds = list(TRAIN_SEEDS) + list(VAL_SEEDS) + list(TEST_SEEDS)
    for name, seeds in (("TRAIN", TRAIN_SEEDS), ("VAL", VAL_SEEDS), ("TEST", TEST_SEEDS)):
        assert len(seeds) == len(set(seeds)), f"{name} contains duplicate seeds"
    assert not set(TRAIN_SEEDS) & set(VAL_SEEDS), "TRAIN and VAL seed sets overlap"
    assert not set(TRAIN_SEEDS) & set(TEST_SEEDS), "TRAIN and TEST seed sets overlap"
    assert not set(VAL_SEEDS) & set(TEST_SEEDS), "VAL and TEST seed sets overlap"
    assert len(set(all_seeds)) == len(all_seeds), "seed count does not equal sum of partitions"
    print(
        f"Seed partition check: PASS (TRAIN {len(TRAIN_SEEDS)}, VAL {len(VAL_SEEDS)}, "
        f"TEST {len(TEST_SEEDS)}, {len(all_seeds)} unique seeds)"
    )


def seed_freeze_check():
    if DEV_RUN:
        print(
            "Seed freeze check: DEV RUN. TEST uses the development range 20000-20999. "
            "This is a code check, not a result."
        )
        return
    for lo, hi in DEVELOPMENT_SEED_RANGES:
        overlap = [seed for seed in TEST_SEEDS if lo <= seed <= hi]
        assert not overlap, f"TEST block overlaps development range {lo}-{hi}"
    print(
        f"Seed freeze check: PASS (TEST block {TEST_SEEDS[0]}-{TEST_SEEDS[-1]} does not overlap "
        "any development seed range)"
    )


def run_sanity_checks(partitions, cache):
    print("\nDATASET SANITY CHECKS")
    for name, seeds in partitions.items():
        total = len(seeds)
        positives = sum(1 for s in seeds if scenario_has_event(cache[s]["ground_truth"]))
        two_events = sum(1 for s in seeds if len(cache[s]["events"]) >= 2)
        with_false = sum(1 for s in seeds if len(cache[s]["false_events"]) > 0)
        phenotypes, categories = {}, {}
        for s in seeds:
            for event in cache[s]["events"]:
                phenotypes[event["phenotype"]] = phenotypes.get(event["phenotype"], 0) + 1
            for event in cache[s]["false_events"]:
                categories[event["category"]] = categories.get(event["category"], 0) + 1
        print(
            f"  {name}: {total} scenarios, {positives} with a true event "
            f"({positives / total * 100:.1f}%), {total - positives} without; "
            f"{two_events} with >=2 true events; {with_false} with false events"
        )
        print(f"    phenotype counts: {dict(sorted(phenotypes.items()))}")
        print(f"    false-event counts: {dict(sorted(categories.items()))}")


# ---------------------------------------------------------------------------
# Conditions, reports and pre-declared claim checks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Weight / K search (TRAIN only)
# ---------------------------------------------------------------------------

COARSE_CONFIGS = unit_configs()[::5]
_SEARCH_STATE = {}


def make_weight_candidates():
    """Default weights, the author's hand-picked set, then fixed-seed random draws."""
    rng = np.random.default_rng(SEARCH_SEED)
    candidates = [
        dict(name="default", W_C=0.40, W_E=0.30, W_P=0.30, K=0.20),
        dict(name="author", **AUTHOR_WEIGHTS),
    ]
    for i in range(max(0, N_CANDIDATES - 2)):
        w = rng.dirichlet([1.0, 1.0, 1.0])
        k = rng.uniform(*K_RANGE)
        candidates.append(dict(name="rand%03d" % i, W_C=float(w[0]), W_E=float(w[1]), W_P=float(w[2]), K=float(k)))
    return candidates


def _score_candidate(job):
    family, idx = job
    cache = _SEARCH_STATE["cache"]
    item = _SEARCH_STATE["items"][family][idx]
    if family == "jor":
        params = {**DEFAULT_PARAMS, "W_C": item["W_C"], "W_E": item["W_E"], "W_P": item["W_P"], "K": item["K"]}
        track_fn = lambda scenario: run_jor(scenario, params, evidence="rel")
    elif family == "cep":
        track_fn = lambda scenario: cep_track(scenario, item)
    else:
        track_fn = lambda scenario: ep_weighted_track(scenario, item["w"])
    tracks = build_tracks(track_fn, TRAIN_SEEDS, cache)
    pick = pick_config(sweep_configs(tracks, TRAIN_SEEDS, cache, COARSE_CONFIGS), COARSE_CONFIGS, FPR_BUDGET)
    return pick["recall"], pick["f1"], pick["fpr"]


def run_weight_search(cache, label):
    """Choose weights / K for JOR, weights for the C/E/P average and w for the E+P average.
    Uses TRAIN scenarios only; ranks candidates by TRAIN recall under the FPR budget."""
    candidates = make_weight_candidates()
    items = {
        "jor": candidates,
        "cep": candidates,
        "ep": [dict(name="w=%.1f" % w, w=float(w)) for w in np.linspace(0.0, 1.0, 11)],
    }
    _SEARCH_STATE["cache"] = cache
    _SEARCH_STATE["items"] = items
    jobs = [(family, i) for family in items for i in range(len(items[family]))]
    outputs = None
    if N_JOBS > 1:
        try:
            with mp.get_context("fork").Pool(N_JOBS) as pool:
                outputs = pool.map(_score_candidate, jobs, chunksize=1)
        except ValueError:
            print("  (fork not available on this platform; running the search serially)", flush=True)
    if outputs is None:
        outputs = [_score_candidate(job) for job in jobs]

    scores = {family: [] for family in items}
    for (family, i), out in zip(jobs, outputs):
        scores[family].append(out)

    def best_index(family):
        ranked = [
            (out[2] <= FPR_BUDGET, out[0], out[1], -i) for i, out in enumerate(scores[family])
        ]
        return max(range(len(ranked)), key=lambda i: ranked[i])

    print("\n  Weight search on TRAIN (%s; coarse threshold grid, FPR budget %.2f)" % (label, FPR_BUDGET))
    chosen = {}
    spread = {}
    for family, title in (("jor", "JOR"), ("cep", "C/E/P avg"), ("ep", "E+P avg")):
        recalls = np.array([out[0] for out in scores[family]])
        idx = best_index(family)
        chosen[family] = items[family][idx]
        spread[family] = float(recalls.max() - np.median(recalls))
        default_recall = scores[family][0][0]
        item = items[family][idx]
        if family == "ep":
            what = "E weight %.1f, P weight %.1f" % (item["w"], 1.0 - item["w"])
        elif family == "jor":
            what = "W_C %.2f W_E %.2f W_P %.2f K %.2f" % (item["W_C"], item["W_E"], item["W_P"], item["K"])
        else:
            what = "W_C %.2f W_E %.2f W_P %.2f" % (item["W_C"], item["W_E"], item["W_P"])
        print(
            "    %-10s %3d candidates; TRAIN recall min %.3f median %.3f max %.3f; chosen %s (%s), TRAIN recall %.3f"
            % (title, len(recalls), recalls.min(), np.median(recalls), recalls.max(), item["name"], what, recalls[idx])
        )
        if family != "ep":
            print("               first candidate (default weights): TRAIN recall %.3f" % default_recall)
    tuned = dict(
        jor={**DEFAULT_PARAMS, "W_C": chosen["jor"]["W_C"], "W_E": chosen["jor"]["W_E"],
             "W_P": chosen["jor"]["W_P"], "K": chosen["jor"]["K"]},
        cep={"W_C": chosen["cep"]["W_C"], "W_E": chosen["cep"]["W_E"], "W_P": chosen["cep"]["W_P"]},
        ep_w=chosen["ep"]["w"],
    )
    return tuned, spread


def carried_tracks(stat, drift, method):
    """Tracks on drifting TEST data built with the STATIONARY-tuned parameters."""
    if method not in TUNED_METHODS:
        return drift["tracks"][method]
    saved = dict(ACTIVE)
    ACTIVE.update(jor=stat["tuned"]["jor"], cep=stat["tuned"]["cep"], ep_w=stat["tuned"]["ep_w"])
    try:
        return build_tracks(ALL_METHODS[method]["track"], TEST_SEEDS, drift["cache"])
    finally:
        ACTIVE.clear()
        ACTIVE.update(saved)


ALL_SEEDS = TRAIN_SEEDS + VAL_SEEDS + TEST_SEEDS
HEADLINE_METHODS = ("JOR (tuned)", "SNR (CFAR excess)", "C/E/P avg (tuned)", "E+P avg (tuned)")
BOOT_PAIRS = (
    ("JOR (tuned)", "SNR (CFAR excess)"),
    ("JOR (tuned)", "C/E/P avg (tuned)"),
    ("JOR (tuned)", "E+P avg (tuned)"),
    ("JOR (tuned)", "JOR (default wts)"),
    ("JOR (tuned)", "E+P avg (0.5/0.5)"),
    ("JOR (tuned)", "ref: JOR abs-SNR (tuned)"),
    ("SNR (CFAR excess)", "ref: SNR abs dB"),
    ("JOR (tuned)", "ref: SNR abs dB"),
    ("JOR+adaptive (tuned)", "JOR (tuned)"),
)


def run_condition(name, drifting):
    print(f"\n[{name}] generating scenarios and sweeping detection settings on TRAIN...", flush=True)
    t0 = time.time()
    cache = {seed: generate_scenario(seed, drifting_floor=drifting) for seed in ALL_SEEDS}
    tuned, spread = run_weight_search(cache, name)
    ACTIVE.update(jor=tuned["jor"], cep=tuned["cep"], ep_w=tuned["ep_w"])
    tracks = {m: build_tracks(spec["track"], ALL_SEEDS, cache) for m, spec in ALL_METHODS.items()}
    sweeps = {
        m: sweep_configs(tracks[m], TRAIN_SEEDS, cache, spec["configs"])
        for m, spec in ALL_METHODS.items()
    }
    picks = {
        m: pick_config(sweeps[m], spec["configs"], FPR_BUDGET) for m, spec in ALL_METHODS.items()
    }
    results = {
        m: {
            "TRAIN": picks[m],
            "VAL": evaluate_tracks(tracks[m], picks[m]["config"], VAL_SEEDS, cache),
            "TEST": evaluate_tracks(tracks[m], picks[m]["config"], TEST_SEEDS, cache),
        }
        for m in ALL_METHODS
    }
    test_rows = {
        m: scenario_outcomes(tracks[m], picks[m]["config"], TEST_SEEDS, cache) for m in ALL_METHODS
    }
    print(f"  done in {time.time() - t0:.0f}s", flush=True)
    return dict(
        name=name, drifting=drifting, cache=cache, tracks=tracks, sweeps=sweeps, tuned=tuned, spread=spread,
        picks=picks, results=results, test_rows=test_rows,
    )


def report_condition(c):
    results, picks = c["results"], c["picks"]
    print("\n" + "=" * 110)
    print(f"CONDITION: {c['name']} noise floor   (weights and thresholds chosen on TRAIN; FPR budget {FPR_BUDGET})")
    print("=" * 110)
    tj, tc = c["tuned"]["jor"], c["tuned"]["cep"]
    print(
        f"  Selected on TRAIN: JOR W_C {tj['W_C']:.2f} W_E {tj['W_E']:.2f} W_P {tj['W_P']:.2f} K {tj['K']:.2f}; "
        f"C/E/P avg W_C {tc['W_C']:.2f} W_E {tc['W_E']:.2f} W_P {tc['W_P']:.2f}; "
        f"E+P avg E weight {c['tuned']['ep_w']:.1f}"
    )
    print(
        f"{'Method':<26}{'Setting':<28}{'TRAIN R / FPR':<16}{'VAL R / FPR':<16}"
        f"{'TEST recall':>12}{'FPR':>8}{'F1':>8}{'Latency':>9}"
    )
    for group, methods in (
        ("headline (tuned)", METHODS), ("untuned reference", UNTUNED_METHODS), ("variant", VARIANT_METHODS),
        ("reference (absolute SNR)", REFERENCE_METHODS),
    ):
        print(f"  -- {group}")
        for m in methods:
            r = results[m]
            lat = r["TEST"]["mean_latency"]
            lat_text = "%.2f" % lat if lat is not None else "n/a"
            print(
                f"{m:<26}{fmt_cfg(picks[m]['config']):<28}"
                f"{r['TRAIN']['recall']:.3f} / {r['TRAIN']['fpr']:.3f}   "
                f"{r['VAL']['recall']:.3f} / {r['VAL']['fpr']:.3f}   "
                f"{r['TEST']['recall']:>12.3f}{r['TEST']['fpr']:>8.3f}{r['TEST']['f1']:>8.3f}"
                f"{lat_text:>9}"
            )
    print("\n  TEST 95% Wilson intervals:")
    for m in ALL_METHODS:
        t = results[m]["TEST"]
        print(
            f"    {m:<26} Recall {fmt_ci(t['tp'], t['tp'] + t['fn'])}   "
            f"FPR {fmt_ci(t['fp'], t['fp'] + t['tn'])}"
        )

    c["boots"] = {
        (a, b): paired_bootstrap(c["test_rows"][a], c["test_rows"][b]) for a, b in BOOT_PAIRS
    }
    print("\n  Paired bootstrap on TEST (2,000 resamples of scenarios), A minus B, 95% interval:")
    for (a, b), d in c["boots"].items():
        print(f"    {a} - {b}:")
        print(
            f"      Recall {d['recall'][0]:+.3f} [{d['recall'][1]:+.3f}, {d['recall'][2]:+.3f}]   "
            f"FPR {d['fpr'][0]:+.3f} [{d['fpr'][1]:+.3f}, {d['fpr'][2]:+.3f}]   "
            f"F1 {d['f1'][0]:+.3f} [{d['f1'][1]:+.3f}, {d['f1'][2]:+.3f}]"
        )

    print("\n  Sensitivity to the FPR budget (setting chosen on TRAIN under each budget; metrics on TEST):")
    print(f"    {'Budget':<8}{'Method':<26}{'TRAIN FPR':>10}{'TEST R':>9}{'TEST FPR':>10}{'TEST F1':>9}")
    for budget in (0.05, 0.08, 0.12):
        for m in HEADLINE_METHODS + tuple(VARIANT_METHODS):
            pk = pick_config(c["sweeps"][m], ALL_METHODS[m]["configs"], budget)
            te = evaluate_tracks(c["tracks"][m], pk["config"], TEST_SEEDS, c["cache"])
            print(
                f"    {budget:<8.2f}{m:<26}{pk['fpr']:>10.3f}{te['recall']:>9.3f}"
                f"{te['fpr']:>10.3f}{te['f1']:>9.3f}"
            )

    rows = {
        m: scenario_outcomes(c["tracks"][m], picks[m]["config"], VAL_SEEDS + TEST_SEEDS, c["cache"])
        for m in HEADLINE_METHODS
    }
    for m in HEADLINE_METHODS:
        for key, outcome in (("tp", "TP"), ("fn", "FN"), ("fp", "FP"), ("tn", "TN")):
            assert (
                sum(1 for _, o, _ in rows[m] if o == outcome)
                == results[m]["VAL"][key] + results[m]["TEST"][key]
            )
    print_breakdown(f"VAL + TEST pooled, {c['name']}", rows, c["cache"], lat_method="JOR (tuned)")


def report_carry_over(stat, drift):
    print("\n" + "=" * 110)
    print("CARRY-OVER: settings tuned on the STATIONARY TRAIN set applied unchanged to DRIFTING TEST data")
    print("=" * 110)
    print(
        f"{'Method':<26}{'Stationary FPR [95% CI]':<26}{'Carried-over FPR [95% CI]':<28}"
        f"{'Carried recall':>15}{'Retuned drift R / FPR':>26}"
    )
    carried = {}
    for m in ALL_METHODS:
        s_res = stat["results"][m]["TEST"]
        t_res = evaluate_tracks(carried_tracks(stat, drift, m), stat["picks"][m]["config"], TEST_SEEDS, drift["cache"])
        r_res = drift["results"][m]["TEST"]
        carried[m] = t_res
        retuned = "%.3f / %.3f" % (r_res["recall"], r_res["fpr"])
        print(
            f"{m:<26}{fmt_ci(s_res['fp'], s_res['fp'] + s_res['tn']):<26}"
            f"{fmt_ci(t_res['fp'], t_res['fp'] + t_res['tn']):<28}"
            f"{t_res['recall']:>15.3f}{retuned:>26}"
        )
    return carried


def report_f_sparse(stat, drift):
    """F_SPARSE diagnostics at the already-selected TRAIN settings (no retuning)."""
    print("\n" + "-" * 100)
    print("TABLE F1: F_SPARSE PHENOTYPE RECALL (first detection; TRAIN-selected settings; TEST seeds)")
    print("-" * 100)
    print(f"{'Method':<24} | {'Stationary':<14} | {'Drifting':<14} | {'TEST scenarios with F_SPARSE':<28}")
    first_recall = {}
    for m in ALL_METHODS:
        a = evaluate_f_sparse_recall(stat["tracks"][m], stat["picks"][m]["config"], TEST_SEEDS, stat["cache"])
        b = evaluate_f_sparse_recall(drift["tracks"][m], drift["picks"][m]["config"], TEST_SEEDS, drift["cache"])
        first_recall[m] = (a["recall"], b["recall"])
        print(f"{m:<24} | {a['recall']:<14.3f} | {b['recall']:<14.3f} | {a['scenarios']:<28d}")
    print(
        "Denominator: TEST scenarios containing at least one F_SPARSE event; the first detection "
        "must match an F_SPARSE event within the lag allowance."
    )

    print("\n" + "-" * 120)
    print("TABLE F2: F_SPARSE LATER-DETECTION / EVENT-LEVEL DIAGNOSTICS (same settings; one-to-one matching)")
    print("-" * 120)
    print(
        f"{'Method':<24} | {'Stat any-scn':<12} | {'Drift any-scn':<13} | {'Stat event rec':<14} | "
        f"{'Drift event rec':<15} | {'Stat matched/total':<18} | {'Drift matched/total':<19}"
    )
    audit = {}
    for m in ALL_METHODS:
        d_stat = evaluate_f_sparse_diagnostics(stat["tracks"][m], stat["picks"][m]["config"], TEST_SEEDS, stat["cache"])
        d_drift = evaluate_f_sparse_diagnostics(drift["tracks"][m], drift["picks"][m]["config"], TEST_SEEDS, drift["cache"])
        audit[m] = (d_stat, d_drift)
        print(
            f"{m:<24} | {d_stat['any_scenario_recall']:<12.3f} | {d_drift['any_scenario_recall']:<13.3f} | "
            f"{d_stat['event_recall']:<14.3f} | {d_drift['event_recall']:<15.3f} | "
            f"{str(d_stat['matched_events']) + '/' + str(d_stat['total_events']):<18} | "
            f"{str(d_drift['matched_events']) + '/' + str(d_drift['total_events']):<19}"
        )
    print("First-detection audit (stationary / drifting): F_SPARSE, other phenotype, unmatched, none.")
    for m in ALL_METHODS:
        d_stat, d_drift = audit[m]
        total = d_stat["scenarios"]
        assert (
            d_stat["first_sparse_true"] + d_stat["first_other_true"] + d_stat["first_unmatched"] + d_stat["no_detection"]
            == total
        )
        print(
            f"  {m}: F_SPARSE {d_stat['first_sparse_true']}/{d_drift['first_sparse_true']}; "
            f"other {d_stat['first_other_true']}/{d_drift['first_other_true']}; "
            f"unmatched {d_stat['first_unmatched']}/{d_drift['first_unmatched']}; "
            f"none {d_stat['no_detection']}/{d_drift['no_detection']}"
        )

    print("\n" + "-" * 112)
    print("TABLE F3: F_SPARSE RAW SENSOR OBSERVABILITY (TEST seeds; independent of any detection method)")
    print("-" * 112)
    print(
        f"{'Condition':<12} | {'Events':>6} | {'Steps':>6} | {'SNR %':>7} | {'RCS %':>7} | {'Power %':>8} | "
        f"{'Doppler %':>9} | {'Range %':>8} | {'DetP %':>7} | {'Any %':>7} | {'3+ %':>7} | {'Mean max run':>12}"
    )
    for label, c in (("Stationary", stat), ("Drifting", drift)):
        obs = evaluate_f_sparse_observability(TEST_SEEDS, c["cache"])
        cp = obs["channel_pct"]
        print(
            f"{label:<12} | {obs['events']:>6d} | {obs['event_steps']:>6d} | {cp['snr']:>6.1f}% | {cp['rcs']:>6.1f}% | "
            f"{cp['power']:>7.1f}% | {cp['doppler']:>8.1f}% | {cp['range_res']:>7.1f}% | {cp['detp']:>6.1f}% | "
            f"{obs['any_channel_pct']:>6.1f}% | {obs['three_plus_pct']:>6.1f}% | {obs['mean_max_three_plus_run']:>12.2f}"
        )
    print("3+ % = steps with at least three of six channels available; max run = longest run of such steps per event.")
    return first_recall


def check_claims(stat, drift, carried, first_recall):
    """Claims written down before the final run (see PREDECLARED_CLAIMS.md)."""
    J, SNR, CEP, EPT = "JOR (tuned)", "SNR (CFAR excess)", "C/E/P avg (tuned)", "E+P avg (tuned)"
    JD, ABS = "JOR (default wts)", "ref: SNR abs dB"

    def boot(c, a, b):
        return c["boots"][(a, b)]["recall"]

    def recall(c, m):
        return c["results"][m]["TEST"]["recall"]

    def fpr(c, m):
        return c["results"][m]["TEST"]["fpr"]

    claims = [
        ("T1", "Stationary: tuned JOR recall exceeds the CFAR SNR tracker (paired 95% CI lower bound > 0)",
         boot(stat, J, SNR)[1] > 0, boot(stat, J, SNR)),
        ("T2", "Drifting: tuned JOR recall exceeds the CFAR SNR tracker (paired 95% CI lower bound > 0)",
         boot(drift, J, SNR)[1] > 0, boot(drift, J, SNR)),
        ("T3", "Stationary: tuned JOR recall exceeds the tuned C/E/P weighted average (CI lower bound > 0)",
         boot(stat, J, CEP)[1] > 0, boot(stat, J, CEP)),
        ("T4", "Drifting: tuned JOR recall exceeds the tuned C/E/P weighted average (CI lower bound > 0)",
         boot(drift, J, CEP)[1] > 0, boot(drift, J, CEP)),
        ("T5", "Stationary: tuned JOR recall exceeds the tuned E+P average (CI lower bound > 0)",
         boot(stat, J, EPT)[1] > 0, boot(stat, J, EPT)),
        ("T6", "Drifting: tuned JOR recall exceeds the tuned E+P average (CI lower bound > 0)",
         boot(drift, J, EPT)[1] > 0, boot(drift, J, EPT)),
        ("T7", "Stationary: tuned JOR recall exceeds default-weight JOR (CI lower bound > 0)",
         boot(stat, J, JD)[1] > 0, boot(stat, J, JD)),
        ("T8", "Weights matter: JOR candidate TRAIN recall, best minus median, is at least 0.03 in both conditions",
         min(stat["spread"]["jor"], drift["spread"]["jor"]) >= 0.03,
         (stat["spread"]["jor"], drift["spread"]["jor"])),
        ("T9", "Stationary: tuned JOR recall is no more than 0.03 below the absolute-SNR tracker (point estimate)",
         recall(stat, J) >= recall(stat, ABS) - 0.03, (recall(stat, J), recall(stat, ABS))),
        ("T10", "Drifting: tuned JOR recall exceeds the retuned absolute-SNR tracker by more than 0.20 (point estimate)",
         recall(drift, J) - recall(drift, ABS) > 0.20, (recall(drift, J), recall(drift, ABS))),
        ("T11", "Carried over to drift, the absolute-SNR tracker's FPR exceeds 0.25",
         carried[ABS]["fpr"] > 0.25, carried[ABS]["fpr"]),
        ("T12", "Carried over to drift (stationary-tuned weights and thresholds), tuned JOR keeps FPR at or below 0.20",
         carried[J]["fpr"] <= 0.20, carried[J]["fpr"]),
        ("T13", "Tuned JOR TEST FPR is at or below 0.15 in both conditions",
         max(fpr(stat, J), fpr(drift, J)) <= 0.15, (fpr(stat, J), fpr(drift, J))),
        ("T14", "Drifting: adaptive decay with tuned weights does not improve on tuned JOR (CI upper bound < 0)",
         boot(drift, "JOR+adaptive (tuned)", J)[2] < 0, boot(drift, "JOR+adaptive (tuned)", J)),
        ("T15", "Stationary F_SPARSE first-detection recall is below 0.10 for every method",
         all(first_recall[m][0] < 0.10 for m in ALL_METHODS), {m: round(first_recall[m][0], 3) for m in ALL_METHODS}),
    ]
    print("\n" + "=" * 110)
    print("PRE-DECLARED CLAIM CHECKS" + ("  (DEV RUN: not a result)" if DEV_RUN else ""))
    print("=" * 110)
    held = 0
    for cid, text, ok, detail in claims:
        held += int(ok)
        if isinstance(detail, tuple) and len(detail) == 3:
            detail_text = f"point {detail[0]:+.3f} [{detail[1]:+.3f}, {detail[2]:+.3f}]"
        elif isinstance(detail, tuple):
            detail_text = ", ".join(f"{x:.3f}" for x in detail)
        elif isinstance(detail, dict):
            detail_text = str(detail)
        else:
            detail_text = f"{detail:.3f}"
        print(f"  {cid:<4} {'HELD    ' if ok else 'NOT HELD'}  {text}\n        {detail_text}")
    print(f"\n  {held} of {len(claims)} pre-declared claims held.")


# ---------------------------------------------------------------------------
# Main benchmark execution
# ---------------------------------------------------------------------------

def main():
    with open(__file__, "rb") as handle:
        code_hash = hashlib.sha256(handle.read()).hexdigest()

    print("=" * 110)
    print("JOR V3.1-enriched: tuned-weights study, stationary and drifting noise floor")
    print(f"Weight search: {N_CANDIDATES} candidates, seed {SEARCH_SEED}; jobs {N_JOBS}")
    print(f"Script SHA-256: {code_hash}")
    print("=" * 110)
    seed_freeze_check()
    run_seed_partition_check()

    stat = run_condition("STATIONARY", False)
    run_sanity_checks(
        {"TRAIN": TRAIN_SEEDS, "VAL": VAL_SEEDS, "TEST": TEST_SEEDS}, stat["cache"]
    )
    drift = run_condition("DRIFTING", True)

    # The drift condition must change only the SNR baseline: events, false events and
    # ground truth are identical for every seed.
    for seed in ALL_SEEDS:
        a, b = stat["cache"][seed], drift["cache"][seed]
        assert a["events"] == b["events"] and a["false_events"] == b["false_events"]
        assert np.array_equal(a["ground_truth"], b["ground_truth"])
    print("\nCondition isolation check: PASS (events, false events and ground truth identical across conditions)")

    report_condition(stat)
    report_condition(drift)
    carried = report_carry_over(stat, drift)
    first_recall = report_f_sparse(stat, drift)
    check_claims(stat, drift, carried, first_recall)
    print("\n" + "=" * 110)
    print("Run complete." + ("  DEV RUN: not a result." if DEV_RUN else ""))


if __name__ == "__main__":
    main()
