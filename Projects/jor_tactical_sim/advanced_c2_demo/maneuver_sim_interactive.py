"""
JOR 4.0 – Maneuver Simulation with Healthy Baseline + Catastrophic Collapse
Interactive command-decision engine with:
  - Early decision triggers
  - Stronger intervention effects
  - Multi-step decision chains
  - Morale as a fourth health dimension
  - Commander skill levels (novice / experienced / expert / AI-assisted)
  - Enemy behavior (tempo, pressure, outcomes, morale)
  - Scenario loader (urban, deep maneuver, logistics constrained, high enemy pressure, baseline)
  - Dynamic enemy posture system
  - Matrix-based state scaling (n-unit readiness vector + outcomes matrix)
  - Timed decision menus (async input clock; timeout forfeits initiative)
"""

import argparse
import select
import sys
import numpy as np
import time
import textwrap

from engine import JOREngine
from logger import FusionLogger
from maneuver_adapter import ManeuverAdapter
from map_telemetry import MapTelemetrySink, build_frame, init_positions
from decision_catalog import describe_decision
from map_view import LiveMapDisplay
from severity import SeverityGauge, severity_band


# ============================================================
# MATRIX STATE SCALING
# ============================================================
#
# Force health is carried as vector / matrix state so the same dynamics run
# for a company (n=3, original) or an expansive theater (n >> 3) without
# rewriting generators or decision effects.
#
#   readiness : shape (n,) float in [0, 1]
#   outcomes  : shape (n, 3) int columns [success, stall, failure]
#
# n_units defaults to 3 so skill=1.0 / feedback=0 / legacy engine still matches
# the original tick-for-tick at the regression level.

DEFAULT_N_UNITS = 3

# outcomes matrix column indices
COL_SUCCESS = 0
COL_STALL = 1
COL_FAILURE = 2


REFERENCE_PEERS = 2  # peers sharing the cost in the original 3-unit force


def redistribute_load(readiness, skill, primary_gain, peer_cost):
    """
    Vector load-redistribution: boost unit 0, charge the remaining units.

    `peer_cost` is the per-peer cost in the original 3-unit force (2 peers).
    The TOTAL cost, peer_cost * 2, is shared among however many peers there are,
    so the move stays a transfer of readiness between units and does not change
    the force's total readiness as N grows. (Charging every peer `peer_cost`
    made the same decision cost the force (N-1) * peer_cost, e.g. -1.35 total
    readiness at N=48.) Bit-identical to the original hardcoded 3-element
    arrays at n=3.
    """
    readiness = np.asarray(readiness, dtype=float)
    n = readiness.shape[0]
    delta = np.zeros(n, dtype=float)
    delta[0] = primary_gain * skill
    if n > 1:
        delta[1:] = -peer_cost * REFERENCE_PEERS / (n - 1) * skill
    return readiness + delta


def scale_outcome_column(outcomes, col, factor):
    """Multiply one outcomes-matrix column by factor and floor at zero (in place)."""
    outcomes[:, col] = np.maximum(0, (outcomes[:, col] * factor).astype(int))


# ============================================================
# SCENARIO LOADER
# ============================================================

# Menu currently open (set just before the decision prompt; read by scripted policies).
CURRENT_DECISION = {"type": None, "t": None}


class Scenario:
    def __init__(self, name):
        self.name = name.lower()

        if self.name == "urban":
            self.readiness_decay = 0.015
            self.pressure_growth = 0.030
            self.enemy_posture = "aggressive"
            self.morale_sensitivity = 1.20
            self.outcome_multiplier = 1.25

        elif self.name == "deep_maneuver":
            self.readiness_decay = 0.010
            self.pressure_growth = 0.020
            self.enemy_posture = "probing"
            self.morale_sensitivity = 1.00
            self.outcome_multiplier = 1.10

        elif self.name == "logistics_constrained":
            self.readiness_decay = 0.020
            self.pressure_growth = 0.015
            self.enemy_posture = "cautious"
            self.morale_sensitivity = 1.40
            self.outcome_multiplier = 1.30

        elif self.name == "high_enemy_pressure":
            self.readiness_decay = 0.012
            self.pressure_growth = 0.040
            self.enemy_posture = "aggressive"
            self.morale_sensitivity = 1.10
            self.outcome_multiplier = 1.40

        else:
            self.readiness_decay = 0.010
            self.pressure_growth = 0.020
            self.enemy_posture = "probing"
            self.morale_sensitivity = 1.00
            self.outcome_multiplier = 1.00

        self.use_distinct_events()

    # ------------------------------------------------------------------
    # Scenario event signatures
    # ------------------------------------------------------------------
    # The local-strain event and the catastrophe used to be identical in every
    # scenario (t=30 and t=110, same severity), so all five ended alike. Each
    # scenario now has its own timing and its own *kind* of catastrophe:
    #   Difficulty for a well-handled expert force, easiest to hardest:
    #   deep_maneuver < baseline < logistics_constrained < urban < high_enemy_pressure.
    #   *_scale values multiply the corresponding terms of the original
    #   catastrophe (readiness kept, pressure added, morale lost, extra
    #   stall/failure outcomes); 1.0 everywhere is the original event.
    # Baseline is unchanged. Keep catastrophe_tick within ~10 ticks of the end
    # of a default run (see COLLAPSE_SUSTAIN_TICKS), or everything collapses.
    EVENT_PROFILES = {
        "baseline": dict(strain_tick=30, catastrophe_tick=110,
                         cat_readiness_scale=1.0, cat_pressure_scale=1.0,
                         cat_morale_scale=1.0, cat_outcome_scale=1.0,
                         strain_cause="", cat_cause=""),
        # open ground, probing enemy: strain shows up late and the break comes late
        # and is a coordination/overextension one (the most forgiving scenario)
        "deep_maneuver": dict(strain_tick=45, catastrophe_tick=113,
                              cat_readiness_scale=1.00, cat_pressure_scale=1.15,
                              cat_morale_scale=0.80, cat_outcome_scale=0.90,
                              strain_cause="extended lines of communication",
                              cat_cause="overextension outruns command and control"),
        # supply: early local shortfall; the break drains readiness rather than morale
        "logistics_constrained": dict(strain_tick=25, catastrophe_tick=106,
                                      cat_readiness_scale=0.50, cat_pressure_scale=0.80,
                                      cat_morale_scale=1.00, cat_outcome_scale=0.80,
                                      strain_cause="local supply shortfall",
                                      cat_cause="sustainment collapses; units run dry"),
        # dense terrain: friction arrives early; the break is casualty-driven
        "urban": dict(strain_tick=20, catastrophe_tick=108,
                      cat_readiness_scale=1.00, cat_pressure_scale=0.90,
                      cat_morale_scale=1.40, cat_outcome_scale=1.50,
                      strain_cause="street-fighting friction",
                      cat_cause="house-to-house attrition overwhelms the lead elements"),
        # adversary initiative: the break comes soonest and is pressure-driven
        # (the hardest scenario)
        "high_enemy_pressure": dict(strain_tick=30, catastrophe_tick=102,
                                    cat_readiness_scale=1.00, cat_pressure_scale=1.25,
                                    cat_morale_scale=1.00, cat_outcome_scale=1.40,
                                    strain_cause="adversary fires on the flank",
                                    cat_cause="coordinated enemy offensive breaks the line"),
    }

    def use_distinct_events(self):
        for k, v in self.EVENT_PROFILES.get(self.name, self.EVENT_PROFILES["baseline"]).items():
            setattr(self, k, v)
        self.events_mode = "distinct"

    def use_legacy_events(self):
        """The original event schedule and severity, identical in every scenario."""
        for k, v in self.EVENT_PROFILES["baseline"].items():
            setattr(self, k, v)
        self.events_mode = "legacy"


# ============================================================
# COMMANDER SKILL MODEL
# ============================================================

class CommanderSkill:
    def __init__(self, level="experienced"):
        self.level = level.lower()

    def multiplier(self):
        if self.level == "novice":
            return 0.6
        if self.level == "experienced":
            return 1.0
        if self.level == "expert":
            return 1.4
        if self.level == "ai_assisted":
            return 1.8
        return 1.0




class EnemyForce:
    def __init__(self, posture="probing"):
        self.posture = posture
        self.readiness = 0.80
        self.pressure = 0.45
        self.morale = 0.75
        self.outcomes = 0.15

    def step(self, t):
        if self.posture == "cautious":
            self.pressure += np.random.normal(0.0, 0.01)
            self.readiness -= 0.003
            self.outcomes += np.random.normal(0.0, 0.01)
            self.morale += np.random.normal(0.0, 0.01)

        elif self.posture == "probing":
            self.pressure += np.random.normal(0.02, 0.02)
            self.readiness -= 0.007
            self.outcomes += np.random.normal(0.02, 0.02)
            self.morale += np.random.normal(0.0, 0.02)

        elif self.posture == "aggressive":
            self.pressure += np.random.normal(0.06, 0.03)
            self.readiness -= 0.015
            self.outcomes += np.random.normal(0.05, 0.03)
            self.morale += np.random.normal(0.01, 0.02)

        elif self.posture == "overextended":
            self.pressure += np.random.normal(0.04, 0.04)
            self.readiness -= 0.020
            self.outcomes += np.random.normal(0.07, 0.04)
            self.morale += np.random.normal(-0.02, 0.03)

        self.readiness = float(np.clip(self.readiness, 0.0, 1.0))
        self.pressure = float(np.clip(self.pressure, 0.0, 1.0))
        self.morale = float(np.clip(self.morale, 0.0, 1.0))
        self.outcomes = float(np.clip(self.outcomes, 0.0, 1.0))

    def apply_friendly_skill_effects(self, skill):
        self.pressure *= (1.0 - 0.05 * (skill - 1.0))
        self.outcomes *= (1.0 + 0.05 * (skill - 1.0))
        self.morale -= 0.02 * (skill - 1.0)

        self.readiness = float(np.clip(self.readiness, 0.0, 1.0))
        self.pressure = float(np.clip(self.pressure, 0.0, 1.0))
        self.morale = float(np.clip(self.morale, 0.0, 1.0))
        self.outcomes = float(np.clip(self.outcomes, 0.0, 1.0))

    def update_posture(self):
        if self.posture == "probing":
            if self.morale > 0.7 and self.readiness > 0.6:
                self.posture = "aggressive"
            elif self.morale < 0.4 or self.readiness < 0.3:
                self.posture = "cautious"

        elif self.posture == "aggressive":
            if self.pressure > 0.85 or self.outcomes > 0.8:
                self.posture = "overextended"
            elif self.morale < 0.4 or self.readiness < 0.3:
                self.posture = "cautious"

        elif self.posture == "overextended":
            if self.morale < 0.3 or self.readiness < 0.2:
                self.posture = "cautious"

        elif self.posture == "cautious":
            if self.morale > 0.6 and self.readiness > 0.5 and self.pressure < 0.7:
                self.posture = "probing"


# ============================================================
# DECISION STATE + MULTI-STEP EFFECTS
# ============================================================

# Ticks before a decision category can prompt again after the commander has
# answered a prompt in it (a response needs time to take effect).
PROMPT_COOLDOWN = 10

# Which decision category each multi-step effect belongs to. The escalated
# prompts are meant to stay quiet while a response of their own category is
# still running.
EFFECT_CATEGORY = {
    "REDUCE_TEMPO_CHAIN": "FATIGUE",
    "REINFORCE_COMMS_CHAIN": "PRESSURE",
    "ROTATE_UNITS_CHAIN": "OUTCOMES",
    "STABILIZATION_CHAIN": "CATASTROPHIC",
}


class DecisionState:
    """
    Tracks which decision prompts have fired and what responses are running.

    guard_mode "fixed" (default): an escalated prompt (FATIGUE / PRESSURE /
        OUTCOMES) fires only when no effect of its category is running AND its
        cooldown has expired.
    guard_mode "legacy": the original guard, which tested for effect types
        starting with "FATIGUE" / "PRESSURE" / "OUTCOMES". No effect is ever
        named that way, so the test was always true and the escalated prompt
        re-fired on every tick its threshold held (66 prompts in 120 ticks in
        one seeded expert run). Kept only so the original can be reproduced.
    """

    def __init__(self, guard_mode="fixed"):
        self.fatigue_prompted = False
        self.pressure_prompted = False
        self.outcomes_prompted = False
        self.catastrophic_prompted = False
        self.active_effects = []
        self.guard_mode = guard_mode
        self.cooldowns = {"FATIGUE": 0, "PRESSURE": 0, "OUTCOMES": 0}

    def tick(self):
        """Advance cooldowns by one tick (call once per simulation tick)."""
        for category in self.cooldowns:
            self.cooldowns[category] = max(0, self.cooldowns[category] - 1)

    def record_decision(self, choice):
        """Start the cooldown for the category of a prompt that was just answered."""
        category = choice.replace("_EARLY", "")
        if category in self.cooldowns:
            self.cooldowns[category] = PROMPT_COOLDOWN

    def can_escalate(self, category):
        if self.guard_mode == "legacy":
            return not any(e["type"].startswith(category) for e in self.active_effects)
        running = any(EFFECT_CATEGORY.get(e["type"]) == category for e in self.active_effects)
        return self.cooldowns[category] == 0 and not running


# ============================================================
# STATE-SPACE FEEDBACK LOOP (MORALE -> READINESS, PRESSURE)
# ============================================================
#
# The generators above produce readiness and pressure as *levels* each tick,
# so on their own nothing carries over from one tick to the next. To make
# morale a true driver instead of a passive readout, this keeps two persistent
# state variables:
#
#   readiness_debt : accumulated readiness loss  (subtracted from readiness)
#   pressure_creep : accumulated pressure growth (added to pressure)
#
# Each tick, whenever morale sits below MORALE_REF, both accumulate in
# proportion to the morale deficit and then slowly recover. Lower readiness
# already drags morale down in the main loop (theta_s < 0.40 rule), so the
# loop closes: low morale -> readiness debt / pressure creep -> worse theta_s
# and theta_c -> lower morale. Caps keep the loop bounded.
#
# strength = 0.0 is an exact no-op (original uncoupled dynamics).

DEFAULT_FEEDBACK = 1.0


class FeedbackState:
    MORALE_REF = 0.60        # morale above this accumulates no debt
    READINESS_GAIN = 0.04    # debt added per tick per unit of morale deficit
    PRESSURE_GAIN = 0.03     # creep added per tick per unit of morale deficit
    RECOVERY = 0.97          # per-tick persistence (3% recovery per tick)
    READINESS_CAP = 0.30
    PRESSURE_CAP = 0.25

    def __init__(self, strength=DEFAULT_FEEDBACK):
        self.strength = float(strength)
        self.readiness_debt = 0.0
        self.pressure_creep = 0.0

    def apply(self, readiness, pressure, morale):
        deficit = max(0.0, self.MORALE_REF - morale)

        self.readiness_debt = min(
            self.READINESS_CAP,
            self.readiness_debt * self.RECOVERY + self.strength * self.READINESS_GAIN * deficit
        )
        self.pressure_creep = min(
            self.PRESSURE_CAP,
            self.pressure_creep * self.RECOVERY + self.strength * self.PRESSURE_GAIN * deficit
        )

        readiness = np.clip(readiness - self.readiness_debt, 0.0, 1.0)
        pressure = float(np.clip(pressure + self.pressure_creep, 0.0, 1.0))
        return readiness, pressure


def apply_active_effects(readiness, pressure, outcomes, morale, decision_state, skill):
    new_effects = []
    for effect in decision_state.active_effects:
        etype = effect["type"]
        params = effect["params"]
        steps_left = effect["steps_left"]

        if etype == "REDUCE_TEMPO_CHAIN":
            pressure -= params.get("pressure_delta", 0.03 * skill)
            readiness = readiness + params.get("readiness_delta", 0.01 * skill)
            morale += params.get("morale_delta", 0.01 * skill)

        elif etype == "REINFORCE_COMMS_CHAIN":
            pressure -= params.get("pressure_delta", 0.04 * skill)

        elif etype == "ROTATE_UNITS_CHAIN":
            scale_outcome_column(outcomes, COL_STALL, params.get("stall_factor", 0.95))
            scale_outcome_column(outcomes, COL_FAILURE, params.get("failure_factor", 0.95))

        elif etype == "STABILIZATION_CHAIN":
            readiness = readiness + params.get("readiness_delta", 0.02 * skill)
            pressure -= params.get("pressure_delta", 0.03 * skill)
            morale += params.get("morale_delta", 0.01 * skill)

        elif etype == "DEBT_CHAIN":
            # Carry-over cost (training vignettes only): readiness borrowed by pushing
            # on, or lost to strain left unaddressed, is paid over the following ticks.
            readiness = readiness - params.get("readiness_delta", 0.0)
            pressure += params.get("pressure_delta", 0.0)

        steps_left -= 1
        if steps_left > 0:
            effect["steps_left"] = steps_left
            new_effects.append(effect)

    decision_state.active_effects = new_effects

    readiness = np.clip(readiness, 0.0, 1.0)
    pressure = float(np.clip(pressure, 0.0, 1.0))
    morale = float(np.clip(morale, 0.0, 1.0))
    return readiness, pressure, outcomes, morale


# ============================================================
# ASCII BAR HELPERS
# ============================================================

def ascii_bar(value, length=20):
    value = max(0.0, min(1.0, float(value)))
    filled = int(value * length)
    empty = length - filled
    return "█" * filled + "░" * empty


# ============================================================
# HEALTHY MANEUVER BASELINE GENERATORS (SCENARIO-AWARE)
# ============================================================

def degraded_indices(degraded_unit, n_units):
    """
    Indices (int array) of the units under strain that exist in this force.

    `degraded_unit` is None (no strain), an int (the original single-unit
    event) or a sequence of ints (a strained group).
    """
    if degraded_unit is None:
        return np.zeros(0, dtype=int)
    idx = np.atleast_1d(np.asarray(degraded_unit, dtype=int))
    return idx[(idx >= 0) & (idx < n_units)]


def is_degraded(degraded_unit):
    """True if any strain is active (an int, or a non-empty group)."""
    return degraded_unit is not None and np.size(degraded_unit) > 0


def strained_units(n_units, strain_fraction=None, first=1):
    """
    Which units the t=30 strain event hits.

    strain_fraction None : the original event, a single unit (`first`, i.e.
        unit 1). Its weight in force-wide averages falls as 1/N, so in a large
        force the event barely registers.
    strain_fraction f    : a contiguous block of max(1, round(f * N)) units
        starting at `first` (shifted left if it would run off the end), so the
        share of the force under strain, and the size of the signal, does not
        depend on N. f = 1/3 gives exactly unit 1 at N = 3.
    """
    if strain_fraction is None:
        return first
    if not 0.0 < strain_fraction <= 1.0:
        raise ValueError("strain_fraction must be in (0, 1]")
    k = min(n_units, max(1, int(round(strain_fraction * n_units))))
    start = max(0, min(first, n_units - k))
    return list(range(start, start + k))


def generate_readiness(t, degraded_unit, scenario, n_units=DEFAULT_N_UNITS):
    """
    Readiness state vector, shape (n_units,).

    Base levels cycle a small healthy pattern so larger theaters still look
    heterogeneous without needing a hand-tuned list per size. At n=3 the first
    three bases match the original [0.85, 0.83, 0.88].
    """
    pattern = np.array([0.85, 0.83, 0.88])
    base = np.resize(pattern, n_units).astype(float)
    decay = np.clip(t / 160, 0.0, 0.35)
    readiness = base - decay
    readiness -= scenario.readiness_decay
    readiness += np.random.normal(0.0, 0.015, size=n_units)

    idx = degraded_indices(degraded_unit, n_units)
    if idx.size:
        readiness[idx] -= 0.12

    return np.clip(readiness, 0.0, 1.0)


def generate_pressure(t, scenario):
    if t < 20:
        base = 0.45
    elif t < 60:
        base = 0.45 + (t - 20) * (0.65 - 0.45) / 40
    else:
        base = 0.65 + np.sin(t / 12.0) * 0.05

    base += scenario.pressure_growth
    base += np.random.normal(0.0, 0.02)
    return float(np.clip(base, 0.0, 1.0))


LARGE_FORCE_UNITS = 64  # above this: vectorized sampler, compact logs, no unrequested map frames


def _generate_outcomes_vectorized(degraded_unit, n_units):
    """Same per-unit distributions as the scalar sampler, drawn array-at-a-time."""
    success = np.random.randint(11, 17, size=n_units)
    stall = np.random.randint(2, 5, size=n_units)
    failure = np.random.randint(0, 2, size=n_units)

    idx = degraded_indices(degraded_unit, n_units)
    if idx.size:
        stall[idx] += np.random.randint(2, 4, size=idx.size)
        failure[idx] += np.random.randint(1, 3, size=idx.size)

    stall += np.where(np.random.rand(n_units) < 0.10, np.random.randint(2, 5, size=n_units), 0)
    failure += np.where(np.random.rand(n_units) < 0.06, np.random.randint(1, 3, size=n_units), 0)
    success += np.where(np.random.rand(n_units) < 0.10, np.random.randint(2, 5, size=n_units), 0)

    return np.column_stack([success, stall, failure]).astype(int)


def generate_outcomes(t, degraded_unit, n_units=DEFAULT_N_UNITS, sampler="auto"):
    """
    Outcomes state matrix, shape (n_units, 3) with columns
    [success, stall, failure].

    sampler "scalar": per-unit draw order is preserved (success → stall →
        failure → conditional spikes), so n_units=3 consumes the RNG stream
        identically to the original list-of-dicts generator and keeps the
        regression suite green.
    sampler "vectorized": same distributions drawn as arrays (~50x faster at
        thousands of units) but a different RNG stream.
    sampler "auto" (default): scalar up to LARGE_FORCE_UNITS units, so every
        seeded result at normal sizes is unchanged; vectorized above that.
    """
    if sampler == "vectorized" or (sampler == "auto" and n_units > LARGE_FORCE_UNITS):
        return _generate_outcomes_vectorized(degraded_unit, n_units)

    degraded_set = set(degraded_indices(degraded_unit, n_units).tolist())
    rows = []
    for u in range(n_units):
        success = np.random.randint(11, 17)
        stall = np.random.randint(2, 5)
        failure = np.random.randint(0, 2)

        if u in degraded_set:
            stall += np.random.randint(2, 4)
            failure += np.random.randint(1, 3)

        if np.random.rand() < 0.10:
            stall += np.random.randint(2, 5)
        if np.random.rand() < 0.06:
            failure += np.random.randint(1, 3)
        if np.random.rand() < 0.10:
            success += np.random.randint(2, 5)

        rows.append((success, stall, failure))

    return np.asarray(rows, dtype=int)


# ============================================================
# NARRATIVE + MISSION STATE
# ============================================================

MAX_NARRATED_UNITS = 6  # above this many flagged units, the narrative summarizes by count


def describe_troop_state(t, readiness, pressure, outcomes, degraded_unit, morale, skill, commander_level, enemy):
    desc = []

    if pressure < 0.5:
        desc.append("Pressure rising; maneuver tempo increasing.")
    elif pressure < 0.7:
        desc.append("High operational pressure; force maintaining tempo.")
    else:
        desc.append("Pressure extreme; coordination load heavy.")

    avg_r = np.mean(readiness)
    if avg_r > 0.70:
        desc.append("Troop readiness strong; units performing well.")
    elif avg_r > 0.55:
        desc.append("Readiness declining; sustained exertion visible.")
    elif avg_r > 0.40:
        desc.append("Troops fatigued; tempo demanding.")
    else:
        desc.append("Troop readiness collapsing; force approaching exhaustion.")

    if morale > 0.75:
        desc.append("Morale high; force resilient.")
    elif morale > 0.50:
        desc.append("Morale steady; force coping with strain.")
    elif morale > 0.30:
        desc.append("Morale fragile; cohesion under stress.")
    else:
        desc.append("Morale collapsing; risk of force break.")

    if commander_level == "expert":
        desc.append("Commander expertise improving force resilience.")
    elif commander_level == "novice":
        desc.append("Inexperienced command reducing force cohesion.")
    elif commander_level == "ai_assisted":
        desc.append("AI-assisted command enhancing stabilization and tempo control.")

    if is_degraded(degraded_unit):
        strained = np.atleast_1d(np.asarray(degraded_unit, dtype=int))
        if strained.size == 1:
            desc.append(f"Unit {int(strained[0])} showing strain.")
        else:
            sample = ", ".join(str(int(i)) for i in strained[:3])
            more = ", ..." if strained.size > 3 else ""
            desc.append(f"{strained.size} units showing strain (e.g. units {sample}{more}).")

    outcomes = np.asarray(outcomes)
    failure = outcomes[:, COL_FAILURE].astype(int)
    stall = outcomes[:, COL_STALL].astype(int)
    heavy = failure > 4
    friction = ~heavy & (stall > 5)
    minor = ~heavy & ~friction & (failure > 1)

    flagged = int(heavy.sum() + friction.sum() + minor.sum())
    if flagged <= MAX_NARRATED_UNITS:
        # Small force: one sentence per unit, in unit order (as originally).
        for i in np.flatnonzero(heavy | friction | minor):
            if heavy[i]:
                desc.append(f"Unit {i} experiencing heavy failures.")
            elif friction[i]:
                desc.append(f"Unit {i} friction increasing.")
            else:
                desc.append(f"Unit {i} showing minor degradation.")
    else:
        # Large force: summarize by category so the text stays bounded.
        for mask, phrase in ((heavy, "experiencing heavy failures"),
                             (friction, "showing increasing friction"),
                             (minor, "showing minor degradation")):
            idx = np.flatnonzero(mask)
            if idx.size:
                sample = ", ".join(str(int(i)) for i in idx[:3])
                more = f", ..." if idx.size > 3 else ""
                desc.append(f"{idx.size} units {phrase} (e.g. units {sample}{more}).")

    if enemy.pressure > 0.7:
        desc.append("Enemy pressure high; adversary pushing tempo.")
    elif enemy.pressure < 0.4:
        desc.append("Enemy pressure moderate; adversary probing.")

    if enemy.morale < 0.3:
        desc.append("Enemy morale degraded; adversary cohesion weakening.")
    elif enemy.morale > 0.7:
        desc.append("Enemy morale strong; adversary confident.")

    desc.append(f"Enemy posture: {enemy.posture}.")

    return " ".join(desc)


# Collapse criteria are objective definitions of force state, so they do not
# depend on who is commanding. (They used to be divided by the skill
# multiplier, which tightened the "<" tests but loosened the ">" tests for
# skilled commanders and made collapse unreachable for novices, e.g.
# theta_c > 0.90 / 0.6 = 1.5. Skill still matters, but through the dynamics
# and the decision effects, not by moving the goalposts.)

def friendly_critical(theta_s, theta_c, morale):
    """Live 3-condition critical state used by the per-tick mission classifier."""
    return theta_s < 0.25 and theta_c > 0.80 and morale < 0.30


def friendly_collapse_flags(theta_s, theta_c, theta_o, morale):
    """The four joint conditions behind the mission-summary collapse signature."""
    return (
        theta_s < 0.20,
        theta_c > 0.90,
        theta_o > 0.40,
        morale < 0.30,
    )


def enemy_collapse_flags(readiness, pressure, morale):
    """The three joint conditions behind the enemy collapse signature."""
    return (
        readiness < 0.20,
        pressure > 0.90,
        morale < 0.30,
    )


# A momentary breach of every collapse condition is not a collapse: a force
# that crosses all the lines for one tick and then recovers has not broken.
# Collapse means the full set of conditions held for COLLAPSE_SUSTAIN_TICKS
# consecutive ticks.
#
# Keep this well below the time between the catastrophe (t=110) and the end of
# a default run (t=119, ten ticks later), or collapse becomes unreachable. In
# the default run, 5 separates a well-handled expert force (survives) from a
# novice or an ignored crisis (collapses); a longer run lets the catastrophe's
# morale crash outlast the stabilization effect and everything collapses.
COLLAPSE_SUSTAIN_TICKS = 5


def collapse_runs(ticks, met):
    """
    Runs of consecutive ticks on which all collapse conditions held.

    ticks: tick numbers, in order; met: matching booleans (all conditions met on
    that tick). A gap in the tick numbers ends a run. Returns [(start_tick, length), ...].
    """
    runs, start, length, prev = [], None, 0, None
    for t, m in zip(ticks, met):
        if m:
            if length and prev is not None and t == prev + 1:
                length += 1
            else:
                if length:
                    runs.append((start, length))
                start, length = t, 1
        elif length:
            runs.append((start, length))
            start, length = None, 0
        prev = t
    if length:
        runs.append((start, length))
    return runs


def sustained_collapse(ticks, met, sustain=COLLAPSE_SUSTAIN_TICKS):
    """
    Did the all-conditions state hold for `sustain` consecutive ticks?

    Returns {"collapsed", "onset_t", "held", "longest", "longest_t"}: whether it
    did; the first tick and full length of the first such run (None / 0 if
    never); and the longest run of any length and where it began (for reporting
    a breach that was too brief to count).
    """
    runs = collapse_runs(ticks, met)
    first = next(((s, n) for s, n in runs if n >= sustain), None)
    longest = max(runs, key=lambda r: r[1], default=None)
    return {
        "collapsed": first is not None,
        "onset_t": first[0] if first else None,
        "held": first[1] if first else 0,
        "longest": longest[1] if longest else 0,
        "longest_t": longest[0] if longest else None,
    }


def classify_mission_state(theta_s, theta_c, theta_o, degraded_unit, morale, skill, enemy):
    # skill is accepted for call compatibility but intentionally unused (see above)
    friendly_critical_now = friendly_critical(theta_s, theta_c, morale)
    enemy_critical = enemy.readiness < 0.25 and enemy.pressure > 0.80 and enemy.morale < 0.30

    if friendly_critical_now and enemy_critical:
        return "MUTUAL_COLLAPSE"
    if friendly_critical_now and not enemy_critical:
        return "FRIENDLY_COLLAPSE"
    if enemy_critical and not friendly_critical_now:
        return "ENEMY_COLLAPSE"

    if is_degraded(degraded_unit):
        return "DEGRADED"
    if theta_c > 0.65:
        return "HIGH_PRESSURE"
    if theta_s < 0.45 or morale < 0.40:
        return "FATIGUED"
    return "STABLE"


# ============================================================
# TACTICAL DECISION ENGINE
# ============================================================

def readiness_prompt_threshold(base, skill):
    """
    Readiness level below which a commander of this skill gets prompted.

    A skilled commander should notice trouble sooner, i.e. at a smaller amount
    of accumulated risk. Pressure and outcomes are already risk-valued
    (higher = worse), so dividing their thresholds by skill does that. Readiness
    is the opposite way round (lower = worse), so dividing its threshold by
    skill made skilled commanders notice fatigue *later* (and novices were
    prompted at t=0). Work in risk space instead: risk = 1 - readiness, the
    risk needed to prompt is (1 - base) / skill, and the prompt threshold is
    1 - that. Unchanged at skill 1.0.
    """
    return 1.0 - (1.0 - base) / skill


def maybe_prompt_user(theta_s, theta_c, theta_o, morale, catastrophic_event, decision_state, skill):
    if theta_s < readiness_prompt_threshold(0.70, skill) and morale < 0.80 and not decision_state.fatigue_prompted:
        decision_state.fatigue_prompted = True
        return "FATIGUE_EARLY"

    if theta_s < readiness_prompt_threshold(0.55, skill) and morale < 0.70 and decision_state.fatigue_prompted and decision_state.can_escalate("FATIGUE"):
        return "FATIGUE"

    if theta_c > (0.55 / skill) and not decision_state.pressure_prompted:
        decision_state.pressure_prompted = True
        return "PRESSURE_EARLY"

    if theta_c > (0.70 / skill) and decision_state.pressure_prompted and decision_state.can_escalate("PRESSURE"):
        return "PRESSURE"

    if theta_o > (0.20 / skill) and not decision_state.outcomes_prompted:
        decision_state.outcomes_prompted = True
        return "OUTCOMES_EARLY"

    if theta_o > (0.35 / skill) and decision_state.outcomes_prompted and decision_state.can_escalate("OUTCOMES"):
        return "OUTCOMES"

    if catastrophic_event and not decision_state.catastrophic_prompted:
        decision_state.catastrophic_prompted = True
        return "CATASTROPHIC"

    return None


def intervention_factor(base, skill):
    """
    Multiplicative reduction applied to stall/failure counts on intervene choices.

    Preserves the original value at skill 1.0, is always <= 1.0 (never increases
    the bad count), and shrinks further as skill rises. Replaces the inverted
    `base / skill` form, which exceeded 1.0 for novices and could make a
    "light rotation" increase stalls.
    """
    return 1.0 - (1.0 - base) * skill


# ============================================================
# ASYNCHRONOUS INPUT TIMER
# ============================================================
#
# Decision menus run under a real-time clock. If the commander does not answer
# before the window closes, the ignore / accept-risk option is selected and
# tactical initiative is forfeited. Headless / scripted runs pass timeout=None
# (or decision_timeout=0) and keep the original blocking behaviour.

DEFAULT_DECISION_TIMEOUT = None  # seconds; None = blocking (no timer)
TIMEOUT_DEFAULT_CHOICE = "3"     # ignore / accept risk / hold course


def timed_input(prompt, timeout=None, default=TIMEOUT_DEFAULT_CHOICE):
    """
    Read a line from stdin, or return `default` if `timeout` seconds elapse.

    Unix: select() on stdin.
    Windows: msvcrt keyboard poll (select cannot watch stdin; a background
    readline thread also races with the next prompt and can "eat" a typed 1/2/3
    after a forfeit). Falls back to blocking input when timeout is None/<=0 or
    stdin is not a tty (e.g. piped scripted input).
    """
    timed_input.expired = False
    if timeout is None or timeout <= 0:
        return input(prompt)

    try:
        if not sys.stdin.isatty():
            return input(prompt)
    except Exception:
        return input(prompt)

    sys.stdout.write(prompt)
    sys.stdout.write(f"  [window {timeout:.0f}s — timeout forfeits initiative]\n")
    sys.stdout.flush()

    timeout = float(timeout)

    if sys.platform != "win32":
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip()
        print(f"\n>>> TIME EXPIRED — initiative forfeited "
              f"(default decision {default}).\n")
        timed_input.expired = True
        return default

    # --- Windows: poll console keys until Enter or timeout ---
    import time
    try:
        import msvcrt
    except ImportError:
        # Extremely unusual; block without timer rather than crash
        return input("")

    deadline = time.monotonic() + timeout
    buf = []
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            # special keys return \x00 or \xe0 then a follow-up — ignore
            if ch in ("\x00", "\xe0"):
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(buf).strip()
            if ch in ("\b", "\x08"):
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            # printable
            if ch.isprintable():
                buf.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
        else:
            time.sleep(0.05)

    print(f"\n>>> TIME EXPIRED — initiative forfeited "
          f"(default decision {default}).\n")
    timed_input.expired = True
    return default



timed_input.expired = False   # set True when the last call ran out the clock


def make_timed_input_fn(timeout, base_input_fn=None):
    """Wrap an input callable with the decision timer when timeout is active."""
    if timeout is None or timeout <= 0:
        return base_input_fn or input

    def _fn(prompt=""):
        # base_input_fn is only used when not applying the wall clock (e.g. tests);
        # interactive runs use timed_input directly.
        if base_input_fn is not None and base_input_fn is not input:
            return base_input_fn(prompt)
        return timed_input(prompt, timeout=timeout)

    return _fn


def apply_user_choice(choice, readiness, pressure, outcomes, morale, decision_state, skill, input_fn=input):
    print("\n--- DECISION POINT ---")

    aborted = False

    if choice == "FATIGUE_EARLY":
        print("Early signs of fatigue. Choose:")
        print("  [1] Slightly reduce tempo")
        print("  [2] Redistribute minor load")
        print("  [3] Ignore (accept strain)")
        sel = input_fn("Decision (1/2/3): ").strip()

        if sel == "1":
            pressure *= (1.0 - 0.08 * skill)
            readiness = readiness + 0.05 * skill
            morale += 0.05 * skill
            decision_state.active_effects.append({
                "type": "REDUCE_TEMPO_CHAIN",
                "steps_left": int(5 * skill),
                "params": {"pressure_delta": 0.02 * skill, "readiness_delta": 0.01 * skill, "morale_delta": 0.01 * skill}
            })
        elif sel == "2":
            readiness = redistribute_load(readiness, skill, primary_gain=0.06, peer_cost=0.03)
            morale += 0.03 * skill
        elif sel == "3":
            pressure *= (1.0 + 0.05 / skill)
            morale -= 0.04 / skill

    elif choice == "FATIGUE":
        print("Force fatigue rising. Choose:")
        print("  [1] Reduce tempo significantly")
        print("  [2] Aggressively redistribute load")
        print("  [3] Push through hard")
        sel = input_fn("Decision (1/2/3): ").strip()

        if sel == "1":
            pressure *= (1.0 - 0.20 * skill)
            readiness = readiness + 0.15 * skill
            morale += 0.08 * skill
            decision_state.active_effects.append({
                "type": "REDUCE_TEMPO_CHAIN",
                "steps_left": int(8 * skill),
                "params": {"pressure_delta": 0.03 * skill, "readiness_delta": 0.015 * skill, "morale_delta": 0.01 * skill}
            })
        elif sel == "2":
            readiness = redistribute_load(readiness, skill, primary_gain=0.12, peer_cost=0.06)
            morale += 0.05 * skill
        elif sel == "3":
            pressure *= (1.0 + 0.15 / skill)
            readiness = readiness - 0.10 / skill
            morale -= 0.10 / skill

    elif choice == "PRESSURE_EARLY":
        print("Coordination strain emerging. Choose:")
        print("  [1] Reinforce comms lightly")
        print("  [2] Slow maneuver slightly")
        print("  [3] Accept higher tempo")
        sel = input_fn("Decision (1/2/3): ").strip()

        if sel == "1":
            pressure -= 0.08 * skill
            morale += 0.04 * skill
            decision_state.active_effects.append({
                "type": "REINFORCE_COMMS_CHAIN",
                "steps_left": int(4 * skill),
                "params": {"pressure_delta": 0.03 * skill}
            })
        elif sel == "2":
            pressure -= 0.10 * skill
            readiness += 0.05 * skill
            morale += 0.03 * skill
        elif sel == "3":
            pressure += 0.05 / skill
            morale -= 0.03 / skill

    elif choice == "PRESSURE":
        print("Coordination load heavy. Choose:")
        print("  [1] Strongly reinforce comms")
        print("  [2] Slow maneuver significantly")
        print("  [3] Accept risk and push")
        sel = input_fn("Decision (1/2/3): ").strip()

        if sel == "1":
            pressure -= 0.15 * skill
            morale += 0.06 * skill
            decision_state.active_effects.append({
                "type": "REINFORCE_COMMS_CHAIN",
                "steps_left": int(6 * skill),
                "params": {"pressure_delta": 0.04 * skill}
            })
        elif sel == "2":
            pressure -= 0.20 * skill
            readiness += 0.10 * skill
            morale += 0.08 * skill
        elif sel == "3":
            pressure += 0.10 / skill
            morale -= 0.08 / skill

    elif choice == "OUTCOMES_EARLY":
        print("Friction emerging. Choose:")
        print("  [1] Rotate units lightly")
        print("  [2] Redistribute tasks")
        print("  [3] Ignore (monitor only)")
        sel = input_fn("Decision (1/2/3): ").strip()

        if sel == "1":
            f = intervention_factor(0.9, skill)
            scale_outcome_column(outcomes, COL_STALL, f)
            scale_outcome_column(outcomes, COL_FAILURE, f)
            morale += 0.04 * skill
            decision_state.active_effects.append({
                "type": "ROTATE_UNITS_CHAIN",
                "steps_left": int(4 * skill),
                "params": {"stall_factor": 0.97, "failure_factor": 0.97}
            })
        elif sel == "2":
            f = intervention_factor(0.95, skill)
            scale_outcome_column(outcomes, COL_STALL, f)
            morale += 0.03 * skill
        elif sel == "3":
            morale -= 0.02 / skill

    elif choice == "OUTCOMES":
        print("Unit friction detected. Choose:")
        print("  [1] Aggressively rotate units")
        print("  [2] Strongly redistribute tasks")
        print("  [3] Ignore (accept failures)")
        sel = input_fn("Decision (1/2/3): ").strip()

        if sel == "1":
            f = intervention_factor(0.7, skill)
            scale_outcome_column(outcomes, COL_STALL, f)
            scale_outcome_column(outcomes, COL_FAILURE, f)
            morale += 0.06 * skill
            decision_state.active_effects.append({
                "type": "ROTATE_UNITS_CHAIN",
                "steps_left": int(6 * skill),
                "params": {"stall_factor": 0.9, "failure_factor": 0.9}
            })
        elif sel == "2":
            f = intervention_factor(0.8, skill)
            scale_outcome_column(outcomes, COL_STALL, f)
            morale += 0.04 * skill
        elif sel == "3":
            outcomes[:, COL_FAILURE] += int(2 / skill)
            morale -= 0.08 / skill

    elif choice == "CATASTROPHIC":
        print("Collapse imminent. Choose:")
        print("  [1] Attempt strong emergency stabilization")
        print("  [2] Abort maneuver")
        print("  [3] Hold course (accept collapse)")
        sel = input_fn("Decision (1/2/3): ").strip()

        if sel == "1":
            readiness = readiness + 0.25 * skill
            pressure -= 0.25 * skill
            morale += 0.12 * skill
            scale_outcome_column(outcomes, COL_FAILURE, intervention_factor(0.6, skill))
            scale_outcome_column(outcomes, COL_STALL, intervention_factor(0.7, skill))
            decision_state.active_effects.append({
                "type": "STABILIZATION_CHAIN",
                "steps_left": int(5 * skill),
                "params": {"readiness_delta": 0.02 * skill, "pressure_delta": 0.03 * skill, "morale_delta": 0.01 * skill}
            })
        elif sel == "2":
            print("\n>>> MANEUVER ABORTED BY COMMAND DECISION.\n")
            morale -= 0.05 / skill
            aborted = True
        elif sel == "3":
            pressure += 0.05 / skill
            morale -= 0.10 / skill

    readiness = np.clip(readiness, 0.0, 1.0)
    pressure = float(np.clip(pressure, 0.0, 1.0))
    morale = float(np.clip(morale, 0.0, 1.0))

    return readiness, pressure, outcomes, morale, aborted


# ============================================================
# MISSION SUMMARY FOOTER
# ============================================================

def print_mission_summary(logger, commander, skill, scenario):
    states = logger.buffer

    if not states:
        print("\n(No mission data logged.)\n")
        return

    nhp_values = [s["nhp"] for s in states]
    sop_values = [s["sop"] for s in states]
    theta_s_values = [s["metadata"]["theta_s"] for s in states]
    theta_c_values = [s["metadata"]["theta_c"] for s in states]
    theta_o_values = [s["metadata"]["theta_o"] for s in states]
    morale_values = [s["metadata"]["morale"] for s in states]
    enemy_r_values = [s["metadata"]["enemy_readiness"] for s in states]
    enemy_p_values = [s["metadata"]["enemy_pressure"] for s in states]
    enemy_o_values = [s["metadata"]["enemy_outcomes"] for s in states]
    enemy_m_values = [s["metadata"]["enemy_morale"] for s in states]
    enemy_postures = [s["metadata"]["enemy_posture"] for s in states]

    alert_count = sum(1 for s in states if s["alert"])
    alert_pct = 100.0 * alert_count / len(states)

    def friendly_flags(s):
        m = s["metadata"]
        return friendly_collapse_flags(m["theta_s"], m["theta_c"], m["theta_o"], m["morale"])

    def enemy_flags(s):
        m = s["metadata"]
        return enemy_collapse_flags(m["enemy_readiness"], m["enemy_pressure"], m["enemy_morale"])

    # Collapse = every joint condition held on COLLAPSE_SUSTAIN_TICKS consecutive
    # ticks (a one-tick breach that recovers is not a collapse).
    ticks = [s["metadata"]["t"] for s in states]
    friendly_run = sustained_collapse(ticks, [all(friendly_flags(s)) for s in states])
    enemy_run = sustained_collapse(ticks, [all(enemy_flags(s)) for s in states])
    friendly_collapse = friendly_run["collapsed"]
    enemy_collapse = enemy_run["collapsed"]

    # --- Near-collapse proximity tier ---
    # Tracks, tick by tick, how many of the collapse conditions were satisfied
    # simultaneously, so a run that got close (e.g. 3 of 4 conditions at once)
    # is distinguishable from one that never came close at all, and a breach of
    # all of them that was too brief to count is reported as such.
    def friendly_condition_count(s):
        return sum(friendly_flags(s))

    def enemy_condition_count(s):
        return sum(enemy_flags(s))

    friendly_counts = [(friendly_condition_count(s), s["metadata"]["t"]) for s in states]
    enemy_counts = [(enemy_condition_count(s), s["metadata"]["t"]) for s in states]

    friendly_max_count, friendly_max_t = max(friendly_counts, key=lambda x: x[0])
    enemy_max_count, enemy_max_t = max(enemy_counts, key=lambda x: x[0])

    FRIENDLY_TOTAL = 4
    ENEMY_TOTAL = 3

    def near_collapse_label(max_count, total, is_full_collapse, at_t, run):
        if is_full_collapse:
            return None  # already reported as full collapse
        if max_count == total:
            return (f"BRIEF BREACH — all {total} conditions met at t={run['longest_t']} for "
                    f"{run['longest']} tick(s); {COLLAPSE_SUSTAIN_TICKS} needed for collapse")
        if max_count == total - 1:
            return f"NEAR-COLLAPSE — {max_count}/{total} conditions aligned at t={at_t}"
        elif max_count > 0:
            return f"Contained — best alignment {max_count}/{total} conditions at t={at_t}"
        else:
            return "No alignment — conditions never overlapped"

    friendly_near = near_collapse_label(friendly_max_count, FRIENDLY_TOTAL, friendly_collapse, friendly_max_t, friendly_run)
    enemy_near = near_collapse_label(enemy_max_count, ENEMY_TOTAL, enemy_collapse, enemy_max_t, enemy_run)

    print("\n==================== MISSION SUMMARY ====================\n")
    print(f"Scenario:               {scenario.name.capitalize()}")
    print(f"Commander Skill Level:  {commander.level.capitalize()}")
    print(f"Skill Multiplier:       {skill:.2f}\n")

    print(f"Baseline SOP (t=0–24): {np.mean(sop_values[:25]):.3f}")
    print(f"Final SOP (t={states[-1]['metadata']['t']}): {sop_values[-1]:.3f}")
    print(f"Final NHP:              {nhp_values[-1]:.3f}")
    print(f"ALERT Duration:         {alert_count} steps ({alert_pct:.1f}%)\n")

    print("Friendly Readiness (theta_s):")
    print(f"  Min: {min(theta_s_values):.3f}  Avg: {np.mean(theta_s_values):.3f}  Max: {max(theta_s_values):.3f}")

    print("Friendly Pressure (theta_c):")
    print(f"  Min: {min(theta_c_values):.3f}  Avg: {np.mean(theta_c_values):.3f}  Max: {max(theta_c_values):.3f}")

    print("Friendly Outcomes (theta_o):")
    print(f"  Min: {min(theta_o_values):.3f}  Avg: {np.mean(theta_o_values):.3f}  Max: {max(theta_o_values):.3f}\n")

    print("Friendly Morale:")
    print(f"  Min: {min(morale_values):.3f}  Avg: {np.mean(morale_values):.3f}  Max: {max(morale_values):.3f}\n")

    print("Enemy Readiness:")
    print(f"  Min: {min(enemy_r_values):.3f}  Avg: {np.mean(enemy_r_values):.3f}  Max: {max(enemy_r_values):.3f}")

    print("Enemy Pressure:")
    print(f"  Min: {min(enemy_p_values):.3f}  Avg: {np.mean(enemy_p_values):.3f}  Max: {max(enemy_p_values):.3f}")

    print("Enemy Outcomes:")
    print(f"  Min: {min(enemy_o_values):.3f}  Avg: {np.mean(enemy_o_values):.3f}  Max: {max(enemy_o_values):.3f}\n")

    print("Enemy Morale:")
    print(f"  Min: {min(enemy_m_values):.3f}  Avg: {np.mean(enemy_m_values):.3f}  Max: {max(enemy_m_values):.3f}\n")

    print(f"Enemy Posture Samples:  {sorted(set(enemy_postures))}\n")

    if friendly_collapse and enemy_collapse:
        print("Collapse Signature:      MUTUAL COLLAPSE")
    elif friendly_collapse and not enemy_collapse:
        print("Collapse Signature:      FRIENDLY COLLAPSE")
    elif enemy_collapse and not friendly_collapse:
        print("Collapse Signature:      ENEMY COLLAPSE")
    else:
        print("Collapse Signature:      NOT DETECTED")

    if friendly_collapse:
        print(f"Friendly Collapse:       onset t={friendly_run['onset_t']}, "
              f"held {friendly_run['held']} ticks (>= {COLLAPSE_SUSTAIN_TICKS} required)")
    if enemy_collapse:
        print(f"Enemy Collapse:          onset t={enemy_run['onset_t']}, "
              f"held {enemy_run['held']} ticks (>= {COLLAPSE_SUSTAIN_TICKS} required)")

    if friendly_near:
        print(f"Friendly Proximity:      {friendly_near}")
    if enemy_near:
        print(f"Enemy Proximity:         {enemy_near}")

    print("\n==========================================================\n")


# ============================================================
# CATASTROPHE SKILL TERM
# ============================================================

def catastrophe_readiness_retention(skill):
    """
    Fraction of readiness that survives the catastrophic event.

    This used to be 0.25 / skill, which gave skilled commanders a *smaller*
    surviving fraction (more readiness lost) than novices, the opposite of the
    pressure (+0.55 / skill) and morale (-0.15 / skill) terms, which both hurt
    skilled commanders less. It now rises with skill and is unchanged at
    skill 1.0.
    """
    return min(1.0, 0.25 * skill)


# ============================================================
# ENGINE PROFILES
# ============================================================
#
# "legacy" is the original tuning (steepness 16). At that steepness the
# recursive update treats every tick as fresh, independent evidence, but SOP is
# strongly autocorrelated, so a persistent deviation of only ~0.05 from the
# calibrated baseline pushes NHP past the alert line within ~6 ticks and to its
# retention ceiling (0.8 * 1 + 0.2 * prior = 0.81) soon after. NHP then reads
# ~0.75-0.80 for most of any mission regardless of scenario, skill, or
# decisions, so it can't express severity.
#
# "graded" lowers steepness to 3.0 and leaves everything else (including the
# 0.80 retention) as tuned. In seeded sweeps this keeps the healthy phase
# normal, lets NHP separate good from poor decisions, and still detects the
# t=110 catastrophe within a tick.

ENGINE_PROFILES = {
    "legacy": dict(prior_nh=0.05, steepness=16.0, upper_th=0.50, lower_th=0.40,
                   retention=0.80, calibration_steps=25),
    "graded": dict(prior_nh=0.05, steepness=3.0, upper_th=0.50, lower_th=0.40,
                   retention=0.80, calibration_steps=25),
    # "training": evidence accumulated with forgetting (see engine.py). Steady-state
    # gain steepness / (1 - forget) = 12.5 log-odds per unit of SOP deviation, so
    # a well-handled force (deviation ~0.10-0.17) reads ~0.15-0.35, a poorly
    # handled one (~0.26-0.35) ~0.55-0.85, and the catastrophe (~0.45-0.5) ~0.93.
    "training": dict(mode="discounted", prior_nh=0.05, steepness=2.5, forget=0.80,
                     upper_th=0.50, lower_th=0.40, calibration_steps=25),
}
DEFAULT_ENGINE_PROFILE = "training"


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(scenario_name="baseline", commander_level="expert", feedback=DEFAULT_FEEDBACK,
                   seed=None, input_fn=input, delay=0.1, log_file=None, n_ticks=120,
                   engine_profile=DEFAULT_ENGINE_PROFILE, engine_kwargs=None,
                   prompt_guards="fixed", n_units=DEFAULT_N_UNITS,
                   decision_timeout=DEFAULT_DECISION_TIMEOUT,
                   map_log=None, map_live=False, collect_map="auto", frame_callback=None, scenario_events="distinct",
                   strain_fraction=None, scenario_overrides=None, tick_hook=None, push_cost=None, neglect_cost=None):
    """
    Run one maneuver mission and return (logger, commander, skill, scenario, map_sink).

    scenario_name : scenario key (urban, deep_maneuver, logistics_constrained,
                    high_enemy_pressure, baseline)
    feedback      : morale feedback-loop strength (0.0 reproduces the original,
                    uncoupled dynamics exactly for a given seed)
    engine_profile: "graded" (default) or "legacy" (original steepness-16 tuning);
                    engine_kwargs overrides individual engine parameters on top
    prompt_guards : "fixed" (default; escalated prompts respect a per-category
                    cooldown and running effects) or "legacy" (original guards,
                    which re-prompted on every tick a threshold held)
    seed          : numpy RNG seed for reproducible runs (None = unseeded)
    input_fn      : callable used for decision prompts (input by default; a
                    scripted policy for headless runs)
    delay         : seconds to sleep between ticks (0 for headless)
    log_file      : JSON-lines log path (default maneuver_log_<scenario>.json)
    n_units       : force size; readiness is (n,), outcomes is (n, 3). Default 3
                    preserves the original company-scale dynamics.
    decision_timeout : seconds allowed on each decision menu. None/0 = blocking
                    (original behaviour). On timeout the ignore option is taken
                    and initiative is forfeited.
    map_log       : optional path for map telemetry JSON-lines frames. Always
                    collected in memory on the returned map_sink; file write only
                    when this path is set.
    map_live      : if True, open a matplotlib window and redraw each tick
                    (interactive / desktop only; ignore in headless CI).
    collect_map   : "auto" (default): build map frames except for large forces
                    (more than LARGE_FORCE_UNITS units) when neither map_log nor
                    map_live asks for them; True / False force it on / off.

    scenario_events: "distinct" (default): each scenario has its own strain timing
                    and catastrophe composition (Scenario.EVENT_PROFILES). "legacy"
                    reproduces the original identical t=30 / t=110 events.
    frame_callback: optional callable(frame) invoked for every map frame as it is
                    produced (used by --serve to feed the browser map live).
                    Asking for it turns frame building on even for large forces.

    strain_fraction: None (default) = the original t=30 event, one strained unit
                    (unit 1). A float in (0, 1] strains that share of the force
                    instead (a contiguous block of units), so the event's size
                    does not shrink as n_units grows; 1/3 matches the original
                    exactly at 3 units.

    scenario_overrides: optional dict of Scenario attributes applied after the event
                    profile (e.g. strain_tick, catastrophe_tick, cat_*_scale). None =
                    no change. Used by the training vignettes.
    tick_hook     : optional callable(info) run as each tick is logged with
                    info = {t, decision, nhp, severity, mission_state, effects}.
                    decision is the resolved decision dict (type, selected,
                    timed_out, ...) or None. Read-only; consumes no RNG.

    push_cost     : None (default) = off. A dict {"readiness": r, "pressure": p,
                    "steps": n} makes pushing on (option 3 at a FATIGUE or PRESSURE
                    menu, including a timeout) borrow against the future: for n ticks
                    readiness falls r and pressure rises p per tick. Without it,
                    pushing before a shock is free because the force recovers fully,
                    so tempo would never be a trade-off. Used by the tempo vignette.
    neglect_cost  : None (default) = off. Same dict format. Ignoring a FATIGUE,
                    PRESSURE or OUTCOMES prompt (option 3, early or escalated, or a
                    timeout) leaves lasting strain: the cost runs for "steps" ticks.
                    Without it nothing carries over from before a shock, so acting on
                    early warnings could never matter. Used by the early-warning vignette.

    Large forces (n_units > LARGE_FORCE_UNITS) also use the vectorized outcomes
    sampler (same distributions, different RNG stream) and compact log lines
    (readiness summary instead of the full per-unit list; positions written once).
    """
    if seed is not None:
        np.random.seed(seed)

    n_units = int(n_units)
    if n_units < 1:
        raise ValueError("n_units must be >= 1")

    compact_log = n_units > LARGE_FORCE_UNITS
    if collect_map == "auto":
        collect_map = (not compact_log) or bool(map_log) or bool(map_live) or bool(frame_callback)
    collect_map = bool(collect_map)
    positions_logged = {"done": False}

    scenario = Scenario(scenario_name)
    if scenario_events == "legacy":
        scenario.use_legacy_events()
    elif scenario_events != "distinct":
        raise ValueError("scenario_events must be 'distinct' or 'legacy'")
    for _k, _v in (scenario_overrides or {}).items():
        if not hasattr(scenario, _k):
            raise ValueError(f"unknown scenario attribute: {_k}")
        setattr(scenario, _k, _v)
    commander = CommanderSkill(level=commander_level)
    skill = commander.multiplier()

    engine_params = dict(ENGINE_PROFILES[engine_profile])
    engine_params.update(engine_kwargs or {})
    engine = JOREngine(**engine_params)

    adapter = ManeuverAdapter()
    logger = FusionLogger(log_file or f"maneuver_log_{scenario.name}.json")
    map_sink = MapTelemetrySink(path=map_log, on_emit=frame_callback)
    live_map = LiveMapDisplay(pause=0.01) if map_live else None
    positions = init_positions(n_units, scenario.name, seed=seed)

    degraded_unit = None
    catastrophic_event = False
    decision_state = DecisionState(guard_mode=prompt_guards)
    feedback_state = FeedbackState(strength=feedback)

    # Skill-scaled decision window: experts get more time, novices less.
    # Unchanged when decision_timeout is None (blocking).
    effective_timeout = decision_timeout
    if decision_timeout is not None and decision_timeout > 0:
        effective_timeout = max(2.0, float(decision_timeout) * skill)

    last_decision = {"choice": None, "selected": None, "timed_out": False}
    _timed_input = make_timed_input_fn(effective_timeout, base_input_fn=input_fn)

    def decision_input(prompt=""):
        timed_input.expired = False
        sel = _timed_input(prompt)
        last_decision["selected"] = (sel or "").strip()
        last_decision["timed_out"] = bool(timed_input.expired)
        return sel

    morale = 0.75

    enemy = EnemyForce(posture=scenario.enemy_posture)

    # Graded severity gauge (see severity.py), one-shot events for the debrief, and
    # a slim per-tick feed that exists even when map frames are not built (large
    # forces), so the debrief never depends on the map.
    gauge = SeverityGauge()
    tick_events = []
    logger.feed = []

    def log_tick(t, sop, nhp, alert, theta_s, theta_c, theta_o, readiness, pressure,
                 morale, outliers, median, worst_ema, outcomes, decision=None):
        if decision is None and last_decision.get("choice"):
            decision = describe_decision(
                last_decision["choice"],
                selected=last_decision.get("selected"),
                timeout_s=effective_timeout,
                timed_out=last_decision.get("timed_out", False),
            )
            last_decision["choice"] = None
        severity = gauge.update(sop, engine.baseline_sop if engine.is_calibrated else None)
        sev_band = severity_band(severity)
        events_now = list(tick_events)
        tick_events.clear()
        mission_state = classify_mission_state(theta_s, theta_c, theta_o, degraded_unit, morale, skill, enemy)
        narrative = describe_troop_state(t, readiness, pressure, outcomes, degraded_unit, morale, skill, commander.level, enemy)
        if compact_log:
            # Large force: per-unit vectors in every line would dominate the log
            # (positions never change within a run, so they are written once).
            unit_fields = {
                "log_mode": "compact",
                "readiness_summary": {
                    "min": float(np.min(readiness)),
                    "mean": float(np.mean(readiness)),
                    "max": float(np.max(readiness)),
                    "lowest_units": [int(i) for i in np.argsort(readiness)[:5]],
                },
            }
            if not positions_logged["done"]:
                unit_fields["positions"] = positions.tolist()
                positions_logged["done"] = True
        else:
            unit_fields = {"readiness": readiness.tolist(), "positions": positions.tolist()}
        logger.log_state(
            sop, nhp, alert,
            metadata={
                "t": t,
                "mission_state": mission_state,
                "theta_s": theta_s,
                "theta_c": theta_c,
                "theta_o": theta_o,
                **unit_fields,
                "pressure": pressure,
                "morale": morale,
                "outliers": outliers,
                "median_tempo": median,
                "worst_ema": worst_ema,
                "severity": round(float(severity), 4),
                "severity_band": sev_band,
                "degraded_unit": degraded_unit,
                "commander_skill": commander.level,
                "skill_multiplier": skill,
                "enemy_readiness": enemy.readiness,
                "enemy_pressure": enemy.pressure,
                "enemy_outcomes": enemy.outcomes,
                "enemy_morale": enemy.morale,
                "enemy_posture": enemy.posture,
                "scenario": scenario.name,
                "feedback_strength": feedback,
                "readiness_debt": feedback_state.readiness_debt,
                "pressure_creep": feedback_state.pressure_creep,
                "n_units": n_units,
                "narrative": narrative
            }
        )
        logger.feed.append({
            "t": t, "scenario": scenario.name, "mission_state": mission_state,
            "nhp": float(nhp), "alert": bool(alert),
            "theta_s": float(theta_s), "theta_c": float(theta_c), "theta_o": float(theta_o),
            "pressure": float(pressure), "morale": float(morale),
            "severity": float(severity), "severity_band": sev_band,
            "enemy": {"readiness": enemy.readiness, "pressure": enemy.pressure,
                      "outcomes": enemy.outcomes, "morale": enemy.morale},
            "degraded": is_degraded(degraded_unit),
            "decision": decision, "events": events_now,
        })
        if collect_map:
            map_sink.emit(build_frame(
                t=t,
                positions=positions,
                readiness=readiness,
                pressure=pressure,
                morale=morale,
                nhp=nhp,
                alert=alert,
                mission_state=mission_state,
                theta_s=theta_s,
                theta_c=theta_c,
                theta_o=theta_o,
                degraded_unit=degraded_unit,
                scenario=scenario.name,
                n_units=n_units,
                outcomes=outcomes,
                enemy={
                    "readiness": enemy.readiness,
                    "pressure": enemy.pressure,
                    "outcomes": enemy.outcomes,
                    "morale": enemy.morale,
                    "posture": enemy.posture,
                },
                decision=decision,
                decision_timeout_s=effective_timeout,
                severity=severity,
                severity_band=sev_band,
                events=events_now,
            ))
        if live_map is not None:
            live_map.update(map_sink.last())
        if tick_hook is not None:
            tick_hook({
                "t": t, "decision": decision, "nhp": float(nhp), "severity": float(severity),
                "mission_state": mission_state,
                "effects": [e["type"] for e in decision_state.active_effects],
            })
        return mission_state, narrative

    for t in range(n_ticks):

        if t == scenario.strain_tick:
            degraded_unit = strained_units(n_units, strain_fraction)
            cause = f" ({scenario.strain_cause})" if scenario.strain_cause else ""
            tick_events.append({"kind": "strain", "text": "Local strain" + (
                f": {scenario.strain_cause}" if scenario.strain_cause else " on a task force")})
            if strain_fraction is None:
                print(f"\n>>> Local Strain Detected{cause}: Unit 1 showing increased friction.\n")
            else:
                print(f"\n>>> Local Strain Detected{cause}: {len(degraded_unit)} unit(s) "
                      f"({degraded_unit[0]}-{degraded_unit[-1]}) showing increased friction.\n")

        if t == scenario.catastrophe_tick:
            catastrophic_event = True
            cause = scenario.cat_cause or "Operational overextension + coordination failure"
            tick_events.append({"kind": "catastrophe", "text": "Catastrophic event: " + cause})
            print(f"\n>>> CATASTROPHIC EVENT: {cause}.\n")

        decision_state.tick()

        readiness = generate_readiness(t, degraded_unit, scenario, n_units=n_units)
        pressure = generate_pressure(t, scenario)
        outcomes = generate_outcomes(t, degraded_unit, n_units=n_units)

        # Morale feedback loop: uses morale as it stood at the START of this tick
        # (one-tick lag), consumes no RNG, and is a strict no-op at strength 0.
        readiness, pressure = feedback_state.apply(readiness, pressure, morale)

        enemy.step(t)
        enemy.apply_friendly_skill_effects(skill)
        enemy.update_posture()

        if catastrophic_event:
            readiness = readiness * min(
                1.0, catastrophe_readiness_retention(skill) * scenario.cat_readiness_scale)
            pressure = min(1.0, pressure + 0.55 / skill * scenario.cat_pressure_scale)
            morale -= 0.15 / skill * scenario.cat_morale_scale
            # Per-unit RNG order preserved (stall then failure) for regression;
            # the scenario scale only multiplies what was drawn.
            for u in range(outcomes.shape[0]):
                outcomes[u, COL_STALL] += int(round(np.random.randint(12, 20) * scenario.cat_outcome_scale))
                outcomes[u, COL_FAILURE] += int(round(np.random.randint(10, 18) * scenario.cat_outcome_scale))

        pressure += enemy.pressure * 0.20
        morale += scenario.morale_sensitivity * (0.05 * (0.5 - enemy.morale))

        readiness, pressure, outcomes, morale = apply_active_effects(readiness, pressure, outcomes, morale, decision_state, skill)

        # Single theta_o computation per tick (compute_theta_o updates the
        # adapter's persistent EMA state as a side effect, so calling it more
        # than once per tick on the same outcomes would double/triple-count
        # that update and distort outlier detection).
        theta_s = adapter.compute_theta_s(readiness)
        theta_c = adapter.compute_theta_c(pressure)
        theta_o, outliers, median, worst_ema = adapter.compute_theta_o(outcomes)
        theta_o *= (1.0 + enemy.outcomes * 0.10)
        theta_o *= scenario.outcome_multiplier

        choice = maybe_prompt_user(
            theta_s, theta_c, theta_o,
            morale, catastrophic_event, decision_state, skill
        )

        if choice:
            last_decision["choice"] = choice
            last_decision["selected"] = None
            CURRENT_DECISION["type"] = choice      # lets scripted policies see which menu is open
            CURRENT_DECISION["t"] = t
            # Emit a PENDING decision frame BEFORE blocking on input so the
            # live dashboard can show the menu + countdown while the commander
            # chooses. Without this, frames only appear after the wait ends.
            try:
                # Preview on a COPY: fusion_step advances the recursive posterior,
                # so calling it on the live engine would double-step the tick.
                import copy as _copy
                _eng = _copy.deepcopy(engine)
                _sop, _nhp, _alert = _eng.fusion_step(theta_s, theta_c, theta_o)
                _sev = gauge.peek(_sop, _eng.baseline_sop if _eng.is_calibrated else None)
                _sev_band = severity_band(_sev)
                _ms = classify_mission_state(
                    theta_s, theta_c, theta_o, degraded_unit, morale, skill, enemy
                )
                pending = describe_decision(
                    choice, selected=None, timeout_s=effective_timeout
                )
                if pending is not None:
                    pending = dict(pending)
                    pending["pending"] = True
                # The feed always records the pre-choice state: the debrief compares
                # it with the state right after the choice to show the immediate effect.
                logger.feed.append({
                    "t": t, "scenario": scenario.name, "mission_state": _ms,
                    "nhp": float(_nhp), "alert": bool(_alert),
                    "theta_s": float(theta_s), "theta_c": float(theta_c), "theta_o": float(theta_o),
                    "pressure": float(pressure), "morale": float(morale),
                    "severity": float(_sev), "severity_band": _sev_band,
                    "enemy": {"readiness": enemy.readiness, "pressure": enemy.pressure,
                              "outcomes": enemy.outcomes, "morale": enemy.morale},
                    "degraded": is_degraded(degraded_unit),
                    "decision": pending, "events": [],
                })
                if collect_map:
                    map_sink.emit(build_frame(
                        t=t,
                        positions=positions,
                        readiness=readiness,
                        pressure=pressure,
                        morale=morale,
                        nhp=_nhp,
                        alert=_alert,
                        mission_state=_ms,
                        theta_s=theta_s,
                        theta_c=theta_c,
                        theta_o=theta_o,
                        degraded_unit=degraded_unit,
                        scenario=scenario.name,
                        n_units=n_units,
                        outcomes=outcomes,
                        enemy={
                            "readiness": enemy.readiness,
                            "pressure": enemy.pressure,
                            "outcomes": enemy.outcomes,
                            "morale": enemy.morale,
                            "posture": enemy.posture,
                        },
                        decision=pending,
                        decision_timeout_s=effective_timeout,
                        severity=_sev,
                        severity_band=_sev_band,
                    ))
            except Exception:
                pass
            readiness, pressure, outcomes, morale, aborted = apply_user_choice(
                choice, readiness, pressure, outcomes, morale, decision_state, skill,
                input_fn=decision_input
            )
            decision_state.record_decision(choice)
            _sel = (last_decision.get("selected") or "").strip()
            _cat = choice.replace("_EARLY", "")
            for _cost, _cats in ((push_cost, ("FATIGUE", "PRESSURE")),
                                 (neglect_cost, ("FATIGUE", "PRESSURE", "OUTCOMES"))):
                if _cost and not aborted and _cat in _cats and _sel == "3":
                    decision_state.active_effects.append({
                        "type": "DEBT_CHAIN", "steps_left": int(_cost.get("steps", 12)),
                        "params": {"readiness_delta": float(_cost.get("readiness", 0.0)),
                                   "pressure_delta": float(_cost.get("pressure", 0.0))}})
            # State changed as a result of the choice, so refresh the metrics once.
            theta_s = adapter.compute_theta_s(readiness)
            theta_c = adapter.compute_theta_c(pressure)
            theta_o, outliers, median, worst_ema = adapter.compute_theta_o(outcomes)
            theta_o *= (1.0 + enemy.outcomes * 0.10)
            theta_o *= scenario.outcome_multiplier

            if aborted:
                sop, nhp, alert = engine.fusion_step(theta_s, theta_c, theta_o)
                log_tick(t, sop, nhp, alert, theta_s, theta_c, theta_o, readiness, pressure,
                         morale, outliers, median, worst_ema, outcomes)
                break

        if theta_o < 0.20 and pressure < 0.60 and theta_s > 0.60:
            morale += 0.02 * skill
        elif theta_o > 0.40:
            morale -= 0.05 / skill

        if pressure > 0.80:
            morale -= 0.04 / skill
        if theta_s > 0.70:
            morale += 0.03 * skill
        elif theta_s < 0.40:
            morale -= 0.05 / skill

        morale = float(np.clip(morale, 0.0, 1.0))

        sop, nhp, alert = engine.fusion_step(theta_s, theta_c, theta_o)

        mission_state, narrative = log_tick(
            t, sop, nhp, alert, theta_s, theta_c, theta_o, readiness, pressure,
            morale, outliers, median, worst_ema, outcomes
        )

        print(f"t={t:3d}  [{mission_state:16s}]")
        print(f"  Readiness (theta_s): {theta_s:.3f}  {ascii_bar(theta_s)}")
        print(f"  Pressure  (theta_c): {theta_c:.3f}  {ascii_bar(theta_c)}")
        print(f"  Outcomes  (theta_o): {theta_o:.3f}  {ascii_bar(theta_o)}")
        print(f"  Morale             : {morale:.3f}  {ascii_bar(morale)}")
        print(f"  Enemy Readiness    : {enemy.readiness:.3f}  {ascii_bar(enemy.readiness)}")
        print(f"  Enemy Pressure     : {enemy.pressure:.3f}  {ascii_bar(enemy.pressure)}")
        print(f"  Enemy Outcomes     : {enemy.outcomes:.3f}  {ascii_bar(enemy.outcomes)}")
        print(f"  Enemy Morale       : {enemy.morale:.3f}  {ascii_bar(enemy.morale)}")
        print(f"  Enemy Posture      : {enemy.posture}")
        print(f"  NHP Posterior : {nhp:.3f}  {'ALERT' if alert else 'Normal'}")
        print(f"  Outliers: {outliers}  Median Tempo: {median:.3f}  Worst EMA: {worst_ema:.3f}")
        if feedback_state.strength > 0:
            print(f"  Feedback      : readiness debt {feedback_state.readiness_debt:.3f}  pressure creep {feedback_state.pressure_creep:.3f}")
        print(textwrap.fill(f">> {narrative}", width=100))
        print()

        if delay > 0:
            time.sleep(delay)

    map_sink.flush()
    if live_map is not None:
        live_map.close()
    return logger, commander, skill, scenario, map_sink


def main():
    parser = argparse.ArgumentParser(description="JOR 4.0 maneuver health simulation")
    parser.add_argument("--feedback", type=float, default=DEFAULT_FEEDBACK,
                        help="morale feedback-loop strength (0 = original uncoupled dynamics)")
    parser.add_argument("--engine", choices=sorted(ENGINE_PROFILES), default=DEFAULT_ENGINE_PROFILE,
                        help="engine tuning profile (legacy = original steepness 16)")
    parser.add_argument("--prompt-guards", choices=["fixed", "legacy"], default="fixed",
                        help="decision-prompt guards (legacy = original, re-prompts every tick)")
    parser.add_argument("--n-units", type=int, default=DEFAULT_N_UNITS,
                        help="force size (readiness vector / outcomes matrix rows); default 3")
    parser.add_argument("--scenario-events", choices=["distinct", "legacy"], default="distinct",
                        help="distinct = each scenario has its own strain timing and catastrophe; "
                             "legacy = the original identical events in every scenario")
    parser.add_argument("--serve", nargs="?", const=8766, type=int, default=None, metavar="PORT",
                        help="also serve the map dashboard live on this port (default 8766) so a "
                             "single-monitor user can keep the terminal and the map side by side")
    parser.add_argument("--no-browser", action="store_true",
                        help="with --serve: do not open the browser automatically")
    parser.add_argument("--strain-fraction", type=float, default=None,
                        help="share of the force hit by the t=30 strain event (0-1]; "
                             "default: the original single unit. 0.333 matches it at 3 units.")
    parser.add_argument("--decision-timeout", type=float, default=0.0,
                        help="seconds allowed on each decision menu (0 = blocking, no timer)")
    parser.add_argument("--map-log", type=str, default=None,
                        help="write map telemetry JSON-lines frames to this path")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["baseline", "urban", "deep_maneuver",
                                 "logistics_constrained", "high_enemy_pressure"],
                        help="scenario key (if omitted, interactive menu is shown)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible runs")
    parser.add_argument("--commander", type=str, default="expert",
                        choices=["novice", "intermediate", "expert"],
                        help="commander skill level (default expert)")
    args = parser.parse_args()

    decision_timeout = args.decision_timeout if args.decision_timeout > 0 else None

    print("=== JOR 4.0 — Maneuver Health Simulation (Morale + Commander Skill + Enemy + Scenarios + Posture) ===\n")

    print("INTERPRETING THE METRICS:")
    print("  Readiness (theta_s):   Troop physical/mental readiness. High = good. Low = fatigue.")
    print("  Pressure  (theta_c):   Operational pressure and coordination load. High = stress.")
    print("  Outcomes  (theta_o):   Maneuver results (success/stall/failure). High = worse outcomes.")
    print("  Morale:                Force cohesion, resilience, and confidence (0–1).")
    print("  Enemy metrics:         Parallel readiness/pressure/outcomes/morale for adversary.")
    print("  NHP Posterior:         Probability the force is in a non-healthy state.")
    print("                         Normal < 0.40, Warning 0.40–0.50, ALERT > 0.50.")
    print(f"  Feedback loop:         Low morale accumulates readiness debt and pressure creep (strength {args.feedback:.2f}).")
    print(f"  Force size:            {args.n_units} units (matrix state).")
    if decision_timeout:
        print(f"  Decision timer:        {decision_timeout:.0f}s window (skill-scaled; timeout = forfeit).")
    else:
        print("  Decision timer:        off (blocking input).")
    print("--------------------------------------------------------------------------\n")

    viewer = None
    if args.serve is not None:
        from stream_server import start_viewer
        viewer = start_viewer(port=args.serve, open_browser=not args.no_browser)

    if args.scenario:
        scenario_name = args.scenario
    else:
        print("Choose scenario:")
        print("  [1] Urban")
        print("  [2] Deep Maneuver")
        print("  [3] Logistics Constrained")
        print("  [4] High Enemy Pressure")
        print("  [5] Baseline")
        sel = input("Scenario (1–5): ").strip()
        scenario_map = {
            "1": "urban",
            "2": "deep_maneuver",
            "3": "logistics_constrained",
            "4": "high_enemy_pressure",
            "5": "baseline",
        }
        scenario_name = scenario_map.get(sel, "baseline")

    logger, commander, skill, scenario, map_sink = run_simulation(
        scenario_name=scenario_name,
        commander_level=args.commander,
        feedback=args.feedback,
        seed=args.seed,
        engine_profile=args.engine,
        prompt_guards=args.prompt_guards,
        n_units=args.n_units,
        decision_timeout=decision_timeout,
        map_log=args.map_log,
        strain_fraction=args.strain_fraction,
        frame_callback=viewer.broadcast if viewer else None,
        scenario_events=args.scenario_events,
    )

    if viewer:
        viewer.finish(map_sink.frames, feed=getattr(logger, "feed", None))

    print_mission_summary(logger, commander, skill, scenario)
    try:
        from debrief import build_debrief, format_debrief
        debrief = build_debrief(getattr(logger, "feed", None) or [])
        print(format_debrief(debrief))
        if args.map_log and map_sink.frames:
            from aar_summary import summarize_frames
            from map_telemetry import append_end_record
            append_end_record(args.map_log, summarize_frames(map_sink.frames), debrief)
    except Exception as exc:          # the debrief is an add-on; never break the summary
        print(f"(debrief unavailable: {exc})")
    if map_sink.frames:
        print(f"Map telemetry frames:   {len(map_sink.frames)}"
              + (f"  → {args.map_log}" if args.map_log else "  (in-memory only)"))


if __name__ == "__main__":
    main()
