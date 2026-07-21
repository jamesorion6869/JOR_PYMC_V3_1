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
"""

import numpy as np
import time
import textwrap

from engine import JOREngine
from logger import FusionLogger
from maneuver_adapter import ManeuverAdapter


# ============================================================
# SCENARIO LOADER
# ============================================================

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

class DecisionState:
    def __init__(self):
        self.fatigue_prompted = False
        self.pressure_prompted = False
        self.outcomes_prompted = False
        self.catastrophic_prompted = False
        self.active_effects = []


def apply_active_effects(readiness, pressure, outcomes, morale, decision_state, skill):
    new_effects = []
    for effect in decision_state.active_effects:
        etype = effect["type"]
        params = effect["params"]
        steps_left = effect["steps_left"]

        if etype == "REDUCE_TEMPO_CHAIN":
            pressure -= params.get("pressure_delta", 0.03 * skill)
            readiness += params.get("readiness_delta", 0.01 * skill)
            morale += params.get("morale_delta", 0.01 * skill)

        elif etype == "REINFORCE_COMMS_CHAIN":
            pressure -= params.get("pressure_delta", 0.04 * skill)

        elif etype == "ROTATE_UNITS_CHAIN":
            for u in outcomes:
                u["stall"] = max(0, int(u["stall"] * params.get("stall_factor", 0.95)))
                u["failure"] = max(0, int(u["failure"] * params.get("failure_factor", 0.95)))

        elif etype == "STABILIZATION_CHAIN":
            readiness += params.get("readiness_delta", 0.02 * skill)
            pressure -= params.get("pressure_delta", 0.03 * skill)
            morale += params.get("morale_delta", 0.01 * skill)

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

def generate_readiness(t, degraded_unit, scenario):
    base = np.array([0.85, 0.83, 0.88])
    decay = np.clip(t / 160, 0.0, 0.35)
    readiness = base - decay
    readiness -= scenario.readiness_decay
    readiness += np.random.normal(0.0, 0.015, size=readiness.shape)

    if degraded_unit is not None:
        readiness[degraded_unit] -= 0.12

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


def generate_outcomes(t, degraded_unit):
    outcomes = []
    for u in range(3):
        success = np.random.randint(11, 17)
        stall = np.random.randint(2, 5)
        failure = np.random.randint(0, 2)

        if u == degraded_unit:
            stall += np.random.randint(2, 4)
            failure += np.random.randint(1, 3)

        if np.random.rand() < 0.10:
            stall += np.random.randint(2, 5)
        if np.random.rand() < 0.06:
            failure += np.random.randint(1, 3)
        if np.random.rand() < 0.10:
            success += np.random.randint(2, 5)

        outcomes.append({"success": success, "stall": stall, "failure": failure})

    return outcomes


# ============================================================
# NARRATIVE + MISSION STATE
# ============================================================

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

    if degraded_unit is not None:
        desc.append(f"Unit {degraded_unit} showing strain.")

    for i, u in enumerate(outcomes):
        if u["failure"] > 4:
            desc.append(f"Unit {i} experiencing heavy failures.")
        elif u["stall"] > 5:
            desc.append(f"Unit {i} friction increasing.")
        elif u["failure"] > 1:
            desc.append(f"Unit {i} showing minor degradation.")

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


def classify_mission_state(theta_s, theta_c, theta_o, degraded_unit, morale, skill, enemy):
    friendly_critical = theta_s < (0.25 / skill) and theta_c > (0.80 / skill) and morale < (0.30 / skill)
    enemy_critical = enemy.readiness < 0.25 and enemy.pressure > 0.80 and enemy.morale < 0.30

    if friendly_critical and enemy_critical:
        return "MUTUAL_COLLAPSE"
    if friendly_critical and not enemy_critical:
        return "FRIENDLY_COLLAPSE"
    if enemy_critical and not friendly_critical:
        return "ENEMY_COLLAPSE"

    if degraded_unit is not None:
        return "DEGRADED"
    if theta_c > 0.65:
        return "HIGH_PRESSURE"
    if theta_s < 0.45 or morale < 0.40:
        return "FATIGUED"
    return "STABLE"


# ============================================================
# TACTICAL DECISION ENGINE
# ============================================================

def maybe_prompt_user(theta_s, theta_c, theta_o, morale, catastrophic_event, decision_state, skill):
    if theta_s < (0.70 / skill) and morale < 0.80 and not decision_state.fatigue_prompted:
        decision_state.fatigue_prompted = True
        return "FATIGUE_EARLY"

    if theta_s < (0.55 / skill) and morale < 0.70 and decision_state.fatigue_prompted and not any(e["type"].startswith("FATIGUE") for e in decision_state.active_effects):
        return "FATIGUE"

    if theta_c > (0.55 / skill) and not decision_state.pressure_prompted:
        decision_state.pressure_prompted = True
        return "PRESSURE_EARLY"

    if theta_c > (0.70 / skill) and decision_state.pressure_prompted and not any(e["type"].startswith("PRESSURE") for e in decision_state.active_effects):
        return "PRESSURE"

    if theta_o > (0.20 / skill) and not decision_state.outcomes_prompted:
        decision_state.outcomes_prompted = True
        return "OUTCOMES_EARLY"

    if theta_o > (0.35 / skill) and decision_state.outcomes_prompted and not any(e["type"].startswith("OUTCOMES") for e in decision_state.active_effects):
        return "OUTCOMES"

    if catastrophic_event and not decision_state.catastrophic_prompted:
        decision_state.catastrophic_prompted = True
        return "CATASTROPHIC"

    return None


def apply_user_choice(choice, readiness, pressure, outcomes, morale, decision_state, skill):
    print("\n--- DECISION POINT ---")

    aborted = False

    if choice == "FATIGUE_EARLY":
        print("Early signs of fatigue. Choose:")
        print("  [1] Slightly reduce tempo")
        print("  [2] Redistribute minor load")
        print("  [3] Ignore (accept strain)")
        sel = input("Decision (1/2/3): ").strip()

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
            readiness = readiness + np.array([0.06 * skill, -0.03 * skill, -0.03 * skill])
            morale += 0.03 * skill
        elif sel == "3":
            pressure *= (1.0 + 0.05 / skill)
            morale -= 0.04 / skill

    elif choice == "FATIGUE":
        print("Force fatigue rising. Choose:")
        print("  [1] Reduce tempo significantly")
        print("  [2] Aggressively redistribute load")
        print("  [3] Push through hard")
        sel = input("Decision (1/2/3): ").strip()

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
            readiness = readiness + np.array([0.12 * skill, -0.06 * skill, -0.06 * skill])
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
        sel = input("Decision (1/2/3): ").strip()

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
        sel = input("Decision (1/2/3): ").strip()

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
        sel = input("Decision (1/2/3): ").strip()

        if sel == "1":
            for u in outcomes:
                u["stall"] = max(0, int(u["stall"] * (0.9 / skill)))
                u["failure"] = max(0, int(u["failure"] * (0.9 / skill)))
            morale += 0.04 * skill
            decision_state.active_effects.append({
                "type": "ROTATE_UNITS_CHAIN",
                "steps_left": int(4 * skill),
                "params": {"stall_factor": 0.97, "failure_factor": 0.97}
            })
        elif sel == "2":
            for u in outcomes:
                u["stall"] = max(0, int(u["stall"] * (0.95 / skill)))
            morale += 0.03 * skill
        elif sel == "3":
            morale -= 0.02 / skill

    elif choice == "OUTCOMES":
        print("Unit friction detected. Choose:")
        print("  [1] Aggressively rotate units")
        print("  [2] Strongly redistribute tasks")
        print("  [3] Ignore (accept failures)")
        sel = input("Decision (1/2/3): ").strip()

        if sel == "1":
            for u in outcomes:
                u["stall"] = max(0, int(u["stall"] * (0.7 / skill)))
                u["failure"] = max(0, int(u["failure"] * (0.7 / skill)))
            morale += 0.06 * skill
            decision_state.active_effects.append({
                "type": "ROTATE_UNITS_CHAIN",
                "steps_left": int(6 * skill),
                "params": {"stall_factor": 0.9, "failure_factor": 0.9}
            })
        elif sel == "2":
            for u in outcomes:
                u["stall"] = max(0, int(u["stall"] * (0.8 / skill)))
            morale += 0.04 * skill
        elif sel == "3":
            for u in outcomes:
                u["failure"] += int(2 / skill)
            morale -= 0.08 / skill

    elif choice == "CATASTROPHIC":
        print("Collapse imminent. Choose:")
        print("  [1] Attempt strong emergency stabilization")
        print("  [2] Abort maneuver")
        print("  [3] Hold course (accept collapse)")
        sel = input("Decision (1/2/3): ").strip()

        if sel == "1":
            readiness = readiness + 0.25 * skill
            pressure -= 0.25 * skill
            morale += 0.12 * skill
            for u in outcomes:
                u["failure"] = max(0, int(u["failure"] * (0.6 / skill)))
                u["stall"] = max(0, int(u["stall"] * (0.7 / skill)))
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

    friendly_collapse = any(
        s["metadata"]["theta_s"] < (0.20 / skill) and
        s["metadata"]["theta_c"] > (0.90 / skill) and
        s["metadata"]["theta_o"] > (0.40 / skill) and
        s["metadata"]["morale"] < (0.30 / skill)
        for s in states
    )

    enemy_collapse = any(
        s["metadata"]["enemy_readiness"] < 0.20 and
        s["metadata"]["enemy_pressure"] > 0.90 and
        s["metadata"]["enemy_morale"] < 0.30
        for s in states
    )

    # --- Near-collapse proximity tier ---
    # Full collapse requires every joint condition to hold on the SAME tick.
    # This tracks, tick by tick, how many of those conditions were satisfied
    # simultaneously, so a run that got close (e.g. 3 of 4 conditions at once)
    # is distinguishable from one that never came close at all.
    def friendly_condition_count(s):
        m = s["metadata"]
        return sum([
            m["theta_s"] < (0.20 / skill),
            m["theta_c"] > (0.90 / skill),
            m["theta_o"] > (0.40 / skill),
            m["morale"] < (0.30 / skill),
        ])

    def enemy_condition_count(s):
        m = s["metadata"]
        return sum([
            m["enemy_readiness"] < 0.20,
            m["enemy_pressure"] > 0.90,
            m["enemy_morale"] < 0.30,
        ])

    friendly_counts = [(friendly_condition_count(s), s["metadata"]["t"]) for s in states]
    enemy_counts = [(enemy_condition_count(s), s["metadata"]["t"]) for s in states]

    friendly_max_count, friendly_max_t = max(friendly_counts, key=lambda x: x[0])
    enemy_max_count, enemy_max_t = max(enemy_counts, key=lambda x: x[0])

    FRIENDLY_TOTAL = 4
    ENEMY_TOTAL = 3

    def near_collapse_label(max_count, total, is_full_collapse, at_t):
        if is_full_collapse:
            return None  # already reported as full collapse
        if max_count == total - 1:
            return f"NEAR-COLLAPSE — {max_count}/{total} conditions aligned at t={at_t}"
        elif max_count > 0:
            return f"Contained — best alignment {max_count}/{total} conditions at t={at_t}"
        else:
            return "No alignment — conditions never overlapped"

    friendly_near = near_collapse_label(friendly_max_count, FRIENDLY_TOTAL, friendly_collapse, friendly_max_t)
    enemy_near = near_collapse_label(enemy_max_count, ENEMY_TOTAL, enemy_collapse, enemy_max_t)

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

    if friendly_near:
        print(f"Friendly Proximity:      {friendly_near}")
    if enemy_near:
        print(f"Enemy Proximity:         {enemy_near}")

    print("\n==========================================================\n")


# ============================================================
# MAIN SIMULATION
# ============================================================

def main():
    print("=== JOR 4.0 — Maneuver Health Simulation (Morale + Commander Skill + Enemy + Scenarios + Posture) ===\n")

    print("INTERPRETING THE METRICS:")
    print("  Readiness (theta_s):   Troop physical/mental readiness. High = good. Low = fatigue.")
    print("  Pressure  (theta_c):   Operational pressure and coordination load. High = stress.")
    print("  Outcomes  (theta_o):   Maneuver results (success/stall/failure). High = worse outcomes.")
    print("  Morale:                Force cohesion, resilience, and confidence (0–1).")
    print("  Enemy metrics:         Parallel readiness/pressure/outcomes/morale for adversary.")
    print("  NHP Posterior:         Probability the force is in a non-healthy state.")
    print("                         Normal < 0.40, Warning 0.40–0.50, ALERT > 0.50.")
    print("--------------------------------------------------------------------------\n")

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
        "5": "baseline"
    }
    scenario = Scenario(scenario_map.get(sel, "baseline"))

    commander = CommanderSkill(level="expert")
    skill = commander.multiplier()

    engine = JOREngine(
        prior_nh=0.05,
        steepness=16.0,
        upper_th=0.50,
        lower_th=0.40,
        retention=0.80,
        calibration_steps=25
    )

    adapter = ManeuverAdapter()
    logger = FusionLogger(f"maneuver_log_{scenario.name}.json")

    degraded_unit = None
    catastrophic_event = False
    decision_state = DecisionState()

    morale = 0.75

    enemy = EnemyForce(posture=scenario.enemy_posture)

    for t in range(120):

        if t == 30:
            degraded_unit = 1
            print("\n>>> Local Strain Detected: Unit 1 showing increased friction.\n")

        if t == 110:
            catastrophic_event = True
            print("\n>>> CATASTROPHIC EVENT: Operational overextension + coordination failure.\n")

        readiness = generate_readiness(t, degraded_unit, scenario)
        pressure = generate_pressure(t, scenario)
        outcomes = generate_outcomes(t, degraded_unit)

        enemy.step(t)
        enemy.apply_friendly_skill_effects(skill)
        enemy.update_posture()

        if catastrophic_event:
            readiness = readiness * (0.25 / skill)
            pressure = min(1.0, pressure + 0.55 / skill)
            morale -= 0.15 / skill
            for u in outcomes:
                u["stall"] += np.random.randint(12, 20)
                u["failure"] += np.random.randint(10, 18)

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
            readiness, pressure, outcomes, morale, aborted = apply_user_choice(
                choice, readiness, pressure, outcomes, morale, decision_state, skill
            )
            # State changed as a result of the choice, so refresh the metrics once.
            theta_s = adapter.compute_theta_s(readiness)
            theta_c = adapter.compute_theta_c(pressure)
            theta_o, outliers, median, worst_ema = adapter.compute_theta_o(outcomes)
            theta_o *= (1.0 + enemy.outcomes * 0.10)
            theta_o *= scenario.outcome_multiplier

            if aborted:
                sop, nhp, alert = engine.fusion_step(theta_s, theta_c, theta_o)

                mission_state = classify_mission_state(theta_s, theta_c, theta_o, degraded_unit, morale, skill, enemy)
                narrative = describe_troop_state(t, readiness, pressure, outcomes, degraded_unit, morale, skill, commander.level, enemy)

                logger.log_state(
                    sop, nhp, alert,
                    metadata={
                        "t": t,
                        "mission_state": mission_state,
                        "theta_s": theta_s,
                        "theta_c": theta_c,
                        "theta_o": theta_o,
                        "readiness": readiness.tolist(),
                        "pressure": pressure,
                        "morale": morale,
                        "outliers": outliers,
                        "median_tempo": median,
                        "worst_ema": worst_ema,
                        "degraded_unit": degraded_unit,
                        "commander_skill": commander.level,
                        "skill_multiplier": skill,
                        "enemy_readiness": enemy.readiness,
                        "enemy_pressure": enemy.pressure,
                        "enemy_outcomes": enemy.outcomes,
                        "enemy_morale": enemy.morale,
                        "enemy_posture": enemy.posture,
                        "scenario": scenario.name,
                        "narrative": narrative
                    }
                )
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

        mission_state = classify_mission_state(theta_s, theta_c, theta_o, degraded_unit, morale, skill, enemy)
        narrative = describe_troop_state(t, readiness, pressure, outcomes, degraded_unit, morale, skill, commander.level, enemy)

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
        print(textwrap.fill(f">> {narrative}", width=100))
        print()

        logger.log_state(
            sop, nhp, alert,
            metadata={
                "t": t,
                "mission_state": mission_state,
                "theta_s": theta_s,
                "theta_c": theta_c,
                "theta_o": theta_o,
                "readiness": readiness.tolist(),
                "pressure": pressure,
                "morale": morale,
                "outliers": outliers,
                "median_tempo": median,
                "worst_ema": worst_ema,
                "degraded_unit": degraded_unit,
                "commander_skill": commander.level,
                "skill_multiplier": skill,
                "enemy_readiness": enemy.readiness,
                "enemy_pressure": enemy.pressure,
                "enemy_outcomes": enemy.outcomes,
                "enemy_morale": enemy.morale,
                "enemy_posture": enemy.posture,
                "scenario": scenario.name,
                "narrative": narrative
            }
        )

        time.sleep(0.1)

    print_mission_summary(logger, commander, skill, scenario)


if __name__ == "__main__":
    main()
