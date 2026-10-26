"""
Decision prompt catalog for telemetry + dashboard Decision Support panel.

Mirrors the menus in apply_user_choice so the right-rail UI can show the same
text the commander saw (or would see) at that tick.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DECISION_MENUS: Dict[str, Dict[str, Any]] = {
    "FATIGUE_EARLY": {
        "title": "Early signs of fatigue. Choose:",
        "options": [
            {"id": "1", "label": "Slightly reduce tempo"},
            {"id": "2", "label": "Redistribute minor load"},
            {"id": "3", "label": "Ignore (accept strain)"},
        ],
        "category": "FATIGUE",
    },
    "FATIGUE": {
        "title": "Force fatigue rising. Choose:",
        "options": [
            {"id": "1", "label": "Reduce tempo significantly"},
            {"id": "2", "label": "Aggressively redistribute load"},
            {"id": "3", "label": "Push through hard"},
        ],
        "category": "FATIGUE",
    },
    "PRESSURE_EARLY": {
        "title": "Coordination strain emerging. Choose:",
        "options": [
            {"id": "1", "label": "Reinforce comms lightly"},
            {"id": "2", "label": "Slow maneuver slightly"},
            {"id": "3", "label": "Accept higher tempo"},
        ],
        "category": "PRESSURE",
    },
    "PRESSURE": {
        "title": "Coordination load heavy. Choose:",
        "options": [
            {"id": "1", "label": "Strongly reinforce comms"},
            {"id": "2", "label": "Slow maneuver significantly"},
            {"id": "3", "label": "Accept risk and push"},
        ],
        "category": "PRESSURE",
    },
    "OUTCOMES_EARLY": {
        "title": "Friction emerging. Choose:",
        "options": [
            {"id": "1", "label": "Rotate units lightly"},
            {"id": "2", "label": "Redistribute tasks"},
            {"id": "3", "label": "Ignore (monitor only)"},
        ],
        "category": "OUTCOMES",
    },
    "OUTCOMES": {
        "title": "Unit friction detected. Choose:",
        "options": [
            {"id": "1", "label": "Aggressively rotate units"},
            {"id": "2", "label": "Strongly redistribute tasks"},
            {"id": "3", "label": "Ignore (accept failures)"},
        ],
        "category": "OUTCOMES",
    },
    "CATASTROPHIC": {
        "title": "Collapse imminent. Choose:",
        "options": [
            {"id": "1", "label": "Attempt strong emergency stabilization"},
            {"id": "2", "label": "Abort maneuver"},
            {"id": "3", "label": "Hold course (accept collapse)"},
        ],
        "category": "CATASTROPHIC",
    },
}


def describe_decision(choice: Optional[str], selected: Optional[str] = None,
                      timeout_s: Optional[float] = None,
                      timed_out: bool = False) -> Optional[Dict[str, Any]]:
    if not choice:
        return None
    menu = DECISION_MENUS.get(choice)
    if not menu:
        return {"type": choice, "title": choice, "options": [], "selected": selected,
                "timeout_s": timeout_s, "timed_out": bool(timed_out)}
    return {
        "type": choice,
        "category": menu["category"],
        "title": menu["title"],
        "options": list(menu["options"]),
        "selected": selected,
        "timeout_s": timeout_s,
        "timed_out": bool(timed_out),
    }
