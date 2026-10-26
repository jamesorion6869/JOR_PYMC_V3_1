"""
Named task-force / unit designations for map + readiness matrix.

Provides a stable, scenario-aware TO&E-style labeling so the dashboard can
show "TF 1-6 INF" rather than only "U0". Roles also drive MIL-style icons.
"""

from __future__ import annotations

from typing import Dict, List

# Rotating designation templates. Index is unit id % len.
# short: matrix badge; name: full card title; role: icon key; echelon: optional
TEMPLATES = [
    {"short": "INF", "name": "TF 1-6 INF", "role": "infantry", "echelon": "BN"},
    {"short": "AR", "name": "TF 3-8 AR", "role": "armor", "echelon": "BN"},
    {"short": "INF", "name": "TF 2-4 INF", "role": "infantry", "echelon": "BN"},
    {"short": "RCN", "name": "TRP A RCN", "role": "recon", "echelon": "TRP"},
    {"short": "LOG", "name": "SPT CO LOG", "role": "logistics", "echelon": "CO"},
    {"short": "AR", "name": "TF 1-72 AR", "role": "armor", "echelon": "BN"},
    {"short": "INF", "name": "TF 1-27 INF", "role": "infantry", "echelon": "BN"},
    {"short": "HQ", "name": "BDE HQ", "role": "hq", "echelon": "BDE"},
    {"short": "SPT", "name": "SPT BDE", "role": "logistics", "echelon": "BDE"},
]


def unit_identity(unit_id: int, scenario: str = "baseline") -> Dict[str, str]:
    """Return display identity for a unit index."""
    base = dict(TEMPLATES[unit_id % len(TEMPLATES)])
    # Scenario tag keeps names unique when multiple of same template appear
    if unit_id >= len(TEMPLATES):
        base["name"] = f"{base['name']}-{unit_id}"
    base["id"] = unit_id
    base["scenario"] = (scenario or "baseline").lower()
    return base


def force_roster(n_units: int, scenario: str = "baseline") -> List[Dict[str, str]]:
    return [unit_identity(i, scenario) for i in range(n_units)]
