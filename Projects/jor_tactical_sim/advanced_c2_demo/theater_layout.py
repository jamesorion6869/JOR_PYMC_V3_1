"""
Synthetic theater layouts: assign 2D positions to units for map display.

Coordinates are in abstract theater units (not lat/lon). Origin is southwest;
x increases east, y increases north. Scenario layouts bias spacing and
clustering so different missions look distinct on the map.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

# (x, y) in theater units
Position = Tuple[float, float]


def unit_positions(n_units: int, scenario: str = "baseline", seed: int = 0) -> np.ndarray:
    """
    Return shape (n_units, 2) float array of unit positions.

    Layouts are deterministic given (n_units, scenario, seed) so the same
    mission always draws the same map geometry.
    """
    scenario = (scenario or "baseline").lower()
    rng = np.random.RandomState(seed + hash(scenario) % 10007)

    if scenario == "urban":
        # Tight cluster with street-grid jitter
        cols = max(1, int(math.ceil(math.sqrt(n_units))))
        rows = int(math.ceil(n_units / cols))
        pts = []
        for i in range(n_units):
            r, c = divmod(i, cols)
            x = 4.0 + c * 1.2 + rng.uniform(-0.25, 0.25)
            y = 4.0 + r * 1.2 + rng.uniform(-0.25, 0.25)
            pts.append((x, y))
        return np.asarray(pts, dtype=float)

    if scenario == "deep_maneuver":
        # Long axis of advance (west → east column)
        pts = []
        for i in range(n_units):
            x = 2.0 + i * (14.0 / max(n_units - 1, 1))
            y = 8.0 + rng.uniform(-1.5, 1.5)
            pts.append((x, y))
        return np.asarray(pts, dtype=float)

    if scenario == "logistics_constrained":
        # Units near a supply node with a stretched tail
        pts = []
        for i in range(n_units):
            if i == 0:
                pts.append((3.0, 3.0))  # logistics hub
            else:
                angle = rng.uniform(0, 2 * math.pi)
                rad = 1.5 + 0.4 * i
                pts.append((3.0 + rad * math.cos(angle), 3.0 + rad * math.sin(angle)))
        return np.asarray(pts, dtype=float)

    if scenario == "high_enemy_pressure":
        # Friendly line facing an implied threat from the north
        pts = []
        for i in range(n_units):
            x = 3.0 + i * (12.0 / max(n_units - 1, 1))
            y = 5.0 + rng.uniform(-0.4, 0.4)
            pts.append((x, y))
        return np.asarray(pts, dtype=float)

    # baseline: open arc / dispersed company
    pts = []
    for i in range(n_units):
        angle = -0.6 + 1.2 * (i / max(n_units - 1, 1))
        rad = 6.0 + rng.uniform(-0.5, 0.5)
        pts.append((8.0 + rad * math.cos(angle), 6.0 + rad * math.sin(angle)))
    return np.asarray(pts, dtype=float)


def theater_bounds(positions: np.ndarray, pad: float = 2.0):
    """Return (xmin, xmax, ymin, ymax) with padding for map framing."""
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return 0.0, 10.0, 0.0, 10.0
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    return float(xmin - pad), float(xmax + pad), float(ymin - pad), float(ymax + pad)
