"""
Per-tick map telemetry frames for geospatial display.

Decouples the simulation loop from any particular renderer: the sim emits
frames; map_view / dashboard consume them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from theater_layout import unit_positions
from force_structure import unit_identity

# Scenario-specific theater origins (lon/lat). Theater x/y map into a local box.
# These are synthetic AOs for demo geography — not operational coordinates.
SCENARIO_GEO = {
    "baseline": {"lon": 35.40, "lat": 33.85, "scale": 0.018},          # coastal Levant
    "urban": {"lon": 44.36, "lat": 33.31, "scale": 0.012},              # dense urban (Baghdad-ish)
    "deep_maneuver": {"lon": 69.15, "lat": 34.55, "scale": 0.022},      # open approaches
    "logistics_constrained": {"lon": 38.00, "lat": 36.20, "scale": 0.016},
    "high_enemy_pressure": {"lon": 47.95, "lat": 29.35, "scale": 0.020},
}
_DEFAULT_GEO = SCENARIO_GEO["baseline"]


def theater_to_lonlat(x: float, y: float, scenario: str = "baseline"):
    """Map abstract theater (x,y) to WGS84 lon/lat for the scenario AO."""
    geo = SCENARIO_GEO.get((scenario or "baseline").lower(), _DEFAULT_GEO)
    lon = geo["lon"] + float(x) * geo["scale"]
    lat = geo["lat"] + float(y) * geo["scale"]
    return lon, lat


def build_frame(
    t: int,
    positions: np.ndarray,
    readiness: np.ndarray,
    pressure: float,
    morale: float,
    nhp: float,
    alert: bool,
    mission_state: str,
    theta_s: float,
    theta_c: float,
    theta_o: float,
    degraded_unit,          # None, an int, or a sequence of ints (strained units)
    scenario: str,
    n_units: int,
    outcomes: Optional[np.ndarray] = None,
    enemy: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
    decision_timeout_s: Optional[float] = None,
    severity: Optional[float] = None,
    severity_band: Optional[str] = None,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    readiness = np.asarray(readiness, dtype=float).ravel()
    positions = np.asarray(positions, dtype=float)
    degraded_set = (
        set() if degraded_unit is None
        else {int(i) for i in np.atleast_1d(np.asarray(degraded_unit, dtype=int))}
    )
    units = []
    for i in range(n_units):
        ident = unit_identity(i, scenario)
        x, y = float(positions[i, 0]), float(positions[i, 1])
        lon, lat = theater_to_lonlat(x, y, scenario)
        u: Dict[str, Any] = {
            "id": i,
            "name": ident["name"],
            "short": ident["short"],
            "role": ident["role"],
            "echelon": ident["echelon"],
            "x": x,
            "y": y,
            "lon": lon,
            "lat": lat,
            "readiness": float(readiness[i]) if i < len(readiness) else float(np.mean(readiness)),
            "degraded": i in degraded_set,
        }
        if outcomes is not None:
            row = np.asarray(outcomes[i], dtype=float)
            u["success"] = int(row[0])
            u["stall"] = int(row[1])
            u["failure"] = int(row[2])
        units.append(u)

    frame: Dict[str, Any] = {
        "t": int(t),
        "scenario": scenario,
        "n_units": int(n_units),
        "mission_state": mission_state,
        "nhp": float(nhp),
        "alert": bool(alert),
        "theta_s": float(theta_s),
        "theta_c": float(theta_c),
        "theta_o": float(theta_o),
        "pressure": float(pressure),
        "morale": float(morale),
        "degraded_unit": degraded_unit,
        "units": units,
        "decision_timeout_s": decision_timeout_s,
        "decision": decision,
    }
    if enemy is not None:
        frame["enemy"] = enemy
    if severity is not None:
        frame["severity"] = float(severity)
        frame["severity_band"] = severity_band
    if events:
        frame["events"] = events
    return frame


class MapTelemetrySink:
    """Collects frames in memory; optional JSON-lines file on flush()."""

    def __init__(self, path: Optional[str] = None, on_emit=None):
        self.path = Path(path) if path else None
        self.frames: List[Dict[str, Any]] = []
        self.on_emit = on_emit      # optional callable(frame), e.g. a live viewer
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, frame: Dict[str, Any]) -> None:
        self.frames.append(frame)
        if self.on_emit is not None:
            try:
                self.on_emit(frame)
            except Exception:
                pass            # a viewer problem must never break the simulation

    def flush(self) -> None:
        if self.path is None:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            for frame in self.frames:
                f.write(json.dumps(frame) + "\n")

    def last(self) -> Optional[Dict[str, Any]]:
        return self.frames[-1] if self.frames else None


def init_positions(n_units: int, scenario: str, seed: Optional[int] = None) -> np.ndarray:
    return unit_positions(n_units, scenario=scenario, seed=0 if seed is None else int(seed))


def append_end_record(path, summary=None, debrief=None) -> None:
    """Append the end-of-run record (summary + debrief) to a saved frames file.

    The dashboard reads it when it plays the file back, so saved playback can show
    the same debrief a live run does. Older files without it still play fine.
    """
    rec = {"type": "end", "t": -1, "summary": summary, "debrief": debrief}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
