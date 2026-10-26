"""
Severity gauge: a graded 0-1 measure of how far the force has moved from healthy.

Why this exists next to NHP
---------------------------
NHP is a posterior *probability* that the force is non-healthy. A probability
answers "how sure are we something is wrong", so once the evidence is strong it
sits near its ceiling and cannot say HOW wrong. Severity answers the second
question. It is the excess of the fused risk score (SOP) over the force's own
calibrated baseline, scaled so that 0 = at baseline and 1 = a fully broken
force, then lightly smoothed. It is not recursive (it does not accumulate), so
it falls as soon as the force recovers and keeps separating a well-handled
force from a poorly handled one through the catastrophe.

It does not touch the engine or NHP, so every existing result is unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

# SOP at which the force is treated as fully broken (readiness ~0.1, pressure ~1,
# outcomes ~0.9 gives ~0.93). Severity 1.0 is reached at or above this.
SOP_BREAKING = 0.90
SMOOTHING = 0.60          # weight on the newest reading (responsive, not jittery)

# (upper bound, band name); the last band is open-ended
SEVERITY_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.20, "NOMINAL"),
    (0.40, "ELEVATED"),
    (0.60, "STRAINED"),
    (0.80, "CRITICAL"),
    (1.01, "BREAKING"),
)


def severity_band(value: float) -> str:
    for upper, name in SEVERITY_BANDS:
        if value < upper:
            return name
    return SEVERITY_BANDS[-1][1]


def raw_severity(sop: float, baseline_sop: Optional[float], breaking: float = SOP_BREAKING) -> float:
    """Unsmoothed severity. 0 while the engine is still calibrating (no baseline yet)."""
    if baseline_sop is None:
        return 0.0
    span = max(breaking - baseline_sop, 1e-6)
    return min(1.0, max(0.0, (sop - baseline_sop) / span))


class SeverityGauge:
    """Stateful gauge: call update() once per tick; peek() previews without advancing."""

    def __init__(self, breaking: float = SOP_BREAKING, smoothing: float = SMOOTHING):
        self.breaking = breaking
        self.smoothing = smoothing
        self.value = 0.0

    def peek(self, sop: float, baseline_sop: Optional[float]) -> float:
        if baseline_sop is None:
            return 0.0
        raw = raw_severity(sop, baseline_sop, self.breaking)
        return self.smoothing * raw + (1.0 - self.smoothing) * self.value

    def update(self, sop: float, baseline_sop: Optional[float]) -> float:
        self.value = self.peek(sop, baseline_sop)
        return self.value
