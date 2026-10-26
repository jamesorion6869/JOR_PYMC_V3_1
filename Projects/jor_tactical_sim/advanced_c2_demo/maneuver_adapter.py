import numpy as np

class ManeuverAdapter:
    """
    Adapter for maneuver operations:
      - θₛ = readiness / morale / logistics
      - θ_c = operational pressure / tempo
      - θₒ = observed outcomes (success / stall / failure)

    Outcomes may be passed as:
      - list of dicts  {"success", "stall", "failure"}  (legacy)
      - ndarray shape (n, 3) with columns [success, stall, failure]
    """

    def __init__(self, iqr_floor=0.10, ema_alpha=0.3, outlier_threshold=1.5):
        self.iqr_floor = iqr_floor
        self.ema_alpha = ema_alpha
        self.outlier_threshold = outlier_threshold
        # Per-unit persistence state as a vector; grows (zero-filled) if more
        # units appear, like the defaultdict it replaces.
        self._ema = np.zeros(0, dtype=float)

    @property
    def unit_ema(self):
        """Per-unit EMA as {unit index: value} (read-only view of the vector state)."""
        return {i: float(v) for i, v in enumerate(self._ema)}

    # ------------------------------
    # θₛ — readiness / morale
    # ------------------------------
    def compute_theta_s(self, readiness_scores):
        readiness_scores = np.clip(np.asarray(readiness_scores, dtype=float), 0.0, 1.0)
        return float(np.mean(readiness_scores))

    # ------------------------------
    # θ_c — operational pressure
    # ------------------------------
    def compute_theta_c(self, pressure):
        return float(np.clip(pressure, 0.0, 1.0))

    # ------------------------------
    # θₒ — observed outcomes
    # ------------------------------
    @staticmethod
    def _outcomes_to_matrix(unit_outcomes):
        """Normalize list-of-dicts or (n, 3) matrix to an (n, 3) int array."""
        if isinstance(unit_outcomes, np.ndarray):
            arr = np.asarray(unit_outcomes, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"outcomes matrix must be (n, 3); got {arr.shape}")
            return arr
        rows = []
        for u in unit_outcomes:
            rows.append((u["success"], u["stall"], u["failure"]))
        return np.asarray(rows, dtype=float)

    def compute_theta_o(self, unit_outcomes):
        """
        unit_outcomes: list of dicts {success, stall, failure}
                       OR ndarray shape (n, 3) columns [success, stall, failure]
        """
        mat = self._outcomes_to_matrix(unit_outcomes)
        totals = np.maximum(mat.sum(axis=1), 1.0)
        # tempo degradation per unit: 0*success + 0.5*stall + 1.0*failure
        unit_scores = (0.5 * mat[:, 1] + 1.0 * mat[:, 2]) / totals

        # Fleet median + IQR floor
        median = float(np.median(unit_scores))
        q1, q3 = np.percentile(unit_scores, [25, 75])
        iqr = max(q3 - q1, self.iqr_floor)

        # EMA persistence + outlier detection
        n = unit_scores.shape[0]
        if self._ema.shape[0] < n:
            self._ema = np.concatenate([self._ema, np.zeros(n - self._ema.shape[0])])

        z = (unit_scores - median) / iqr
        # Same element-wise operations, in the same order, as the per-unit loop
        # this replaces (so results are bit-identical). Units beyond this call's
        # n (if the force ever shrinks) keep their stored value, as before.
        self._ema[:n] = (
            self.ema_alpha * np.maximum(z, 0.0) +
            (1 - self.ema_alpha) * self._ema[:n]
        )
        persistent_outliers = int(np.count_nonzero(self._ema[:n] > self.outlier_threshold))

        # Severity boost
        worst_ema = float(self._ema.max()) if self._ema.size else 0.0
        severity_boost = np.clip(
            (worst_ema - self.outlier_threshold) / max(self.outlier_threshold, 1e-6),
            0.0, 1.0
        )

        # Fraction boost
        fraction_boost = np.clip(persistent_outliers / len(unit_scores), 0.0, 1.0)

        # Combine boosts
        combined = np.clip(
            0.7 * max(severity_boost, fraction_boost) +
            0.3 * ((severity_boost + fraction_boost) / 2.0),
            0.0, 1.0
        )

        # Base θₒ = fleet median tempo degradation
        theta_o_base = float(np.clip(median, 0.0, 1.0))

        # Boosted θₒ
        theta_o = float(np.clip(theta_o_base + 0.7 * combined, 0.0, 1.0))

        return theta_o, persistent_outliers, median, worst_ema
