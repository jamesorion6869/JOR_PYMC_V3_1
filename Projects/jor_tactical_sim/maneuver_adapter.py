import numpy as np
from collections import defaultdict

class ManeuverAdapter:
    """
    Adapter for maneuver operations:
      - θₛ = readiness / morale / logistics
      - θ_c = operational pressure / tempo
      - θₒ = observed outcomes (success / stall / failure)
    """

    def __init__(self, iqr_floor=0.10, ema_alpha=0.3, outlier_threshold=1.5):
        self.iqr_floor = iqr_floor
        self.ema_alpha = ema_alpha
        self.outlier_threshold = outlier_threshold
        self.unit_ema = defaultdict(float)

    # ------------------------------
    # θₛ — readiness / morale
    # ------------------------------
    def compute_theta_s(self, readiness_scores):
        readiness_scores = np.clip(readiness_scores, 0.0, 1.0)
        return float(np.mean(readiness_scores))

    # ------------------------------
    # θ_c — operational pressure
    # ------------------------------
    def compute_theta_c(self, pressure):
        return float(np.clip(pressure, 0.0, 1.0))

    # ------------------------------
    # θₒ — observed outcomes
    # ------------------------------
    def compute_theta_o(self, unit_outcomes):
        """
        unit_outcomes: list of dicts:
            { "success": int, "stall": int, "failure": int }
        """

        # Convert outcomes → tempo degradation score per unit
        unit_scores = []
        for u in unit_outcomes:
            total = max(u["success"] + u["stall"] + u["failure"], 1)
            tempo = (
                0.0 * (u["success"] / total) +
                0.5 * (u["stall"] / total) +
                1.0 * (u["failure"] / total)
            )
            unit_scores.append(tempo)

        unit_scores = np.array(unit_scores)

        # Fleet median + IQR floor
        median = float(np.median(unit_scores))
        q1, q3 = np.percentile(unit_scores, [25, 75])
        iqr = max(q3 - q1, self.iqr_floor)

        # EMA persistence + outlier detection
        persistent_outliers = 0
        for idx, score in enumerate(unit_scores):
            z = (score - median) / iqr
            self.unit_ema[idx] = (
                self.ema_alpha * max(z, 0) +
                (1 - self.ema_alpha) * self.unit_ema[idx]
            )
            if self.unit_ema[idx] > self.outlier_threshold:
                persistent_outliers += 1

        # Severity boost
        worst_ema = max(self.unit_ema.values(), default=0.0)
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
