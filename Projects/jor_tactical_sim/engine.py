import math

class JOREngine:
    """
    Maneuver‑tuned JOR Engine (Risk‑coded SOP + Catastrophic Override)

    - readiness collapse → high risk
    - pressure spike     → high risk
    - failures surge     → high risk

    Adds catastrophic override:
      If SOP is far above baseline (delta > 0.40),
      retention increases so NHP can saturate near 1.0.
    """

    def __init__(self, prior_nh=0.05, steepness=16.0, upper_th=0.50, lower_th=0.40,
                 retention=0.80, calibration_steps=25, baseline_sop=None):

        # Recursive posterior state
        self.prior_nh = prior_nh
        self.p_final = prior_nh

        # Maneuver sensitivity tuning
        self.steepness = steepness
        self.upper_th = upper_th
        self.lower_th = lower_th
        self.retention = retention

        # Alert state
        self.alert_status = False

        # Baseline calibration
        self.calibration_steps = calibration_steps
        self._calibration_buffer = []
        self.baseline_sop = baseline_sop
        self.is_calibrated = baseline_sop is not None
        self.calibrating = not self.is_calibrated


    def fusion_step(self, theta_s, theta_c, theta_o):
        """
        Battlefield risk fusion:

        theta_s : readiness (high = good)
        theta_c : pressure  (high = bad)
        theta_o : outcomes  (high = bad)

        Convert readiness into risk:
            risk_s = 1 - theta_s
        """

        # Convert evidence → risk
        risk_s = 1.0 - theta_s     # readiness collapse → high risk
        risk_c = theta_c           # pressure spike → high risk
        risk_o = theta_o           # failures → high risk

        # Weighted battlefield risk score
        sop = (0.40 * risk_s) + (0.30 * risk_c) + (0.30 * risk_o)

        # --- Calibration phase ---
        if not self.is_calibrated:
            self._calibration_buffer.append(sop)

            if len(self._calibration_buffer) >= self.calibration_steps:
                self.baseline_sop = sum(self._calibration_buffer) / len(self._calibration_buffer)
                self.is_calibrated = True
                self.calibrating = False

            return sop, self.p_final, self.alert_status

        # --- Normal operation ---
        delta = sop - self.baseline_sop

        # Logistic likelihood: higher SOP → higher non‑healthy likelihood
        likelihood_nh = 1.0 / (1.0 + math.exp(-self.steepness * delta))
        likelihood_not_nh = 1.0 - likelihood_nh

        numerator = likelihood_nh * self.p_final
        denominator = numerator + (likelihood_not_nh * (1.0 - self.p_final))
        posterior = numerator / (denominator + 1e-9)

        # --- Catastrophic override retention ---
        effective_retention = self.retention

        # If SOP is far above baseline, force NHP to rise sharply
        if delta > 0.40:     # catastrophic deviation threshold
            effective_retention = 0.95

        # Retention‑blended posterior
        self.p_final = (effective_retention * posterior) + ((1 - effective_retention) * self.prior_nh)

        # Hysteresis alert logic
        if self.p_final > self.upper_th:
            self.alert_status = True
        elif self.p_final < self.lower_th:
            self.alert_status = False

        return sop, self.p_final, self.alert_status
