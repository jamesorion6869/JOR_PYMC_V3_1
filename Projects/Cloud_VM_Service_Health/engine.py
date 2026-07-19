import math


class JOREngine:
    """
    JOR-derived Cloud VM Service Health Fusion Engine.

    Recursive evidence-fusion engine for estimating cloud service
    health state from system, operational, and observed service metrics.

    Inputs:
       theta_s : system-state evidence
       theta_c : operational-context evidence
       theta_o : observed service-health evidence

    Outputs:
       SOP   : fused evidence state
       NHP   : recursive posterior non-healthy state estimate
       Alert : hysteresis-controlled alert state

    Features:
       - self-calibrating baseline estimation
       - recursive Bayesian-style state updating
       - retention-based stabilization
       - hysteresis alert thresholds

    Research prototype demonstrating JOR architecture adaptation
    to cloud telemetry monitoring.

    """

    def __init__(self, prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                 retention=0.70, calibration_steps=20, baseline_sop=None):
        self.prior_nh = prior_nh
        self.p_final = prior_nh
        self.steepness = steepness
        self.upper_th = upper_th
        self.lower_th = lower_th
        self.retention = retention
        self.alert_status = False

        # --- Self-calibrating baseline ---
        # If baseline_sop is explicitly provided, use it (e.g. if you already
        # know the healthy operating point from prior data). Otherwise, the
        # engine learns it from the first `calibration_steps` observations.
        self.calibration_steps = calibration_steps
        self._calibration_buffer = []
        self.baseline_sop = baseline_sop
        self.is_calibrated = baseline_sop is not None
        self.calibrating = not self.is_calibrated

    def fusion_step(self, theta_s, theta_c, theta_o):
        sop = (0.40 * theta_s) + (0.30 * theta_c) + (0.30 * theta_o)

        # --- Calibration phase: observe, don't alert yet ---
        if not self.is_calibrated:
            self._calibration_buffer.append(sop)
            if len(self._calibration_buffer) >= self.calibration_steps:
                self.baseline_sop = sum(self._calibration_buffer) / len(self._calibration_buffer)
                self.is_calibrated = True
                self.calibrating = False
            # During calibration, p_final stays at the prior and no alert fires --
            # we don't yet know what "normal" looks like for this machine.
            return sop, self.p_final, self.alert_status

        # --- Normal operation, post-calibration ---
        delta = sop - self.baseline_sop
        likelihood_nh = 1.0 / (1.0 + math.exp(-self.steepness * delta))
        likelihood_not_nh = 1.0 - likelihood_nh

        numerator = likelihood_nh * self.p_final
        denominator = numerator + (likelihood_not_nh * (1.0 - self.p_final))
        posterior = numerator / (denominator + 1e-9)

        # Retention blend stabilizes recursive state evolution.
        self.p_final = (self.retention * posterior) + ((1 - self.retention) * self.prior_nh)

        if self.p_final > self.upper_th:
            self.alert_status = True
        elif self.p_final < self.lower_th:
            self.alert_status = False

        return sop, self.p_final, self.alert_status
