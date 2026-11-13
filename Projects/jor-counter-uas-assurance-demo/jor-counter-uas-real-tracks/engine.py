import math
from statistics import median


class JOREngine:
    """
    JOR-derived Counter-UAS (C-UAS) sensing-assurance engine.

    A meta-layer: it does not detect or defeat drones and makes no
    statement about threat level. It estimates how reliable the
    detection-and-tracking capability currently is, from:

        theta_s : sensor & platform integrity evidence (incl. feed liveness)
        theta_c : load on the sensing system (saturation / clutter)
        theta_o : observed detection & track-quality evidence

    Outputs:
        SOP   : fused evidence state (0-1)
        NHP   : Non-Healthy Posterior - a recursive, posterior-style score
                for "detect/track capability is degraded". It follows
                Bayesian update arithmetic but is NOT calibrated against
                ground truth, so read it as an index, not a literal
                probability.
        Alert : hysteresis-controlled "UNRELIABLE" state
        Band  : NOMINAL / DEGRADED / UNRELIABLE, with hysteresis on both
                the DEGRADED and UNRELIABLE boundaries
        status(): structured, JSON-safe report for an operator / C2

    Robustness features:
        - delta_deadband : SOP excursions smaller than this from the
          calibrated baseline are treated as normal operating variation
        - non-finite inputs are rejected: state is held, `data_fault`
          is set, and after `fault_alert_steps` consecutive faulted
          steps the alert fails safe to ON
        - calibration uses the median (not the mean) and records a
          robust spread; `calibration_suspect` flags a contaminated window
        - nhp_floor / nhp_ceiling expose the reachable NHP range, which is
          bounded by the retention blend
    """

    def __init__(self, prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                 retention=0.70, calibration_steps=20, baseline_sop=None,
                 delta_deadband=0.0, fault_alert_steps=3, degraded_th=0.20,
                 degraded_exit_th=0.15):
        self.prior_nh = prior_nh
        self.p_final = prior_nh
        self.steepness = steepness
        self.upper_th = upper_th
        self.lower_th = lower_th
        self.retention = retention
        self.alert_status = False
        self.delta_deadband = delta_deadband
        self.fault_alert_steps = fault_alert_steps
        self.degraded_th = degraded_th
        self.degraded_exit_th = degraded_exit_th
        self.degraded_status = False

        # Self-calibrating baseline of the fused SOP under healthy conditions
        self.calibration_steps = calibration_steps
        self._calibration_buffer = []
        self.baseline_sop = baseline_sop
        self.baseline_spread = None
        self.baseline_channels = None          # per-channel calibrated medians
        self._calibration_channels = []
        self.last_thetas = None
        self.calibration_suspect = False
        self.is_calibrated = baseline_sop is not None
        self.calibrating = not self.is_calibrated

        # Data-fault tracking
        self.data_fault = False
        self.consecutive_faults = 0
        self.last_sop = baseline_sop if baseline_sop is not None else 0.0

    # Reachable NHP range: p_final = retention*posterior + (1-retention)*prior
    @property
    def nhp_floor(self):
        return (1.0 - self.retention) * self.prior_nh

    @property
    def nhp_ceiling(self):
        return self.retention + self.nhp_floor

    # Channel keys appear in status() and in viz.py; weights are the single
    # source of truth for the SOP fusion below.
    CHANNEL_WEIGHTS = (("sensor_integrity", 0.35),
                       ("sensing_load", 0.25),
                       ("track_quality", 0.40))

    @staticmethod
    def _finite(v):
        try:
            return math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    def fusion_step(self, theta_s, theta_c, theta_o):
        """
        Single recursive fusion step.

        SOP is the weighted sum of the three channels (see CHANNEL_WEIGHTS):
          - 0.35 sensor / platform integrity
          - 0.25 load on the sensing system
          - 0.40 observed track / detection quality
        Weights can be tuned for C-UAS doctrine.
        """
        # Reject non-finite inputs: hold state rather than poison p_final
        if not all(self._finite(v) for v in (theta_s, theta_c, theta_o)):
            self.data_fault = True
            self.consecutive_faults += 1
            if self.is_calibrated and self.consecutive_faults >= self.fault_alert_steps:
                self.alert_status = True      # fail safe: no data = no assurance
            return self.last_sop, self.p_final, self.alert_status
        self.data_fault = False
        self.consecutive_faults = 0

        sop = sum(w * th for (_, w), th in zip(self.CHANNEL_WEIGHTS, (theta_s, theta_c, theta_o)))
        self.last_sop = sop
        self.last_thetas = (float(theta_s), float(theta_c), float(theta_o))

        # Calibration phase: observe only, do not alert
        if not self.is_calibrated:
            self._calibration_buffer.append(sop)
            self._calibration_channels.append(self.last_thetas)
            if len(self._calibration_buffer) >= self.calibration_steps:
                buf = self._calibration_buffer
                med = median(buf)
                mad = median(abs(x - med) for x in buf)
                self.baseline_sop = med
                self.baseline_spread = 1.4826 * mad
                cols = list(zip(*self._calibration_channels))
                self.baseline_channels = {name: median(col) for (name, _), col
                                          in zip(self.CHANNEL_WEIGHTS, cols)}
                limit = med + 4.0 * max(self.baseline_spread, 0.02)
                self.calibration_suspect = max(buf) > limit
                self.is_calibrated = True
                self.calibrating = False
            return sop, self.p_final, self.alert_status

        # Normal operation
        delta = sop - self.baseline_sop
        if self.delta_deadband > 0.0:
            delta = math.copysign(max(abs(delta) - self.delta_deadband, 0.0), delta)
        likelihood_nh = 1.0 / (1.0 + math.exp(-self.steepness * delta))
        likelihood_not_nh = 1.0 - likelihood_nh

        numerator = likelihood_nh * self.p_final
        denominator = numerator + (likelihood_not_nh * (1.0 - self.p_final))
        posterior = numerator / (denominator + 1e-9)

        # Retention blend stabilises the recursive state
        self.p_final = (self.retention * posterior) + ((1 - self.retention) * self.prior_nh)

        if self.p_final > self.upper_th:
            self.alert_status = True
        elif self.p_final < self.lower_th:
            self.alert_status = False

        if self.p_final >= self.degraded_th:
            self.degraded_status = True
        elif self.p_final < self.degraded_exit_th:
            self.degraded_status = False

        return sop, self.p_final, self.alert_status

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def channel_excess(self):
        """Weighted SOP contribution of each channel above its calibrated baseline."""
        if self.baseline_channels is None or self.last_thetas is None:
            return None
        if any(name not in self.baseline_channels for name, _ in self.CHANNEL_WEIGHTS):
            return None                      # state saved by an older version
        return {name: w * (th - self.baseline_channels[name])
                for (name, w), th in zip(self.CHANNEL_WEIGHTS, self.last_thetas)}

    def dominant_channel(self, min_excess=0.01):
        ex = self.channel_excess()
        if not ex:
            return None
        name, val = max(ex.items(), key=lambda kv: kv[1])
        total_up = sum(v for v in ex.values() if v > 0)
        # Only name a driver once the excursion is outside normal variation
        return name if (val > min_excess and total_up > self.delta_deadband) else None

    def state_band(self):
        if not self.is_calibrated:
            return "CALIBRATING"
        if self.alert_status:
            return "UNRELIABLE"
        if self.data_fault or self.degraded_status:
            return "DEGRADED"
        return "NOMINAL"

    def status(self, ts=None, details=None):
        """JSON-safe status report for an operator display or higher-level C2."""
        ex = self.channel_excess()
        return {
            "ts": ts,
            "state": self.state_band(),
            "nhp": round(float(self.p_final), 4),
            "nhp_range": [round(self.nhp_floor, 4), round(self.nhp_ceiling, 4)],
            "sop": round(float(self.last_sop), 4),
            "baseline_sop": None if self.baseline_sop is None else round(float(self.baseline_sop), 4),
            "alert": bool(self.alert_status),
            "data_fault": bool(self.data_fault),
            "consecutive_faults": int(self.consecutive_faults),
            # A driver is only meaningful while something is actually degraded
            "dominant_channel": (self.dominant_channel()
                                 if self.state_band() in ("DEGRADED", "UNRELIABLE") else None),
            "channel_excess": None if ex is None else {k: round(float(v), 4) for k, v in ex.items()},
            "details": details or {},
        }

    @staticmethod
    def status_is_stale(status, now, max_age=5.0):
        """True if a status report is missing a timestamp or older than max_age.
        Lets C2 treat a silent monitor as a fault in its own right."""
        ts = None if status is None else status.get("ts")
        return ts is None or (now - ts) > max_age
