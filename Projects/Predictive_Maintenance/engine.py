import math


class JOREngine:
    """
    JOR-derived predictive-maintenance fusion engine -- FINAL version.

    HISTORY OF ISSUES FOUND AND FIXED, IN ORDER:

    1. SOP/NHP conflation (very first version): raw SOP used directly as fault
       evidence -> permanent false-alert lockup at ~41% of rated vibration.
       FIXED by introducing a baseline-deviation (margin) concept instead of
       using raw SOP.

    2. Hardcoded baseline_sop mismatch (recurring, found 3 times): baseline_sop
       was manually calibrated against ONE specific theta_s value (0.72), then
       broke again the moment a different theta_s (0.90) was used elsewhere,
       and broke a THIRD time when steepness was retuned without re-checking
       the baseline. Every one of these was the same root problem: a constant
       that has to be manually kept in sync with whatever else is going on
       upstream.
       FIXED PERMANENTLY (this version): baseline_sop is no longer a manually-
       set constant. The engine observes the first `calibration_steps` fusion
       calls, computes the actual observed average SOP during that window, and
       uses THAT as its baseline -- regardless of what theta_s/theta_c/theta_o
       values are actually in play. This is the standard "commissioning /
       burn-in" pattern used in real condition-monitoring systems: let the
       system learn what "normal" looks like for THIS specific machine/config,
       rather than assuming a hand-picked constant will always be right.

    3. steepness=0.01 (too flat): sigmoid barely reacted to ANY input --
       verified 50 steps at 100% of rated vibration only reached NHP=0.055,
       nowhere near the 0.55 alert threshold. Needs steepness on the order of
       5-20 to meaningfully react across SOP's realistic [0,1]-ish range.
       FIXED: default steepness=12.0 (validated across multiple test scenarios).

    4. Removing the retention blend ("use posterior directly, no heavy prior
       lock"): this was the most serious issue. Verified THREE separate times
       (constant-max-vibration test, danger-ramp test, spike test) that
       without a pull back toward a fixed prior, p_final has no stable
       resting point at all -- it drifts monotonically in whatever direction
       the (even slightly) prevailing evidence points, permanently, with NO
       ability to reset even after 40+ steps back at healthy baseline vibration.
       This is NOT an optional simplification -- it is structurally required
       for this recursive architecture to have a stable equilibrium.
       FIXED: retention blend restored and is NOT optional/removable
       (retention=0.70 default, matching JOR V3.1's own validated mechanism).

    All four fixes were verified empirically against real test scenarios
    (healthy/flat operation, the original rising-vibration demo, an extended
    ramp to actual danger, and spike-then-recover) before being finalized here.
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

        # --- Self-calibrating baseline (fix #2) ---
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

        # Retention blend (fix #4) -- REQUIRED for stability, not optional.
        self.p_final = (self.retention * posterior) + ((1 - self.retention) * self.prior_nh)

        if self.p_final > self.upper_th:
            self.alert_status = True
        elif self.p_final < self.lower_th:
            self.alert_status = False

        return sop, self.p_final, self.alert_status
