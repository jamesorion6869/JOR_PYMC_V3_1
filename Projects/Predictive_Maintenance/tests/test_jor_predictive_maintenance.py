"""
test_jor_predictive_maintenance.py

Automated regression tests for the predictive-maintenance JOR pipeline
(adapters.py, engine.py, logger.py). Run with:

    pytest test_jor_predictive_maintenance.py -v

These exist to catch regressions if anyone edits zone boundaries, engine
constants, or logger behavior later -- everything here was previously only
verified with one-off manual scripts.
"""
import json
import math
import numpy as np
import pytest

from adapters import VibrationAdapter
from engine import JOREngine
from logger import FusionLogger


# ============================================================
# VibrationAdapter -- Criterion I (zone boundaries)
# ============================================================

class TestCriterionIZoneBoundaries:
    def test_zone_label_at_exact_boundaries(self):
        a = VibrationAdapter()
        # <= semantics: a reading exactly AT a boundary belongs to the
        # lower (safer) zone, not the upper one.
        assert a._zone_label(a.zone_ab) == "A"
        assert a._zone_label(a.zone_ab + 0.001) == "B"
        assert a._zone_label(a.zone_bc) == "B"
        assert a._zone_label(a.zone_bc + 0.001) == "C"
        assert a._zone_label(a.zone_cd) == "C"
        assert a._zone_label(a.zone_cd + 0.001) == "D"

    def test_zone_label_deep_in_each_zone(self):
        a = VibrationAdapter()
        assert a._zone_label(0.0) == "A"
        assert a._zone_label(2.0) == "B"
        assert a._zone_label(5.0) == "C"
        assert a._zone_label(50.0) == "D"

    def test_criterion_i_theta_matches_zone_transition_points(self):
        a = VibrationAdapter()
        # Boundary values should land exactly at the documented transition points.
        assert a._criterion_i_theta(0.0) == pytest.approx(0.0)
        assert a._criterion_i_theta(a.zone_ab) == pytest.approx(0.25)
        assert a._criterion_i_theta(a.zone_bc) == pytest.approx(0.50)
        assert a._criterion_i_theta(a.zone_cd) == pytest.approx(0.85)

    def test_criterion_i_theta_saturates_at_1_for_extreme_velocity(self):
        a = VibrationAdapter()
        assert a._criterion_i_theta(a.zone_cd * 2) == pytest.approx(1.0)
        assert a._criterion_i_theta(a.zone_cd * 100) == pytest.approx(1.0)

    def test_criterion_i_theta_is_monotonically_non_decreasing(self):
        a = VibrationAdapter()
        velocities = np.linspace(0, 30, 500)
        thetas = [a._criterion_i_theta(v) for v in velocities]
        diffs = np.diff(thetas)
        assert np.all(diffs >= -1e-9), "theta_o must never decrease as velocity increases"

    def test_criterion_i_theta_always_in_unit_interval(self):
        a = VibrationAdapter()
        for v in np.linspace(0, 1000, 200):
            theta = a._criterion_i_theta(v)
            assert 0.0 <= theta <= 1.0


# ============================================================
# VibrationAdapter -- Criterion II (rate of change from baseline)
# ============================================================

class TestCriterionII:
    def _make_buf(self, peak_accel_g, freq=180, n=10000, fs=10000, noise=0.0):
        t = np.linspace(0, 1, n)
        signal = peak_accel_g * np.sin(2 * np.pi * freq * t)
        if noise > 0:
            signal = signal + np.random.normal(0, noise, n)
        return signal

    def test_threshold_is_25_percent_of_bc_boundary(self):
        a = VibrationAdapter(zone_bc=2.8)
        assert a.criterion_ii_threshold_mm_s == pytest.approx(0.25 * 2.8)

    def test_no_flag_before_baseline_established(self):
        a = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(5):  # fewer than baseline_window
            feats = a.extract_features(self._make_buf(0.3))
            assert feats['criterion_ii_flag'] is False
            assert feats['criterion_ii_delta_mm_s'] == 0.0
        assert a.baseline_established is False

    def test_baseline_establishes_after_window(self):
        a = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(20):
            a.extract_features(self._make_buf(0.3))
        assert a.baseline_established is True
        assert a.baseline_velocity_mm_s is not None
        assert a.baseline_velocity_mm_s > 0

    def test_small_change_does_not_flag(self):
        a = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(20):
            a.extract_features(self._make_buf(0.30))
        # Small bump, well under 25%-of-B/C threshold
        feats = a.extract_features(self._make_buf(0.31))
        assert feats['criterion_ii_flag'] is False

    def test_large_same_zone_jump_flags(self):
        """
        The core Criterion II case: a jump big enough to exceed the 25%
        threshold while STILL remaining in the same absolute zone --
        Criterion I alone would miss this.
        """
        a = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(20):
            a.extract_features(self._make_buf(0.30))
        baseline_zone = a._zone_label(a.baseline_velocity_mm_s)

        feats = a.extract_features(self._make_buf(0.42))
        assert feats['criterion_ii_flag'] is True
        assert feats['iso_zone'] == baseline_zone, (
            "test setup should produce a same-zone jump; adjust peak_accel_g values if this fails"
        )

    def test_criterion_ii_boost_increases_theta_o_over_criterion_i_alone(self):
        a = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(20):
            a.extract_features(self._make_buf(0.30))
        feats = a.extract_features(self._make_buf(0.42))
        theta_i_only = a._criterion_i_theta(feats['velocity_rms_mm_s'])
        assert feats['theta_o'] >= theta_i_only

    def test_reset_baseline_clears_state(self):
        a = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(20):
            a.extract_features(self._make_buf(0.30))
        assert a.baseline_established is True

        a.reset_baseline()
        assert a.baseline_established is False
        assert a.baseline_velocity_mm_s is None
        assert a._baseline_buffer == []

    def test_get_state_returns_baseline_info(self):
        a = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(20):
            a.extract_features(self._make_buf(0.30))
        state = a.get_state()
        assert state["baseline_established"] is True
        assert state["baseline_velocity_mm_s"] == pytest.approx(a.baseline_velocity_mm_s)
        assert len(state["_baseline_buffer"]) == 20

    def test_load_state_restores_established_baseline(self):
        a1 = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(20):
            a1.extract_features(self._make_buf(0.30))
        saved_state = a1.get_state()

        # Fresh adapter instance, as if the process had restarted.
        a2 = VibrationAdapter(sample_rate=10000, baseline_window=20)
        assert a2.baseline_established is False

        a2.load_state(saved_state)
        assert a2.baseline_established is True
        assert a2.baseline_velocity_mm_s == pytest.approx(a1.baseline_velocity_mm_s)

    def test_load_state_none_is_a_safe_no_op(self):
        """Older state files (pre-Criterion-II-persistence) won't have an
        'adapter' key -- load_state(None) must not crash, and must leave
        the adapter to re-establish its baseline fresh, same as before."""
        a = VibrationAdapter(sample_rate=10000, baseline_window=20)
        a.load_state(None)
        assert a.baseline_established is False

    def test_restored_adapter_immediately_applies_criterion_ii(self):
        """
        The actual bug this persistence fix addresses: after a restart, the
        adapter should NOT need another baseline_window calls before
        Criterion II becomes active again -- it should be active from the
        very next call, using the restored baseline.
        """
        a1 = VibrationAdapter(sample_rate=10000, baseline_window=20)
        for _ in range(20):
            a1.extract_features(self._make_buf(0.30))
        saved_state = a1.get_state()

        a2 = VibrationAdapter(sample_rate=10000, baseline_window=20)
        a2.load_state(saved_state)

        # First call after "restart" -- should already compare against the
        # restored baseline, not start re-buffering from zero.
        feats = a2.extract_features(self._make_buf(0.42))
        assert feats['criterion_ii_flag'] is True

    def test_boost_is_capped_at_boost_weight(self):
        a = VibrationAdapter(sample_rate=10000, baseline_window=20,
                              criterion_ii_boost_weight=0.15)
        for _ in range(20):
            a.extract_features(self._make_buf(0.30))
        # Extreme jump
        feats = a.extract_features(self._make_buf(5.0))
        theta_i_only = a._criterion_i_theta(feats['velocity_rms_mm_s'])
        boost_applied = feats['theta_o'] - theta_i_only
        # boost should never exceed the configured weight (allowing for the
        # final clip(0,1) potentially truncating it further, never exceeding it)
        assert boost_applied <= 0.15 + 1e-9


# ============================================================
# VibrationAdapter -- edge cases / robustness
# ============================================================

class TestNemaThetaC:
    """theta_c is grounded in NEMA MG-1 Class F / 1.15 SF thermal and load limits."""

    def test_load_severity_at_documented_breakpoints(self):
        a = VibrationAdapter()
        assert a._load_severity(0) == pytest.approx(0.0)
        assert a._load_severity(100) == pytest.approx(0.50)   # rated load boundary
        assert a._load_severity(115) == pytest.approx(0.85)   # service factor boundary

    def test_load_severity_saturates_at_130_percent(self):
        a = VibrationAdapter()
        assert a._load_severity(130) == pytest.approx(1.0)
        assert a._load_severity(200) == pytest.approx(1.0)

    def test_load_severity_is_monotonically_non_decreasing(self):
        a = VibrationAdapter()
        loads = np.linspace(0, 200, 100)
        severities = [a._load_severity(l) for l in loads]
        assert np.all(np.diff(severities) >= -1e-9)

    def test_ambient_severity_zero_at_and_below_reference(self):
        a = VibrationAdapter()
        assert a._ambient_severity(40) == pytest.approx(0.0)
        assert a._ambient_severity(20) == pytest.approx(0.0)  # cooler than reference: still 0, not negative

    def test_ambient_severity_saturates_when_rise_budget_fully_consumed(self):
        a = VibrationAdapter()
        # rated_ambient_c=40, rise_budget_c=105 by default -> saturates at 145C
        assert a._ambient_severity(145) == pytest.approx(1.0)
        assert a._ambient_severity(200) == pytest.approx(1.0)

    def test_ambient_severity_is_monotonically_non_decreasing(self):
        a = VibrationAdapter()
        temps = np.linspace(0, 300, 100)
        severities = [a._ambient_severity(t) for t in temps]
        assert np.all(np.diff(severities) >= -1e-9)

    def test_reference_conditions_give_low_but_nonzero_theta_c_at_rated_load(self):
        """At exactly the NEMA reference ambient (40C) and 100% rated load,
        severity should be moderate (right at the rated-load boundary) but
        nowhere near saturated."""
        a = VibrationAdapter()
        theta_c = a.normalize_context(machine_load=100, ambient_temp=40)
        assert 0.3 < theta_c < 0.6

    def test_overload_and_overtemp_together_is_worse_than_either_alone(self):
        a = VibrationAdapter()
        both = a.normalize_context(machine_load=120, ambient_temp=100)
        load_only = a.normalize_context(machine_load=120, ambient_temp=40)
        temp_only = a.normalize_context(machine_load=60, ambient_temp=100)
        assert both > load_only
        assert both > temp_only

    def test_theta_c_always_in_unit_interval(self):
        a = VibrationAdapter()
        for load in np.linspace(0, 300, 20):
            for temp in np.linspace(0, 300, 20):
                tc = a.normalize_context(load, temp)
                assert 0.0 <= tc <= 1.0


class TestAdapterEdgeCases:
    def test_empty_buffer_does_not_crash(self):
        a = VibrationAdapter()
        feats = a.extract_features(np.array([]))
        assert feats['theta_o'] == 0.0
        assert feats['iso_zone'] == 'A'

    def test_all_zero_buffer_does_not_crash(self):
        a = VibrationAdapter()
        feats = a.extract_features(np.zeros(1000))
        assert feats['theta_o'] == 0.0
        assert feats['velocity_rms_mm_s'] == 0.0

    def test_nan_values_are_sanitized(self):
        a = VibrationAdapter()
        buf = np.array([0.1, np.nan, 0.2, np.nan, 0.15] * 200)
        feats = a.extract_features(buf)
        assert not math.isnan(feats['theta_o'])
        assert not math.isnan(feats['velocity_rms_mm_s'])

    def test_inf_values_are_sanitized_and_saturate_high(self):
        a = VibrationAdapter()
        buf = np.array([np.inf, -np.inf] * 500)
        feats = a.extract_features(buf)
        assert not math.isnan(feats['theta_o'])
        assert not math.isinf(feats['theta_o'])
        assert feats['iso_zone'] == 'D'  # posinf sentinel is deliberately extreme

    def test_normalize_context_bounds(self):
        a = VibrationAdapter()
        assert a.normalize_context(machine_load=0, ambient_temp=0) == pytest.approx(0.0)
        assert a.normalize_context(machine_load=1000, ambient_temp=1000) == pytest.approx(1.0)
        # mid-range sanity check
        mid = a.normalize_context(machine_load=50, ambient_temp=40)
        assert 0.0 <= mid <= 1.0


# ============================================================
# JOREngine
# ============================================================

class TestJOREngine:
    def test_no_alert_during_calibration_regardless_of_input(self):
        engine = JOREngine(calibration_steps=10)
        for _ in range(9):  # one less than calibration_steps
            sop, nhp, alert = engine.fusion_step(0.0, 1.0, 1.0)  # worst-case inputs
            assert alert is False
            assert engine.is_calibrated is False

    def test_calibration_completes_after_calibration_steps(self):
        engine = JOREngine(calibration_steps=10)
        for _ in range(10):
            engine.fusion_step(0.5, 0.5, 0.5)
        assert engine.is_calibrated is True
        assert engine.baseline_sop == pytest.approx(0.5)

    def test_baseline_sop_is_average_of_calibration_window(self):
        engine = JOREngine(calibration_steps=5)
        # sop = 0.4*ts + 0.3*tc + 0.3*to; use fixed inputs, sop is constant each step
        for _ in range(5):
            engine.fusion_step(1.0, 1.0, 1.0)
        assert engine.baseline_sop == pytest.approx(1.0)

    def test_provided_baseline_sop_skips_calibration(self):
        engine = JOREngine(baseline_sop=0.5)
        assert engine.is_calibrated is True
        sop, nhp, alert = engine.fusion_step(0.5, 0.5, 0.5)
        # Should immediately run normal-operation math, not calibration buffering.
        assert len(engine._calibration_buffer) == 0

    def test_alert_fires_above_upper_threshold(self):
        engine = JOREngine(calibration_steps=5, upper_th=0.55, lower_th=0.45)
        for _ in range(5):
            engine.fusion_step(0.5, 0.5, 0.5)  # calibrate at a moderate baseline
        # Now push evidence strongly toward non-healthy repeatedly.
        alert = False
        for _ in range(50):
            _, _, alert = engine.fusion_step(0.0, 1.0, 1.0)
        assert alert is True

    def test_alert_clears_below_lower_threshold_after_recovery(self):
        engine = JOREngine(calibration_steps=5, upper_th=0.55, lower_th=0.45)
        for _ in range(5):
            engine.fusion_step(0.5, 0.5, 0.5)
        for _ in range(50):
            _, _, alert = engine.fusion_step(0.0, 1.0, 1.0)
        assert alert is True
        # Recover: feed healthy evidence for a long stretch.
        for _ in range(50):
            _, _, alert = engine.fusion_step(1.0, 0.0, 0.0)
        assert alert is False

    def test_retention_blend_keeps_posterior_stable_at_baseline(self):
        """
        Regression test for the bug documented in engine.py's own history:
        without retention blending, p_final drifts monotonically even when
        held exactly at baseline. This should NOT happen.
        """
        engine = JOREngine(calibration_steps=10, retention=0.70)
        for _ in range(10):
            engine.fusion_step(0.5, 0.5, 0.5)  # establish baseline at sop=0.5

        posteriors = []
        for _ in range(100):
            _, nhp, _ = engine.fusion_step(0.5, 0.5, 0.5)  # stay exactly at baseline
            posteriors.append(nhp)

        # Should hover near the prior, not drift toward 0 or 1 over time.
        assert posteriors[-1] == pytest.approx(posteriors[-10], abs=1e-6)
        assert posteriors[-1] < 0.5  # shouldn't have drifted into alert territory

    def test_sop_formula_matches_documented_weights(self):
        engine = JOREngine(calibration_steps=1000)  # stay in calibration to inspect raw sop
        sop, _, _ = engine.fusion_step(theta_s=0.8, theta_c=0.6, theta_o=0.2)
        expected = 0.40 * 0.8 + 0.30 * 0.6 + 0.30 * 0.2
        assert sop == pytest.approx(expected)


# ============================================================
# FusionLogger
# ============================================================

class TestFusionLogger:
    def test_log_state_writes_valid_json_line(self, tmp_path):
        log_file = tmp_path / "test_log.json"
        logger = FusionLogger(str(log_file))
        result = logger.log_state(0.5, 0.3, False, metadata={"step": 1})
        assert result is True

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["sop_fused"] == 0.5
        assert entry["nhp_posterior"] == 0.3
        assert entry["alert_active"] is False
        assert entry["metadata"]["step"] == 1

    def test_multiple_calls_append_multiple_lines(self, tmp_path):
        log_file = tmp_path / "test_log.json"
        logger = FusionLogger(str(log_file))
        for i in range(5):
            logger.log_state(0.1 * i, 0.2 * i, i % 2 == 0, metadata={"step": i})

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["metadata"]["step"] == i

    def test_default_metadata_is_empty_dict(self, tmp_path):
        log_file = tmp_path / "test_log.json"
        logger = FusionLogger(str(log_file))
        logger.log_state(0.5, 0.5, True)
        entry = json.loads(log_file.read_text().strip())
        assert entry["metadata"] == {}

    def test_values_are_rounded_to_4_decimals(self, tmp_path):
        log_file = tmp_path / "test_log.json"
        logger = FusionLogger(str(log_file))
        logger.log_state(0.123456789, 0.987654321, False)
        entry = json.loads(log_file.read_text().strip())
        assert entry["sop_fused"] == 0.1235
        assert entry["nhp_posterior"] == 0.9877


# ============================================================
# Integration: adapter -> engine, using realistic demo amplitudes
# ============================================================

class TestIntegration:
    def _make_buf(self, peak_accel_g, freq=180, n=10000, fs=10000):
        t = np.linspace(0, 1, n)
        return peak_accel_g * np.sin(2 * np.pi * freq * t)

    def test_healthy_amplitude_stays_zone_a_and_normal(self):
        """
        Regression test for the amplitude rescaling: healthy operation
        (0.12g) must land in Zone A and never trigger ALERT, matching
        main.py's documented expectation.
        """
        adapter = VibrationAdapter(sample_rate=10000)
        engine = JOREngine(calibration_steps=20)

        for _ in range(20):
            feats = adapter.extract_features(self._make_buf(0.12))
            engine.fusion_step(0.72, 0.5, feats['theta_o'])
        assert engine.is_calibrated is True

        for _ in range(15):
            feats = adapter.extract_features(self._make_buf(0.12))
            _, _, alert = engine.fusion_step(0.72, 0.5, feats['theta_o'])
            assert feats['iso_zone'] == 'A'
            assert alert is False

    def test_escalation_eventually_triggers_alert(self):
        """
        Regression test for the full demo arc: ramping amplitude from
        healthy toward Zone D must eventually trigger ALERT.
        """
        adapter = VibrationAdapter(sample_rate=10000)
        engine = JOREngine(calibration_steps=20)

        for _ in range(20):
            feats = adapter.extract_features(self._make_buf(0.12))
            engine.fusion_step(0.72, 0.5, feats['theta_o'])

        alert_seen = False
        for i in range(20):
            peak_accel_g = 0.12 + i * 0.07
            feats = adapter.extract_features(self._make_buf(peak_accel_g))
            _, _, alert = engine.fusion_step(0.72, 0.5, feats['theta_o'])
            if alert:
                alert_seen = True
        assert alert_seen is True

    def test_recovery_returns_to_normal(self):
        adapter = VibrationAdapter(sample_rate=10000)
        engine = JOREngine(calibration_steps=20)

        for _ in range(20):
            feats = adapter.extract_features(self._make_buf(0.12))
            engine.fusion_step(0.72, 0.5, feats['theta_o'])
        for i in range(20):  # escalate to trigger alert
            peak_accel_g = 0.12 + i * 0.07
            feats = adapter.extract_features(self._make_buf(peak_accel_g))
            engine.fusion_step(0.72, 0.5, feats['theta_o'])

        alert = True
        for _ in range(25):  # recovery phase
            feats = adapter.extract_features(self._make_buf(0.12))
            _, _, alert = engine.fusion_step(0.72, 0.5, feats['theta_o'])
        assert alert is False
