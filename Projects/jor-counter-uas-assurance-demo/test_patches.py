import math
import numpy as np
from engine import JOREngine
from cuas_adapter import CUASHealthAdapter


# ---- no targets vs no data -------------------------------------------------
def test_empty_tracks_alive_is_neutral():
    a = CUASHealthAdapter()
    r = a.extract_features(np.array([]), np.array([]), 0.10, 0.0)
    assert r["no_targets"] and not r["data_missing"] and r["theta_o"] < 0.05


def test_dead_feed_is_degraded():
    a = CUASHealthAdapter()
    r = a.extract_features(np.array([]), np.array([]), 0.10, 0.0, feed_alive=False)
    assert r["data_missing"] and r["theta_o"] >= 0.5


def test_all_nan_tracks_not_healthy():
    a = CUASHealthAdapter()
    r = a.extract_features(np.full(10, np.nan), np.full(10, np.nan), 0.1, 0.0)
    assert r["data_missing"] and r["theta_o"] >= 0.5


def test_partial_nan_dropped_not_masked():
    a = CUASHealthAdapter()
    cont = np.array([0.5, 0.5, np.nan, np.nan, np.nan])
    r = a.extract_features(cont, np.full(5, 0.85), 0.1, 0.0)
    assert not r["data_missing"] and abs(r["median_continuity"] - 0.5) < 1e-9


def _srcs(bad=None, empty=()):
    d = {}
    for i in range(8):
        if i in empty:
            d[f"s{i}"] = {"continuities": np.array([]), "confidences": np.array([])}
        else:
            c = 0.42 if i == bad else 0.94
            d[f"s{i}"] = {"continuities": np.full(25, c), "confidences": np.full(25, 0.88)}
    return d


def _beats(stale=()):
    return {f"s{i}": (9.0 if i in stale else 0.3) for i in range(8)}


def test_all_sources_idle_and_alive_is_no_targets():
    a = CUASHealthAdapter()
    r = a.extract_features_per_source(_srcs(empty=range(8)), 0.1, 0.0, _beats())
    assert r["no_targets"] and not r["data_missing"] and len(r["idle_sources"]) == 8


def test_all_sources_stale_is_no_data():
    a = CUASHealthAdapter()
    r = a.extract_features_per_source(_srcs(empty=range(8)), 0.1, 0.0, _beats(stale=range(8)))
    assert r["data_missing"] and len(r["stale_sources"]) == 8


def test_no_sources_configured_is_no_data():
    a = CUASHealthAdapter()
    assert a.extract_features_per_source({}, 0.1, 0.0)["data_missing"]


def test_malformed_source_flagged():
    a = CUASHealthAdapter()
    s = _srcs()
    s["s2"] = {"continuities": np.full(25, np.nan), "confidences": np.full(25, np.nan)}
    r = a.extract_features_per_source(s, 0.1, 0.0)
    assert r["malformed_sources"] == ["s2"] and not r["data_missing"]


def test_stale_sources_excluded_and_named():
    a = CUASHealthAdapter()
    r = a.extract_features_per_source(_srcs(empty=(1, 2)), 0.1, 0.0, _beats(stale=(1, 2)))
    assert r["stale_sources"] == ["s1", "s2"] and not r["data_missing"]


def test_stale_heartbeat_raises_theta_s_despite_good_reported_health():
    a = CUASHealthAdapter()
    health = {f"s{i}": 0.95 for i in range(8)}
    ok = a.normalize_system_state(health, 0.97, 0.05, heartbeat_ages=_beats())
    bad = a.normalize_system_state(health, 0.97, 0.05, heartbeat_ages=_beats(stale=(1, 2, 5)))
    assert bad > ok + 0.25


def test_one_dead_sensor_not_averaged_away():
    a = CUASHealthAdapter()
    health = {f"s{i}": 1.0 for i in range(8)}
    health["s0"] = 0.0
    assert a.normalize_system_state(health) > 0.25


# ---- persistence / attribution --------------------------------------------
def test_outlier_requires_persistence_and_is_attributed():
    a = CUASHealthAdapter(persistence_steps=2)
    rs = [a.extract_features_per_source(_srcs(3), 0.13, 0.02) for _ in range(5)]
    counts = [r["persistent_outliers"] for r in rs]
    assert counts[0] == 0 and counts[1] == 0 and counts[2] == 1
    assert rs[2]["degraded_sources"] == ["s3"]


def test_single_bad_step_not_persistent():
    a = CUASHealthAdapter(persistence_steps=2)
    a.extract_features_per_source(_srcs(3), 0.13, 0.02)
    for _ in range(4):
        r = a.extract_features_per_source(_srcs(None), 0.13, 0.02)
    assert r["persistent_outliers"] == 0 and r["degraded_sources"] == []


def test_zero_traffic_has_no_cooperative_penalty():
    a = CUASHealthAdapter()
    assert a.normalize_context(0, 0, 0.0) == 0.0


# ---- engine ----------------------------------------------------------------
def test_nan_theta_holds_state_then_fails_safe():
    e = JOREngine(baseline_sop=0.07)
    e.fusion_step(0.04, 0.2, 0.01)
    p = e.p_final
    for _ in range(2):
        e.fusion_step(float("nan"), 0.2, 0.01)
        assert e.data_fault and e.p_final == p and not e.alert_status
    e.fusion_step(float("nan"), 0.2, 0.01)
    assert e.alert_status and math.isfinite(e.p_final)
    e.fusion_step(0.04, 0.2, 0.01)
    assert not e.data_fault and math.isfinite(e.p_final)


def test_calibration_flags_contamination():
    e = JOREngine(calibration_steps=20)
    for k in range(20):
        e.fusion_step(0.04, 0.2, 0.30 if k == 7 else 0.01)
    assert e.is_calibrated and e.calibration_suspect and e.baseline_sop < 0.1


def test_clean_calibration_not_flagged():
    e = JOREngine(calibration_steps=20)
    for k in range(20):
        e.fusion_step(0.04, 0.2 + 0.002 * k, 0.01)
    assert not e.calibration_suspect


def test_nhp_range():
    e = JOREngine(prior_nh=0.05, retention=0.70)
    assert abs(e.nhp_floor - 0.015) < 1e-9 and abs(e.nhp_ceiling - 0.715) < 1e-9


def test_deadband_ignores_small_delta():
    e = JOREngine(baseline_sop=0.07, delta_deadband=0.03)
    for _ in range(50):
        e.fusion_step(0.2, 0.2, 0.0)      # sop .12 -> delta .05 -> .02 after deadband
    assert e.p_final < 0.25


# ---- status / bands --------------------------------------------------------
def _calibrated():
    e = JOREngine(calibration_steps=10, delta_deadband=0.03)
    for _ in range(10):
        e.fusion_step(0.04, 0.2, 0.01)
    return e


def test_state_bands_and_status_json_safe():
    import json
    e = JOREngine(calibration_steps=10)
    assert e.state_band() == "CALIBRATING"
    e = _calibrated()
    assert e.state_band() == "NOMINAL"
    for _ in range(6):
        e.fusion_step(0.6, 0.2, 0.7)
    assert e.state_band() == "UNRELIABLE"
    st = e.status(ts=12.0)
    json.dumps(st)
    assert st["state"] == "UNRELIABLE" and st["alert"] and st["nhp_range"] == [0.015, 0.715]


def test_dominant_channel_attribution():
    e = _calibrated()
    e.fusion_step(0.6, 0.2, 0.01)          # sensor integrity collapses
    assert e.dominant_channel() == "sensor_integrity"
    e.fusion_step(0.04, 0.2, 0.9)          # track quality collapses
    assert e.dominant_channel() == "track_quality"
    e.fusion_step(0.04, 0.2, 0.01)         # back to baseline
    assert e.dominant_channel() is None


def test_fault_reports_degraded_band():
    e = _calibrated()
    e.fusion_step(float("nan"), 0.2, 0.01)
    assert e.state_band() == "DEGRADED" and e.status()["data_fault"]


def test_status_staleness_helper():
    st = {"ts": 100.0}
    assert not JOREngine.status_is_stale(st, now=103.0, max_age=5.0)
    assert JOREngine.status_is_stale(st, now=110.0, max_age=5.0)
    assert JOREngine.status_is_stale({"ts": None}, now=1.0)
    assert JOREngine.status_is_stale(None, now=1.0)


def test_per_source_flag():
    a = CUASHealthAdapter()
    assert a.extract_features(np.full(5, 0.9), np.full(5, 0.9), 0.1, 0.0)["per_source"] is False
    assert a.extract_features_per_source(_srcs(), 0.1, 0.0)["per_source"] is True


# ---- log runs / viz loading ------------------------------------------------
def _write_runs(path):
    from logger import FusionLogger
    for rid, n in (("run-A", 3), ("run-B", 5)):
        lg = FusionLogger(path, run_id=rid)
        for i in range(n):
            lg.log_state(0.1, 0.05, False, metadata={
                "phase": "p", "theta_s": 0.1, "theta_c": 0.1, "theta_o": 0.1,
                "status": {"nhp": 0.05, "state": "NOMINAL", "sop": 0.1, "baseline_sop": 0.08,
                           "dominant_channel": None, "details": {}, "nhp_range": [0.015, 0.715]}})


def test_log_runs_selected_by_id():
    import os, tempfile
    import viz
    path = os.path.join(tempfile.mkdtemp(), "log.jsonl")
    _write_runs(path)
    assert len(viz.load_log(path)) == 5                    # latest run only
    assert len(viz.load_log(path, "all")) == 8
    assert len(viz.load_log(path, "run-A")) == 3


def test_viz_friendly_errors():
    import os, tempfile
    import viz
    path = os.path.join(tempfile.mkdtemp(), "log.jsonl")
    _write_runs(path)
    for bad in (lambda: viz.load_log(path + ".missing"), lambda: viz.load_log(path, "nope")):
        try:
            bad()
        except SystemExit:
            continue
        raise AssertionError("expected SystemExit")


# ---- math / labels ---------------------------------------------------------
def test_update_is_log_odds_shift_plus_retention():
    e = JOREngine(prior_nh=0.05, steepness=12.0, retention=0.70, baseline_sop=0.05, delta_deadband=0.0)
    e.p_final = 0.20
    th = (0.30, 0.20, 0.10)
    sop, p, _ = e.fusion_step(*th)
    assert abs(sop - (0.35 * 0.30 + 0.25 * 0.20 + 0.40 * 0.10)) < 1e-12
    odds = (0.20 / 0.80) * math.exp(12.0 * (sop - 0.05))
    expected = 0.70 * (odds / (1 + odds)) + 0.30 * 0.05
    assert abs(p - expected) < 1e-6


def test_nhp_stays_inside_reachable_range():
    rng = np.random.default_rng(0)
    e = JOREngine(calibration_steps=10, delta_deadband=0.02)
    for _ in range(10):
        e.fusion_step(0.04, 0.2, 0.01)
    for _ in range(3000):
        e.fusion_step(*rng.uniform(0, 1, 3))
        assert e.nhp_floor - 1e-9 <= e.p_final <= e.nhp_ceiling + 1e-9


def test_channel_weights_sum_to_one_and_drive_sop():
    assert abs(sum(w for _, w in JOREngine.CHANNEL_WEIGHTS) - 1.0) < 1e-12
    e = JOREngine(baseline_sop=0.05)
    sop, _, _ = e.fusion_step(0.2, 0.4, 0.6)
    assert abs(sop - (0.35 * 0.2 + 0.25 * 0.4 + 0.40 * 0.6)) < 1e-12


def test_channel_keys_match_viz_labels():
    import viz
    keys = {name for name, _ in JOREngine.CHANNEL_WEIGHTS}
    assert keys == {k for k in viz.CHANNEL_COLORS if k is not None}
    assert keys == {k for k in viz.CHANNEL_LABELS if k is not None}


def test_degraded_band_has_hysteresis():
    e = _calibrated()
    e.p_final, e.degraded_status = 0.30, True
    seen = []
    for _ in range(12):
        e.fusion_step(0.04, 0.2, 0.01)           # sits on baseline: NHP decays
        seen.append((e.state_band(), e.p_final))
    for band, p in seen:
        if band == "DEGRADED":
            assert p >= e.degraded_exit_th
        if band == "NOMINAL":
            assert p < e.degraded_exit_th
    assert any(b == "DEGRADED" and p < e.degraded_th for b, p in seen)   # held inside the hysteresis gap
    assert seen[-1][0] == "NOMINAL"


def test_state_from_older_version_does_not_crash():
    e = _calibrated()
    e.baseline_channels = {"sensor_integrity": 0.04, "airspace_load": 0.2, "track_quality": 0.01}
    assert e.channel_excess() is None and e.dominant_channel() is None
    e.status()


def test_status_names_driver_only_when_not_nominal():
    e = _calibrated()
    e.last_thetas = (0.6, 0.2, 0.01)           # sensor integrity well above baseline
    e.p_final, e.alert_status, e.degraded_status = 0.05, False, False
    assert e.dominant_channel() == "sensor_integrity"      # the raw attribution exists
    assert e.state_band() == "NOMINAL" and e.status()["dominant_channel"] is None
    e.p_final, e.degraded_status = 0.30, True
    assert e.state_band() == "DEGRADED" and e.status()["dominant_channel"] == "sensor_integrity"
