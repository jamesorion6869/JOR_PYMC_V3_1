import numpy as np
from collections import defaultdict


class CUASHealthAdapter:
    """
    Domain adapter for C-UAS *sensing-assurance* monitoring.

    This layer never detects or classifies drones. It scores how reliable the
    detection-and-tracking picture currently is, using only quality metadata
    (continuity, confidence, noise, liveness, load), and produces the three
    evidence channels required by the JOR engine:

        theta_s  - sensor & platform integrity (incl. feed liveness)
        theta_c  - load on the sensing system (saturation, RF clutter,
                   non-cooperative share) - NOT a threat-level estimate
        theta_o  - observed detection & track quality (incl. per-source
                   anomaly scoring)

    Three situations are kept distinct:
        no targets : feeds are alive but report nothing (normal in quiet
                     sky)           -> neutral, flagged `no_targets`
        no data    : feed stale / malformed / absent
                                     -> degraded, flagged `data_missing`
        degraded   : feed alive, track quality poor
                                     -> scored normally

    Per-source outlier logic is retained so a single degraded radar sector,
    noisy RF channel, or anomalous track set can raise theta_o without
    waiting for the whole fused picture to collapse.
    """

    def __init__(self,
                 # Track-quality baselines
                 track_continuity_baseline=0.92,
                 track_continuity_critical=0.55,
                 classification_conf_baseline=0.85,
                 classification_conf_critical=0.40,
                 # Noise / clutter
                 noise_floor_baseline=0.15,          # normalised 0-1
                 noise_floor_critical=0.70,
                 # Outlier / anomaly detection
                 outlier_iqr_threshold=1.5,
                 ema_alpha=0.30,
                 min_sources_for_outlier=3,
                 min_iqr=0.05,
                 max_iqr_score=4.0,          # cap so one sample cannot saturate the EMA
                 persistence_steps=2,        # consecutive steps above threshold to count as persistent
                 # Liveness / missing-data handling
                 heartbeat_timeout=2.0,      # seconds without a heartbeat -> stale feed
                 missing_data_theta_o=0.60,  # no usable data != healthy
                 worst_sensor_weight=0.60,   # one dead sensor must not be averaged away
                 # System health
                 sensor_health_critical=0.40):

        self.track_continuity_baseline = track_continuity_baseline
        self.track_continuity_critical = track_continuity_critical
        self.classification_conf_baseline = classification_conf_baseline
        self.classification_conf_critical = classification_conf_critical
        self.noise_floor_baseline = noise_floor_baseline
        self.noise_floor_critical = noise_floor_critical
        self.outlier_iqr_threshold = outlier_iqr_threshold
        self.ema_alpha = ema_alpha
        self.min_sources_for_outlier = min_sources_for_outlier
        self.min_iqr = min_iqr
        self.max_iqr_score = max_iqr_score
        self.persistence_steps = persistence_steps
        self.heartbeat_timeout = heartbeat_timeout
        self.missing_data_theta_o = missing_data_theta_o
        self.worst_sensor_weight = worst_sensor_weight
        self.sensor_health_critical = sensor_health_critical

        # Persistent per-source state
        self.source_ema = defaultdict(float)
        self.source_streak = defaultdict(int)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _attr_defaults(result):
        result.setdefault('persistent_outliers', 0)
        result.setdefault('pending_outliers', 0)
        result.setdefault('degraded_sources', [])
        result.setdefault('stale_sources', [])
        result.setdefault('malformed_sources', [])
        result.setdefault('idle_sources', [])
        result.setdefault('per_source', False)   # True only when per-source detail was evaluated
        return result

    def _noise_norm(self, noise_floor):
        return float(np.clip(
            (noise_floor - self.noise_floor_baseline) /
            (self.noise_floor_critical - self.noise_floor_baseline + 1e-9),
            0.0, 1.0))

    def _missing(self, noise_floor, false_alarm_rate):
        """No usable data: report elevated theta_o and flag it."""
        nf = float(noise_floor) if np.isfinite(noise_floor) else 0.0
        fa = float(false_alarm_rate) if np.isfinite(false_alarm_rate) else 0.0
        return self._attr_defaults({
            'theta_o': float(self.missing_data_theta_o),
            'median_continuity': 0.0,
            'median_confidence': 0.0,
            'noise_floor': nf,
            'false_alarm_rate': fa,
            'data_missing': True,
            'no_targets': False,
        })

    def _no_targets(self, noise_floor, false_alarm_rate):
        """Feeds alive, nothing to track: only the noise floor carries evidence."""
        nf = float(noise_floor)
        return self._attr_defaults({
            'theta_o': float(np.clip(0.25 * self._noise_norm(nf), 0.0, 1.0)),
            'median_continuity': 0.0,
            'median_confidence': 0.0,
            'noise_floor': nf,
            'false_alarm_rate': float(false_alarm_rate),
            'data_missing': False,
            'no_targets': True,
        })

    def reset_source_state(self):
        """Clear per-source EMA/persistence state (e.g. after a long feed gap)."""
        self.source_ema = defaultdict(float)
        self.source_streak = defaultdict(int)

    def _prune(self, current_ids):
        current_ids = set(current_ids)
        self.source_ema = defaultdict(
            float, {sid: self.source_ema[sid] for sid in current_ids if sid in self.source_ema})
        self.source_streak = defaultdict(
            int, {sid: self.source_streak[sid] for sid in current_ids if sid in self.source_streak})

    # ------------------------------------------------------------------
    # theta_o - Observed detection / track quality + anomalies
    # ------------------------------------------------------------------
    def extract_features(self,
                         track_continuities: np.ndarray,
                         classification_confidences: np.ndarray,
                         noise_floor: float,
                         false_alarm_rate: float = 0.0,
                         feed_alive: bool = True):
        """
        Aggregate-only observation channel.

        track_continuities          : per-track continuity scores (0-1)
        classification_confidences  : per-track classification confidences (0-1)
        noise_floor                 : normalised sensor-noise / clutter level (0-1)
        false_alarm_rate            : fraction of tracks later discarded as false
        feed_alive                  : False if the feed's liveness check failed

        Empty arrays with a live feed mean "no targets" (neutral).
        Non-finite-only arrays, a dead feed, or non-finite scalars mean "no data".
        """
        rc = np.asarray(track_continuities, dtype=float)
        rf = np.asarray(classification_confidences, dtype=float)
        c = rc[np.isfinite(rc)]
        f = rf[np.isfinite(rf)]

        if (not feed_alive
                or not np.isfinite(noise_floor) or not np.isfinite(false_alarm_rate)
                or (rc.size > 0 and c.size == 0) or (rf.size > 0 and f.size == 0)):
            return self._missing(noise_floor, false_alarm_rate)
        if rc.size == 0 and rf.size == 0:
            return self._no_targets(noise_floor, false_alarm_rate)
        if c.size == 0 or f.size == 0:       # one channel present without the other
            return self._missing(noise_floor, false_alarm_rate)

        med_cont = float(np.median(c))
        med_conf = float(np.median(f))

        # Continuity degradation (higher = worse)
        cont_norm = np.clip(
            (self.track_continuity_baseline - med_cont) /
            (self.track_continuity_baseline - self.track_continuity_critical + 1e-9),
            0.0, 1.0
        )
        # Classification confidence degradation
        conf_norm = np.clip(
            (self.classification_conf_baseline - med_conf) /
            (self.classification_conf_baseline - self.classification_conf_critical + 1e-9),
            0.0, 1.0
        )
        noise_norm = self._noise_norm(noise_floor)
        fa_norm = np.clip(false_alarm_rate / 0.25, 0.0, 1.0)   # 25 % FAR -> full scale

        # Blend: continuity & confidence dominate, noise and FAR add pressure
        theta_o = np.clip(
            0.35 * max(cont_norm, conf_norm) +
            0.25 * ((cont_norm + conf_norm) / 2.0) +
            0.25 * noise_norm +
            0.15 * fa_norm,
            0.0, 1.0
        )

        return self._attr_defaults({
            'theta_o': float(theta_o),
            'median_continuity': med_cont,
            'median_confidence': med_conf,
            'noise_floor': float(noise_floor),
            'false_alarm_rate': float(false_alarm_rate),
            'data_missing': False,
            'no_targets': False,
        })

    def extract_features_per_source(self,
                                    source_tracks: dict,
                                    noise_floor: float,
                                    false_alarm_rate: float = 0.0,
                                    heartbeat_ages: dict = None):
        """
        Per-source (radar sector / RF channel / EO camera / acoustic array)
        version for gray-failure / track-anomaly detection.

        source_tracks : {source_id: {'continuities': array, 'confidences': array}}
        heartbeat_ages: {source_id: seconds since last heartbeat}; sources older
                        than `heartbeat_timeout` (or missing) are stale feeds.
                        None = liveness unknown, sources assumed alive.

        Result carries attribution lists: degraded_sources (persistent
        outliers), stale_sources, malformed_sources, idle_sources.
        """
        if not source_tracks:
            self._prune([])
            res = self._missing(noise_floor, false_alarm_rate)
            res['per_source'] = True
            return res

        stale, malformed, idle, active = [], [], [], {}
        for sid, v in source_tracks.items():
            if heartbeat_ages is not None:
                age = heartbeat_ages.get(sid, np.inf)
                if not (age <= self.heartbeat_timeout):      # also catches NaN
                    stale.append(sid)
                    continue
            rc = np.asarray(v['continuities'], dtype=float)
            rf = np.asarray(v['confidences'], dtype=float)
            c, f = rc[np.isfinite(rc)], rf[np.isfinite(rf)]
            if (rc.size > 0 and c.size == 0) or (rf.size > 0 and f.size == 0):
                malformed.append(sid)
                continue
            if c.size == 0:
                idle.append(sid)
                continue
            active[sid] = {'continuities': c, 'confidences': f}

        self._prune(active.keys())

        if active:
            all_cont = np.concatenate([v['continuities'] for v in active.values()])
            confs = [v['confidences'] for v in active.values() if len(v['confidences']) > 0]
            all_conf = np.concatenate(confs) if confs else np.array([])
            aggregate = self.extract_features(all_cont, all_conf, noise_floor, false_alarm_rate)
        elif idle:
            aggregate = self._no_targets(noise_floor, false_alarm_rate)
        else:
            aggregate = self._missing(noise_floor, false_alarm_rate)

        aggregate['per_source'] = True
        aggregate['stale_sources'] = stale
        aggregate['malformed_sources'] = malformed
        aggregate['idle_sources'] = idle

        if (len(active) < self.min_sources_for_outlier or aggregate.get('data_missing')
                or aggregate.get('no_targets')):
            return aggregate

        # Per-source median continuity (primary anomaly signal)
        source_medians = {sid: float(np.median(v['continuities'])) for sid, v in active.items()}

        fleet = np.array(list(source_medians.values()))
        fleet_median = float(np.median(fleet))
        q1, q3 = np.percentile(fleet, [25, 75])
        iqr = max(q3 - q1, self.min_iqr)

        persistent_ids = []
        pending = 0
        for sid, med in source_medians.items():
            # Higher continuity is better -> invert so positive score = degradation.
            # Capped so a single catastrophic sample cannot saturate the EMA.
            iqr_score = min((fleet_median - med) / iqr, self.max_iqr_score)
            self.source_ema[sid] = (self.ema_alpha * max(iqr_score, 0.0) +
                                    (1.0 - self.ema_alpha) * self.source_ema[sid])
            if self.source_ema[sid] > self.outlier_iqr_threshold:
                self.source_streak[sid] += 1
            else:
                self.source_streak[sid] = 0
            if self.source_streak[sid] >= self.persistence_steps:
                persistent_ids.append(sid)
            elif self.source_streak[sid] > 0:
                pending += 1

        persistent_outliers = len(persistent_ids)
        outlier_fraction = persistent_outliers / max(len(source_medians), 1)
        worst_ema = max((self.source_ema[sid] for sid in source_medians), default=0.0)

        # Only sources that have persisted drive the boost
        persistent_worst = max((self.source_ema[sid] for sid in persistent_ids), default=0.0)
        severity_boost = (np.clip((persistent_worst - self.outlier_iqr_threshold) /
                                  max(self.outlier_iqr_threshold, 1e-6), 0.0, 1.0)
                          if persistent_ids else 0.0)
        outlier_boost = np.clip(outlier_fraction * 1.8, 0.0, 1.0)
        combined_boost = np.clip(
            0.70 * max(severity_boost, outlier_boost) +
            0.30 * ((severity_boost + outlier_boost) / 2.0),
            0.0, 1.0
        )

        result = aggregate.copy()
        result['theta_o'] = float(np.clip(aggregate['theta_o'] + 0.65 * combined_boost, 0.0, 1.0))
        result['outlier_fraction'] = float(outlier_fraction)
        result['persistent_outliers'] = persistent_outliers
        result['pending_outliers'] = pending
        result['degraded_sources'] = persistent_ids
        result['fleet_median_continuity'] = fleet_median
        result['worst_source_ema'] = float(worst_ema)
        result['severity_boost'] = float(severity_boost)
        return result

    # ------------------------------------------------------------------
    # theta_c - load on the sensing system (not a threat estimate)
    # ------------------------------------------------------------------
    def normalize_context(self,
                          cooperative_tracks: int,
                          total_volume_estimate: int,
                          rf_emitter_density: float,
                          capacity_tracks: int = 120,
                          saturation_knee: float = 0.70):
        """
        Load on the sensing system: how hard the environment makes
        detection and tracking, independent of whether anything is hostile.

        cooperative_tracks     : ADS-B / Remote-ID / known-friendly tracks
        total_volume_estimate  : estimated airborne objects (cooperative + non-cooperative)
        rf_emitter_density     : normalised RF spectral occupancy (0-1)
        capacity_tracks        : design capacity of the C-UAS picture
        """
        if capacity_tracks <= 0:
            return 0.0

        # Volume utilisation (saturation of the picture)
        util = np.clip(total_volume_estimate / capacity_tracks, 0.0, 1.8)
        if util <= saturation_knee:
            volume_norm = float(np.clip((util / saturation_knee) * 0.35, 0.0, 0.35))
        else:
            remaining = max(1.0 - saturation_knee, 1e-6)
            over = np.clip((util - saturation_knee) / remaining, 0.0, 1.0)
            volume_norm = float(np.clip(0.35 + 0.65 * over, 0.0, 1.0))

        # Non-cooperative share adds association / classification burden.
        # With no traffic at all there is no burden.
        if total_volume_estimate <= 0:
            coop_penalty = 0.0
        else:
            coop_ratio = np.clip(cooperative_tracks / total_volume_estimate, 0.0, 1.0)
            coop_penalty = (1.0 - coop_ratio) * 0.25

        rf_norm = np.clip(rf_emitter_density, 0.0, 1.0) * 0.30

        theta_c = np.clip(volume_norm + coop_penalty + rf_norm, 0.0, 1.0)
        return float(theta_c)

    # ------------------------------------------------------------------
    # theta_s - Sensor & platform integrity (incl. liveness)
    # ------------------------------------------------------------------
    def normalize_system_state(self,
                               sensor_health_scores: dict,
                               platform_health: float = 1.0,
                               pipeline_backlog: float = 0.0,
                               max_backlog: float = 1.0,
                               heartbeat_ages: dict = None):
        """
        sensor_health_scores : {source_id: health 0-1}  (1 = fully healthy)
        platform_health      : overall platform / power / thermal / data-link (0-1)
        pipeline_backlog     : normalised processing backlog (0-1)
        heartbeat_ages       : {source_id: seconds since last heartbeat}; stale
                               sources are forced to health 0 regardless of
                               what their last report said.
        """
        health = dict(sensor_health_scores or {})
        if heartbeat_ages is not None:
            for sid, age in heartbeat_ages.items():
                if not (age <= self.heartbeat_timeout):
                    health[sid] = 0.0

        if health:
            vals = np.clip(np.nan_to_num(np.array(list(health.values()), dtype=float), nan=0.0),
                           0.0, 1.0)
            avg_deg = 1.0 - float(np.mean(vals))
            worst_deg = 1.0 - float(np.min(vals))
        else:
            avg_deg, worst_deg = 0.0, 0.0

        # One dead sensor must not be averaged away by seven healthy ones
        sensor_deg = np.clip(max(avg_deg, self.worst_sensor_weight * worst_deg), 0.0, 1.0)
        platform_deg = np.clip(1.0 - platform_health, 0.0, 1.0)
        backlog_norm = np.clip(pipeline_backlog / max(max_backlog, 1e-6), 0.0, 1.0)

        # Worst-signal-dominates blend
        theta_s = np.clip(
            0.55 * max(sensor_deg, platform_deg) +
            0.25 * ((sensor_deg + platform_deg) / 2.0) +
            0.20 * backlog_norm,
            0.0, 1.0
        )
        return float(theta_s)
