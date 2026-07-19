import numpy as np
from collections import defaultdict
import json

class ServiceHealthAdapter:
    """
    Domain adapter for cloud VM / small service health monitoring.

    Now supports per-instance latency tracking for robust gray-failure detection
    (e.g., one bad backend instance) while preserving the original aggregate behavior.
    """

    def __init__(self, latency_baseline_ms=50.0, latency_critical_ms=500.0,
                 error_rate_critical=0.05, outlier_iqr_threshold=1.5,
                 ema_alpha=0.3, min_instances_for_outlier=3, min_iqr_ms=None):

        # Ensure critical latency is always above baseline to avoid divide-by-zero
        if latency_critical_ms <= latency_baseline_ms:
            latency_critical_ms = latency_baseline_ms * 10

        self.latency_baseline_ms = latency_baseline_ms
        self.latency_critical_ms = latency_critical_ms
        self.error_rate_critical = error_rate_critical
        
        # Per-instance parameters
        self.outlier_iqr_threshold = outlier_iqr_threshold  # How many IQRs above fleet median
        self.ema_alpha = ema_alpha  # Smoothing for persistence
        self.min_instances_for_outlier = min_instances_for_outlier

        # With small fleets (or any fleet, on an unlucky sample), the
        # observed IQR across instance medians can collapse to a tiny
        # value purely by chance -- e.g. all instances happening to cluster
        # tightly that particular step. When IQR is that small,
        # (deviation / iqr) explodes even for a normal, healthy deviation,
        # producing a false outlier flag. Flooring IQR at a sensible
        # absolute minimum prevents this amplification. Defaults to 10% of
        # latency_baseline_ms if not given -- e.g. 5ms at a 50ms baseline,
        # small enough not to mask genuine tight-fleet degradation, large
        # enough to stop near-zero IQR from blowing up the score.
        self.min_iqr_ms = min_iqr_ms if min_iqr_ms is not None else max(latency_baseline_ms * 0.1, 1.0)
        
        # Persistent state for per-instance tracking
        self.instance_ema = defaultdict(float)  # instance_id -> smoothed outlier score
        self.instance_count_history = []  # For detecting fleet changes

    def extract_features(self, latency_samples_ms: np.ndarray, error_count: int, request_count: int):
        """Original aggregate-only method (unchanged)."""
        latency_samples_ms = np.nan_to_num(
            latency_samples_ms, nan=self.latency_baseline_ms,
            posinf=self.latency_critical_ms, neginf=0.0
        )

        if len(latency_samples_ms) == 0 or request_count == 0:
            return {'theta_o': 0.0, 'p95_latency_ms': 0.0, 'error_rate': 0.0, 'median_latency_ms': 0.0}

        latency_metric = float(np.median(latency_samples_ms))
        error_rate = error_count / request_count

        latency_norm = np.clip(
            (latency_metric - self.latency_baseline_ms) /
            (self.latency_critical_ms - self.latency_baseline_ms),
            0.0, 1.0
        )
        error_norm = np.clip(error_rate / self.error_rate_critical, 0.0, 1.0)

        theta_o = np.clip(0.6 * max(latency_norm, error_norm) +
                           0.4 * ((latency_norm + error_norm) / 2.0), 0.0, 1.0)

        return {
            'theta_o': float(theta_o),
            'p95_latency_ms': float(np.percentile(latency_samples_ms, 95)),
            'median_latency_ms': latency_metric,
            'error_rate': float(error_rate),
        }

    def extract_features_per_instance(self, instance_latencies: dict, error_count: int, request_count: int):
        """
        Per-instance version for gray-failure detection.
        
        instance_latencies: {instance_id: np.array of latencies for that instance}
        """
        if not instance_latencies:
            return self.extract_features(np.array([]), error_count, request_count)

        # Flatten for aggregate metrics
        all_latencies = np.concatenate(list(instance_latencies.values()))
        aggregate = self.extract_features(all_latencies, error_count, request_count)
        
        # Per-instance outlier detection
        instance_medians = {inst: float(np.median(lat)) for inst, lat in instance_latencies.items() if len(lat) > 0}
        if len(instance_medians) < self.min_instances_for_outlier:
            return aggregate  # Fallback if too few instances

        fleet_medians = np.array(list(instance_medians.values()))
        fleet_median = float(np.median(fleet_medians))
        q1, q3 = np.percentile(fleet_medians, [25, 75])
        iqr = max(q3 - q1, self.min_iqr_ms)
        
        outlier_fraction = 0.0
        persistent_outliers = 0
        
        for inst_id, med in instance_medians.items():
            iqr_score = (med - fleet_median) / iqr
            # Update EMA for persistence
            self.instance_ema[inst_id] = (self.ema_alpha * max(iqr_score, 0) + 
                                        (1 - self.ema_alpha) * self.instance_ema[inst_id])
            
            if self.instance_ema[inst_id] > self.outlier_iqr_threshold:
                persistent_outliers += 1
        
        if len(instance_medians) > 0:
            outlier_fraction = persistent_outliers / len(instance_medians)

        # The EMA-smoothed IQR score for the WORST currently-tracked
        # instance -- this carries actual severity information that
        # outlier_fraction alone discards. A fraction only says "1 of how
        # many are flagged," which shrinks toward zero as fleet size grows
        # regardless of how badly that one instance is degraded. The worst
        # instance's own EMA score doesn't dilute with fleet size -- an
        # 18x-latency instance registers the same severity whether it's
        # 1-of-3 or 1-of-50.
        current_instance_ids = set(instance_medians.keys())
        worst_ema = max(
            (self.instance_ema[inst_id] for inst_id in current_instance_ids),
            default=0.0
        )
        # Scale relative to the configured threshold, same normalization
        # spirit as the rest of the adapter's other normalize_* methods.
        severity_boost = np.clip(
            (worst_ema - self.outlier_iqr_threshold) / max(self.outlier_iqr_threshold, 1e-6),
            0.0, 1.0
        )

        # Fraction-based boost -- kept, since MULTIPLE bad instances (a
        # widespread problem, not just one) should still count for more
        # than a single bad instance does. Fraction and severity capture
        # different things; neither alone is sufficient.
        outlier_boost = np.clip(outlier_fraction * 1.8, 0.0, 1.0)
        # Combine: the worse of severity vs. fraction dominates, but both
        # still contribute a little -- same blend pattern used elsewhere
        # in this adapter (worst-signal-dominates, not winner-take-all).
        combined_boost = np.clip(0.7 * max(severity_boost, outlier_boost) +
                                  0.3 * ((severity_boost + outlier_boost) / 2.0), 0.0, 1.0)
        theta_o_boosted = np.clip(aggregate['theta_o'] + 0.7 * combined_boost, 0.0, 1.0)
        
        result = aggregate.copy()
        result['theta_o'] = float(theta_o_boosted)
        result['outlier_fraction'] = float(outlier_fraction)
        result['persistent_outliers'] = persistent_outliers
        result['fleet_median_latency'] = fleet_median
        result['worst_instance_ema'] = float(worst_ema)
        result['severity_boost'] = float(severity_boost)

        # Prune EMA state to only the instances seen in THIS call. Real
        # deployments rotate instance IDs constantly (container restarts,
        # autoscaling events all create fresh IDs) -- without pruning,
        # self.instance_ema would grow without bound over the life of the
        # process, and a long-dead instance's stale score would never be
        # cleared even if its ID were ever reused. Since every call already
        # provides the full current fleet, anything not in this call's
        # instance_medians is no longer part of the fleet and its state
        # can be safely dropped.
        self.instance_ema = defaultdict(
            float,
            {inst_id: self.instance_ema[inst_id] for inst_id in current_instance_ids}
        )
        
        return result

    def normalize_context(self, requests_per_sec: float, capacity_rps: float, saturation_knee=0.80):
        if capacity_rps <= 0:
            return 0.0
        utilization = np.clip(requests_per_sec / capacity_rps, 0.0, 1.5)
        if utilization <= saturation_knee:
            return float(np.clip((utilization / saturation_knee) * 0.3, 0.0, 0.3))
        remaining = max(1.0 - saturation_knee, 1e-6)
        over_knee = np.clip((utilization - saturation_knee) / remaining, 0.0, 1.0)
        return float(np.clip(0.3 + 0.7 * over_knee, 0.0, 1.0))

    def normalize_system_state(self, memory_pct: float, disk_queue_depth: float, max_queue_depth=10.0):
        mem_norm = np.clip(memory_pct / 100.0, 0.0, 1.0)
        queue_norm = np.clip(disk_queue_depth / max_queue_depth, 0.0, 1.0)
        return float((mem_norm + queue_norm) / 2.0)
