import numpy as np

class VibrationAdapter:
    """
    theta_o (vibration severity) is derived from ISO 20816-3 Group 2
    evaluation criteria (medium power, rigid foundation -- e.g. a standard
    industrial motor or pump bolted to a concrete pad).

    ISO 20816-1 requires BOTH criteria be assessed together:

    Criterion I -- absolute zone boundaries (RMS velocity, mm/s):
        Zone A (0        - zone_ab):  new/reconditioned machine
        Zone B (zone_ab  - zone_bc):  acceptable for unrestricted long-term operation
        Zone C (zone_bc  - zone_cd):  not suitable for continuous long-term
                                       operation; plan corrective action
        Zone D (> zone_cd):           immediate action needed

    Criterion II -- change from a machine's own established baseline:
        A change exceeding 25% of the zone B/C boundary, even if the
        absolute reading stays within the same zone, indicates a developing
        problem worth investigating. This is the standard's own example:
        a fan jumping from 1.5 to 2.5 mm/s stays in Zone B under Criterion I
        alone, but the 1.0 mm/s change (vs. a 25%-of-B/C threshold) is
        flagged as significant under Criterion II. Criterion I alone would
        miss this; that's the entire point of assessing both together.

    Defaults (zone_ab=1.4, zone_bc=2.8, zone_cd=7.1) are the commonly cited
    Group 2 boundaries. Swap these for a different machine group's published
    boundaries if modeling different equipment.

    IMPORTANT UNIT NOTE: ISO 20816-3 zones are defined in RMS *velocity*
    (mm/s), not RMS acceleration. This adapter's raw input signal
    (raw_buffer) is treated as acceleration in g's and converted to an
    approximate RMS velocity before zone lookup. That conversion
    (a_rms / omega, where omega = 2*pi*peak_frequency) is only exact for a
    single dominant sinusoidal frequency component; it's a reasonable
    approximation for a synthetic single-frequency demo signal, but a real
    deployment should measure velocity directly (or properly integrate a
    real broadband waveform) rather than relying on this shortcut.

    theta_c (operational/thermal stress) is grounded in NEMA MG-1's
    published thermal and load limits for a general-purpose Class F
    insulation motor (the overwhelmingly common insulation class on modern
    three-phase AC motors):

    Load severity: NEMA defines a motor's nameplate service factor (SF) as
    the load multiplier a motor can carry beyond its rated load. A common
    general-purpose SF is 1.15 -- i.e. up to 115% of rated load is within
    the motor's designed (if derated-life) operating envelope; above that
    is genuine overload the motor wasn't designed to sustain.

    Ambient severity: NEMA MG-1's reference ambient is 40 degrees C, with a
    105 degree C allowable winding temperature rise for Class F insulation
    at 1.0 SF. NEMA's own derating rule: for every degree the ambient
    exceeds 40C, reduce the allowable temperature rise by the same amount
    (a 1:1 reduction in thermal headroom). Ambient at or below 40C is
    treated as using none of that headroom.

    Defaults (rated_ambient_c=40.0, rise_budget_c=105.0, service_factor=1.15)
    are the commonly cited Class F / general-purpose-motor values. Swap
    these for a specific motor's actual nameplate insulation class and
    service factor if modeling different equipment.

    SIMPLIFICATION NOTE: this treats load severity and ambient severity as
    two independent linear bands combined by a worst-dominates blend,
    matching the combining pattern used elsewhere in the JOR adapter
    family. Real winding temperature rise scales roughly with load squared
    (I^2R losses) and interacts with ambient non-linearly through the
    motor's actual thermal design -- this is a reasonable severity proxy
    grounded in NEMA's published limits, not a full thermal model.
    """

    G_TO_MS2 = 9.80665  # standard gravity, m/s^2 per g

    def __init__(self, sample_rate=10000, zone_ab=1.4, zone_bc=2.8, zone_cd=7.1,
                 baseline_window=20, criterion_ii_boost_weight=0.15,
                 machine_group_label="ISO 20816-3 Group 2 (medium power, rigid foundation)",
                 rated_ambient_c=40.0, rise_budget_c=105.0, service_factor=1.15,
                 insulation_class_label="NEMA MG-1 Class F, 1.0 SF (105C rise / 40C ambient); 1.15 service factor"):
        self.fs = sample_rate
        self.zone_ab = zone_ab
        self.zone_bc = zone_bc
        self.zone_cd = zone_cd
        self.machine_group_label = machine_group_label

        # --- Criterion II state ---
        self.baseline_window = baseline_window
        self.criterion_ii_threshold_mm_s = 0.25 * zone_bc  # ISO's own "25% of B/C" rule
        self.criterion_ii_boost_weight = criterion_ii_boost_weight
        self._baseline_buffer = []
        self.baseline_velocity_mm_s = None
        self.baseline_established = False

        # --- NEMA MG-1 thermal/load parameters (theta_c) ---
        self.rated_ambient_c = rated_ambient_c
        self.rise_budget_c = rise_budget_c
        self.service_factor = service_factor
        self.insulation_class_label = insulation_class_label

    def reset_baseline(self):
        """
        Clears the established Criterion II baseline, e.g. after maintenance
        or a machine rebuild, so the adapter re-learns "normal" from scratch
        over the next `baseline_window` calls instead of comparing against
        a now-outdated baseline.
        """
        self._baseline_buffer = []
        self.baseline_velocity_mm_s = None
        self.baseline_established = False

    def get_state(self):
        """
        Returns the adapter's persistable Criterion II state. Pairs with
        load_state() so the baseline survives a process restart -- without
        this, the engine's own baseline_sop would correctly persist across
        restarts while the adapter's baseline_velocity_mm_s silently reset
        to "unestablished" and re-learned from scratch, leaving the two
        halves of the same fusion pipeline out of sync with each other.
        """
        return {
            "baseline_velocity_mm_s": self.baseline_velocity_mm_s,
            "baseline_established": self.baseline_established,
            "_baseline_buffer": list(self._baseline_buffer),
        }

    def load_state(self, state):
        """Restores Criterion II baseline state saved by get_state()."""
        if state is None:
            return
        self.baseline_velocity_mm_s = state.get("baseline_velocity_mm_s", self.baseline_velocity_mm_s)
        self.baseline_established = state.get("baseline_established", self.baseline_established)
        self._baseline_buffer = list(state.get("_baseline_buffer", self._baseline_buffer))

    def _accel_g_rms_to_velocity_mm_s(self, accel_rms_g, freq_hz):
        freq_hz = max(freq_hz, 1.0)  # guard against divide-by-zero / near-DC estimates
        accel_rms_ms2 = accel_rms_g * self.G_TO_MS2
        omega = 2.0 * np.pi * freq_hz
        velocity_rms_ms = accel_rms_ms2 / omega
        return velocity_rms_ms * 1000.0  # m/s -> mm/s

    def _criterion_i_theta(self, v_rms_mm_s):
        """
        Maps RMS velocity (mm/s) through the ISO zone boundaries onto a
        continuous 0-1 severity, preserving the zones' relative jumps
        rather than a single straight line:
            Zone A: 0.00 - 0.25   Zone B: 0.25 - 0.50
            Zone C: 0.50 - 0.85   Zone D: 0.85 - 1.00 (saturates at 2x C/D)
        """
        ab, bc, cd = self.zone_ab, self.zone_bc, self.zone_cd

        if v_rms_mm_s <= ab:
            frac = v_rms_mm_s / ab if ab > 0 else 0.0
            return 0.00 + frac * 0.25
        elif v_rms_mm_s <= bc:
            frac = (v_rms_mm_s - ab) / (bc - ab)
            return 0.25 + frac * 0.25
        elif v_rms_mm_s <= cd:
            frac = (v_rms_mm_s - bc) / (cd - bc)
            return 0.50 + frac * 0.35
        else:
            saturation_point = cd * 2.0
            frac = min((v_rms_mm_s - cd) / (saturation_point - cd), 1.0)
            return 0.85 + frac * 0.15

    def _zone_label(self, v_rms_mm_s):
        if v_rms_mm_s <= self.zone_ab:
            return "A"
        elif v_rms_mm_s <= self.zone_bc:
            return "B"
        elif v_rms_mm_s <= self.zone_cd:
            return "C"
        else:
            return "D"

    def _apply_criterion_ii(self, v_rms_mm_s):
        """
        Compares the current reading against the established baseline.
        Returns (delta_mm_s, flagged, boost) -- boost is a smooth 0 to
        criterion_ii_boost_weight addition once the 25%-of-B/C threshold is
        exceeded, scaling up as the excess grows (capped at 2x threshold),
        so a borderline crossing nudges theta_o gently while a severe
        same-zone jump nudges it harder.
        """
        if not self.baseline_established:
            return 0.0, False, 0.0

        delta = abs(v_rms_mm_s - self.baseline_velocity_mm_s)
        threshold = self.criterion_ii_threshold_mm_s
        flagged = delta > threshold

        if not flagged:
            return delta, False, 0.0

        excess_ratio = np.clip((delta - threshold) / threshold, 0.0, 1.0)
        boost = self.criterion_ii_boost_weight * excess_ratio
        return delta, True, float(boost)

    def extract_features(self, raw_buffer: np.ndarray):
        raw_buffer = np.nan_to_num(raw_buffer, nan=0.0, posinf=self.zone_cd * 10, neginf=0.0)

        if len(raw_buffer) == 0:
            return {
                'theta_o': 0.0,
                'rms_g': 0.0,
                'peak_freq_hz': 0.0,
                'velocity_rms_mm_s': 0.0,
                'iso_zone': 'A',
                'baseline_velocity_mm_s': self.baseline_velocity_mm_s,
                'criterion_ii_delta_mm_s': 0.0,
                'criterion_ii_threshold_mm_s': self.criterion_ii_threshold_mm_s,
                'criterion_ii_flag': False,
            }

        rms_val = np.sqrt(np.mean(raw_buffer ** 2))
        fft_vals = np.abs(np.fft.rfft(raw_buffer))
        freqs = np.fft.rfftfreq(len(raw_buffer), 1 / self.fs)

        if len(fft_vals) > 1:
            peak_idx = np.argmax(fft_vals[1:]) + 1
            peak_freq = freqs[peak_idx]
        else:
            peak_freq = 0.0

        velocity_rms_mm_s = self._accel_g_rms_to_velocity_mm_s(rms_val, peak_freq)
        zone = self._zone_label(velocity_rms_mm_s)
        theta_o_criterion_i = self._criterion_i_theta(velocity_rms_mm_s)

        # --- Establish or apply Criterion II baseline ---
        if not self.baseline_established:
            self._baseline_buffer.append(velocity_rms_mm_s)
            if len(self._baseline_buffer) >= self.baseline_window:
                self.baseline_velocity_mm_s = float(np.mean(self._baseline_buffer))
                self.baseline_established = True
            delta, flagged, boost = 0.0, False, 0.0
        else:
            delta, flagged, boost = self._apply_criterion_ii(velocity_rms_mm_s)

        theta_o = float(np.clip(theta_o_criterion_i + boost, 0.0, 1.0))

        return {
            'theta_o': theta_o,
            'rms_g': float(rms_val),
            'peak_freq_hz': float(peak_freq),
            'velocity_rms_mm_s': float(velocity_rms_mm_s),
            'iso_zone': zone,
            'baseline_velocity_mm_s': self.baseline_velocity_mm_s,
            'criterion_ii_delta_mm_s': float(delta),
            'criterion_ii_threshold_mm_s': self.criterion_ii_threshold_mm_s,
            'criterion_ii_flag': bool(flagged),
        }

    def _load_severity(self, machine_load_pct):
        """
        machine_load_pct: percent of the motor's RATED load (100 = nameplate
        rated load, not "percent of some arbitrary max").

        0   - 100%  (0.0 - 1.0 ratio):  0.00 - 0.50 severity (normal operating range)
        100 - 115%  (up to service_factor): 0.50 - 0.85 severity (within nameplate
                                             SF, but derates insulation life --
                                             not intended for indefinite operation)
        > 115%:                             0.85 - 1.00 severity, saturating at
                                             1.30x rated (30% overload)
        """
        load_ratio = machine_load_pct / 100.0
        sf = self.service_factor

        if load_ratio <= 1.0:
            frac = np.clip(load_ratio, 0.0, 1.0)
            return 0.00 + frac * 0.50
        elif load_ratio <= sf:
            frac = (load_ratio - 1.0) / (sf - 1.0)
            return 0.50 + frac * 0.35
        else:
            saturation_point = 1.30
            frac = np.clip((load_ratio - sf) / max(saturation_point - sf, 1e-6), 0.0, 1.0)
            return 0.85 + frac * 0.15

    def _ambient_severity(self, ambient_temp_c):
        """
        NEMA MG-1's own derating rule: allowable temperature rise shrinks
        1:1 for every degree C ambient exceeds the 40C reference. Modeled
        here as how much of the rise BUDGET that excess ambient has already
        consumed, saturating at 1.0 once the excess alone would exceed the
        entire rise budget (i.e. zero thermal headroom left for load-driven
        heating at all).
        """
        excess = max(ambient_temp_c - self.rated_ambient_c, 0.0)
        return float(np.clip(excess / self.rise_budget_c, 0.0, 1.0))

    def normalize_context(self, machine_load: float, ambient_temp: float):
        """
        machine_load: percent of rated load (100 = nameplate rated load).
        ambient_temp: ambient temperature in degrees C.

        Combines load severity and ambient severity with the same
        worst-dominates blend used elsewhere in the JOR adapter family:
        the more severe of the two drives most of the result, with a
        smaller contribution from their average so a machine that's
        moderately stressed on BOTH fronts still registers higher than one
        stressed on only one front.
        """
        load_sev = self._load_severity(machine_load)
        ambient_sev = self._ambient_severity(ambient_temp)
        combined = 0.7 * max(load_sev, ambient_sev) + 0.3 * ((load_sev + ambient_sev) / 2.0)
        return float(np.clip(combined, 0.0, 1.0))
