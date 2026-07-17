import numpy as np

class VibrationAdapter:
    def __init__(self, sample_rate=10000, safe_rms_g=10.0):
        self.fs = sample_rate
        self.safe_rms_g = safe_rms_g
        
    def extract_features(self, raw_buffer: np.ndarray):
        raw_buffer = np.nan_to_num(raw_buffer, nan=0.0, posinf=self.safe_rms_g, neginf=0.0)
        
        if len(raw_buffer) == 0:
            return {'theta_o': 0.0, 'rms_g': 0.0, 'peak_freq_hz': 0.0}
            
        rms_val = np.sqrt(np.mean(raw_buffer ** 2))
        fft_vals = np.abs(np.fft.rfft(raw_buffer))
        freqs = np.fft.rfftfreq(len(raw_buffer), 1/self.fs)
        
        if len(fft_vals) > 1:
            peak_idx = np.argmax(fft_vals[1:]) + 1
            peak_freq = freqs[peak_idx]
        else:
            peak_freq = 0.0
        
        # Normalized vibration level (0.0 = silent, 1.0 = at rated safe limit)
        theta_o = np.clip(rms_val / self.safe_rms_g, 0.0, 1.0)
        
        return {
            'theta_o': float(theta_o), 
            'rms_g': float(rms_val), 
            'peak_freq_hz': float(peak_freq)
        }

    def normalize_context(self, machine_load: float, ambient_temp: float):
        load_norm = np.clip(machine_load / 100.0, 0.0, 1.0)
        temp_norm = np.clip(ambient_temp / 80.0, 0.0, 1.0)
        return float((load_norm + temp_norm) / 2.0)
