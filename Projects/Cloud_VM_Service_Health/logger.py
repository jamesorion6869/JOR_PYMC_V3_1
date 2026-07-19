import json
from datetime import datetime
from pathlib import Path

class FusionLogger:
    def __init__(self, log_file="maintenance_log.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_state(self, sop: float, nhp: float, alert_status: bool, metadata: dict = None):
        if metadata is None:
            metadata = {}

        log_entry = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),

            # Core fusion outputs
            "sop_fused": round(float(sop), 4),            # fused SOP (0–1)
            "nhp_posterior": round(float(nhp), 4),        # posterior probability of non‑healthy

            # Engine state
            "alert_active": bool(alert_status),           # hysteresis-controlled alert state

            # Additional context
            "metadata": metadata
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            return True
        except Exception as e:
            print(f"Logging error: {e}")
            return False
