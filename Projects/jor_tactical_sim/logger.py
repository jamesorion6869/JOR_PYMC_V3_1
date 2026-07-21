import json
from datetime import datetime
from pathlib import Path

class FusionLogger:
    def __init__(self, log_file="cloud_vm_log.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # NEW: in‑memory buffer for mission summary
        self.buffer = []

    def log_state(self, sop: float, nhp: float, alert_status: bool, metadata: dict = None):
        if metadata is None:
            metadata = {}

        log_entry = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),

            # Core fusion outputs
            "sop": round(float(sop), 4),
            "nhp": round(float(nhp), 4),

            # Engine state
            "alert": bool(alert_status),

            # Additional context
            "metadata": metadata
        }

        # NEW: store in memory
        self.buffer.append(log_entry)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            return True
        except Exception as e:
            print(f"Logging error: {e}")
            return False
