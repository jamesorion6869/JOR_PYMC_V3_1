import json
from datetime import datetime
from pathlib import Path


class FusionLogger:
    """Simple append-only JSONL logger for C-UAS fusion state.

    The file is appended to across runs, so each record carries a `run_id`
    (when one is given) that lets readers separate runs."""

    def __init__(self, log_file="cuas_airspace_log.jsonl", run_id=None):
        self.run_id = run_id
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_state(self, sop: float, nhp: float, alert_status: bool, metadata: dict = None):
        if metadata is None:
            metadata = {}

        log_entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "sop_fused": round(float(sop), 4),
            "nhp_posterior": round(float(nhp), 4),
            "alert_active": bool(alert_status),
            "metadata": metadata,
        }
        if self.run_id is not None:
            log_entry["run_id"] = self.run_id

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            return True
        except Exception as e:
            print(f"Logging error: {e}")
            return False
