import json

log_file = "maintenance_log.json"

with open(log_file, "r") as f:
    for line in f:
        entry = json.loads(line)
        # nhp_posterior is what FusionLogger.log_state() actually writes.
        nhp = entry.get("nhp_posterior", entry.get("nhp"))
        print(f"Time: {entry['timestamp']} | NHP: {nhp} | Alert: {entry['alert_active']}")