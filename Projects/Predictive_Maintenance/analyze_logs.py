import json

log_file = "maintenance_log.json"

with open(log_file, "r") as f:
    for line in f:
        entry = json.loads(line)
        meta = entry.get("metadata", {})

        # nhp_posterior is what FusionLogger.log_state() actually writes.
        nhp = entry.get("nhp_posterior", entry.get("nhp"))
        alert = entry.get("alert_active")

        # Richer fields added when theta_o was grounded in ISO 20816-3
        # (Criterion I zone + Criterion II rate-of-change). Older log files
        # written before that change won't have these -- .get(...) with
        # None defaults means this script still runs fine against them,
        # it just won't have anything extra to show for those lines.
        phase = meta.get("phase")
        step = meta.get("step")
        zone = meta.get("iso_zone")
        v_rms = meta.get("velocity_rms_mm_s")
        crit_ii_flag = meta.get("criterion_ii_flag")

        line_out = f"Time: {entry['timestamp']} | NHP: {nhp} | Alert: {alert}"

        if phase is not None:
            line_out += f" | Phase: {phase:<11} Step: {step}"
        if zone is not None:
            line_out += f" | Zone: {zone}"
        if v_rms is not None:
            line_out += f" | v_rms: {v_rms:6.3f} mm/s"
        if crit_ii_flag:
            line_out += " | CritII: FLAG"

        print(line_out)
