"""
Real-time local service health test.

Runs several tiny local HTTP "instances" on your own machine (localhost,
different ports) and a client that makes REAL requests against them,
measuring GENUINE latency -- not simulated numbers. Feeds that real data
into the actual engine.py + service_adapter.py, live, the same way
real_time_sensor_test.py fed real CPU% into the vibration engine.

No cloud account, no external service, no extra installs needed --
everything here is Python standard library.

Usage:
    python real_time_service_test.py
    Press Ctrl+C to stop.
"""

import http.server
import socketserver
import threading
import time
import urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from engine import JOREngine
from service_adapter import ServiceHealthAdapter
from logger import FusionLogger

NUM_INSTANCES = 5
BASE_PORT = 9001
BAD_INSTANCE_INDEX = 2
BAD_AFTER_SECONDS = 25
RECOVER_AFTER_SECONDS = 45
REQUESTS_PER_INSTANCE_PER_STEP = 8
STEP_INTERVAL_SECONDS = 1.0

# Shared state so handler threads know whether THEIR instance should be slow.
instance_delay_ms = {i: 0.0 for i in range(NUM_INSTANCES)}


def make_handler(instance_idx):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            delay = instance_delay_ms.get(instance_idx, 0.0)
            if delay > 0:
                time.sleep(delay / 1000.0)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    return Handler


def start_instance(instance_idx):
    port = BASE_PORT + instance_idx
    server = socketserver.ThreadingTCPServer(("127.0.0.1", port), make_handler(instance_idx))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def measure_instance(port, n_requests=REQUESTS_PER_INSTANCE_PER_STEP, timeout=2.0):
    url = f"http://127.0.0.1:{port}/"

    def one_request():
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                resp.read()
            return (time.perf_counter() - start) * 1000.0, False
        except Exception:
            return timeout * 1000.0, True

    with ThreadPoolExecutor(max_workers=n_requests) as pool:
        results = list(pool.map(lambda _: one_request(), range(n_requests)))

    latencies = np.array([r[0] for r in results])
    errors = sum(1 for r in results if r[1])
    return latencies, errors


def measure_all_instances(num_instances, base_port, executor):
    futures = {
        i: executor.submit(measure_instance, base_port + i)
        for i in range(num_instances)
    }
    instance_latencies = {}
    total_errors = 0
    total_requests = 0

    for i, future in futures.items():
        lat, err = future.result()
        instance_latencies[f"inst_{i}"] = lat
        total_errors += err
        total_requests += len(lat)

    return instance_latencies, total_errors, total_requests


def main():
    print(f"=== JOR 4.0 -- REAL-TIME Local Service Health Test ===")
    print(f"Starting {NUM_INSTANCES} local instances on ports "
          f"{BASE_PORT}-{BASE_PORT + NUM_INSTANCES - 1}...")

    servers = []
    for i in range(NUM_INSTANCES):
        server, port = start_instance(i)
        servers.append(server)
        print(f"  instance_{i} listening on 127.0.0.1:{port}")

    print(f"\nInstance {BAD_INSTANCE_INDEX} will go slow after "
          f"{BAD_AFTER_SECONDS}s, recover after {RECOVER_AFTER_SECONDS}s.")
    print("Press Ctrl+C to stop.\n")

    engine = JOREngine(
        prior_nh=0.05,
        steepness=12.0,
        upper_th=0.55,
        lower_th=0.45,
        retention=0.70,
        calibration_steps=20
    )

    adapter = ServiceHealthAdapter()
    logger = FusionLogger("real_time_service_log.json")

    start_time = time.time()
    step = 0

    try:
        with ThreadPoolExecutor(max_workers=NUM_INSTANCES) as executor:
            while True:
                elapsed = time.time() - start_time

                # Inject degradation / recovery
                if BAD_AFTER_SECONDS <= elapsed < RECOVER_AFTER_SECONDS:
                    instance_delay_ms[BAD_INSTANCE_INDEX] = 900.0
                else:
                    instance_delay_ms[BAD_INSTANCE_INDEX] = 0.0

                instance_latencies, total_errors, total_requests = measure_all_instances(
                    NUM_INSTANCES, BASE_PORT, executor)

                obs = adapter.extract_features_per_instance(
                    instance_latencies, total_errors, total_requests)

                theta_c = 0.2
                theta_s = 0.2

                sop, nhp, alert = engine.fusion_step(theta_s, theta_c, obs['theta_o'])

                # --- FIXED PHASE LOGIC ---
                if not engine.is_calibrated:
                    phase = "Calibrating"
                elif BAD_AFTER_SECONDS <= elapsed < RECOVER_AFTER_SECONDS:
                    phase = "DEGRADED"
                elif obs.get("persistent_outliers", 0) > 0:
                    phase = "RECOVERY"
                else:
                    phase = "HEALTHY"

                outliers = obs.get('persistent_outliers', 0)
                median = obs.get('median_latency_ms', obs.get('p95_latency_ms', 0))

                status = "ALERT" if alert else "Normal"

                print(f"t={elapsed:5.1f}s  [{phase:10s}]  "
                      f"theta_o={obs['theta_o']:.3f}  "
                      f"outliers={outliers}  median={median:6.1f}ms  "
                      f"NHP={nhp:.3f}  {status}")

                logger.log_state(sop, nhp, alert, metadata={
                    "step": step,
                    "elapsed_s": round(elapsed, 1),
                    "phase": phase,
                    "persistent_outliers": outliers,
                })

                step += 1
                time.sleep(STEP_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nStopped. Shutting down local instances...")
        for server in servers:
            server.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
