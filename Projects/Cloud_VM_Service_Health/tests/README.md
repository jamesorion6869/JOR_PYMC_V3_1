# Tests: Cloud VM Service Health

Simulation test harnesses run against `ServiceHealthAdapter` (and, for some, `JOREngine`) for the Cloud VM Service Health domain. All tests use synthetic latency, error, and traffic data generated with NumPy.

## Tests

### Per-instance detection

- **`baseline_scaling_test.py`**: Checks that the IQR floor rule (`max(baseline * 0.1, 1.0)`) works at baselines of 5, 50, 500, and 5000 ms. Each baseline runs a healthy fleet, a gray failure on one instance, sustained degradation, and recovery.
- **`iqr_floor_scale_test.py`**: Runs the IQR floor at 5, 50, and 500 ms baselines under noise-only, one bad instance, two bad instances, a rotating bad instance, heterogeneous fleet baselines, and autoscaling fleet sizes (3, 5, 10, 20).
- **`explicit_recovery_test.py`**: Degrades one instance, then removes it from the fleet. Checks that the per-instance EMA resets and no sticky alerts remain after full recovery.
- **`service_instant_churn_test.py`**: Checks detection when instances appear and disappear mid-run: fleet shrink and expansion, churn during degradation and recovery, heterogeneous baselines, and a rotating bad instance with a shrinking fleet.

### Stability and false alerts

- **`long_duration_stability_test.py`**: Runs 2000 healthy cycles with rare small latency spikes. Checks that `theta_o`, the outlier fraction, and the worst-instance EMA stay low, with no alert lock or drift. Prints **PASS/FAIL**.
- **`per_instance_calibration_mismatch_test.py`**: Calibrates on a very low-variance fleet, then operates on a higher but still healthy variance. Counts false alerts across fleet sizes 4 to 20 and four seeds. No bad instance is ever introduced.
- **`service_noise_repair_test.py`**: Feeds the adapter heavy-tailed latency noise (occasional slow requests during healthy operation) and checks for alert flapping. Also confirms a genuine incident is still caught. Currently runs the baseline check against the unmodified adapter only.
- **`service_noisy_test.py`**: End-to-end noisy and bursty run: calibration, ordinary noise, two genuine incidents (a bad deploy and a memory leak), and recovery. Reports ALERT/Normal transitions and flags rapid alternation (state runs of 2 steps or fewer).

### Context and fusion behavior

- **`service_highload_context_test.py`**: Tests context normalization (`theta_c`) near saturation, at full capacity, under 150% overload, oscillating around the saturation knee, with overload plus gray failures, and during recovery. Prints **PASS/FAIL**.
- **`service_isolation_test.py`**: Degrades each evidence channel alone (`theta_o`, `theta_c`, `theta_s`) while the other two stay healthy. Checks whether the fusion weights behave sensibly, especially for a silent resource leak with no user-facing symptoms.

## Running

The tests import `engine.py` and `service_adapter.py` from the parent folder, so run them from the `Cloud_VM_Service_Health` folder with that folder on the Python path:

macOS / Linux:

```
PYTHONPATH=. python tests/<test_name>.py
```

Windows (PowerShell):

```
$env:PYTHONPATH = "."
python tests\<test_name>.py
```

Requirements: Python 3 and NumPy.

Most tests print metrics (`fleet_median`, `outlier_fraction`, `persistent_outliers`, `worst_instance_ema`, `severity_boost`, `theta_o`) for review rather than asserting a result. Only the long-duration and high-load tests report an explicit PASS/FAIL.
