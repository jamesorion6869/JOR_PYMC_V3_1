"""
Soak test for the JOR C-UAS sensing-assurance layer (synthetic data).

A: long busy-but-healthy runs inside the calibrated load range, with jitter
   on every input -> expect no false alarms
B: load sweep beyond the calibrated range -> where does the state change?
C: gray failure / feed loss injected into a busy healthy soak -> still detected?
D: bursty load spikes -> transient behaviour
E: slow drift (single source, sensor health, common-mode, platform, noise),
   with and without the deadband   (python soak_test.py --drift-only)
"""
import sys
import numpy as np
from engine import JOREngine
from cuas_adapter import CUASHealthAdapter

N_SRC = 8


def make(deadband=0.02):
    eng = JOREngine(prior_nh=0.05, steepness=12.0, upper_th=0.55, lower_th=0.45,
                    retention=0.70, calibration_steps=20, delta_deadband=deadband)
    return eng, CUASHealthAdapter()


def one_step(eng, ad, rng, vol, bad=None, stale=(), bad_cont=0.42, drift=None):
    dr = drift or {}
    vol = float(np.clip(vol, 0, 150))
    coop = vol * rng.uniform(0.4, 0.8)
    rf = float(np.clip(0.08 + 0.0057 * vol + rng.normal(0, 0.02), 0, 1))
    noise = float(np.clip(rng.normal(0.12 + 0.002 * max(vol - 30, 0) + dr.get('noise_add', 0.0), 0.03), 0, 1))
    fa = float(rng.uniform(0.0, 0.04))
    n_per = max(4, int(vol / N_SRC * rng.uniform(0.6, 1.4)) + 4)

    tracks = {}
    for i in range(N_SRC):
        sid = f"src_{i}"
        if i in stale or rng.random() < 0.03:         # lost feed or an idle sector
            c, f = np.array([]), np.array([])
        elif i == bad:
            c = np.clip(rng.normal(bad_cont, 0.06, n_per), 0.05, 1)
            f = np.clip(rng.normal(0.55, 0.08, n_per), 0.05, 1)
        else:
            off = rng.normal(0, 0.01)
            mean = 0.93 + off + dr.get('fleet_cont', 0.0)
            if i == dr.get('src_idx'):
                mean = dr['src_cont'] + off
            c = np.clip(rng.normal(mean, 0.03, n_per), 0.05, 1)
            f = np.clip(rng.normal(0.88 - 0.7 * max(0.93 - mean, 0.0), 0.04, n_per), 0.05, 1)
        tracks[sid] = {"continuities": c, "confidences": f}
    beats = {f"src_{i}": (6.0 if i in stale else float(rng.uniform(0.1, 0.9))) for i in range(N_SRC)}
    health = {f"src_{i}": float(np.clip(rng.normal(0.96, 0.02), 0, 1)) for i in range(N_SRC)}
    if bad is not None:
        health[f"src_{bad}"] = 0.55
    if 'health_idx' in dr:
        health[f"src_{dr['health_idx']}"] = dr['health_val']

    obs = ad.extract_features_per_source(tracks, noise, fa, heartbeat_ages=beats)
    th_c = ad.normalize_context(int(coop), int(vol), rf)
    th_s = ad.normalize_system_state(health, float(np.clip(rng.normal(dr.get('platform', 0.97), 0.01), 0, 1)),
                                     float(abs(rng.normal(0.05, 0.03))), heartbeat_ages=beats)
    eng.fusion_step(th_s, th_c, obs["theta_o"])
    return eng.state_band(), eng.p_final


def calibrate(eng, ad, rng):
    for i in range(20):                        # same ramp the demo uses
        one_step(eng, ad, rng, 16 + (36 - 16) * i / 19)
    assert eng.is_calibrated


def ar1_load(rng, mean, sd, n, rho=0.9, lo=None, hi=None):
    x = np.empty(n); x[0] = mean
    for k in range(1, n):
        x[k] = mean + rho * (x[k - 1] - mean) + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
    if lo is not None:
        x = np.clip(x, lo, hi)
    return x


def run(seeds, n, load_fn, **kw):
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        eng, ad = make(); calibrate(eng, ad, rng)
        loads = load_fn(rng, n)
        bands, nhp = [], []
        for v in loads:
            b, p = one_step(eng, ad, rng, v, **kw)
            bands.append(b); nhp.append(p)
        out.append((bands, nhp))
    return out


def frac(res, band):
    tot = sum(len(b) for b, _ in res)
    return sum(sum(1 for x in b if x == band) for b, _ in res) / tot


DRIFTS = {
    # name: function(frac) -> drift dict, frac in [0,1] of the way to the final value
    "one source's continuity 0.93 -> 0.45":
        lambda f: dict(src_idx=3, src_cont=0.93 + (0.45 - 0.93) * f),
    "one sensor's self-reported health 0.96 -> 0.40":
        lambda f: dict(health_idx=3, health_val=0.96 + (0.40 - 0.96) * f),
    "ALL sources' continuity 0.93 -> 0.68 (common mode)":
        lambda f: dict(fleet_cont=(0.68 - 0.93) * f),
    "platform health 0.97 -> 0.75":
        lambda f: dict(platform=0.97 + (0.75 - 0.97) * f),
    "noise floor +0.35":
        lambda f: dict(noise_add=0.35 * f),
}


def drift_run(seed, deadband, fn, D, onset=200, hold=200):
    rng = np.random.default_rng(9000 + seed)
    eng, ad = make(deadband); calibrate(eng, ad, rng)
    loads = ar1_load(rng, 28, 4, onset + D + hold, lo=16, hi=36)
    first_deg = first_unr = None
    pre = 0
    for t, v in enumerate(loads):
        frac = 0.0 if t < onset else min((t - onset) / D, 1.0)
        b, _ = one_step(eng, ad, rng, v, drift=fn(frac) if t >= onset else None)
        if t < onset:
            pre += b in ("DEGRADED", "UNRELIABLE")
        else:
            if first_deg is None and b in ("DEGRADED", "UNRELIABLE"):
                first_deg = frac
            if first_unr is None and b == "UNRELIABLE":
                first_unr = frac
    return first_deg, first_unr, pre


def drift_scenarios(S=15):
    print("E. Slow drift into a busy healthy soak (mean load 28), onset at step 200, then held")
    print("   'at x%' = how far along the drift (to its final value) when the state first changed\n")
    for name, fn in DRIFTS.items():
        print(f"   {name}")
        for D in (150, 600):
            for db in (0.02, 0.0):
                r = [drift_run(s, db, fn, D) for s in range(S)]
                deg = [x[0] for x in r if x[0] is not None]
                unr = [x[1] for x in r if x[1] is not None]
                pre = sum(x[2] for x in r)
                m = lambda v: f"{np.median(v) * 100:3.0f}%" if v else "  - "
                print(f"     drift over {D:3d} steps, deadband {db:.2f}:  DEGRADED {len(deg):2d}/{S} (median at {m(deg)})"
                      f"   UNRELIABLE {len(unr):2d}/{S} (median at {m(unr)})   pre-onset false steps {pre}")
        print()


def main():
    if "--drift-only" in sys.argv:
        drift_scenarios()
        return
    S, N = 20, 1000
    print(f"Soak test: {S} seeds x {N} steps per scenario (synthetic data)\n")

    # A. Busy but healthy, inside calibrated range
    print("A. Healthy soak, load inside calibrated range (16-36 tracks)")
    for mean, sd in [(20, 4), (26, 5), (32, 3)]:
        res = run(S, N, lambda r, n: ar1_load(r, mean, sd, n, lo=16, hi=36))
        mx = max(max(p) for _, p in res)
        print(f"   mean {mean:2d}: UNRELIABLE {frac(res,'UNRELIABLE')*100:5.2f}%  "
              f"DEGRADED {frac(res,'DEGRADED')*100:5.2f}%  max NHP {mx:.3f}  "
              f"seeds with any alert {sum('UNRELIABLE' in b for b,_ in res)}/{S}")

    # B. Beyond the calibrated range
    print("\nB. Load sweep above the calibrated range (healthy sensors, steady load)")
    print("   tracks  UNRELIABLE  DEGRADED  mean NHP")
    for vol in (36, 40, 45, 50, 60, 70, 95):
        res = run(S, 300, lambda r, n, v=vol: np.full(n, v) + r.normal(0, 1.5, n))
        mean_nhp = np.mean([np.mean(p) for _, p in res])
        print(f"   {vol:5d}   {frac(res,'UNRELIABLE')*100:8.1f}%  {frac(res,'DEGRADED')*100:7.1f}%  {mean_nhp:.3f}")

    # C. Detection under busy-healthy background
    print("\nC. Fault injection into a busy healthy soak (mean load 28), onset at step 200")
    for name, kw in [("gray failure src_3", dict(bad=3)), ("feed loss src_1,2,5", dict(stale=(1, 2, 5)))]:
        lat, pre_alerts, detected = [], 0, 0
        for s in range(S):
            rng = np.random.default_rng(5000 + s)
            eng, ad = make(); calibrate(eng, ad, rng)
            loads = ar1_load(rng, 28, 4, 400, lo=16, hi=36)
            first = None
            for t, v in enumerate(loads):
                b, _ = one_step(eng, ad, rng, v, **(kw if t >= 200 else {}))
                if b == "UNRELIABLE":
                    if t < 200:
                        pre_alerts += 1
                    elif first is None:
                        first = t - 200
            if first is not None:
                detected += 1; lat.append(first)
        print(f"   {name:20s} detected {detected}/{S}  steps to UNRELIABLE: "
              f"median {np.median(lat):.0f}, max {max(lat)}  | false UNRELIABLE steps before onset: {pre_alerts}")

    # D. Bursty load
    print("\nD. Bursty load: base 24, 5-step spikes to 45 every ~60 steps")
    def bursty(rng, n):
        x = ar1_load(rng, 24, 3, n, lo=16, hi=36)
        for t0 in range(30, n, 60):
            x[t0:t0 + 5] = 45
        return x
    res = run(S, N, bursty)
    print(f"   UNRELIABLE {frac(res,'UNRELIABLE')*100:.2f}%  DEGRADED {frac(res,'DEGRADED')*100:.2f}%  "
          f"max NHP {max(max(p) for _, p in res):.3f}")

    print()
    drift_scenarios()


if __name__ == "__main__":
    main()
