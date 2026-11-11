"""One-command runner (Windows, macOS, Linux).
Usage (from this folder, with LSS-HSR-L/ next to it):
    python run_all.py                 # v2 weak front end, seeds 7 11 23
    python run_all.py --var v1 --seeds 7
Steps: build routes.pkl (once) -> segs.pkl (once) -> 5 folds per seed -> combine.
Each fold takes a few minutes; already-finished folds are skipped."""
import argparse, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
ap = argparse.ArgumentParser(); ap.add_argument("--var", default="v2", choices=["v1", "v2"])
ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23]); a = ap.parse_args()
py = sys.executable
if not os.path.isdir("LSS-HSR-L"): sys.exit("Put the LSS-HSR-L folder (with air_routes.json) in this folder first.")
if not os.path.exists("routes.pkl"): print("building routes.pkl ..."); subprocess.check_call([py, "build_routes.py"])
if not os.path.exists("segs.pkl"): print("building segs.pkl ..."); subprocess.check_call([py, "hsr_prep.py"])
pre = "v2_" if a.var == "v2" else ""
for s in a.seeds:
    for f in range(5):
        out = f"hsr_{pre}s{s}_f{f}.pkl"
        if os.path.exists(out): print("skip", out); continue
        env = dict(os.environ, SEED=str(s), FOLD=str(f)); env.pop("VAR", None)
        if a.var == "v2": env["VAR"] = "v2"
        print(f"seed {s} fold {f} ...", flush=True)
        with open(f"log_{pre}s{s}_f{f}.txt", "w") as lg: subprocess.check_call([py, "hsr_fold.py"], env=env, stdout=lg, stderr=subprocess.STDOUT)
env = dict(os.environ, PRE=pre)
subprocess.check_call([py, "combine_hsr.py"] + [str(s) for s in a.seeds], env=env)
print("done: results in hsr_results_" + pre + "_".join(map(str, a.seeds)) + ".txt (and .json)")
