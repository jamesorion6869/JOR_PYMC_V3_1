JOR 4.0 — complete package (latest)

LIVE DEMO
  python run_demo.py
    Empty map: "Click Live stream to begin the mission"
    Live stream starts the run; saves map_frames_baseline.jsonl

VIGNETTES
  python vignettes.py run shock
  python vignettes.py run tempo
  python vignettes.py run early_warning

SCENARIOS (CLI)
  python maneuver_sim_interactive.py --scenario baseline --seed 42 --n-units 6 --commander expert
  python maneuver_sim_interactive.py --scenario deep_maneuver --seed 42 --n-units 6 --commander expert
  python maneuver_sim_interactive.py --scenario urban --seed 7 --n-units 6 --commander expert
  python maneuver_sim_interactive.py --scenario logistics_constrained --seed 7 --n-units 6 --commander expert
  python maneuver_sim_interactive.py --scenario high_enemy_pressure --seed 7 --n-units 6 --commander expert

  python maneuver_sim_interactive.py --scenario baseline --seed 42 --n-units 6 --map-log my_run.jsonl
  python maneuver_sim_interactive.py --serve --scenario baseline --seed 42 --n-units 6

TESTS
  python test_suite.py
  python test_regression.py
