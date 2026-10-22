# SUMO mobility assets

The canonical SPARC-V2X repository uses SUMO for reproducible mobility traces. SUMO generates vehicle trajectories; the Python Mode-2 simulator remains responsible for live radio outcomes.

Tracked assets:

- `mobility/sumo_runner.py`: invokes SUMO and writes deterministic traces.
- `mobility/trace_loader.py`: loads FCD traces for replay.
- `configs/sumo_highway.yaml` and `configs/sumo_manhattan.yaml`: reproducible scenario definitions.
- `mobility_output/*_seed42.fcd.xml`: reference traces included for immediate replay.

To regenerate traces, set `SUMO_HOME` and run `python experiments/generate_sumo_mobility.py --scenario highway --seed 42` (or `manhattan`). Generated auxiliary network/route files are reproducible and remain untracked.
