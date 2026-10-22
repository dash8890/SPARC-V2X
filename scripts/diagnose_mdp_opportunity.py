import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

import task_layer
from controllers.rl.pb_crl_packer import price_budget_pack
from controllers.rl.hq_spo_baselines import build_baseline_pack

print("=" * 80)
print("AUDIT: TASK GENERATION & MDP STATE TRANSITION ANALYSIS (DENSITY 15)")
print("=" * 80)

density = 15
task_seed = 186100
layer = task_layer.TaskLayer(density, task_seed, tasks_per_source=4, spread_arrivals=True)

print(f"Total tasks in TaskLayer: {len(layer.tasks)}")
print(f"Total helpers: {len(layer.helpers)}")
print(f"Tasks per source: 4 | Sources: {len(set(t.source_id for t in layer.tasks))}")

active_epochs = []
for ep in range(100):
    layer.set_epoch(ep)
    cands = layer.candidates(0.04, ep / 100.0)
    if cands:
        active_epochs.append((ep, len(cands), len(set(c["task_id"] for c in cands))))

print(f"\nActive Epochs with Candidates: {len(active_epochs)} out of 100 epochs:")
for ep, n_cands, n_tasks in active_epochs:
    print(f"  Epoch {ep:2d}: {n_cands:2d} candidate pairs across {n_tasks} unique tasks")
