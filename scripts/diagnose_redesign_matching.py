import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from experiments.wilab_closed_loop.task_layer import TaskLayer
from controllers.rl.structured_priority_policy import score_and_match

# Run B1 vs Structured_Priority_Aware
l_b1 = TaskLayer(15, 186100, tasks_per_source=12, arrival_mode="streaming")
l_sp = TaskLayer(15, 186100, tasks_per_source=12, arrival_mode="streaming")

b1_picks = {}
sp_picks = {}

for ep in range(100):
    l_b1.set_epoch(ep)
    cands_b1 = l_b1.candidates(0.04)
    if cands_b1:
        cands_sorted = sorted(cands_b1, key=lambda c: float(c["task_value"]) / max(1, int(c["payload_bytes"])), reverse=True)
        pack_b1 = []
        used_s = set()
        tot_b = 0
        for c in cands_sorted:
            if c["source_id"] not in used_s and tot_b + c["payload_bytes"] <= 175000:
                pack_b1.append(c)
                used_s.add(c["source_id"])
                tot_b += c["payload_bytes"]
        if pack_b1:
            b1_picks[ep] = [(p["task_id"], p["helper_id"]) for p in pack_b1]
            l_b1.apply_radio_outcomes(pack_b1, [p["task_id"] for p in pack_b1], 10.0)

    l_sp.set_epoch(ep)
    cands_sp = l_sp.candidates(0.04)
    if cands_sp:
        pack_sp, meta = score_and_match(cands_sp, weights=[3.5, 2.5, 3.0, 0.5], beta_throttle=0.95, unique_sources=True)
        if pack_sp:
            sp_picks[ep] = [(p["task_id"], p["helper_id"]) for p in pack_sp]
            l_sp.apply_radio_outcomes(pack_sp, [p["task_id"] for p in pack_sp], 10.0)

print(f"B1 Completed Value: {l_b1.completed_value():.2f}")
print(f"SP Completed Value: {l_sp.completed_value():.2f}")

all_eps = sorted(set(list(b1_picks.keys()) + list(sp_picks.keys())))
diffs = 0
for ep in all_eps:
    p1 = b1_picks.get(ep, [])
    p2 = sp_picks.get(ep, [])
    if p1 != p2:
        diffs += 1
        print(f"Epoch {ep}: B1 picked {p1} | SP picked {p2}")

print(f"Total epochs with different assignments: {diffs}")
