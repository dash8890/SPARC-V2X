import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from experiments.wilab_closed_loop.task_layer import TaskLayer
from controllers.rl.structured_priority_policy import score_and_match

print("=" * 80)
print("EVALUATING STRUCTURED PRIORITY LOAD BALANCING UNDER INCREASING ARRIVAL LOAD")
print("=" * 80)

for tps in [10, 12, 14, 16]:
    b1_vals = []
    sp_vals = []
    diff_count_total = 0

    for s in [73300, 73301, 73302, 73303]:
        t_seed = 186100 + (s - 73300)
        l_b1 = TaskLayer(15, t_seed, tasks_per_source=tps, arrival_mode="streaming")
        l_sp = TaskLayer(15, t_seed, tasks_per_source=tps, arrival_mode="streaming")

        diffs = 0
        for ep in range(100):
            # B1
            l_b1.set_epoch(ep)
            c_b1 = l_b1.candidates(0.04)
            if c_b1:
                c_b1_sorted = sorted(c_b1, key=lambda c: float(c["task_value"]) / max(1, int(c["payload_bytes"])), reverse=True)
                p_b1 = []
                used_s = set()
                tot_b = 0
                for c in c_b1_sorted:
                    if c["source_id"] not in used_s and tot_b + c["payload_bytes"] <= 175000:
                        p_b1.append(c)
                        used_s.add(c["source_id"])
                        tot_b += c["payload_bytes"]
                if p_b1:
                    l_b1.apply_radio_outcomes(p_b1, [p["task_id"] for p in p_b1], 10.0)

            # SP
            l_sp.set_epoch(ep)
            c_sp = l_sp.candidates(0.04)
            if c_sp:
                p_sp, _ = score_and_match(c_sp, weights=[3.5, 2.5, 3.0, 0.5], beta_throttle=0.95, unique_sources=True)
                if p_sp:
                    l_sp.apply_radio_outcomes(p_sp, [p["task_id"] for p in p_sp], 10.0)

            t1 = [(p["task_id"], p["helper_id"]) for p in (p_b1 if c_b1 else [])]
            t2 = [(p["task_id"], p["helper_id"]) for p in (p_sp if c_sp else [])]
            if t1 != t2:
                diffs += 1

        l_b1.set_epoch(100)
        l_sp.set_epoch(100)
        b1_vals.append(l_b1.completed_value())
        sp_vals.append(l_sp.completed_value())
        diff_count_total += diffs

    deltas = [s - b for s, b in zip(sp_vals, b1_vals)]
    print(f"TPS={tps:2d} => B1 Mean V = {np.mean(b1_vals):.2f} | SP Mean V = {np.mean(sp_vals):.2f} | Delta = {np.mean(sp_vals)-np.mean(b1_vals):+.2f} | Per-seed deltas: {[round(d, 2) for d in deltas]} | Divergent Epochs = {diff_count_total}")
