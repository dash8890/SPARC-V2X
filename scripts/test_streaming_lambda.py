import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from experiments.wilab_closed_loop.task_layer import TaskLayer
from controllers.rl.pb_crl_packer import price_budget_pack

seeds = [73300, 73301, 73302, 73303]
for lam in [0.0, 3.0, 5.0, 8.0, 10.0]:
    v_list = []
    for s in seeds:
        t_seed = 186100 + (s - 73300)
        layer = TaskLayer(15, t_seed, tasks_per_source=12, arrival_mode="streaming")
        for ep in range(100):
            layer.set_epoch(ep)
            cands = layer.candidates(0.0, 0.0)
            if cands:
                pack = price_budget_pack(cands, lam=lam, beta_exec=1.0, b_max=175000, unique_sources=True)
                if pack:
                    delivered = [a["task_id"] for a in pack]
                    layer.apply_radio_outcomes(pack, delivered, 10.0)
        layer.set_epoch(100)
        v_list.append(layer.completed_value())
    print(f"lam={lam:4.1f} => Mean V = {sum(v_list)/len(v_list):.2f} | Per seed: {[round(v, 2) for v in v_list]}")
