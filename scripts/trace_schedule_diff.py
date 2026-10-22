import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_repaired_mdp_workloads import RepairedTaskLayer
from controllers.rl.pb_crl_packer import price_budget_pack

print("=" * 80)
print("TRACING B4 vs PB-CRL SCHEDULE DIFFERENCE ON SEED 73301")
print("=" * 80)

# Trace B4
l_b4 = RepairedTaskLayer(15, 186101, n_tasks_per_source=10, arrival_mode="poisson")
b4_sched = {}
for ep in range(100):
    l_b4.set_epoch(ep)
    c = l_b4.candidates()
    if c:
        p = price_budget_pack(c, lam=0.0, beta_exec=1.0, b_max=175000, unique_sources=True)
        if p:
            b4_sched[ep] = p
            l_b4.apply_assignments(p)
l_b4.set_epoch(100)
v_b4 = l_b4.completed_value()

# Trace PB-CRL (lam=5.0)
l_pb = RepairedTaskLayer(15, 186101, n_tasks_per_source=10, arrival_mode="poisson")
pb_sched = {}
for ep in range(100):
    l_pb.set_epoch(ep)
    c = l_pb.candidates()
    if c:
        p = price_budget_pack(c, lam=5.0, beta_exec=1.0, b_max=175000, unique_sources=True)
        if p:
            pb_sched[ep] = p
            l_pb.apply_assignments(p)
l_pb.set_epoch(100)
v_pb = l_pb.completed_value()

print(f"B4 Completed Value:     {v_b4:.2f}")
print(f"PB-CRL Completed Value: {v_pb:.2f} (Delta = {v_pb - v_b4:+.2f})")

# Find the first epoch where decisions diverged
all_eps = sorted(set(list(b4_sched.keys()) + list(pb_sched.keys())))
for ep in all_eps:
    p_b4 = b4_sched.get(ep, [])
    p_pb = pb_sched.get(ep, [])
    t_b4 = [(x["task_id"], x["helper_id"]) for x in p_b4]
    t_pb = [(x["task_id"], x["helper_id"]) for x in p_pb]
    if t_b4 != t_pb:
        print(f"\n--- DIVERGENCE AT EPOCH {ep} ---")
        print(f"  B4 scheduled:     {t_b4}")
        print(f"  PB-CRL scheduled: {t_pb}")
        # Inspect tasks at this epoch
        for x in p_b4:
            tid = x["task_id"]
            task = next(t for t in l_b4.tasks if t["task_id"] == tid)
            print(f"    Task {tid}: val={task['value']:.2f}, bytes={task['payload_bytes']}, dl={task['deadline_ms']:.1f}ms, cyc={task['cycles']/1e6:.1f}M")
