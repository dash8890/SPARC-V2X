"""Detailed utility pipeline breakdown: Candidates -> Selected -> Transmitted -> Delivered -> Completed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base_s6 = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step6_baseline_comparison"

eids = {
    "B4_ce_milp": 80001,
    "B5_prime": 80002,
    "PB_CRL_repaired": 80003,
}

print("=" * 100)
print("UTILITY LOSS PIPELINE AUDIT (SEED 73300, DENSITY 35, 100 EPOCHS)")
print("=" * 100)

for name, eid in eids.items():
    matches = list(base_s6.rglob(f"episode_{eid:04d}"))
    if not matches:
        continue
    d = matches[0]
    audit = json.loads((d / "bridge_audit.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "episode_summary.json").read_text(encoding="utf-8"))

    # Extract per-epoch actions
    lambdas = []
    betas = []
    offered_bytes = []
    selected_tasks_total = set()
    delivered_tasks_total = set()
    n_cands_total = 0

    for ep, a in enumerate(audit):
        pb = a.get("pb_crl") or {}
        lam = float(pb.get("lam", 0.0) or 0.0)
        beta = float(pb.get("beta_exec", 1.0) or 1.0)
        lambdas.append(lam)
        betas.append(beta)
        b = int(a.get("computation_bytes_offered", 0))
        offered_bytes.append(b)

        sel = a.get("selected_assignments") or a.get("assignments") or []
        for s in sel:
            if "task_id" in s:
                selected_tasks_total.add(int(s["task_id"]))

        deliv = a.get("delivered_task_ids") or []
        if isinstance(deliv, (int, float)):
            deliv = [int(deliv)]
        for t in deliv:
            delivered_tasks_total.add(int(t))

        n_cands_total += int(a.get("n_candidates", 0))

    print(f"\n--- {name} (Eid {eid}) ---")
    print(f"  Offered Candidate Tasks in Horizon: {summary.get('n_tasks')} (Offered Value V_off = {summary.get('V_off'):.2f})")
    print(f"  Selected Tasks by Policy:           {len(selected_tasks_total)} / {summary.get('n_tasks')} ({len(selected_tasks_total)/summary.get('n_tasks')*100:.1f}%)")
    print(f"  Delivered Tasks over Radio:         {len(delivered_tasks_total)} / {len(selected_tasks_total)} ({len(delivered_tasks_total)/max(1, len(selected_tasks_total))*100:.1f}%)")
    print(f"  Deadline-Valid Completed Tasks:     {summary.get('n_deadline_valid_completed')} / {len(delivered_tasks_total)} ({summary.get('n_deadline_valid_completed')/max(1, len(delivered_tasks_total))*100:.1f}%)")
    print(f"  Deadline-Missed Incomplete Tasks:   {summary.get('n_incomplete_tasks')}")
    print(f"  Final Completed Utility V_done:     {summary.get('V_done'):.2f} (Value Completion Ratio R_val = {summary.get('R_value'):.3f})")
    print(f"  Total Computation Bytes Executed:   {summary.get('executed_bytes_sum'):,d} B")
    print(f"  Average lambda action:              {sum(lambdas)/len(lambdas):.2f} (min={min(lambdas):.2f}, max={max(lambdas):.2f})")
    print(f"  Average beta action:                {sum(betas)/len(betas):.3f} (min={min(betas):.3f}, max={max(betas):.3f})")
    print(f"  Safety P_viol (PRR < 0.95):         {summary.get('safety_P_viol'):.4f} ({summary.get('safety_P_viol')*100:.1f}%)")
    print(f"  Mean Radio PRR:                     {summary.get('safety_mean_prr_valid'):.4f}")
    print(f"  Governor Interventions:             {summary.get('governor_intervention_epochs')} epochs")
