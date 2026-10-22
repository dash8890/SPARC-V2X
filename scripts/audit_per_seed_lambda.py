"""Inspect per-seed lambda and task admission across all 4 validation seeds."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base_s6 = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step6_baseline_comparison"

eids_s6 = {
    80001: ("B4_ce_milp", 73300),
    80002: ("B5_prime", 73300),
    80003: ("PB_CRL_repaired", 73300),
    80004: ("B4_ce_milp", 73301),
    80005: ("B5_prime", 73301),
    80006: ("PB_CRL_repaired", 73301),
    80007: ("B4_ce_milp", 73302),
    80008: ("B5_prime", 73302),
    80009: ("PB_CRL_repaired", 73302),
    80010: ("B4_ce_milp", 73303),
    80011: ("B5_prime", 73303),
    80012: ("PB_CRL_repaired", 73303),
}

print("=" * 115)
print("PER-SEED TASK SELECTION, LAMBDA, AND VALUE ADMISSION AUDIT")
print("=" * 115)

for eid, (m, s) in eids_s6.items():
    matches = list(base_s6.rglob(f"episode_{eid:04d}"))
    if not matches:
        continue
    d = matches[0]
    audit = json.loads((d / "bridge_audit.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "episode_summary.json").read_text(encoding="utf-8"))

    # Track lambdas when candidates were present
    lams_with_cands = []
    betas_with_cands = []
    tasks_admitted = set()
    cands_seen = set()

    for a in audit:
        cands = int(a.get("n_candidates", 0))
        pb = a.get("pb_crl") or {}
        lam = float(pb.get("lam", 0.0) or 0.0)
        beta = float(pb.get("beta_exec", 1.0) or 1.0)
        if cands > 0:
            lams_with_cands.append(lam)
            betas_with_cands.append(beta)

        sel = a.get("selected_assignments") or a.get("assignments") or []
        for item in sel:
            if "task_id" in item:
                tasks_admitted.add(int(item["task_id"]))

    mean_lam = np.mean(lams_with_cands) if lams_with_cands else 0.0
    mean_beta = np.mean(betas_with_cands) if betas_with_cands else 1.0

    print(f"Eid {eid} | {m:<16} | Seed {s} | V_done={summary.get('V_done'):5.2f}/{summary.get('V_off'):5.2f} | "
          f"Tasks Admitted={len(tasks_admitted):2d}/{summary.get('n_tasks'):2d} | "
          f"Compl={summary.get('n_deadline_valid_completed'):2d} | "
          f"Bytes={summary.get('executed_bytes_sum'):6d} | "
          f"Lam={mean_lam:5.2f} | Beta={mean_beta:5.3f} | P_viol={summary.get('safety_P_viol'):.4f}")
