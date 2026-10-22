"""Deep-dive quantitative reconciliation and utility-loss decomposition."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base_s6 = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step6_baseline_comparison"
base_s5 = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step5_matched_comparison"

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

print("=" * 110)
print("ACCURATE EPISODE-BY-EPISODE METRICS (RECONCILED FROM EPISODE_SUMMARY.JSON)")
print("=" * 110)

s6_data = {}
for eid, (m, s) in eids_s6.items():
    matches = list(base_s6.rglob(f"episode_{eid:04d}"))
    if not matches:
        continue
    d = matches[0]
    sm_file = d / "episode_summary.json"
    if not sm_file.is_file():
        continue
    sm = json.loads(sm_file.read_text(encoding="utf-8"))
    
    # Also load audit for CBR
    audit_file = d / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    cbr = [float(e.get("cbr", 0.0)) for e in valid]
    mean_cbr = float(np.mean(cbr)) if cbr else 0.0

    rec = {
        "eid": eid,
        "method": m,
        "seed": s,
        "V_done": sm.get("V_done", sm.get("return", 0.0)),
        "V_off": sm.get("V_off", 0.0),
        "R_value": sm.get("R_value", 0.0),
        "n_tasks": sm.get("n_tasks", 0),
        "n_completed": sm.get("n_deadline_valid_completed", 0),
        "n_incomplete": sm.get("n_incomplete_tasks", 0),
        "completion_rate": sm.get("task_completion_rate", 0.0),
        "executed_bytes": sm.get("executed_bytes_sum", 0),
        "delivered_bytes": sm.get("delivered_bytes_sum", 0),
        "safety_P_viol": sm.get("safety_P_viol", 0.0),
        "mean_prr": sm.get("safety_mean_prr_valid", 1.0),
        "mean_cbr": mean_cbr,
        "gov_interventions": sm.get("governor_intervention_epochs", 0),
        "path": str(d.relative_to(base_s6)),
    }
    s6_data[(m, s)] = rec
    print(f"Eid {eid} | {m:<16} | Seed {s} | V_done={rec['V_done']:6.2f} | V_off={rec['V_off']:6.2f} | "
          f"Tasks={rec['n_completed']:2d}/{rec['n_tasks']:2d} ({rec['completion_rate']*100:4.1f}%) | "
          f"Bytes={rec['executed_bytes']:6d} | PRR={rec['mean_prr']:.4f} | CBR={rec['mean_cbr']:.4f} | "
          f"P_viol={rec['safety_P_viol']:.4f}")

# Produce Reconciled Step 6 Summary Table
print("\n" + "=" * 105)
print(f"{'Method':<16} | {'Seed':<6} | {'V_done':<7} | {'V_off':<7} | {'Compl%':<7} | {'Tasks':<7} | {'Bytes':<8} | {'PRR':<7} | {'CBR':<7} | {'P_viol':<7}")
print("-" * 105)

methods = ["B4_ce_milp", "B5_prime", "PB_CRL_repaired"]
seeds = [73300, 73301, 73302, 73303]

for m in methods:
    for s in seeds:
        r = s6_data.get((m, s))
        if not r: continue
        print(f"{m:<16} | {s:<6d} | {r['V_done']:<7.2f} | {r['V_off']:<7.2f} | {r['completion_rate']*100:<6.1f}% | "
              f"{r['n_completed']:2d}/{r['n_tasks']:2d} | {r['executed_bytes']:<8d} | {r['mean_prr']:<7.4f} | {r['mean_cbr']:<7.4f} | {r['safety_P_viol']:<7.4f}")
    m_recs = [s6_data[(m, s)] for s in seeds if (m, s) in s6_data]
    mV = np.mean([r["V_done"] for r in m_recs])
    mVoff = np.mean([r["V_off"] for r in m_recs])
    mRate = np.mean([r["completion_rate"] for r in m_recs])
    mBytes = np.mean([r["executed_bytes"] for r in m_recs])
    mPRR = np.mean([r["mean_prr"] for r in m_recs])
    mCBR = np.mean([r["mean_cbr"] for r in m_recs])
    mPviol = np.mean([r["safety_P_viol"] for r in m_recs])
    print("-" * 105)
    print(f"{m + ' (MEAN)':<16} | {'ALL':<6} | {mV:<7.2f} | {mVoff:<7.2f} | {mRate*100:<6.1f}% | {'--':<7} | {mBytes:<8.0f} | {mPRR:<7.4f} | {mCBR:<7.4f} | {mPviol:<7.4f}")
    print("-" * 105)
