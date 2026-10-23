"""Detailed dump of the 5 violating windows in SC-RLGMI."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for seed, viols_idx in [(73300, [12, 22, 27, 57]), (73301, [13])]:
    eval_dir = ROOT / f"sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval/val_s{seed}"
    audit_files = list(eval_dir.glob("**/bridge_audit.json"))
    p = audit_files[0]
    data = json.loads(p.read_text(encoding="utf-8"))
    valid = [e for e in data if e.get("cbr_valid") and e.get("delay_valid")]
    
    print(f"\n=======================================================")
    print(f"SEED {seed}: Detailed Analysis of Violating Windows")
    print(f"=======================================================")
    for v_idx in viols_idx:
        v = valid[v_idx]
        prr = float(v.get("window_prr", v.get("prr", 1.0)))
        cbr = float(v.get("cbr", 0.0))
        p95 = v.get("p95_delay_ms")
        rx_cum = v.get("safety_rx_cumulative")
        err_cum = v.get("safety_error_cumulative")
        num_sel = v.get("num_selected")
        prop_b = v.get("proposed_bytes")
        exec_b = v.get("executed_bytes")
        prop_p = v.get("proposed_packets")
        exec_p = v.get("executed_packets")
        sel_assign = v.get("selected_assignments")
        gov_interv = v.get("governor_intervened")
        gov_reason = v.get("governor_reason")
        enq_b = v.get("wilab_enqueued_bytes")
        pend_p = v.get("wilab_pending_packets")
        
        # Also check preceding window (v_idx - 1)
        prev_v = valid[v_idx - 1] if v_idx > 0 else None
        prev_prr = float(prev_v.get("window_prr", prev_v.get("prr", 1.0))) if prev_v else None
        prev_exec_b = prev_v.get("executed_bytes") if prev_v else None
        prev_sel = prev_v.get("num_selected") if prev_v else None
        
        print(f"\n--- Window Index: {v_idx} ---")
        print(f"  PRR: {prr:.4f} | P95 Delay: {p95} ms | CBR: {cbr:.4f}")
        print(f"  Safety RX: {rx_cum} | Safety Err: {err_cum}")
        print(f"  Selected: {num_sel} tasks | Proposed Bytes: {prop_b} | Executed Bytes: {exec_b}")
        print(f"  Selected Assignments: {sel_assign}")
        print(f"  WiLab Enqueued Bytes: {enq_b} | Pending Packets: {pend_p}")
        print(f"  Governor Intervened: {gov_interv} ({gov_reason})")
        print(f"  Preceding Window (idx {v_idx-1}): PRR={prev_prr}, Executed Bytes={prev_exec_b}, Selected Tasks={prev_sel}")
