"""Concise trace of all 5 violating windows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for seed, target_windows in [(73300, [12, 22, 27, 57]), (73301, [13])]:
    eval_dir = ROOT / f"sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval/val_s{seed}"
    audit_files = list(eval_dir.glob("**/bridge_audit.json"))
    p = audit_files[0]
    data = json.loads(p.read_text(encoding="utf-8"))
    
    print(f"\n=======================================================")
    print(f"SEED {seed}: Violating Windows Sequence")
    print(f"=======================================================")
    
    for tw in target_windows:
        start_idx = max(0, tw - 3)
        end_idx = min(len(data), tw + 3)
        print(f"\n--- Around Window {tw} ---")
        for i in range(start_idx, end_idx):
            entry = data[i]
            prr = float(entry.get("window_prr", entry.get("prr", 1.0)))
            cbr = float(entry.get("cbr", 0.0))
            p95 = entry.get("p95_delay_ms")
            rx_cum = entry.get("safety_rx_cumulative")
            err_cum = entry.get("safety_error_cumulative")
            prop_b = entry.get("proposed_bytes", 0)
            exec_b = entry.get("executed_bytes", 0)
            pend_p = entry.get("wilab_pending_packets", 0)
            sel_len = len(entry.get("selected_assignments", []))
            marker = " *** VIOLATION ***" if prr < 0.95 else ""
            print(f"  [{i:02d}]{marker} PRR={prr:.4f} CBR={cbr:.4f} P95={p95}ms | Exec_B={exec_b:5d} | Pend_P={pend_p:2d} | Err={err_cum:2d} Rx={rx_cum:3d} | Sel={sel_len}")
