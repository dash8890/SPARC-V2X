"""Inspect Seed 73200 under C1 to see why violations happened."""
import json
from pathlib import Path

p = Path("sparc_performance_evaluation/results/raw/stage_b_candidates/C1_train/val_s73200/episode_97101/episode_97101/bridge_audit.json")
data = json.loads(p.read_text(encoding="utf-8"))
valid = [e for e in data if e.get("cbr_valid") and e.get("delay_valid")]
viols = [e for e in valid if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95]

print(f"Seed 73200 C1: total={len(data)}, valid={len(valid)}, viols={len(viols)}")
print("First 5 violating windows:")
for v in viols[:5]:
    idx = valid.index(v)
    prr = float(v.get("window_prr", v.get("prr", 1.0)))
    cbr = float(v.get("cbr", 0.0))
    p95 = v.get("p95_delay_ms")
    pend_p = v.get("wilab_pending_packets", 0)
    exec_b = v.get("executed_bytes", 0)
    sel = v.get("selected_assignments", [])
    print(f"  Window {idx:02d}: PRR={prr:.4f}, CBR={cbr:.4f}, Exec_B={exec_b}, Pend_P={pend_p}, Sel={len(sel)}")
