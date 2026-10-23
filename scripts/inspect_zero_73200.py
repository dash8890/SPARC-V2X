"""Inspect zero offload on seed 73200."""
import json
from pathlib import Path

p = Path("sparc_performance_evaluation/results/raw/test_zero_offload_73200/episode_99990/bridge_audit.json")
data = json.loads(p.read_text(encoding="utf-8"))
valid = [e for e in data if e.get("cbr_valid") and e.get("delay_valid")]
viols = [e for e in valid if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95]
print(f"Seed 73200 ZERO OFFLOAD: total={len(data)}, valid={len(valid)}, viols={len(viols)} ({len(viols)/len(valid)*100:.2f}%)")
for v in viols[:5]:
    idx = valid.index(v)
    prr = float(v.get("window_prr", v.get("prr", 1.0)))
    print(f"  Window {idx:02d}: PRR={prr:.4f}")
