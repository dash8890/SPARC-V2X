"""Check structured_priority_meta in bridge audit files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for seed in [73300, 73301, 73302, 73303]:
    eval_dir = ROOT / f"sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval/val_s{seed}"
    audit_files = list(eval_dir.glob("**/bridge_audit.json"))
    p = audit_files[0]
    data = json.loads(p.read_text(encoding="utf-8"))
    metas = [e.get("structured_priority_meta") for e in data if e.get("structured_priority_meta")]
    modes = [m.get("selected_mode") for m in metas]
    overrides = sum(1 for m in modes if m == "sc_rlgmi_alternative")
    ref_kept = sum(1 for m in modes if m == "reference_b4")
    empty = sum(1 for m in modes if m == "empty")
    print(f"Seed {seed}: total_metas={len(metas)}, overrides={overrides}, ref_kept={ref_kept}, empty={empty}")
    if metas and overrides > 0:
        sample_override = next(m for m in metas if m.get("selected_mode") == "sc_rlgmi_alternative")
        print(f"   Sample override: {sample_override}")
