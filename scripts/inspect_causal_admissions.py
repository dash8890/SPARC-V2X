"""Check whether the 5 violating decisions came from reference pi_B or an advantage override."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Let's inspect the selected_assignments at the causal epochs:
# Seed 73300: epoch 10, epoch 20, epoch 24, epoch 54
# Seed 73301: epoch 12

eval_dir_73300 = ROOT / "sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval/val_s73300"
audit_file_73300 = list(eval_dir_73300.glob("**/bridge_audit.json"))[0]
data_73300 = json.loads(audit_file_73300.read_text(encoding="utf-8"))

eval_dir_73301 = ROOT / "sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval/val_s73301"
audit_file_73301 = list(eval_dir_73301.glob("**/bridge_audit.json"))[0]
data_73301 = json.loads(audit_file_73301.read_text(encoding="utf-8"))

print("=== SEED 73300 CAUSAL ADMISSIONS ===")
for ep in [10, 20, 24, 54]:
    entry = data_73300[ep]
    print(f"\nEpoch {ep}: Admitted {entry.get('executed_bytes')} bytes, {entry.get('executed_packets')} packets")
    for a in entry.get("selected_assignments", []):
        print(f"  Task {a.get('task_id')} -> Helper {a.get('helper_id')}: {a.get('payload_bytes')} B, val={a.get('task_value'):.2f}, dist={a.get('_dist'):.1f}m, risk={a.get('_risk'):.4f}")

print("\n=== SEED 73301 CAUSAL ADMISSIONS ===")
for ep in [12]:
    entry = data_73301[ep]
    print(f"\nEpoch {ep}: Admitted {entry.get('executed_bytes')} bytes, {entry.get('executed_packets')} packets")
    for a in entry.get("selected_assignments", []):
        print(f"  Task {a.get('task_id')} -> Helper {a.get('helper_id')}: {a.get('payload_bytes')} B, val={a.get('task_value'):.2f}, dist={a.get('_dist'):.1f}m, risk={a.get('_risk'):.4f}")
