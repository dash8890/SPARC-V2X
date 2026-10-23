"""Inspect entry at epoch 10 in seed 73300."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval/val_s73300/episode_93001/episode_93001/bridge_audit.json"
data = json.loads(p.read_text(encoding="utf-8"))
print("Entry 10:")
print(json.dumps(data[10], indent=2))
