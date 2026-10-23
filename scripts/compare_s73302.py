"""Analyze what was rejected in Seed 73302 under Candidate C."""
import json
from pathlib import Path

p_ref = Path("sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval/val_s73302/episode_93003/episode_93003/bridge_audit.json")
p_c = Path("sparc_performance_evaluation/results/raw/stage_b_candidates/Candidate_C_Surgical/val_s73302/episode_98103/episode_98103/bridge_audit.json")

data_ref = json.loads(p_ref.read_text(encoding="utf-8"))
data_c = json.loads(p_c.read_text(encoding="utf-8"))

ref_bytes = sum(e.get("executed_bytes", 0) for e in data_ref)
c_bytes = sum(e.get("executed_bytes", 0) for e in data_c)
print(f"Seed 73302 Total Executed Bytes: Ref = {ref_bytes} B | Candidate C = {c_bytes} B (Loss = {ref_bytes - c_bytes} B)")

ref_tasks = sum(e.get("n_executed", 0) for e in data_ref)
c_tasks = sum(e.get("n_executed", 0) for e in data_c)
print(f"Seed 73302 Total Executed Tasks: Ref = {ref_tasks} | Candidate C = {c_tasks}")
