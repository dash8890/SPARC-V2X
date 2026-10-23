"""Evaluate C_0 (frozen reference) on training seeds 73200 and 73201."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from experiments.wilab_closed_loop.train_sc_rlgmi import run_single_ep
from controllers.rl.sc_rlgmi.sc_rlgmi_policy import SCRLGMIPolicy

ckpt_path = ROOT / "checkpoints" / "sc_rlgmi_best.pt"
policy = SCRLGMIPolicy(b_max=175000, r_thresh=0.20, delta=0.0, depth=2, beam=8)
if ckpt_path.is_file():
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.advantage_model.load_state_dict(ckpt["advantage_model"])
    print(f"Loaded advantage model from {ckpt_path}")

out_base = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "test_train_seeds"

for s, eid in [(73200, 96000), (73201, 96001)]:
    c_val, pv, audit = run_single_ep(policy, s, eid, out_base / f"val_s{s}", deterministic=True)
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    viols = sum(1 for p in prrs if p < 0.95)
    print(f"Train Seed {s}: V_done = {c_val:.2f} | P_viol = {pv*100:.2f}% ({viols}/{len(valid)} windows)")
