"""Candidate D: Principled Spatio-Temporal Risk Calibration.

Tests if tuning the continuous physical parameters (r_thresh=0.15, cell_edge_dist=180m):
- Preserves high-value close-range offloading in safe conditions (Seeds 73302, 73303)
- Cuts high spatio-temporal interference bursts in congested conditions (Seeds 73300, 73301)
"""
import copy
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode
from experiments.wilab_closed_loop.train_sc_rlgmi import run_single_ep
from controllers.rl.sc_rlgmi.sc_rlgmi_core import canonical, improve
from controllers.rl.sc_rlgmi.sc_rlgmi_policy import SCRLGMIPolicy
from controllers.rl.scql.action_mask import CandidateEdge, candidate_from_dict
from controllers.rl.causal_safety_guard import causal_sidelink_risk


def make_risk_calibrated_guard(r_thresh: float = 0.15, edge_dist: float = 180.0, edge_bytes: int = 30000):
    def guard_fn(candidates, layer=None, r_thresh_in=r_thresh, r_comm=250.0, b_max=175000, startup_sec=0.025, cbr_val=0.04):
        now_sec = 0.0
        if layer is not None and hasattr(layer, "current_epoch"):
            now_sec = (10.0 * float(layer.current_epoch)) / 1000.0

        filtered = []
        rejected = 0
        has_far = False

        for c in candidates:
            l_kj = float(c.get("lat_pressure", c.get("L_kj", 0.0)))
            contact_feas = bool(c.get("contact_feasible", True))
            if l_kj > 1.0 or not contact_feas:
                continue

            b_k = int(c.get("payload_bytes", 0))
            sid = c.get("source_id")
            hid = c.get("helper_id")

            dist = float(c.get("_dist") or 0.0)
            if layer is not None and hasattr(layer, "source_kinematics") and sid in layer.source_kinematics:
                src_kin = layer.source_kinematics[sid]
                hlp = next((h for h in getattr(layer, "helpers", []) if getattr(h, "helper_id", None) == hid), None)
                if hlp is not None and getattr(hlp, "kinematics", None) is not None:
                    x_s = src_kin.x(now_sec)
                    y_s = src_kin.y
                    x_h = hlp.kinematics.x(now_sec)
                    y_h = hlp.kinematics.y
                    dist = math.hypot(x_s - x_h, y_s - y_h)

            risk = float(c.get("_risk") or causal_sidelink_risk(dist, b_k, r_comm=r_comm, b_max=float(b_max)))

            # Principled Risk Rules
            is_risky = (
                (risk > r_thresh) or
                (dist > edge_dist and b_k > edge_bytes) or
                (now_sec < startup_sec and b_k > 10000) or
                (b_k > 80000 and dist > 35.0)
            )
            if is_risky:
                rejected += 1
                continue

            c_copy = dict(c)
            c_copy["_dist"] = dist
            c_copy["_risk"] = risk
            if dist > 180.0:
                has_far = True
            filtered.append(c_copy)

        eff_b = 55000 if has_far else b_max
        return filtered, rejected, eff_b

    return guard_fn


class RiskCalibratedPolicy(SCRLGMIPolicy):
    def __init__(self, guard_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_fn = guard_fn

    def select_matching(
        self, candidates, b_max=175000, unique_sources=True, unique_helpers=True, layer=None, state=None, deterministic=True
    ):
        self.total_decisions += 1
        filtered, rej, eff_b = self.guard_fn(candidates, layer=layer, b_max=b_max)
        parsed_cands = [candidate_from_dict(c) for c in filtered]

        if not parsed_cands or eff_b <= 0:
            return [], {
                "n_pack": 0, "bytes": 0, "b_limit": eff_b, "selected_mode": "empty",
                "policy_mode": "sc_rlgmi", "predicted_gain": 0.0, "rejected_geometry": rej,
            }

        ref_matching = self._build_reference_matching(
            parsed_cands, b_max=float(eff_b), unique_sources=unique_sources, unique_helpers=unique_helpers
        )
        cur_epoch = int(getattr(layer, "current_epoch", 0) or 0)
        tot_epochs = int(getattr(layer, "controller_epochs", 100) or 100)

        d_state, edge_map = self._extract_decision_state(
            parsed_cands, cbr=0.04, b_safe=float(eff_b), current_epoch=cur_epoch, total_epochs=tot_epochs
        )
        oracle = self._make_feasibility_oracle(edge_map, b_safe=float(eff_b))

        selection = improve(
            d_state, baseline=ref_matching, check=oracle, model=self.advantage_model,
            depth=self.depth, beam=self.beam, min_predicted_gain=self.delta,
        )
        selected_matching = selection.selected
        pred_gain = float(selection.predicted_gain)
        searched_count = int(selection.searched)

        if selected_matching != ref_matching and pred_gain > self.delta:
            sel_mode = "sc_rlgmi_alternative"
            self.override_decisions += 1
        else:
            sel_mode = "reference_b4"
            selected_matching = ref_matching
            self.reference_decisions += 1

        selected_pack = [dict(edge_map[e].raw_dict) for e in selected_matching if e in edge_map]
        tot_bytes = sum(c.payload_bytes for c in [edge_map[e] for e in selected_matching if e in edge_map])

        meta = {
            "n_pack": len(selected_pack), "bytes": tot_bytes, "b_limit": eff_b,
            "selected_mode": sel_mode, "policy_mode": "sc_rlgmi", "predicted_gain": pred_gain,
            "searched_nodes": searched_count, "rejected_geometry": rej,
        }
        return selected_pack, meta

    def _sequential_match(self, candidates, b_max=175000, unique_sources=True, unique_helpers=True, layer=None, deterministic=True):
        return self.select_matching(
            candidates, b_max=b_max, unique_sources=unique_sources, unique_helpers=unique_helpers,
            layer=layer, deterministic=deterministic
        )


if __name__ == "__main__":
    ckpt_path = ROOT / "checkpoints" / "sc_rlgmi_best.pt"
    g_d = make_risk_calibrated_guard(r_thresh=0.15, edge_dist=180.0, edge_bytes=30000)
    policy = RiskCalibratedPolicy(g_d, b_max=175000, r_thresh=0.15)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        policy.advantage_model.load_state_dict(ckpt["advantage_model"])
    policy.eval()

    dev_seeds = [73300, 73301, 73302, 73303]
    out_base = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "stage_b_candidates" / "Candidate_D_RiskCalibrated"

    print("\n" + "=" * 60)
    print("EVALUATING CANDIDATE D (r_thresh=0.15) ON DEV SEEDS")
    print("=" * 60)

    results = []
    for i, s in enumerate(dev_seeds):
        eid = 98200 + i + 1
        comp, pv, audit = run_single_ep(policy, s, eid, out_base / f"val_s{s}", deterministic=True)
        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        mean_prr = float(np.mean(prrs)) if prrs else 1.0
        viols = sum(1 for p in prrs if p < 0.95)
        results.append({"seed": s, "V_done": comp, "P_viol": pv, "mean_prr": mean_prr, "n_viols": viols})
        print(f"  Seed {s}: V_done = {comp:6.2f} | P_viol = {pv*100:5.2f}% ({viols}/{len(valid)}) | PRR = {mean_prr:.4f}")

    mean_v = float(np.mean([r["V_done"] for r in results]))
    mean_pv = float(np.mean([r["P_viol"] for r in results]))
    print("-" * 60)
    print(f"SUMMARY Candidate D: Mean V_done = {mean_v:.2f} | Mean P_viol = {mean_pv*100:.2f}%")
    print(f"  Meets Preferred Safety (0.00%)?  {mean_pv == 0.0}")
    print(f"  Meets Mandatory Safety (<=2.0%)? {mean_pv <= 0.02}")
    print(f"  Beats Best Baseline B1 (39.20)?  {mean_v > 39.20} (Delta: {mean_v - 39.20:+.2f})")
    print("=" * 60 + "\n")
