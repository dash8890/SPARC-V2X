"""Stage B: Test candidate guard configurations on training seeds (73200, 73201).

Investigates candidate interventions:
- C_1: Priority 1 (Accounting & CBR wiring + single burst cap 50kB)
- C_2: C_1 + Strengthened cell-edge cut (dist > 150m, bk > 20kB) + far epoch cap 35kB
- C_3: C_2 + Epoch aggregate byte cap 75kB
- C_4: C_3 + CBR-adaptive dynamic throttling
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode
from experiments.wilab_closed_loop.train_sc_rlgmi import run_single_ep
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.sc_rlgmi.sc_rlgmi_core import (
    canonical,
    improve,
    AdvantageModel,
)
from controllers.rl.sc_rlgmi.sc_rlgmi_policy import SCRLGMIPolicy
from controllers.rl.scql.action_mask import CandidateEdge, candidate_from_dict
from controllers.rl.causal_safety_guard import causal_sidelink_risk


def make_custom_guard(
    max_single_burst: int = 50000,
    monopolize_dist: float = 35.0,
    monopolize_bytes: int = 38000,
    cell_edge_dist: float = 150.0,
    cell_edge_bytes: int = 20000,
    far_dist: float = 180.0,
    far_epoch_cap: int = 35000,
    epoch_byte_cap: int = 75000,
    cbr_adaptive: bool = True,
    r_thresh: float = 0.20,
):
    """Factory creating parameterized causal safety guard filter."""
    def guard_fn(
        candidates: List[Dict[str, Any]],
        layer: Any = None,
        r_thresh_in: float = r_thresh,
        r_comm: float = 250.0,
        b_max: int = 175000,
        startup_sec: float = 0.025,
        cbr_val: float = 0.04,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
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

            # Parameterized Safety Rules
            is_risky = (
                (risk > r_thresh_in) or
                (b_k > max_single_burst) or  # Hard single-task burst cap
                (b_k > monopolize_bytes and dist > monopolize_dist) or  # Close/medium monopolization
                (dist > cell_edge_dist and b_k > cell_edge_bytes) or   # Cell edge cut
                (now_sec < startup_sec and b_k > 10000)
            )
            if is_risky:
                rejected += 1
                continue

            c_copy = dict(c)
            c_copy["_dist"] = dist
            c_copy["_risk"] = risk
            if dist > far_dist:
                has_far = True
            filtered.append(c_copy)

        eff_b = far_epoch_cap if has_far else min(b_max, epoch_byte_cap)
        if cbr_adaptive and cbr_val > 0.05:
            # Linear throttle under elevated CBR
            throttle = max(0.40, 1.0 - 5.0 * (cbr_val - 0.05))
            eff_b = int(float(eff_b) * throttle)

        return filtered, rejected, eff_b

    return guard_fn


class CandidateSCRLGMIPolicy(SCRLGMIPolicy):
    """SC-RLGMI Policy with pluggable guard function and CBR wiring."""

    def __init__(self, guard_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_fn = guard_fn
        self._current_cbr = 0.04

    def update_history(self, cbr: float, **kwargs):
        super().update_history(cbr, **kwargs)
        self._current_cbr = float(cbr)

    def select_matching(
        self,
        candidates: List[Dict[str, Any]],
        b_max: int = 175000,
        unique_sources: bool = True,
        unique_helpers: bool = True,
        layer: Any = None,
        state: Any = None,
        deterministic: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.total_decisions += 1
        cbr_val = self._current_cbr

        # 1. Apply pluggable causal safety guard
        filtered, rej, eff_b = self.guard_fn(
            candidates,
            layer=layer,
            r_thresh_in=self.r_thresh,
            b_max=b_max,
            cbr_val=cbr_val,
        )
        parsed_cands = [candidate_from_dict(c) for c in filtered]

        if not parsed_cands or eff_b <= 0:
            return [], {
                "n_pack": 0,
                "bytes": 0,
                "b_limit": eff_b,
                "selected_mode": "empty",
                "policy_mode": "sc_rlgmi",
                "predicted_gain": 0.0,
                "rejected_geometry": rej,
            }

        # 2. Build complete feasible reference matching M_B
        ref_matching = self._build_reference_matching(
            parsed_cands, b_max=float(eff_b), unique_sources=unique_sources, unique_helpers=unique_helpers
        )

        cur_epoch = int(getattr(layer, "current_epoch", 0) or 0)
        tot_epochs = int(getattr(layer, "controller_epochs", 100) or 100)

        # 3. Construct DecisionState and Feasibility Oracle
        d_state, edge_map = self._extract_decision_state(
            parsed_cands, cbr=cbr_val, b_safe=float(eff_b), current_epoch=cur_epoch, total_epochs=tot_epochs
        )
        oracle = self._make_feasibility_oracle(edge_map, b_safe=float(eff_b))

        # 4. Search
        selection = improve(
            d_state,
            baseline=ref_matching,
            check=oracle,
            model=self.advantage_model,
            depth=self.depth,
            beam=self.beam,
            min_predicted_gain=self.delta,
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
            "n_pack": len(selected_pack),
            "bytes": tot_bytes,
            "b_limit": eff_b,
            "selected_mode": sel_mode,
            "policy_mode": "sc_rlgmi",
            "predicted_gain": pred_gain,
            "searched_nodes": searched_count,
            "rejected_geometry": rej,
        }
        return selected_pack, meta

    def _sequential_match(self, candidates, b_max=175000, unique_sources=True, unique_helpers=True, layer=None, deterministic=True):
        return self.select_matching(
            candidates,
            b_max=b_max,
            unique_sources=unique_sources,
            unique_helpers=unique_helpers,
            layer=layer,
            state={"cbr": self._current_cbr},
            deterministic=deterministic,
        )


def eval_candidate(policy: CandidateSCRLGMIPolicy, seeds: List[int], eid_base: int, out_name: str):
    ckpt_path = ROOT / "checkpoints" / "sc_rlgmi_best.pt"
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        policy.advantage_model.load_state_dict(ckpt["advantage_model"])
    policy.eval()

    out_base = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "stage_b_candidates" / out_name
    results = []

    for i, s in enumerate(seeds):
        eid = eid_base + i + 1
        comp, pv, audit = run_single_ep(policy, s, eid, out_base / f"val_s{s}", deterministic=True)
        results.append({"seed": s, "V_done": comp, "P_viol": pv})
        print(f"    Seed {s}: V_done = {comp:6.2f} | P_viol = {pv*100:5.2f}%")

    mean_v = float(np.mean([r["V_done"] for r in results]))
    mean_pv = float(np.mean([r["P_viol"] for r in results]))
    print(f"  --> MEAN: V_done = {mean_v:6.2f} | P_viol = {mean_pv*100:5.2f}%\n")
    return mean_v, mean_pv, results


if __name__ == "__main__":
    train_seeds = [73200, 73201]

    # Candidate C_1: Priority 1 (Burst cap 50kB, CBR wiring)
    g1 = make_custom_guard(max_single_burst=50000, cell_edge_dist=220.0, epoch_byte_cap=175000, cbr_adaptive=False)
    p1 = CandidateSCRLGMIPolicy(g1, b_max=175000)
    print("\n--- Evaluating Candidate C_1 on Train Seeds ---")
    eval_candidate(p1, train_seeds, 97100, "C1_train")

    # Candidate C_2: C_1 + Cell-edge cut (150m, 20kB) + far cap 35kB
    g2 = make_custom_guard(max_single_burst=50000, cell_edge_dist=150.0, cell_edge_bytes=20000, far_epoch_cap=35000, epoch_byte_cap=175000, cbr_adaptive=False)
    p2 = CandidateSCRLGMIPolicy(g2, b_max=175000)
    print("\n--- Evaluating Candidate C_2 on Train Seeds ---")
    eval_candidate(p2, train_seeds, 97200, "C2_train")

    # Candidate C_3: C_2 + Epoch aggregate cap 75kB + monopolization 38kB
    g3 = make_custom_guard(max_single_burst=48000, monopolize_dist=30.0, monopolize_bytes=38000, cell_edge_dist=150.0, cell_edge_bytes=20000, far_epoch_cap=30000, epoch_byte_cap=75000, cbr_adaptive=False)
    p3 = CandidateSCRLGMIPolicy(g3, b_max=175000)
    print("\n--- Evaluating Candidate C_3 on Train Seeds ---")
    eval_candidate(p3, train_seeds, 97300, "C3_train")
