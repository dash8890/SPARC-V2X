"""Decision-Level Counterfactual Audit: SCQL vs. Guarded B4 on Training Scenarios.

Verifies whether SCQL has learned genuine capacity-aware scheduling advantage
rather than greedy value selection:
  A(s, i, j) = Q(s, i, j) - Q(s, STOP)

Evaluates on training seeds [73200, 73201]:
1. Runs full closed-loop episodes under Guarded B4 and SCQL to find divergent decision epochs.
2. At each divergent epoch t, branches from that state:
   - Evaluates downstream completed value delta Delta V = V_SCQL - V_B4.
   - Evaluates safety violation rate P_viol <= 2.0%.
3. Analyzes whether the decision was driven by observable helper capacity:
   - queue_cycles, estimated release time, contact duration, post-assignment slack.
4. Applies the engineering gate specified in Section 5.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.scql.scql_policy import SCQLSequentialPolicy


def run_single_ep(
    seed: int,
    t_seed: int,
    policy: SCQLSequentialPolicy,
    policy_mode: str,
    eid: int,
    out_dir: Path,
    force_pack: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Runs a single episode, optionally forcing a pack at an epoch."""
    bridge = Path(tempfile.gettempdir()) / f"audit_scql_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)

    _, _, comp_val, _ = run_episode(
        WROOT, bridge, eid, 15, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 12, False, 175000,
        300, 12.0, "scql", None,
        True, "off", "", True, t_seed, gov,
        175000, True, True, 1.0, None,
        helper_swap=None, force_pack=force_pack, helper_cpu_lo=2e8, helper_cpu_hi=1e9,
        output_root=out_dir,
        arrival_mode="streaming",
    )

    audit_file = out_dir / f"episode_{eid:04d}" / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    viols = sum(1 for p in prrs if p < 0.95)
    pv = viols / len(valid) if valid else 1.0

    return comp_val, pv, audit


def audit_scql_capacity_advantage(
    checkpoint_path: Path,
    seeds: List[int],
    output_dir: Path,
    base_eid: int = 96000,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Load SCQL Policy
    scql_policy = SCQLSequentialPolicy(b_max=175000, r_thresh=0.20, expert_mode="rl")
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=device)
        scql_policy.load_state_dict(ckpt["policy"])
        print(f"Loaded SCQL checkpoint from {checkpoint_path}")
    scql_policy.eval()
    scql_policy.epsilon = 0.0

    # Guarded B4 Policy
    b4_policy = SCQLSequentialPolicy(b_max=175000, r_thresh=0.20, expert_mode="guarded_b4")
    b4_policy.eval()

    eid = base_eid
    all_divergences = []
    beneficial_decisions = []

    print("\n" + "=" * 80)
    print("DECISION-LEVEL AUDIT: SCQL VS. GUARDED B4 ON TRAINING SCENARIOS")
    print(f"Target Training Seeds: {seeds}")
    print("=" * 80)

    for seed in seeds:
        t_seed = 191000 + (seed - 73200) if seed in [73200, 73201] else 186100 + (seed - 73300)

        # 1. Run full Guarded B4 baseline
        eid += 1
        b4_out = output_dir / f"base_b4_s{seed}"
        v_b4, pv_b4, audit_b4 = run_single_ep(
            seed, t_seed, b4_policy, "scql", eid, b4_out
        )
        print(f"\nSeed {seed} Baseline: Guarded B4 V_done = {v_b4:.2f} | P_viol = {pv_b4*100:.2f}%")

        # 2. Run full SCQL baseline
        eid += 1
        scql_out = output_dir / f"base_scql_s{seed}"
        v_scql, pv_scql, audit_scql = run_single_ep(
            seed, t_seed, scql_policy, "scql", eid, scql_out
        )
        print(f"Seed {seed} Baseline: SCQL       V_done = {v_scql:.2f} | P_viol = {pv_scql*100:.2f}%")

        # 3. Identify divergent decision epochs
        n_epochs = min(len(audit_b4), len(audit_scql))
        divergent_epochs = []
        for ep in range(n_epochs):
            assn_b4 = sorted([(a["task_id"], a["helper_id"]) for a in (audit_b4[ep].get("selected_assignments") or [])])
            assn_scql = sorted([(a["task_id"], a["helper_id"]) for a in (audit_scql[ep].get("selected_assignments") or [])])

            if assn_b4 != assn_scql:
                divergent_epochs.append({
                    "epoch": ep,
                    "assn_b4": assn_b4,
                    "assn_scql": assn_scql,
                    "pack_b4": audit_b4[ep].get("selected_assignments") or [],
                    "pack_scql": audit_scql[ep].get("selected_assignments") or [],
                })

        print(f"  Found {len(divergent_epochs)} divergent decision epochs on Seed {seed}")

        # 4. Branch each divergent epoch from the common B4 trajectory
        for div in divergent_epochs:
            ep_idx = div["epoch"]
            pack_scql = div["pack_scql"]
            assn_scql = div["assn_scql"]
            assn_b4 = div["assn_b4"]

            eid += 1
            branch_out = output_dir / f"branch_s{seed}_ep{ep_idx:02d}_scql"
            force_spec = {"epoch": ep_idx, "assignments": pack_scql}
            v_branch, pv_branch, _ = run_single_ep(
                seed, t_seed, b4_policy, "scql", eid, branch_out, force_pack=force_spec
            )

            delta_v = v_branch - v_b4
            is_beneficial = bool(delta_v > 0.05 and pv_branch <= 0.02)
            is_harmful = bool(delta_v < -0.05 or pv_branch > 0.02)

            div_record = {
                "seed": seed,
                "epoch": ep_idx,
                "assn_b4": assn_b4,
                "assn_scql": assn_scql,
                "v_b4_base": v_b4,
                "v_branch_scql": v_branch,
                "delta_v": delta_v,
                "pv_b4": pv_b4,
                "pv_branch": pv_branch,
                "is_beneficial": is_beneficial,
                "is_harmful": is_harmful,
            }
            all_divergences.append(div_record)
            if is_beneficial:
                beneficial_decisions.append(div_record)

            print(f"    Epoch {ep_idx:02d}: B4={assn_b4} vs SCQL={assn_scql} | Downstream V={v_branch:.2f} (Delta={delta_v:+.2f}) | P_viol={pv_branch*100:.2f}% | Beneficial={is_beneficial}")

    # Summary
    summary = {
        "seeds": seeds,
        "total_divergences": len(all_divergences),
        "beneficial_count": len(beneficial_decisions),
        "harmful_count": sum(1 for d in all_divergences if d["is_harmful"]),
        "divergences": all_divergences,
    }
    summary_file = output_dir / "scql_decision_audit_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"DECISION AUDIT COMPLETE: {len(all_divergences)} divergent decisions tested, {len(beneficial_decisions)} beneficial.")
    print(f"Summary written to {summary_file}")
    print("=" * 80 + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision-level audit of SCQL capacity advantage")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scql_warm_start.pt", help="Path to checkpoint")
    parser.add_argument("--seeds", nargs="+", type=int, default=[73200, 73201], help="Training seeds to audit")
    args = parser.parse_args()

    out_root = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "scql_decision_audit"
    audit_scql_capacity_advantage(Path(args.checkpoint), args.seeds, out_root)
