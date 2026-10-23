"""Phase A: Matched Decision Replay & Counterfactual Generation.

Executes deterministic closed-loop simulations under:
  - Guarded B4 policy (Reference Heuristic)
  - Trained Guarded Structured Priority policy (Current Model)
Identifies divergent decision epochs where assignments differ.
Branches at each divergent epoch t:
  - Branch SP: executes Guarded SP action
  - Branch Alt (B4 or Alternative): forces alternative action at epoch t, then continues under Guarded SP
Measures exact downstream completed value V_done and safety cost P_viol.
Records full state features, kinematics, queue backlogs, and outcomes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.causal_safety_guard import (
    CausalGuardedPolicy,
    GuardedB4Policy,
    filter_candidates_causal_guard,
    score_and_match_causal_guard,
)
from controllers.rl.structured_priority_policy import score_and_match


def get_checkpoint_policy(ckpt_path: Path) -> CausalGuardedPolicy:
    policy = CausalGuardedPolicy(r_thresh=0.20)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        policy.load_state_dict(ckpt["policy"])
        print(f"Loaded trained policy from {ckpt_path}")
    else:
        print(f"Warning: {ckpt_path} not found; using initialized policy")
    policy.eval()
    return policy


def run_counterfactual_branch(
    seed: int,
    t_seed: int,
    policy: CausalGuardedPolicy,
    branch_epoch: int,
    forced_pack: List[Dict[str, Any]],
    eid: int,
    runs_dir: Path,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Runs a counterfactual trajectory forcing forced_pack at branch_epoch."""
    bridge = Path(tempfile.gettempdir()) / f"cf_branch_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    out_dir = runs_dir / f"branch_ep_{eid:04d}"
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    force_spec = {"epoch": branch_epoch, "assignments": forced_pack}

    _, _, comp_val, _ = run_episode(
        WROOT, bridge, eid, 15, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 12, False, 175000,
        300, 12.0, "structured_priority", None,
        True, "off", "", True, t_seed, gov,
        175000, True, True, 1.0, None,
        helper_swap=None, force_pack=force_spec, helper_cpu_lo=2e8, helper_cpu_hi=1e9,
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


def collect_replay_dataset(
    seeds: List[int],
    output_dir: Path,
    ckpt_path: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sp_policy = get_checkpoint_policy(ckpt_path)
    b4_policy = GuardedB4Policy(r_thresh=0.20)
    b4_policy.eval()

    base_eid = 90000
    dataset = []

    print(f"\nCollecting matched decision traces on seeds: {seeds}")

    for seed in seeds:
        t_seed = 186100 + (seed - 73300) if seed >= 73300 else 191000 + (seed - 73200)

        # 1. Run unperturbed baseline for SP (or load if cached)
        eid_sp = base_eid + 1
        base_eid += 1
        out_sp = output_dir / f"base_sp_seed_{seed}"
        audit_sp_file = list(out_sp.glob("episode_*/bridge_audit.json"))
        sum_sp_file = list(out_sp.glob("episode_*/episode_summary.json"))
        if audit_sp_file and sum_sp_file:
            audit_sp = json.loads(audit_sp_file[0].read_text(encoding="utf-8"))
            v_sp_base = float(json.loads(sum_sp_file[0].read_text(encoding="utf-8")).get("V_done", 0.0))
            valid = [e for e in audit_sp if e.get("cbr_valid") and e.get("delay_valid")]
            pv_sp = sum(1 for e in valid if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95) / len(valid) if valid else 1.0
            print(f"Loaded existing SP baseline for seed {seed}")
        else:
            bridge_sp = Path(tempfile.gettempdir()) / f"cf_base_sp_{os.getpid()}_{eid_sp}"
            if bridge_sp.exists():
                shutil.rmtree(bridge_sp, ignore_errors=True)
            gov = SafetyGovernor(175000, horizon_envelope_only=True)
            _, _, v_sp_base, _ = run_episode(
                WROOT, bridge_sp, eid_sp, 15, sp_policy, 900.0,
                False, False, 100.0, 20.0, seed, 100,
                None, 12, False, 175000,
                300, 12.0, "structured_priority", None,
                True, "off", "", True, t_seed, gov,
                175000, True, True, 1.0, None, None, None, 2e8, 1e9,
                output_root=out_sp, arrival_mode="streaming",
            )
            audit_sp = json.loads((out_sp / f"episode_{eid_sp:04d}" / "bridge_audit.json").read_text(encoding="utf-8"))
            valid = [e for e in audit_sp if e.get("cbr_valid") and e.get("delay_valid")]
            pv_sp = sum(1 for e in valid if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95) / len(valid) if valid else 1.0

        # 2. Run unperturbed baseline for B4 (or load if cached)
        eid_b4 = base_eid + 1
        base_eid += 1
        out_b4 = output_dir / f"base_b4_seed_{seed}"
        audit_b4_file = list(out_b4.glob("episode_*/bridge_audit.json"))
        sum_b4_file = list(out_b4.glob("episode_*/episode_summary.json"))
        if audit_b4_file and sum_b4_file:
            audit_b4 = json.loads(audit_b4_file[0].read_text(encoding="utf-8"))
            v_b4_base = float(json.loads(sum_b4_file[0].read_text(encoding="utf-8")).get("V_done", 0.0))
            valid = [e for e in audit_b4 if e.get("cbr_valid") and e.get("delay_valid")]
            pv_b4 = sum(1 for e in valid if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95) / len(valid) if valid else 1.0
            print(f"Loaded existing B4 baseline for seed {seed}")
        else:
            bridge_b4 = Path(tempfile.gettempdir()) / f"cf_base_b4_{os.getpid()}_{eid_b4}"
            if bridge_b4.exists():
                shutil.rmtree(bridge_b4, ignore_errors=True)
            gov = SafetyGovernor(175000, horizon_envelope_only=True)
            _, _, v_b4_base, _ = run_episode(
                WROOT, bridge_b4, eid_b4, 15, b4_policy, 900.0,
                False, False, 100.0, 20.0, seed, 100,
                None, 12, False, 175000,
                300, 12.0, "structured_priority", None,
                True, "off", "", True, t_seed, gov,
                175000, True, True, 1.0, None, None, None, 2e8, 1e9,
                output_root=out_b4, arrival_mode="streaming",
            )
            audit_b4 = json.loads((out_b4 / f"episode_{eid_b4:04d}" / "bridge_audit.json").read_text(encoding="utf-8"))
            valid = [e for e in audit_b4 if e.get("cbr_valid") and e.get("delay_valid")]
            pv_b4 = sum(1 for e in valid if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95) / len(valid) if valid else 1.0

        print(f"Seed {seed}: SP Base V={v_sp_base:.2f} (P_viol={pv_sp*100:.1f}%) | B4 Base V={v_b4_base:.2f} (P_viol={pv_b4*100:.1f}%)")

        # 3. Identify divergent decision epochs
        divergent_epochs = []
        for epoch in range(min(len(audit_sp), len(audit_b4))):
            e_sp = audit_sp[epoch]
            e_b4 = audit_b4[epoch]
            assn_sp = sorted([(a["task_id"], a["helper_id"]) for a in (e_sp.get("selected_assignments") or [])])
            assn_b4 = sorted([(a["task_id"], a["helper_id"]) for a in (e_b4.get("selected_assignments") or [])])
            if assn_sp != assn_b4:
                divergent_epochs.append({
                    "epoch": epoch,
                    "assn_sp": assn_sp,
                    "assn_b4": assn_b4,
                    "b4_pack": e_b4.get("selected_assignments") or [],
                    "sp_pack": e_sp.get("selected_assignments") or [],
                    "cbr": float(e_sp.get("cbr", 0.04)),
                    "window_prr": float(e_sp.get("window_prr", e_sp.get("prr", 1.0))),
                    "sp_weights": e_sp.get("weights"),
                })

        print(f"  Found {len(divergent_epochs)} divergent decision epochs on seed {seed}")

        # 4. Evaluate counterfactual branch for each divergence
        # Branch: At epoch t, execute B4 action instead of SP action, then continue under Guarded SP policy
        for div in divergent_epochs:
            ep_idx = div["epoch"]
            b4_pack = div["b4_pack"]
            eid_branch = base_eid + 1
            base_eid += 1

            branch_out = output_dir / f"branch_s{seed}_ep{ep_idx:02d}"
            branch_audit_file = list(branch_out.glob("episode_*/bridge_audit.json"))
            branch_sum_file = list(branch_out.glob("episode_*/episode_summary.json"))
            if branch_audit_file and branch_sum_file:
                v_branch = float(json.loads(branch_sum_file[0].read_text(encoding="utf-8")).get("V_done", 0.0))
                audit_branch = json.loads(branch_audit_file[0].read_text(encoding="utf-8"))
                valid_br = [e for e in audit_branch if e.get("cbr_valid") and e.get("delay_valid")]
                pv_branch = sum(1 for e in valid_br if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95) / len(valid_br) if valid_br else 1.0
                print(f"    Loaded cached branch for seed {seed} ep {ep_idx:02d}")
            else:
                v_branch, pv_branch, audit_branch = run_counterfactual_branch(
                    seed, t_seed, sp_policy, ep_idx, b4_pack, eid_branch, branch_out
                )

            delta_v_sp = v_branch - v_sp_base
            delta_v_b4 = v_branch - v_b4_base
            is_beneficial = bool(delta_v_sp > 0.05 and pv_branch <= 0.02)
            is_harmful = bool(delta_v_sp < -0.05 or pv_branch > 0.02)

            record = {
                "seed": seed,
                "epoch": ep_idx,
                "v_sp_base": v_sp_base,
                "v_b4_base": v_b4_base,
                "v_branch": v_branch,
                "delta_v_sp": delta_v_sp,
                "delta_v_b4": delta_v_b4,
                "delta_v": delta_v_sp,
                "pv_base_sp": pv_sp,
                "pv_base_b4": pv_b4,
                "pv_branch": pv_branch,
                "is_beneficial": is_beneficial,
                "is_harmful": is_harmful,
                "assn_sp": div["assn_sp"],
                "assn_b4": div["assn_b4"],
                "b4_pack": b4_pack,
                "sp_pack": div["sp_pack"],
                "cbr": div["cbr"],
                "window_prr": div["window_prr"],
                "sp_weights": div["sp_weights"],
            }
            dataset.append(record)
            print(f"    Epoch {ep_idx:02d}: Delta V_SP = {delta_v_sp:+.2f} (Branch V={v_branch:.2f}, SP Base={v_sp_base:.2f}, B4 Base={v_b4_base:.2f}) | P_viol={pv_branch*100:.2f}% | Beneficial={is_beneficial}")

    summary = {
        "seeds": seeds,
        "n_decisions_evaluated": len(dataset),
        "n_beneficial": sum(1 for d in dataset if d["is_beneficial"]),
        "n_harmful": sum(1 for d in dataset if d["is_harmful"]),
        "n_neutral": sum(1 for d in dataset if not d["is_beneficial"] and not d["is_harmful"]),
        "mean_beneficial_gain": float(np.mean([d["delta_v"] for d in dataset if d["is_beneficial"]])) if any(d["is_beneficial"] for d in dataset) else 0.0,
        "records": dataset,
    }

    out_file = output_dir / "counterfactual_replay_dataset.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {len(dataset)} counterfactual replay decisions to {out_file}")
    print(f"Beneficial opportunities: {summary['n_beneficial']} | Harmful: {summary['n_harmful']} | Neutral: {summary['n_neutral']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build counterfactual replay dataset")
    parser.add_argument("--seeds", nargs="+", type=int, default=[73300, 73301, 73302, 73303], help="Seeds to evaluate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/structured_priority_best_feasible.pt", help="Path to SP policy")
    parser.add_argument("--out-dir", type=str, default="sparc_performance_evaluation/results/raw/counterfactual_replay", help="Output directory")
    args = parser.parse_args()

    collect_replay_dataset(args.seeds, ROOT / args.out_dir, ROOT / args.checkpoint)


if __name__ == "__main__":
    main()
