"""Safety-Constrained Reinforcement Learning-Guided Matching Improvement (SC-RLGMI) Training & Evaluation.

Implements Algorithm 1 (Training) and Algorithm 2 (Evaluation) from the Final Specification:
1. Combinatorial Matching Search over Neighborhood Operators (Add, Drop, Reassign, Exchange, Replace)
2. Matched Counterfactual Return Learning: ΔG = G(M) - G(M_B) under identical exogenous continuations
3. Iterative Policy Improvement: Beam Search guided by antisymmetric advantage network A_θ
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.sc_rlgmi.sc_rlgmi_core import (
    Edge,
    Matching,
    canonical,
    DecisionState,
    AdvantageModel,
    PairExample,
    train_pairs,
    neighbors,
)
from controllers.rl.sc_rlgmi.sc_rlgmi_policy import SCRLGMIPolicy

# Safe, non-degenerate corrected-environment training seeds (screened at
# 5 ms / 20 MHz / 300 B TB / 10-30 kB): 73211 V=11.22 P_viol_long=0.00,
# 73219 V=13.72 P_viol_long=0.00.  The original 73200/73201 pair was
# degenerate (1-2 completions, 100% safety violation).
# 73212 is retained as a high-value, safety-stressed seed (V=23.25,
# P_viol_long=0.168) so the counterfactual set contains non-zero
# safety-driven alternatives; 73219 is the safe reference anchor.
TRAIN_SEEDS = [73212, 73219]
DEV_SEEDS = [73300, 73301, 73302, 73303]
# Corrected-environment guarded reference baseline (B1/B3/B5' value-density
# reference, dev seeds 73300-73303): 17.03, 16.59, 8.74, 15.58 -> mean 14.486.
# The old 49.15 target belonged to the legacy 10 MHz / 2 kB / 20 ms workload
# and is not comparable.
GUARDED_B4_TARGET = 14.486
SAFETY_LIMIT = 0.020


def run_single_ep(
    policy: SCRLGMIPolicy,
    seed: int,
    eid: int,
    out_dir: Path,
    deterministic: bool = True,
    force_pack: Optional[Dict[str, Any]] = None,
    simulation_time: float = 12.0,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Runs a single closed-loop simulation episode, optionally forcing a pack at an epoch."""
    t_seed = 186100 + (seed - 73300) if seed >= 73300 else 191000 + (seed - 73200)
    bridge = Path(tempfile.gettempdir()) / f"bridge_sc_rlgmi_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    ep_dir = out_dir / f"episode_{eid:04d}"

    _, _, comp_val, _ = run_episode(
        WROOT,
        bridge,
        eid,
        15,
        policy,
        900.0,
        False,
        False,
        100.0,
        20.0,
        seed,
        100,
        None,
        12,
        False,
        175000,
        300,
        simulation_time,
        "scql",  # Uses _sequential_match hook
        None,
        deterministic,
        "off",
        "",
        True,
        t_seed,
        gov,
        175000,
        True,
        True,
        1.0,
        None,
        helper_swap=None,
        force_pack=force_pack,
        helper_cpu_lo=1e9,
        helper_cpu_hi=4e9,
        tb_bytes=300,
        segment_bytes=2000,
        reservation_ms=5.0,
        bw_mhz=20,
        payload_bytes_lo=10_000,
        payload_bytes_hi=30_000,
        deadline_ms_lo=100.0,
        deadline_ms_hi=500.0,
        cycles_lo=5e6,
        cycles_hi=5e7,
        packets_per_epoch=4,
        output_root=ep_dir,
        arrival_mode="streaming",
    )

    audit_file = ep_dir / f"episode_{eid:04d}" / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    summary_file = ep_dir / f"episode_{eid:04d}" / "episode_summary.json"
    if summary_file.is_file():
        ep_summary = json.loads(summary_file.read_text(encoding="utf-8"))
        pv = float(ep_summary.get("safety_P_viol_long", ep_summary.get("safety_P_viol", 1.0)))
    else:
        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        viols = sum(1 for p in prrs if p < 0.95)
        pv = viols / len(valid) if valid else 1.0

    return comp_val, pv, audit


def collect_counterfactual_rollouts(
    policy: SCRLGMIPolicy,
    seeds: List[int],
    out_dir: Path,
    base_eid: int = 87000,
    max_epochs_per_seed: int = 5,
    max_neighbors_per_epoch: int = 4,
) -> List[PairExample]:
    """Generates matched counterfactual rollouts across training seeds [73200, 73201]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset: List[PairExample] = []
    eid = base_eid

    print("\n" + "=" * 70)
    print("COLLECTING MATCHED COUNTERFACTUAL DATASET ON TRAINING SEEDS")
    print(f"Seeds: {seeds} | Max Epochs/Seed: {max_epochs_per_seed} | Neighbors/Epoch: {max_neighbors_per_epoch}")
    print("=" * 70)

    for seed in seeds:
        eid += 1
        ref_out = out_dir / f"ref_s{seed}"
        policy.expert_mode = "reference"
        policy.record_decisions = True
        policy.recorded_decisions.clear()
        v_ref_full, pv_ref_full, audit_ref = run_single_ep(
            policy, seed, eid, ref_out, deterministic=True, simulation_time=2.5
        )
        policy.record_decisions = False
        recorded = dict(policy.recorded_decisions)
        print(f"\nSeed {seed} Reference: V = {v_ref_full:.2f} | P_viol = {pv_ref_full*100:.2f}% | Recorded Decision Epochs = {len(recorded)}")

        candidate_epochs = sorted(recorded.keys())
        # Subsample diverse decision epochs
        if len(candidate_epochs) > max_epochs_per_seed:
            selected_epochs = list(np.linspace(0, len(candidate_epochs) - 1, max_epochs_per_seed, dtype=int))
            chosen_epochs = [candidate_epochs[i] for i in selected_epochs]
        else:
            chosen_epochs = candidate_epochs

        print(f"  Selected {len(chosen_epochs)} epochs for counterfactual branching: {chosen_epochs}")

        for ep in chosen_epochs:
            rec = recorded[ep]
            d_state = rec["d_state"]
            edge_map = rec["edge_map"]
            ref_matching = rec["ref_matching"]
            b_safe = rec["b_safe"]
            oracle = policy._make_feasibility_oracle(edge_map, b_safe=b_safe)

            # Generate neighborhood moves.  Prefer alternatives that change the
            # task set (add/drop/replace) over pure helper reassignments:
            # reassigning the same task usually still completes it and yields
            # a degenerate Delta_G = 0 target.
            cand_neighbors = neighbors(d_state, ref_matching, oracle)
            if not cand_neighbors:
                continue

            ref_tasks = {k for k, _ in ref_matching}

            def _alt_rank(m):
                same_tasks = {k for k, _ in m} == ref_tasks
                same_size = len(m) == len(ref_matching)
                return (same_tasks, same_size, -len(m), m)

            ordered = sorted(cand_neighbors, key=_alt_rank)
            sub_neighbors = ordered[:max_neighbors_per_epoch]
            print(f"    Epoch {ep:02d}: Evaluated {len(cand_neighbors)} neighbors -> branching top {len(sub_neighbors)}")

            for alt_matching in sub_neighbors:
                # An empty matching is a valid MDP action but it produces a
                # zero-byte action file that the MATLAB bridge cannot consume
                # ("Invalid message - buffer too small 0 < 8").  It also has
                # no counterfactual content when the reference already
                # admitted the only completion, so skip it.
                if not alt_matching:
                    continue
                eid += 1
                alt_pack = [dict(edge_map[e].raw_dict) for e in alt_matching if e in edge_map]
                force_spec = {"epoch": ep, "assignments": alt_pack}
                branch_out = out_dir / f"branch_s{seed}_ep{ep:02d}_m{eid}"

                v_alt, pv_alt, _ = run_single_ep(
                    policy, seed, eid, branch_out, deterministic=True,
                    force_pack=force_spec, simulation_time=2.5,
                )

                delta_g = float(v_alt - v_ref_full)
                example = PairExample(
                    state=d_state,
                    alternative=alt_matching,
                    reference=ref_matching,
                    advantage=delta_g,
                    alt_violation=pv_alt,
                    ref_violation=pv_ref_full,
                )
                dataset.append(example)

                print(f"      Alt (size {len(alt_matching)}): V = {v_alt:5.2f} (Delta_G = {delta_g:+5.2f}) | P_viol = {pv_alt*100:4.1f}%")

    print(f"\nCollected {len(dataset)} counterfactual pair examples.\n")
    return dataset


def evaluate_sc_rlgmi_on_dev(
    policy: SCRLGMIPolicy,
    dev_seeds: List[int],
    out_dir: Path,
    base_eid: int = 93000,
) -> Dict[str, Any]:
    """Evaluates SC-RLGMI policy on all 4 development validation seeds."""
    policy.eval()
    policy.expert_mode = "rl"

    results = []
    print("\n" + "=" * 70)
    print("  SC-RLGMI DEVELOPMENT VALIDATION EVALUATION (SEEDS 73300-73303)")
    print(f"  Advantage Threshold: delta = {policy.delta:.2f} | Beam: {policy.beam} | Depth: {policy.depth}")
    print("=" * 70)

    for i, s in enumerate(dev_seeds):
        eid = base_eid + i + 1
        comp, pv, audit = run_single_ep(
            policy, s, eid, out_dir / f"val_s{s}", deterministic=True
        )
        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        mean_prr = float(np.mean(prrs)) if prrs else 1.0

        modes = []
        for e in audit:
            meta = e.get("scql_meta")
            if isinstance(meta, dict):
                modes.append(meta.get("selected_mode"))
        overrides = sum(1 for m in modes if m == "sc_rlgmi_alternative")
        ref_kept = sum(1 for m in modes if m == "reference_b4")

        results.append({
            "seed": s,
            "V_done": comp,
            "P_viol": pv,
            "mean_prr": mean_prr,
            "overrides": overrides,
            "reference_kept": ref_kept,
            "total_decisions": len(modes),
        })
        print(f"  Seed {s}: V_done = {comp:6.2f} | P_viol = {pv*100:5.2f}% | PRR = {mean_prr:.4f} | Overrides = {overrides:2d}/{len(modes)}")

    mean_v = float(np.mean([r["V_done"] for r in results]))
    mean_pv = float(np.mean([r["P_viol"] for r in results]))
    tot_overrides = sum(r["overrides"] for r in results)
    tot_decs = sum(r["total_decisions"] for r in results)
    override_rate = tot_overrides / max(1, tot_decs)

    beats_b4 = (mean_v > GUARDED_B4_TARGET) and (mean_pv <= SAFETY_LIMIT)

    summary = {
        "mean_V_done": mean_v,
        "mean_P_viol": mean_pv,
        "results": results,
        "guarded_b4_target": GUARDED_B4_TARGET,
        "delta_vs_b4": mean_v - GUARDED_B4_TARGET,
        "beats_b4": beats_b4,
        "safety_compliant": mean_pv <= SAFETY_LIMIT,
        "override_rate": override_rate,
        "total_overrides": tot_overrides,
        "delta_threshold": policy.delta,
        "beam_width": policy.beam,
        "search_depth": policy.depth,
    }

    print("-" * 70)
    print(f"  Mean Completed Value: {mean_v:.2f} (Guarded B4 Target: {GUARDED_B4_TARGET:.2f})")
    print(f"  Delta vs Guarded B4 : {mean_v - GUARDED_B4_TARGET:+.2f}")
    print(f"  Mean Safety Violation: {mean_pv*100:.2f}% (Safety Limit: {SAFETY_LIMIT*100:.1f}%)")
    print(f"  Override Rate       : {override_rate*100:.2f}% ({tot_overrides}/{tot_decs} decisions)")
    print(f"  BEATS GUARDED B4    : {beats_b4}")
    print("=" * 70 + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SC-RLGMI Training and Evaluation")
    parser.add_argument("--epochs", type=int, default=40, help="Advantage model training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--delta", type=float, default=0.0, help="Advantage override threshold")
    parser.add_argument("--depth", type=int, default=2, help="Beam search depth")
    parser.add_argument("--beam", type=int, default=8, help="Beam search width")
    parser.add_argument("--max-epochs-per-seed", type=int, default=4, help="Max decision epochs to branch per seed")
    parser.add_argument("--max-neighbors-per-epoch", type=int, default=3, help="Max neighbors to branch per epoch")
    parser.add_argument("--eval-only", action="store_true", help="Only run development evaluation")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path")
    args = parser.parse_args()

    out_root = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "sc_rlgmi_training"
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else checkpoints_dir / "sc_rlgmi_best.pt"

    policy = SCRLGMIPolicy(
        b_max=175000,
        r_thresh=0.20,
        delta=args.delta,
        depth=args.depth,
        beam=args.beam,
    )

    if args.eval_only:
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            policy.advantage_model.load_state_dict(ckpt["advantage_model"])
            print(f"Loaded trained advantage model from {ckpt_path}")
        summary = evaluate_sc_rlgmi_on_dev(policy, DEV_SEEDS, out_root / "eval")
    else:
        # Step 1: Collect matched counterfactual dataset on training seeds
        examples = collect_counterfactual_rollouts(
            policy,
            TRAIN_SEEDS,
            out_root / "rollouts",
            max_epochs_per_seed=args.max_epochs_per_seed,
            max_neighbors_per_epoch=args.max_neighbors_per_epoch,
        )

        # Step 2: Train advantage network on paired differences
        print("Training Advantage Network on Paired Counterfactual Targets...")
        losses = train_pairs(policy.advantage_model, examples, epochs=args.epochs, lr=args.lr)
        print(f"Trained for {args.epochs} epochs. Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")

        # Save checkpoint
        torch.save({
            "advantage_model": policy.advantage_model.state_dict(),
            "n_examples": len(examples),
            "final_loss": losses[-1],
            "delta": args.delta,
        }, ckpt_path)
        print(f"Saved trained SC-RLGMI checkpoint to {ckpt_path}")

        # Step 3: Evaluate on development validation seeds
        summary = evaluate_sc_rlgmi_on_dev(policy, DEV_SEEDS, out_root / "eval")

    summary_file = out_root / "sc_rlgmi_development_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Development summary written to {summary_file}")


if __name__ == "__main__":
    main()
