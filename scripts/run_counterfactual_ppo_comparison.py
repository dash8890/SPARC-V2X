"""Phase D: Bounded Evaluation of Standard Guarded PPO vs Counterfactual-Augmented Guarded PPO.

Compares:
  - Baseline: Standard Guarded PPO (alpha_pref = 0.0)
  - Proposed: Counterfactual-Augmented Guarded PPO (alpha_pref > 0.0)
Under identical:
  - Initial checkpoint: checkpoints/structured_priority_best_feasible.pt (47.70 at 1.25%)
  - Training seeds: [73200, 73201]
  - Update budget: <= 10 PPO updates
  - Evaluation seeds: [73300, 73301, 73302, 73303]
  - Physical simulator & Refined Safety Guard (frozen)
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
from controllers.rl.causal_safety_guard import CausalGuardedPolicy, filter_candidates_causal_guard
from controllers.rl.structured_priority_policy import score_and_match
from controllers.rl.structured_priority_ppo import structured_priority_ppo_update
from controllers.rl.lagrangian_ppo import dual_ascent

ALPHA = 0.02
GUARDED_B4_TARGET = 49.15


def run_eval(
    policy: CausalGuardedPolicy,
    seeds: List[int],
    out_dir: Path,
    base_eid: int = 80000,
) -> Tuple[float, float, List[Dict[str, Any]], int]:
    policy.eval()
    results = []
    eid = base_eid

    for seed in seeds:
        eid += 1
        t_seed = 186100 + (seed - 73300) if seed >= 73300 else 191000 + (seed - 73200)
        bridge = Path(tempfile.gettempdir()) / f"eval_cf_{os.getpid()}_{eid}"
        if bridge.exists():
            shutil.rmtree(bridge, ignore_errors=True)
        gov = SafetyGovernor(175000, horizon_envelope_only=True)
        ep_dir = out_dir / f"eval_s{seed}_ep{eid}"

        _, _, comp, _ = run_episode(
            WROOT, bridge, eid, 15, policy, 900.0,
            False, False, 100.0, 20.0, seed, 100,
            None, 12, False, 175000,
            300, 12.0, "structured_priority", None,
            True, "off", "", True, t_seed, gov,
            175000, True, True, 1.0, None, None, None, 2e8, 1e9,
            output_root=ep_dir, arrival_mode="streaming",
        )

        audit_file = ep_dir / f"episode_{eid:04d}" / "bridge_audit.json"
        audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        viols = sum(1 for p in prrs if p < 0.95)
        pv = viols / len(valid) if valid else 1.0
        mean_prr = float(np.mean(prrs)) if prrs else 1.0

        results.append({
            "seed": seed,
            "V_done": comp,
            "P_viol": pv,
            "mean_prr": mean_prr,
        })
        print(f"      Validation Seed {seed}: V = {comp:.2f} | P_viol = {pv*100:.2f}% | PRR = {mean_prr:.4f}")

    mean_v = float(np.mean([r["V_done"] for r in results]))
    mean_pv = float(np.mean([r["P_viol"] for r in results]))
    return mean_v, mean_pv, results, eid


def train_and_evaluate(
    initial_checkpoint: Path,
    out_dir: Path,
    updates: int = 10,
    lr: float = 3e-4,
    alpha_pref: float = 0.05,
    train_seeds: List[int] = [73200, 73201],
    val_seeds: List[int] = [73300, 73301, 73302, 73303],
    mode_name: str = "counterfactual_ppo",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Starting {mode_name.upper()} (Alpha_Pref = {alpha_pref})")
    print(f"Updates: {updates} | LR: {lr} | Train Seeds: {train_seeds}")
    print(f"=======================================================\n")

    policy = CausalGuardedPolicy(r_thresh=0.20)
    ckpt = torch.load(initial_checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy"])
    print(f"Loaded initial checkpoint from {initial_checkpoint}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    lam_L = float(ckpt.get("lam_L", 0.0))
    base_eid = 70000 if alpha_pref > 0 else 65000

    # Evaluate initial performance
    print("  Initial Validation Performance:")
    init_v, init_pv, init_rows, base_eid = run_eval(policy, val_seeds, runs_dir, base_eid)
    print(f"  Init Mean V = {init_v:.2f} | Init P_viol = {init_pv*100:.2f}%\n")

    history = []

    # Target weights for counterfactual guidance (high value density, lower latency/queue penalty)
    target_w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    w_norm = torch.clamp(target_w / policy.w_max, 1e-4, 1.0 - 1e-4)
    target_z_w = torch.logit(w_norm).tolist()
    target_z_b = float(torch.logit(torch.tensor(0.85)))

    for u in range(1, updates + 1):
        t0 = time.perf_counter()
        traces = []
        train_vs = []

        # 1. Rollout training episodes
        for s in train_seeds:
            base_eid += 1
            t_seed = 191000 + (s - 73200) + u * 10
            bridge = Path(tempfile.gettempdir()) / f"tr_{os.getpid()}_{base_eid}"
            if bridge.exists():
                shutil.rmtree(bridge, ignore_errors=True)
            gov = SafetyGovernor(175000, horizon_envelope_only=True)
            ep_out = runs_dir / f"tr_u{u}_s{s}"

            _, _, comp, tr = run_episode(
                WROOT, bridge, base_eid, 15, policy, 900.0,
                False, False, 100.0, 20.0, s, 100,
                None, 12, False, 175000,
                300, 12.0, "structured_priority", None,
                True, "off", "", True, t_seed, gov,
                175000, True, True, 1.0, None, None, None, 2e8, 1e9,
                output_root=ep_out, arrival_mode="streaming",
            )
            train_vs.append(comp)
            if tr:
                for item in tr:
                    if not torch.is_tensor(item.get("logp")):
                        item["logp"] = torch.tensor(float(item.get("logp") or 0.0))
                    if not torch.is_tensor(item.get("value")):
                        item["value"] = torch.tensor(float(item.get("value") or 0.0))
                    if not torch.is_tensor(item.get("cost_value")):
                        item["cost_value"] = torch.tensor(float(item.get("cost_value") or 0.0))
                traces.append(tr)

        # 2. Extract counterfactual targets if alpha_pref > 0
        cf_targets = []
        if alpha_pref > 0.0:
            for tr in traces:
                for step in tr:
                    sp_obs = step.get("sp_obs")
                    if sp_obs is not None:
                        # Identify transitions where helper was idle or safe high-value candidate existed
                        cands = step.get("candidate_features") or []
                        if len(cands) > 1 and len(step.get("assignments") or []) == 0:
                            cf_targets.append({
                                "sp_obs": sp_obs,
                                "target_z_w": target_z_w,
                                "target_z_b": target_z_b,
                            })

        # 3. PPO Update
        metrics = structured_priority_ppo_update(
            policy,
            optimizer,
            traces,
            lam_L,
            epochs=2,
            minibatch_size=32,
            target_kl=0.02,
            detach_trunk=True,
            cf_targets=cf_targets if cf_targets else None,
            alpha_pref=alpha_pref if cf_targets else 0.0,
        )

        mean_cost = float(metrics.get("mean_epoch_violation_rate", 0.0))
        lam_L = dual_ascent(lam_L, mean_cost, cost_limit=ALPHA, lr_dual=0.05)
        el = time.perf_counter() - t0

        record = {
            "update": u,
            "train_v": float(np.mean(train_vs)),
            "actor_loss": metrics.get("actor_loss", 0.0),
            "value_loss": metrics.get("value_loss", 0.0),
            "cost_loss": metrics.get("cost_loss", 0.0),
            "pref_loss": metrics.get("pref_loss", 0.0),
            "approx_kl": metrics.get("approx_kl", 0.0),
            "grad_norm": metrics.get("grad_norm", 0.0),
            "lam_L": lam_L,
            "n_cf_targets": len(cf_targets),
            "wall_s": el,
        }
        history.append(record)
        print(f"  Update {u:02d}/{updates} in {el:.1f}s | Train V = {record['train_v']:.2f} | KL = {record['approx_kl']:.4f} | Actor L = {record['actor_loss']:.4f} | Pref L = {record['pref_loss']:.4f} | Cost L = {record['cost_loss']:.4f} | lam_L = {lam_L:.4f}")

    # Final Validation Evaluation
    print(f"\n  Final Validation Evaluation on {val_seeds}:")
    final_v, final_pv, final_rows, base_eid = run_eval(policy, val_seeds, runs_dir, base_eid)
    print(f"  Final Mean V = {final_v:.2f} | Final P_viol = {final_pv*100:.2f}% | Target B4 = {GUARDED_B4_TARGET:.2f}\n")

    summary = {
        "mode": mode_name,
        "alpha_pref": alpha_pref,
        "updates": updates,
        "init_val_v": init_v,
        "init_val_pv": init_pv,
        "final_val_v": final_v,
        "final_val_pv": final_pv,
        "delta_b4": final_v - GUARDED_B4_TARGET,
        "delta_init": final_v - init_v,
        "pass_target": bool(final_v > GUARDED_B4_TARGET and final_pv <= ALPHA),
        "history": history,
        "final_rows": final_rows,
    }

    out_file = out_dir / f"{mode_name}_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Bounded evaluation of Guarded PPO vs Counterfactual PPO")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/structured_priority_best_feasible.pt")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-pref", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="sparc_performance_evaluation/results/raw/counterfactual_learning")
    args = parser.parse_args()

    out_base = ROOT / args.out_dir
    ckpt_path = ROOT / args.checkpoint

    # Condition A: Standard Guarded PPO (alpha_pref = 0.0)
    res_std = train_and_evaluate(
        ckpt_path, out_base / "standard_ppo",
        updates=args.updates, lr=args.lr, alpha_pref=0.0,
        mode_name="standard_guarded_ppo",
    )

    # Condition B: Counterfactual-Augmented Guarded PPO (alpha_pref = 0.05)
    res_cf = train_and_evaluate(
        ckpt_path, out_base / "counterfactual_ppo",
        updates=args.updates, lr=args.lr, alpha_pref=args.alpha_pref,
        mode_name="counterfactual_guarded_ppo",
    )

    # Comparison summary
    print("\n" + "=" * 80)
    print("PHASE D: FINAL HEAD-TO-HEAD BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Guarded B4 (Ref)':<18} | {'Standard Guarded PPO':<22} | {'Counterfactual Guarded PPO':<24}")
    print("-" * 100)
    print(f"{'Mean Completed Value':<30} | {GUARDED_B4_TARGET:<18.2f} | {res_std['final_val_v']:<22.2f} | {res_cf['final_val_v']:<24.2f}")
    print(f"{'Safety Violations (P_viol)':<30} | {'1.25%':<18} | {res_std['final_val_pv']*100:<21.2f}% | {res_cf['final_val_pv']*100:<23.2f}%")
    print(f"{'Delta vs Guarded B4':<30} | {'0.00':<18} | {res_std['delta_b4']:<+22.2f} | {res_cf['delta_b4']:<+24.2f}")
    print(f"{'Feasible (P_viol <= 2%)':<30} | {'PASS':<18} | {str(res_std['final_val_pv'] <= 0.02):<22} | {str(res_cf['final_val_pv'] <= 0.02):<24}")
    print(f"{'Beats Guarded B4 (>49.15)':<30} | {'--':<18} | {str(res_std['pass_target']):<22} | {str(res_cf['pass_target']):<24}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
