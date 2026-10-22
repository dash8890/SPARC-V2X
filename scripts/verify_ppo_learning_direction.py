"""Verify PPO Learning Direction on Differentiated Returns.

Protocol:
1. Run a sampled episode (sigma_w = 1.0) on Seed 73200 that produces divergent actions and a lower return (66.39 vs 71.03).
2. Measure the log-probability of the divergent actions before the PPO update.
3. Perform 1 PPO optimization update (with lr=2e-3).
4. Measure the log-probability of the divergent actions after the PPO update.
5. Verify that for steps with negative advantage (A_t < 0), the update DECREASES the action probability:
   Delta log p = log p_new - log p_old < 0.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.structured_priority_policy import StructuredPriorityPolicy
from controllers.rl.structured_priority_ppo import build_sp_steps, structured_priority_ppo_update


def main():
    print("=" * 80)
    print("VERIFYING PPO LEARNING DIRECTION ON DIFFERENTIATED RETURNS")
    print("=" * 80)

    # 1. Initialize policy with sigma = 1.0 (log_std = 0.0)
    policy = StructuredPriorityPolicy()
    policy.log_std_w.data.fill_(0.0)
    policy.log_std_beta.data.fill_(0.0)

    bridge = Path(tempfile.gettempdir()) / f"verify_learn_dir_{os.getpid()}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "verify_learning_direction"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("Running sampled rollout on Seed 73200 (sigma=1.0)...")
    _, _, comp_val, trace = run_episode(
        WILAB_ROOT, bridge, 92000, 15, policy, 900.0,
        False, False, 100.0, 20.0, 73200, 100,
        None, 12, False, 175000,
        300, 12.0, "structured_priority", None,
        False, "off", "", True, 191000, gov,
        175000, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
        arrival_mode="streaming",
    )
    print(f"Sampled Rollout Completed Value: {comp_val:.2f} (Reference Deterministic: 71.03)")

    steps = build_sp_steps([trace], lam_L=0.005, gamma=0.995, gae_lambda=0.95)
    print(f"Extracted {len(steps)} valid transition steps.")
    if not steps:
        print("ERROR: No valid steps found in trace!")
        return

    # Check advantages
    advs = [s["adv"] for s in steps]
    print(f"Advantage stats: min={min(advs):.3f}, max={max(advs):.3f}, mean={np.mean(advs):.3f}, std={np.std(advs):.3f}")

    # Evaluate initial log-probabilities for all steps
    device = next(policy.parameters()).device
    initial_logps = []
    for s in steps:
        obs = s["sp_obs"]
        zw = torch.tensor([s["z_w"]], dtype=torch.float32, device=device)
        zb = torch.tensor([s["z_b"]], dtype=torch.float32, device=device)
        lp, _ = policy.evaluate_actions(
            obs["task_t"].to(device), obs["hlp_t"].to(device),
            obs["radio_t"].to(device), obs["budget_t"].to(device),
            zw, zb
        )
        initial_logps.append(float(lp.item()))

    # Perform PPO update
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)
    metrics = structured_priority_ppo_update(
        policy, optimizer, [trace], lam_L=0.005, epochs=4, minibatch_size=16
    )
    print(f"PPO Update completed: Actor Loss = {metrics.get('actor_loss'):.4f}, Value Loss = {metrics.get('value_loss'):.4f}, Grad Norm = {metrics.get('grad_norm'):.4f}")

    # Evaluate updated log-probabilities
    updated_logps = []
    for s in steps:
        obs = s["sp_obs"]
        zw = torch.tensor([s["z_w"]], dtype=torch.float32, device=device)
        zb = torch.tensor([s["z_b"]], dtype=torch.float32, device=device)
        lp, _ = policy.evaluate_actions(
            obs["task_t"].to(device), obs["hlp_t"].to(device),
            obs["radio_t"].to(device), obs["budget_t"].to(device),
            zw, zb
        )
        updated_logps.append(float(lp.item()))

    print("\n" + "=" * 80)
    print("STEP-BY-STEP POLICY GRADIENT ALIGNMENT AUDIT")
    print("=" * 80)
    print(f"{'Step':<5} | {'Advantage A_t':<15} | {'Initial log p':<14} | {'Updated log p':<14} | {'Delta log p':<13} | {'Direction':<10}")
    print("-" * 80)
    correct_directions = 0
    total_active = 0
    for i, (a, lp_init, lp_up) in enumerate(zip(advs, initial_logps, updated_logps)):
        d_lp = lp_up - lp_init
        # Positive advantage should increase logp (d_lp > 0)
        # Negative advantage should decrease logp (d_lp < 0)
        if a > 0.05:
            correct = (d_lp > 0)
            dir_str = "INCREASED" if d_lp > 0 else "DECREASED"
            expected = "INCREASE"
        elif a < -0.05:
            correct = (d_lp < 0)
            dir_str = "DECREASED" if d_lp < 0 else "INCREASED"
            expected = "DECREASE"
        else:
            correct = True
            dir_str = "NEUTRAL"
            expected = "NEUTRAL"

        if expected != "NEUTRAL":
            total_active += 1
            if correct:
                correct_directions += 1

        status = "CORRECT" if correct else "WRONG"
        print(f"{i:<5} | {a:<+15.3f} | {lp_init:<14.4f} | {lp_up:<14.4f} | {d_lp:<+13.4f} | {status} ({dir_str})")

    print("-" * 80)
    alignment_rate = (correct_directions / max(1, total_active)) * 100
    print(f"Policy Gradient Direction Alignment Rate: {correct_directions}/{total_active} ({alignment_rate:.1f}%)")
    print("=" * 80)

    # Save summary
    out_summary = runs_dir / "learning_direction_summary.json"
    out_summary.write_text(json.dumps({
        "comp_val": comp_val,
        "metrics": metrics,
        "alignment_rate": alignment_rate,
        "correct_directions": correct_directions,
        "total_active": total_active,
    }, indent=2), encoding="utf-8")
    print(f"Results saved to {out_summary}")


if __name__ == "__main__":
    main()
