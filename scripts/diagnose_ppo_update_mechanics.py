"""Detailed Diagnostic of the PPO Update Mechanics.

Measures:
1. Actor-only gradient norm vs Critic gradient norm vs Total network gradient norm before clipping.
2. Clipping frequency and clipping compression factor.
3. Actual parameter delta per PPO update epoch on actor heads (head_w, head_beta).
4. Probability ratio distribution: r_t(theta) = exp(logp - old_logp), clipping fraction at epsilon=0.2.
5. Advantage attribution: Correlation between GAE advantages, timely task rewards, and safety violation costs.
"""
from __future__ import annotations

import copy
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

from controllers.rl.structured_priority_policy import StructuredPriorityPolicy
from controllers.rl.structured_priority_ppo import build_sp_steps
from controllers.rl.lagrangian_ppo import episode_violation_costs


def diagnose_trace_update(traces: list[list[dict]], lam_L: float = 0.005):
    print("=" * 80)
    print("DETAILED PPO UPDATE MECHANICS DIAGNOSTIC")
    print("=" * 80)

    policy = StructuredPriorityPolicy()
    init_state = copy.deepcopy(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)

    steps = build_sp_steps(traces, lam_L=lam_L, gamma=0.995, gae_lambda=0.95)
    print(f"\n1. Step Extraction: {len(steps)} valid transition steps extracted from {len(traces)} traces.")

    if not steps:
        print("ERROR: No valid steps found!")
        return

    # Check advantage alignment
    advs = np.array([s["adv"] for s in steps])
    rets_r = np.array([s["ret_r"] for s in steps])
    rets_c = np.array([s["ret_c"] for s in steps])
    print(f"   Advantage Range: [{advs.min():.3f}, {advs.max():.3f}] | Mean: {advs.mean():.3f} | Std: {advs.std():.3f}")
    print(f"   Reward Return Range: [{rets_r.min():.2f}, {rets_r.max():.2f}] | Mean: {rets_r.mean():.2f}")
    print(f"   Cost Return Range:   [{rets_c.min():.2f}, {rets_c.max():.2f}] | Mean: {rets_c.mean():.2f}")

    # Inspect gradients on 1 minibatch
    batch = steps[:min(32, len(steps))]
    device = next(policy.parameters()).device

    total_actor_loss = torch.tensor(0.0, device=device)
    total_value_loss = torch.tensor(0.0, device=device)
    total_cost_loss = torch.tensor(0.0, device=device)
    ratios = []

    for s in batch:
        obs = s["sp_obs"]
        task_t = obs["task_t"].to(device)
        hlp_t = obs["hlp_t"].to(device)
        radio_t = obs["radio_t"].to(device)
        budget_t = obs["budget_t"].to(device)

        zw = torch.tensor([s["z_w"]], dtype=torch.float32, device=device)
        zb = torch.tensor([s["z_b"]], dtype=torch.float32, device=device)

        logp, ent = policy.evaluate_actions(task_t, hlp_t, radio_t, budget_t, zw, zb)
        old_logp = torch.tensor(s["old_logp"], dtype=torch.float32, device=device)
        adv = torch.tensor(s["adv"], dtype=torch.float32, device=device)
        target_r = torch.tensor(s["ret_r"], dtype=torch.float32, device=device)
        target_c = torch.tensor(s["ret_c"], dtype=torch.float32, device=device)

        ratio = torch.exp(logp - old_logp)
        ratios.append(float(ratio.item()))
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
        policy_loss = -torch.min(surr1, surr2) - 0.01 * ent

        pred_r = policy.value(task_t, hlp_t, radio_t, budget_t)
        pred_c = policy.cost_value(task_t, hlp_t, radio_t, budget_t)
        v_loss = 0.5 * (pred_r.squeeze() - target_r) ** 2
        c_loss = 0.5 * (pred_c.squeeze() - target_c) ** 2

        total_actor_loss += policy_loss.squeeze()
        total_value_loss += v_loss.squeeze()
        total_cost_loss += c_loss.squeeze()

    bsz = len(batch)
    actor_loss = total_actor_loss / bsz
    value_loss = 0.1 * (total_value_loss / bsz)
    cost_loss = 0.1 * (total_cost_loss / bsz)
    total_loss = actor_loss + value_loss + cost_loss

    optimizer.zero_grad()
    total_loss.backward()

    # Decompose gradient norms
    actor_params = list(policy.head_w.parameters()) + list(policy.head_beta.parameters())
    critic_params = list(policy.v_r_head.parameters()) + list(policy.v_c_head.parameters())
    trunk_params = list(policy.trunk.parameters()) + list(policy.task_enc.parameters()) + list(policy.helper_enc.parameters()) + list(policy.radio_gru.parameters())

    def param_norm(params):
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        return torch.sqrt(sum(g.norm()**2 for g in grads)).item()

    norm_actor = param_norm(actor_params)
    norm_critic = param_norm(critic_params)
    norm_trunk = param_norm(trunk_params)
    norm_total = param_norm(policy.parameters())

    print(f"\n2. Gradient Decomposition (Before Clipping):")
    print(f"   Actor Heads Grad Norm:   {norm_actor:.4f}")
    print(f"   Critic Heads Grad Norm:  {norm_critic:.4f}")
    print(f"   Trunk & Encoders Norm:   {norm_trunk:.4f}")
    print(f"   Total Network Grad Norm: {norm_total:.4f}")

    clip_factor = min(1.0, 0.5 / max(norm_total, 1e-6))
    print(f"   Global Clip (max=0.5) Factor: {clip_factor:.4f} (gradients scaled to {clip_factor*100:.2f}% of original)")
    print(f"   Effective Actor Head Norm After Clipping: {norm_actor * clip_factor:.6f}")

    # Step optimizer and measure parameter delta
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()

    updated_state = policy.state_dict()
    print(f"\n3. Parameter Shifts (After 1 Adam Step with lr=2e-3):")
    for k in ["head_w.weight", "head_w.bias", "head_beta.weight", "head_beta.bias"]:
        d = (updated_state[k] - init_state[k]).abs().max().item()
        print(f"   {k}: max delta = {d:.6f}")

    print(f"\n4. Probability Ratios r_t(theta):")
    r_arr = np.array(ratios)
    clipped_frac = np.mean((r_arr < 0.8) | (r_arr > 1.2))
    print(f"   Ratio Mean: {r_arr.mean():.4f} | Min: {r_arr.min():.4f} | Max: {r_arr.max():.4f}")
    print(f"   Fraction Clipped by [0.8, 1.2]: {clipped_frac*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
    from controllers.rl.safety_governor import SafetyGovernor

    print("Generating matched training trace on Seed 73200...")
    bridge = Path(tempfile.gettempdir()) / f"diag_trace_{os.getpid()}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    policy = StructuredPriorityPolicy()

    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "diag_trace_run"
    runs_dir.mkdir(parents=True, exist_ok=True)

    _, _, comp_val, trace = run_episode(
        WILAB_ROOT, bridge, 88888, 15, policy, 900.0,
        False, False, 100.0, 20.0, 73200, 100,
        None, 12, False, 175000,
        300, 12.0, "structured_priority", None,
        False, "off", "", True, 191000, gov,
        175000, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
        arrival_mode="streaming",
    )
    print(f"Rollout complete! Completed Value: {comp_val:.2f}, Trace length: {len(trace)}")
    diagnose_trace_update([trace], lam_L=0.005)
