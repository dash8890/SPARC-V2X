"""Causal Diagnostic of Gradient Alignment: Utility vs. Safety Penalty.

Measures:
1. Exact gradient vectors:
   - g_R: Policy gradient of deadline-valid completed utility (A^r)
   - g_C: Policy gradient of safety violation cost (A^c)
2. Cosine similarity between g_R and -g_C:
   - cos(g_R, -g_C): Are utility and safety updates aligned (>0), orthogonal (~0), or conflicting (<0)?
3. Counterfactual parameter step evaluation:
   - theta_R = theta_0 + alpha * g_R (Pure utility update)
   - theta_mixed = theta_0 + alpha * (g_R - lam_L * g_C) (Lagrangian update)
4. Closed-loop validation on seeds [73300, 73301, 73302, 73303] under both checkpoints.
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

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.structured_priority_policy import StructuredPriorityPolicy
from controllers.rl.structured_priority_ppo import build_sp_steps
from controllers.rl.lagrangian_ppo import episode_violation_costs
from metrics_logger import safety_from_audit_list


def evaluate_policy_closed_loop(policy, seeds, runs_dir, prefix_eid):
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    vals = []
    pviols = []
    for i, s in enumerate(seeds):
        bridge = Path(tempfile.gettempdir()) / f"eval_{prefix_eid}_{os.getpid()}_{s}"
        eid = prefix_eid + i
        t_seed = 186100 + (s - 73300)
        _, _, v, _ = run_episode(
            WILAB_ROOT, bridge, eid, 15, policy, 900.0,
            False, False, 100.0, 20.0, s, 100,
            None, 12, False, 175000,
            300, 12.0, "structured_priority", None,
            True, "off", "", True, t_seed, gov,
            175000, True, True, 1.0, None, None, None, 2e8, 1e9,
            output_root=runs_dir, arrival_mode="streaming"
        )
        ep = runs_dir / f"episode_{eid:04d}"
        audit = json.loads((ep / "bridge_audit.json").read_text(encoding="utf-8"))
        safety = safety_from_audit_list(audit)
        vals.append(v)
        pviols.append(float(safety.get("P_viol", 0.0)))
    return np.mean(vals), np.mean(pviols), vals, pviols


def main():
    print("=" * 80)
    print("CAUSAL GRADIENT ALIGNMENT DIAGNOSTIC: UTILITY VS. SAFETY PENALTY")
    print("=" * 80)

    # 1. Base policy at default initialization (59.79 B4 parity)
    policy_base = StructuredPriorityPolicy()
    device = next(policy_base.parameters()).device

    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "gradient_alignment_diag"
    runs_dir.mkdir(parents=True, exist_ok=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)

    # 2. Collect 2 training episodes on training seeds (e.g. 73200, 73201)
    print("Collecting training transitions on Seeds 73200 (even) and 73201 (odd)...")
    traces = []
    for i, s in enumerate([73200, 73201]):
        bridge = Path(tempfile.gettempdir()) / f"tr_align_{os.getpid()}_{s}"
        eid = 97000 + i
        t_seed = 191000 + i
        _, _, comp_val, tr = run_episode(
            WILAB_ROOT, bridge, eid, 15, policy_base, 900.0,
            False, False, 100.0, 20.0, s, 100,
            None, 12, False, 175000,
            300, 12.0, "structured_priority", None,
            False, "off", "", True, t_seed, gov,
            175000, True, True, 1.0, None, None, None, 2e8, 1e9,
            output_root=runs_dir, arrival_mode="streaming"
        )
        traces.append(tr)
        print(f"  Training Seed {s}: Completed Value = {comp_val:.2f}, Trace length = {len(tr)}")

    # 3. Extract separate reward advantages A^r and cost advantages A^c
    steps_pure_r = build_sp_steps(traces, lam_L=0.0, gamma=0.995, gae_lambda=0.95)
    print(f"\nExtracted {len(steps_pure_r)} informative transition steps.")

    # Compute raw grad of utility: g_R
    policy_r = copy.deepcopy(policy_base)
    optimizer_r = torch.optim.Adam(policy_r.parameters(), lr=2e-3)

    total_loss_r = torch.tensor(0.0, device=device)
    for s in steps_pure_r:
        obs = s["sp_obs"]
        zw = torch.tensor([s["z_w"]], dtype=torch.float32, device=device)
        zb = torch.tensor([s["z_b"]], dtype=torch.float32, device=device)
        logp, ent = policy_r.evaluate_actions(
            obs["task_t"].to(device), obs["hlp_t"].to(device),
            obs["radio_t"].to(device), obs["budget_t"].to(device), zw, zb
        )
        old_logp = torch.tensor(s["old_logp"], dtype=torch.float32, device=device)
        adv_r = torch.tensor(s["adv"], dtype=torch.float32, device=device)
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * adv_r
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_r
        loss = -torch.min(surr1, surr2).squeeze() - 0.01 * ent.squeeze()
        total_loss_r += loss

    total_loss_r = total_loss_r / len(steps_pure_r)
    optimizer_r.zero_grad()
    total_loss_r.backward()

    # Extract grad vector g_R on actor heads
    actor_params_r = list(policy_r.head_w.parameters()) + list(policy_r.head_beta.parameters())
    g_R = torch.cat([p.grad.flatten() for p in actor_params_r if p.grad is not None])

    # Compute raw grad of safety cost: g_C
    policy_c = copy.deepcopy(policy_base)
    optimizer_c = torch.optim.Adam(policy_c.parameters(), lr=2e-3)

    # Use pure cost advantages
    costs_all = [float(x) for tr in traces for x in episode_violation_costs(tr)]
    mean_cost = float(np.mean(costs_all)) if costs_all else 0.0
    print(f"Mean training cost violation rate: {mean_cost*100:.1f}%")

    # Build cost-only steps
    steps_cost = copy.deepcopy(steps_pure_r)
    # Compute GAE cost advantages
    for tr, s_subset in zip(traces, [steps_cost[:len(steps_pure_r)//2], steps_cost[len(steps_pure_r)//2:]]):
        c_costs = torch.tensor(episode_violation_costs(tr), dtype=torch.float32)
        old_c = torch.tensor([float(x.get("cost_value") or 0.0) for x in tr], dtype=torch.float32)
        adv_c = torch.zeros_like(c_costs)
        gae_c = torch.tensor(0.0)
        next_c = torch.tensor(0.0)
        for i in reversed(range(len(tr))):
            delta_c = c_costs[i] + 0.995 * next_c - old_c[i]
            gae_c = delta_c + 0.995 * 0.95 * gae_c
            adv_c[i] = gae_c
            next_c = old_c[i]

    total_loss_c = torch.tensor(0.0, device=device)
    for s in steps_pure_r:
        obs = s["sp_obs"]
        zw = torch.tensor([s["z_w"]], dtype=torch.float32, device=device)
        zb = torch.tensor([s["z_b"]], dtype=torch.float32, device=device)
        logp, ent = policy_c.evaluate_actions(
            obs["task_t"].to(device), obs["hlp_t"].to(device),
            obs["radio_t"].to(device), obs["budget_t"].to(device), zw, zb
        )
        old_logp = torch.tensor(s["old_logp"], dtype=torch.float32, device=device)
        # Gradient of cost: minimizing cost is maximizing -cost, so loss = +ratio * adv_c
        ret_c = torch.tensor(s["ret_c"], dtype=torch.float32, device=device)
        ratio = torch.exp(logp - old_logp)
        surr = ratio * ret_c  # cost penalty direction
        loss = surr.squeeze()
        total_loss_c += loss

    total_loss_c = total_loss_c / len(steps_pure_r)
    optimizer_c.zero_grad()
    total_loss_c.backward()

    actor_params_c = list(policy_c.head_w.parameters()) + list(policy_c.head_beta.parameters())
    g_C = torch.cat([p.grad.flatten() for p in actor_params_c if p.grad is not None])

    # 4. Measure alignment and cosine similarity
    norm_R = g_R.norm().item()
    norm_C = g_C.norm().item()
    dot_prod = torch.dot(g_R, g_C).item()
    cos_sim = dot_prod / (norm_R * norm_C + 1e-8)

    print("\n" + "=" * 80)
    print("GRADIENT ALIGNMENT ANALYSIS:")
    print(f"  Utility Gradient Norm ||g_R||:      {norm_R:.4f}")
    print(f"  Safety Cost Gradient Norm ||g_C||:  {norm_C:.4f}")
    print(f"  Dot Product <g_R, g_C>:             {dot_prod:+.4f}")
    print(f"  Cosine Similarity cos(g_R, -g_C):  {-cos_sim:+.4f}")
    if -cos_sim < -0.2:
        print("  -> CONFLICT: Utility and Safety gradients are in direct opposition!")
    elif abs(cos_sim) <= 0.2:
        print("  -> ORTHOGONAL: Utility and Safety gradients are operating in independent subspaces.")
    else:
        print("  -> ALIGNED: Utility and Safety gradients push in the same direction.")
    print("=" * 80)

    # 5. Take 1 step on pure utility vs 1 step with dual penalty (lam_L = 0.01)
    lam_L_test = 0.01
    policy_mixed = copy.deepcopy(policy_base)
    optimizer_mixed = torch.optim.Adam(policy_mixed.parameters(), lr=2e-3)

    total_loss_mixed = torch.tensor(0.0, device=device)
    for s in steps_pure_r:
        obs = s["sp_obs"]
        zw = torch.tensor([s["z_w"]], dtype=torch.float32, device=device)
        zb = torch.tensor([s["z_b"]], dtype=torch.float32, device=device)
        logp, ent = policy_mixed.evaluate_actions(
            obs["task_t"].to(device), obs["hlp_t"].to(device),
            obs["radio_t"].to(device), obs["budget_t"].to(device), zw, zb
        )
        old_logp = torch.tensor(s["old_logp"], dtype=torch.float32, device=device)
        adv_r = torch.tensor(s["adv"], dtype=torch.float32, device=device)
        ret_c = torch.tensor(s["ret_c"], dtype=torch.float32, device=device)
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * adv_r
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_r
        surr_u = -torch.min(surr1, surr2).squeeze() - 0.01 * ent.squeeze()
        surr_c = ratio.squeeze() * ret_c
        total_loss_mixed += (surr_u + lam_L_test * surr_c)

    total_loss_mixed = total_loss_mixed / len(steps_pure_r)
    optimizer_mixed.zero_grad()
    total_loss_mixed.backward()

    optimizer_r.step()
    optimizer_mixed.step()

    # 6. Evaluate both updated policies on validation seeds [73300, 73301, 73302, 73303]
    val_seeds = [73300, 73301, 73302, 73303]
    print("\nEvaluating Base Policy on Validation Seeds...")
    mean_base_v, mean_base_pv, vals_base, _ = evaluate_policy_closed_loop(policy_base, val_seeds, runs_dir, 98000)
    print(f"  Base Policy:  Mean V = {mean_base_v:.2f} (Per-seed: {[round(x, 2) for x in vals_base]}) | P_viol = {mean_base_pv*100:.2f}%")

    print("\nEvaluating Pure Utility Policy (theta_R, lam_L=0) on Validation Seeds...")
    mean_r_v, mean_r_pv, vals_r, _ = evaluate_policy_closed_loop(policy_r, val_seeds, runs_dir, 98100)
    print(f"  Pure Utility: Mean V = {mean_r_v:.2f} (Per-seed: {[round(x, 2) for x in vals_r]}) | P_viol = {mean_r_pv*100:.2f}%")

    print(f"\nEvaluating Lagrangian Mixed Policy (theta_mixed, lam_L={lam_L_test}) on Validation Seeds...")
    mean_m_v, mean_m_pv, vals_m, _ = evaluate_policy_closed_loop(policy_mixed, val_seeds, runs_dir, 98200)
    print(f"  Mixed Policy: Mean V = {mean_m_v:.2f} (Per-seed: {[round(x, 2) for x in vals_m]}) | P_viol = {mean_m_pv*100:.2f}%")

    print("\n" + "=" * 80)
    print("COUNTERFACTUAL COMPARISON SUMMARY:")
    print(f"{'Metric':<25} | {'Base (Init)':<15} | {'Pure Utility (lam=0)':<22} | {'Lagrangian (lam=0.01)':<22}")
    print("-" * 80)
    print(f"{'Mean Completed Value':<25} | {mean_base_v:<15.2f} | {mean_r_v:<22.2f} | {mean_m_v:<22.2f}")
    print(f"{'Delta vs. B4 (59.79)':<25} | {mean_base_v - 59.79:<+15.2f} | {mean_r_v - 59.79:<+22.2f} | {mean_m_v - 59.79:<+22.2f}")
    print(f"{'Seed 73301 Value':<25} | {vals_base[1]:<15.2f} | {vals_r[1]:<22.2f} | {vals_m[1]:<22.2f}")
    print(f"{'Mean Safety Violation':<25} | {mean_base_pv*100:<14.2f}% | {mean_r_pv*100:<21.2f}% | {mean_m_pv*100:<21.2f}%")
    print("=" * 80)

    out_file = runs_dir / "gradient_alignment_summary.json"
    out_file.write_text(json.dumps({
        "norm_R": norm_R,
        "norm_C": norm_C,
        "dot_prod": dot_prod,
        "cos_sim": cos_sim,
        "mean_base_v": mean_base_v,
        "mean_r_v": mean_r_v,
        "mean_m_v": mean_m_v,
        "vals_base": vals_base,
        "vals_r": vals_r,
        "vals_m": vals_m,
    }, indent=2), encoding="utf-8")
    print(f"Saved summary to {out_file}")


if __name__ == "__main__":
    main()
