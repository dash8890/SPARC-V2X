"""Unit test demonstrating increased likelihood of a known beneficial representable action after one update."""
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.rl.structured_priority_policy import StructuredPriorityPolicy
from controllers.rl.structured_priority_ppo import structured_priority_ppo_update


def test_counterfactual_preference_loss_increases_likelihood():
    torch.manual_seed(42)
    policy = StructuredPriorityPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 1. Setup representative observation state
    task_t = torch.randn(1, 4, 5)
    hlp_t = torch.randn(1, 6, 4)
    radio_t = torch.randn(1, 10, 4)
    budget_t = torch.randn(1, 3)

    sp_obs = {
        "task_t": task_t,
        "hlp_t": hlp_t,
        "radio_t": radio_t,
        "budget_t": budget_t,
        "value": 0.0,
        "cost_value": 0.0,
    }

    # 2. Known beneficial target action (high value density, lower latency/queue penalty)
    # Target weights w = [3.5, 0.2, 0.2, 0.2], beta = 0.95
    target_w = torch.tensor([3.5, 0.2, 0.2, 0.2])
    target_z_w = torch.logit(torch.clamp(target_w / policy.w_max, 1e-4, 1.0 - 1e-4)).tolist()
    target_z_b = float(torch.logit(torch.tensor(0.95)))

    # Initial log prob under current policy
    policy.eval()
    with torch.no_grad():
        zw_t = torch.tensor([target_z_w], dtype=torch.float32)
        zb_t = torch.tensor([target_z_b], dtype=torch.float32)
        init_logp, _ = policy.evaluate_actions(task_t, hlp_t, radio_t, budget_t, zw_t, zb_t)
        init_logp_val = float(init_logp.item())

    # 3. Build a standard trace step
    with torch.no_grad():
        w_vec, beta, base_lp, z_w_sample, z_b_sample, _ = policy.forward_actions(task_t, hlp_t, radio_t, budget_t)

    trace_step = {
        "sp_obs": sp_obs,
        "z_w": z_w_sample.squeeze(0).tolist(),
        "z_b": float(z_b_sample.item()),
        "logp": base_lp,
        "reward": 1.0,
        "value": torch.tensor(0.0),
        "cost_value": torch.tensor(0.0),
        "cbr_valid": True,
        "delay_valid": True,
    }
    traces = [[trace_step]]

    cf_targets = [{
        "sp_obs": sp_obs,
        "target_z_w": target_z_w,
        "target_z_b": target_z_b,
    }]

    # 4. Perform one single update with auxiliary preference loss
    metrics = structured_priority_ppo_update(
        policy,
        optimizer,
        traces,
        lam_L=0.0,
        epochs=1,
        minibatch_size=1,
        cf_targets=cf_targets,
        alpha_pref=0.20,
    )

    # 5. Measure post-update log prob
    policy.eval()
    with torch.no_grad():
        post_logp, _ = policy.evaluate_actions(task_t, hlp_t, radio_t, budget_t, zw_t, zb_t)
        post_logp_val = float(post_logp.item())

    delta_logp = post_logp_val - init_logp_val
    print(f"\nInitial Target LogP: {init_logp_val:.4f} -> Post-Update Target LogP: {post_logp_val:.4f} (Delta = {delta_logp:+.4f})")
    assert delta_logp > 0.0, f"Expected target action likelihood to increase, but delta was {delta_logp:+.4f}"
    assert metrics.get("pref_loss", 0.0) != 0.0, "Expected pref_loss to be tracked in metrics"
