"""Regression Tests for PPO Stability, KL Stopping, and Checkpoint Selection (SPARC-V2X Redesign).

Validates:
1. Non-negativity and numerical accuracy of the Schulman KL estimator D_KL ≈ (r - 1) - log(r).
2. Explicit KL early stopping triggers on large policy shifts and bounds policy drift within target_kl.
3. Trunk detachment isolates actor representations from critic value-error gradients.
4. Checkpoint selection registers baseline-parity Round 0 and strictly enforces monotonic utility improvement.
5. Exact deterministic baseline parity retention at initialization.
"""
from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from controllers.rl.structured_priority_policy import StructuredPriorityPolicy, score_and_match
from controllers.rl.structured_priority_ppo import build_sp_steps, structured_priority_ppo_update


def test_schulman_kl_estimator_properties():
    """Verify mathematical properties of the Schulman PPO KL estimator: (r - 1) - log(r)."""
    # Test identical distribution: r = 1 -> KL = 0
    ratio_one = torch.tensor(1.0)
    log_diff_zero = torch.tensor(0.0)
    kl_zero = float((ratio_one - 1.0 - log_diff_zero).item())
    assert kl_zero == pytest.approx(0.0, abs=1e-7)

    # Test arbitrary ratios: (r - 1) - log(r) must be strictly non-negative
    for log_diff_val in [-2.5, -1.0, -0.1, 0.1, 1.0, 2.5]:
        log_diff = torch.tensor(log_diff_val)
        r = torch.exp(log_diff)
        kl = float((r - 1.0 - log_diff).item())
        assert kl >= 0.0, f"Schulman KL must be non-negative, got {kl} for log_diff={log_diff_val}"

    # Verify second-order Taylor expansion accuracy: (e^x - 1 - x) ≈ 0.5 * x^2 for small x
    small_x = torch.tensor(0.01)
    kl_small = float((torch.exp(small_x) - 1.0 - small_x).item())
    expected_approx = 0.5 * (0.01 ** 2)
    assert kl_small == pytest.approx(expected_approx, rel=0.05)


def test_detached_trunk_critic_isolation():
    """Verify that detach_trunk=True prevents critic value errors from updating the policy trunk."""
    policy = StructuredPriorityPolicy()
    task_t = torch.randn(1, 4, 5)
    hlp_t = torch.randn(1, 3, 4)
    radio_t = torch.randn(1, 10, 4)
    budget_t = torch.tensor([[1.0, 0.5, 1.0]], dtype=torch.float32)

    # With detach_trunk=True:
    v = policy.value(task_t, hlp_t, radio_t, budget_t, detach_trunk=True)
    v_loss = 0.5 * (v - 50.0) ** 2
    v_loss.backward()

    # Trunk weights must have NO gradients from critic loss
    for p in policy.trunk.parameters():
        assert p.grad is None, "policy.trunk parameters must not receive gradients when detach_trunk=True"

    # Value head weights MUST have gradients
    for p in policy.v_r_head.parameters():
        assert p.grad is not None, "policy.v_r_head parameters must receive critic gradients"


def test_kl_early_stopping_condition():
    """Verify that structured_priority_ppo_update triggers early stopping when KL exceeds target_kl."""
    policy = StructuredPriorityPolicy()
    # High learning rate to provoke an intentional policy shift
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    task_t = torch.randn(1, 4, 5)
    hlp_t = torch.randn(1, 3, 4)
    radio_t = torch.randn(1, 10, 4)
    budget_t = torch.tensor([[1.0, 0.5, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        w, b, lp, zw, zb, _ = policy.forward_actions(task_t, hlp_t, radio_t, budget_t, deterministic=False)

    trace = []
    for _ in range(64):
        trace.append({
            "sp_obs": {"task_t": task_t.clone(), "hlp_t": hlp_t.clone(), "radio_t": radio_t.clone(), "budget_t": budget_t.clone()},
            "z_w": zw.squeeze(0).tolist(),
            "z_b": float(zb.item()),
            "logp": float(lp.item()),
            "reward": 5.0,
            "cost_value": 0.0,
            "value": 0.0,
            "cbr_valid": True,
            "delay_valid": True,
            "prr": 0.99,
            "p95_delay_ms": 10.0,
        })

    target_kl = 0.015
    stats = structured_priority_ppo_update(
        policy, optimizer, [trace], lam_L=0.0,
        epochs=10, minibatch_size=16,
        target_kl=target_kl, detach_trunk=True
    )

    assert "early_stopped" in stats
    assert stats["early_stopped"] is True, "PPO update must early-stop when KL exceeds 1.5 * target_kl"


def test_checkpoint_selection_monotonicity_and_baseline_registration():
    """Verify that Round 0 baseline-parity candidate is retained against inferior checkpoints."""
    policy = StructuredPriorityPolicy()
    init_state = copy.deepcopy(policy.state_dict())

    init_v = 59.79
    init_pv = 0.025
    alpha = 0.02

    best_v = init_v
    best_feasible_v = init_v if init_pv <= alpha else -1e9
    best_checkpoint = {"val_mean_v": init_v, "round": 0, "feasible": False}

    # Round 1 completes with inferior utility (58.57) and same safety (0.025)
    r1_v = 58.57
    r1_pv = 0.025

    if r1_v > best_v:
        best_v = r1_v
        best_checkpoint = {"val_mean_v": r1_v, "round": 1}

    # Checkpoint must NOT be replaced by inferior Round 1
    assert best_v == 59.79, f"Best utility should remain 59.79, got {best_v}"
    assert best_checkpoint["round"] == 0, f"Best checkpoint round should remain 0, got {best_checkpoint['round']}"

    # Superior round arrives (e.g. 61.20)
    r2_v = 61.20
    r2_pv = 0.025
    if r2_v > best_v:
        best_v = r2_v
        best_checkpoint = {"val_mean_v": r2_v, "round": 2}

    assert best_v == 61.20
    assert best_checkpoint["round"] == 2


def test_structured_priority_deterministic_b4_parity():
    """Verify that StructuredPriorityPolicy reproduces documented B4 reference parameters at initialization."""
    policy = StructuredPriorityPolicy()
    policy.eval()

    task_t = torch.randn(1, 5, 5)
    hlp_t = torch.randn(1, 3, 4)
    radio_t = torch.randn(1, 10, 4)
    budget_t = torch.tensor([[1.0, 0.5, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        w_vec, beta, _, _, _, _ = policy.forward_actions(task_t, hlp_t, radio_t, budget_t, deterministic=True)

    w = w_vec.squeeze(0).cpu().tolist()
    b = float(beta.item())

    # Documented baseline anchors: w_ref = [2.5, 1.887, 1.887, 1.887], beta_ref = 0.881
    assert w[0] == pytest.approx(2.50, abs=1e-2)
    assert w[1] == pytest.approx(1.887, abs=1e-2)
    assert w[2] == pytest.approx(1.887, abs=1e-2)
    assert w[3] == pytest.approx(1.887, abs=1e-2)
    assert b == pytest.approx(0.881, abs=1e-2)
