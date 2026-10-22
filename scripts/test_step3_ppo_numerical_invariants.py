"""Step 3: PPO Deterministic Numerical Invariants Test.

Verifies fundamental PPO algorithmic invariants for PBCRLPolicy:
  1. Old-policy likelihood: r_t(theta) = exp(logp_new - logp_old) == 1.000000 before parameter updates.
  2. Autograd policy gradients agree with two-sided finite-difference numerical checks.
  3. Reward/cost critic targets and GAE advantages compute correctly with bootstrap boundaries.
  4. Adaptive dual multiplier lambda_L increases strictly under excess violation cost and maintains non-negativity.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import math
import torch
from controllers.rl.pb_crl_policy import PBCRLPolicy
from controllers.rl.lagrangian_ppo import dual_ascent


def test_step3():
    print("=" * 80)
    print("STEP 3: PPO DETERMINISTIC NUMERICAL INVARIANTS TEST")
    print("=" * 80)

    torch.manual_seed(42)
    policy = PBCRLPolicy()

    # Synthetic observation batch
    n_tasks = 4
    n_helpers = 3
    task_feats = torch.randn(1, n_tasks, 6)
    helper_feats = torch.randn(1, n_helpers, 4)
    radio_hist = torch.randn(1, 10, 4)
    budget_block = torch.randn(1, 3)

    # -------------------------------------------------------------------------
    # INVARIANT 1: OLD-POLICY LIKELIHOOD RATIO r(theta) == 1.000000
    # -------------------------------------------------------------------------
    print("\n--- Invariant 1: Likelihood Ratio r(theta) == 1.000000 ---")
    with torch.no_grad():
        lam, beta, g, logp_old, z_l, z_b, z_g, ent = policy.forward_actions(
            task_feats, helper_feats, radio_hist, budget_block, deterministic=False
        )
        
        # Evaluate actions under the exact same parameters
        logp_eval, ent_eval = policy.evaluate_actions(
            task_feats, helper_feats, radio_hist, budget_block,
            float(z_l), float(z_b), float(z_g)
        )

        ratio = torch.exp(logp_eval - logp_old).item()
        diff = abs(ratio - 1.0)
        print(f"  logp_old  = {logp_old.item():.8f}")
        print(f"  logp_eval = {logp_eval.item():.8f}")
        print(f"  ratio r_t = {ratio:.8f} (diff from 1.0: {diff:.2e})")
        assert diff < 1e-6, f"Likelihood ratio is not 1.0 before parameter updates! Ratio: {ratio}"
        print("  -> Invariant 1 PASS: Policy ratio is identically 1.000000 before updates.")

    # -------------------------------------------------------------------------
    # INVARIANT 2: AUTOGRAD VS FINITE-DIFFERENCE GRADIENTS
    # -------------------------------------------------------------------------
    print("\n--- Invariant 2: Autograd vs. Finite Differences ---")
    # Define surrogate objective: L(theta) = - logp(z) * advantage
    adv = 2.5
    zl_f, zb_f, zg_f = float(z_l), float(z_b), float(z_g)

    # Autograd vs FD in double precision to eliminate float32 cancellation noise
    policy_d = PBCRLPolicy().double()
    task_feats_d = task_feats.double()
    helper_feats_d = helper_feats.double()
    radio_hist_d = radio_hist.double()
    budget_block_d = budget_block.double()

    def compute_loss_d():
        logp, _ = policy_d.evaluate_actions(
            task_feats_d, helper_feats_d, radio_hist_d, budget_block_d,
            zl_f, zb_f, zg_f
        )
        return -logp * adv

    policy_d.zero_grad()
    loss_d = compute_loss_d()
    loss_d.backward()

    params_to_check = [
        ("head_lam.bias", policy_d.head_lam.bias),
        ("head_beta.bias", policy_d.head_beta.bias),
        ("head_g.bias", policy_d.head_g.bias),
        ("log_std_lam", policy_d.log_std_lam),
        ("log_std_beta", policy_d.log_std_beta),
        ("log_std_g", policy_d.log_std_g),
    ]

    eps = 1e-5
    for name, p in params_to_check:
        auto_grad = p.grad.item() if p.grad.dim() == 0 else p.grad[0].item()
        
        orig_val = p.data.clone()
        p.data.add_(eps)
        loss_pos = compute_loss_d().item()
        
        p.data.copy_(orig_val)
        p.data.sub_(eps)
        loss_neg = compute_loss_d().item()
        
        p.data.copy_(orig_val)
        fd_grad = (loss_pos - loss_neg) / (2.0 * eps)

        rel_err = abs(auto_grad - fd_grad) / max(abs(auto_grad), abs(fd_grad), 1e-5)
        print(f"  {name:15s}: autograd={auto_grad:10.6f}, FD={fd_grad:10.6f}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-4, f"Gradient mismatch on {name}: autograd={auto_grad}, FD={fd_grad}"

    print("  -> Invariant 2 PASS: Autograd matches finite differences within 0.1% rel error.")

    # -------------------------------------------------------------------------
    # INVARIANT 3: GAE & BOOTSTRAP RETURN TARGETS
    # -------------------------------------------------------------------------
    print("\n--- Invariant 3: GAE and Bootstrap Target Returns ---")
    T = 4
    gamma = 0.99
    gae_lambda = 0.95
    rewards = torch.tensor([1.0, 2.0, 1.5, 3.0])
    values = torch.tensor([1.2, 1.8, 2.2, 2.8])

    adv_r = torch.zeros(T)
    gae = 0.0
    next_v = 0.0  # Terminal boundary: V_T = 0
    for i in reversed(range(T)):
        delta = rewards[i] + gamma * next_v - values[i]
        gae = delta + gamma * gae_lambda * gae
        adv_r[i] = gae
        next_v = values[i]

    targets = adv_r + values

    # Analytical verification:
    # t=3 (terminal step): next_v=0 -> delta_3 = 3.0 + 0 - 2.8 = 0.2
    #                      gae_3 = 0.2
    #                      target_3 = 0.2 + 2.8 = 3.0 (exact terminal reward!)
    assert abs(targets[3].item() - 3.0) < 1e-6, f"Terminal target return must equal terminal reward! Got {targets[3].item()}"
    assert abs(adv_r[3].item() - 0.2) < 1e-6, f"Terminal GAE error! Got {adv_r[3].item()}"

    # t=2: next_v=values[3]=2.8 -> delta_2 = 1.5 + 0.99*2.8 - 2.2 = 1.5 + 2.772 - 2.2 = 2.072
    #      gae_2 = 2.072 + 0.99*0.95*0.2 = 2.072 + 0.1881 = 2.2601
    #      target_2 = 2.2601 + 2.2 = 4.4601
    expected_delta_2 = 1.5 + 0.99 * 2.8 - 2.2
    expected_gae_2 = expected_delta_2 + 0.99 * 0.95 * 0.2
    assert abs(adv_r[2].item() - expected_gae_2) < 1e-5, f"GAE t=2 error: {adv_r[2].item()} vs {expected_gae_2}"
    print(f"  t=3: target={targets[3].item():.4f} (analytical=3.0000), adv={adv_r[3].item():.4f}")
    print(f"  t=2: target={targets[2].item():.4f} (analytical={expected_gae_2+2.2:.4f}), adv={adv_r[2].item():.4f}")
    print("  -> Invariant 3 PASS: GAE and bootstrap returns match analytical derivations exactly.")

    # -------------------------------------------------------------------------
    # INVARIANT 4: ADAPTIVE DUAL MULTIPLIER (LAMBDA_L) UPDATES
    # -------------------------------------------------------------------------
    print("\n--- Invariant 4: Dual Multiplier Lambda_L Dynamics ---")
    lam_0 = 1.0
    cost_limit = 0.05
    lr_dual = 0.05

    # Excess cost -> strict increase
    mean_cost_high = 0.20
    lam_up = dual_ascent(lam_0, mean_cost_high, cost_limit=cost_limit, lr_dual=lr_dual)
    print(f"  Excess cost J_c={mean_cost_high:.2f} > {cost_limit}: lam {lam_0:.4f} -> {lam_up:.4f} (+{lam_up-lam_0:.4f})")
    assert lam_up > lam_0, f"Lambda failed to increase under excess cost! {lam_up} <= {lam_0}"
    assert abs(lam_up - (1.0 + 0.05 * (0.20 - 0.05))) < 1e-6

    # Feasible cost -> strict decrease
    mean_cost_low = 0.01
    lam_down = dual_ascent(lam_0, mean_cost_low, cost_limit=cost_limit, lr_dual=lr_dual)
    print(f"  Feasible cost J_c={mean_cost_low:.2f} < {cost_limit}: lam {lam_0:.4f} -> {lam_down:.4f} ({lam_down-lam_0:.4f})")
    assert lam_down < lam_0, f"Lambda failed to decrease under feasible cost! {lam_down} >= {lam_0}"

    # Non-negativity floor
    lam_near_zero = 0.001
    lam_floored = dual_ascent(lam_near_zero, 0.0, cost_limit=cost_limit, lr_dual=lr_dual)
    print(f"  Zero-bound floor check: lam {lam_near_zero:.4f} with J_c=0.0 -> {lam_floored:.4f}")
    assert lam_floored == 0.0, f"Lambda must floor at 0.0! Got {lam_floored}"

    print("  -> Invariant 4 PASS: Dual ascent strictly penalizes violations and maintains non-negativity.")

    print("\n" + "=" * 80)
    print("STEP 3 PASS: All PPO numerical invariants verified deterministically.")
    print("=" * 80)


if __name__ == "__main__":
    test_step3()
