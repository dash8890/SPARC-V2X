"""Unshielded Lagrangian PPO helpers (B4)."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from controllers.rl.residual_ppo import PPOStep, _sequence_logp


@dataclass
class LagPPOStep(PPOStep):
    cost_target: float = 0.0


def episode_violation_costs(trace, prr_floor=0.95, p95_ceiling=50.0):
    """Per-epoch union safety-violation costs; invalid windows → cost 0."""
    costs = []
    for item in trace:
        valid = bool(item.get("cbr_valid")) and bool(item.get("delay_valid"))
        if not valid:
            costs.append(0.0)
            continue
        prr = item.get("prr")
        p95 = item.get("p95_delay_ms")
        viol = False
        if prr is not None and float(prr) < prr_floor:
            viol = True
        if p95 is not None and float(p95) > p95_ceiling:
            viol = True
        costs.append(1.0 if viol else 0.0)
    return costs


def build_lagrangian_steps(traces, lam: float, gamma=0.99, gae_lambda=0.95,
                           suite_advantage: bool = False):
    """GAE on reward and cost.

    Default: advantage = A^r - λ A^c (legacy).
    suite_advantage=True (HQ-SPO B5): advantage = (A^r - λ A^c) / (1+λ).
    """
    raw = []
    value_targets = []
    cost_targets = []
    for trace in traces:
        if not trace:
            continue
        # Prefer raw reward increments (deadline-valid value); fall back to normalized.
        rewards = torch.tensor([
            float(x.get("reward", x.get("normalized_reward", 0.0))) for x in trace
        ])
        costs = torch.tensor(episode_violation_costs(trace), dtype=torch.float32)
        old_values = torch.tensor([float(x["value"].detach()) for x in trace])
        old_costs = torch.tensor([
            float(x["cost_value"].detach()) if torch.is_tensor(x.get("cost_value"))
            else float(x.get("cost_value", 0.0))
            for x in trace
        ])
        adv_r = torch.zeros_like(rewards)
        adv_c = torch.zeros_like(costs)
        gae_r = torch.tensor(0.0)
        gae_c = torch.tensor(0.0)
        next_v = torch.tensor(0.0)
        next_c = torch.tensor(0.0)
        for index in reversed(range(len(trace))):
            delta_r = rewards[index] + gamma * next_v - old_values[index]
            gae_r = delta_r + gamma * gae_lambda * gae_r
            adv_r[index] = gae_r
            next_v = old_values[index]
            delta_c = costs[index] + gamma * next_c - old_costs[index]
            gae_c = delta_c + gamma * gae_lambda * gae_c
            adv_c[index] = gae_c
            next_c = old_costs[index]
        ret_r = adv_r + old_values
        ret_c = adv_c + old_costs
        for item, a_r, a_c, tr, tc in zip(trace, adv_r, adv_c, ret_r, ret_c):
            value_targets.append((item, float(tr)))
            cost_targets.append((item, float(tc)))
            if not item.get("selected_candidate_indices"):
                continue
            mixed = float(a_r) - float(lam) * float(a_c)
            if suite_advantage:
                mixed = mixed / (1.0 + float(lam))
            raw.append((item, float(item["logp"].detach()), mixed, float(tr), float(tc)))
    if not raw:
        return [], value_targets, cost_targets
    adv = np.asarray([x[2] for x in raw], dtype=np.float64)
    adv = (adv - adv.mean()) / max(float(adv.std()), 1e-6)
    steps = [LagPPOStep(item=x[0], old_logp=x[1], advantage=float(a),
                        target_return=x[3], cost_target=x[4])
             for x, a in zip(raw, adv)]
    return steps, value_targets, cost_targets


def lagrangian_ppo_update(policy, optimizer, traces, envelope_bytes: int, lam: float,
                          epochs=4, minibatch_size=64, clip_ratio=0.2, value_coef=0.5,
                          cost_coef=0.5, entropy_coef=0.02, grad_clip=1.0,
                          unique_sources=True, target_kl=0.015,
                          gamma: float = 0.99, gae_lambda: float = 0.95,
                          suite_advantage: bool = False):
    steps, value_targets, cost_targets = build_lagrangian_steps(
        traces, lam=lam, gamma=gamma, gae_lambda=gae_lambda, suite_advantage=suite_advantage,
    )
    if not steps:
        return None
    losses = []
    early_stop = False
    for _ in range(int(epochs)):
        if early_stop:
            break
        random.shuffle(steps)
        for start in range(0, len(steps), int(minibatch_size)):
            batch = steps[start:start + int(minibatch_size)]
            policy_terms, entropy_terms, value_terms, cost_terms = [], [], [], []
            for step in batch:
                evaluated = _sequence_logp(policy, step.item, envelope_bytes, unique_sources)
                if evaluated is None:
                    continue
                new_logp, entropy = evaluated
                ratio = torch.exp(new_logp - torch.tensor(step.old_logp))
                advantage = torch.tensor(step.advantage)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
                policy_terms.append(-torch.minimum(unclipped, clipped))
                entropy_terms.append(entropy)
                obs = torch.tensor(step.item["observation"], dtype=torch.float32)
                value_terms.append((policy.value(obs).reshape(()) - torch.tensor(step.target_return)).square())
                if hasattr(policy, "cost_value"):
                    cost_terms.append(
                        (policy.cost_value(obs).reshape(()) - torch.tensor(step.cost_target)).square()
                    )
            if not policy_terms:
                continue
            actor_loss = torch.stack(policy_terms).mean()
            value_loss = torch.stack(value_terms).mean()
            entropy = torch.stack(entropy_terms).mean()
            cost_loss = torch.stack(cost_terms).mean() if cost_terms else torch.tensor(0.0)
            loss = actor_loss + value_coef * value_loss + cost_coef * cost_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip))
            optimizer.step()
            kl_terms = []
            with torch.no_grad():
                for step in batch:
                    evaluated = _sequence_logp(policy, step.item, envelope_bytes, unique_sources)
                    if evaluated is not None:
                        kl_terms.append(torch.tensor(step.old_logp) - evaluated[0])
            approx_kl = float(torch.stack(kl_terms).mean()) if kl_terms else 0.0
            losses.append((float(actor_loss.detach()), float(value_loss.detach()),
                           float(entropy.detach()), grad, approx_kl, float(cost_loss.detach())))
            if target_kl is not None and approx_kl > float(target_kl):
                early_stop = True
                break
    if not losses:
        return None
    arr = np.asarray(losses)
    mean_ep_cost = float(np.mean([
        sum(episode_violation_costs(tr)) / max(1, len(tr)) for tr in traces if tr
    ]))
    return {
        "actor_loss": float(arr[:, 0].mean()),
        "value_loss": float(arr[:, 1].mean()),
        "entropy": float(arr[:, 2].mean()),
        "gradient_norm": float(arr[:, 3].mean()),
        "approx_kl": float(arr[:, 4].mean()),
        "cost_loss": float(arr[:, 5].mean()),
        "mean_epoch_violation_rate": mean_ep_cost,
        "lambda": float(lam),
        "action_steps": len(steps),
        "ppo_minibatches": int(len(losses)),
    }


def dual_ascent(lam: float, mean_cost: float, cost_limit: float = 0.05, lr_dual: float = 0.05):
    """Nonnegative λ update: λ ← max(0, λ + η (J_c - κ))."""
    return float(max(0.0, lam + lr_dual * (mean_cost - cost_limit)))
