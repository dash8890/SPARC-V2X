"""PPO Update Module for Structured Priority Controller (SPARC-V2X Redesign).

Implements Lagrangian PPO with twin critics (V_r, V_c) for StructuredPriorityPolicy:
  - Extracts causal observations (task_t, hlp_t, radio_t, budget_t) and action latents (z_w, z_b)
  - Computes GAE advantages for timely utility and safety violation cost
  - Evaluates PPO clipped surrogate objective on joint action log-probabilities
  - Updates policy trunk, actor heads (w, beta), and twin critics (V_r, V_c)
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from controllers.rl.lagrangian_ppo import episode_violation_costs


def build_sp_steps(
    traces: List[List[dict]],
    lam_L: float,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> List[dict]:
    raw = []
    for trace in traces:
        if not trace:
            continue
        rewards = torch.tensor([float(x.get("reward", 0.0)) for x in trace], dtype=torch.float32)
        costs = torch.tensor(episode_violation_costs(trace), dtype=torch.float32)
        old_v = torch.tensor([
            float(x["value"].detach().cpu()) if torch.is_tensor(x.get("value"))
            else float(x.get("value") or 0.0)
            for x in trace
        ], dtype=torch.float32)
        old_c = torch.tensor([
            float(x["cost_value"].detach().cpu()) if torch.is_tensor(x.get("cost_value"))
            else float(x.get("cost_value") or 0.0)
            for x in trace
        ], dtype=torch.float32)

        adv_r = torch.zeros_like(rewards)
        adv_c = torch.zeros_like(costs)
        gae_r = torch.tensor(0.0)
        gae_c = torch.tensor(0.0)
        next_v = torch.tensor(0.0)
        next_c = torch.tensor(0.0)

        for i in reversed(range(len(trace))):
            delta_r = rewards[i] + gamma * next_v - old_v[i]
            gae_r = delta_r + gamma * gae_lambda * gae_r
            adv_r[i] = gae_r
            next_v = old_v[i]

            delta_c = costs[i] + gamma * next_c - old_c[i]
            gae_c = delta_c + gamma * gae_lambda * gae_c
            adv_c[i] = gae_c
            next_c = old_c[i]

        ret_r = adv_r + old_v
        ret_c = adv_c + old_c

        for item, a_r, a_c, tr, tc in zip(trace, adv_r, adv_c, ret_r, ret_c):
            sp_obs = item.get("sp_obs")
            z_w = item.get("z_w")
            z_b = item.get("z_b")
            if sp_obs is None or z_w is None or z_b is None:
                continue

            mixed = (float(a_r) - float(lam_L) * float(a_c)) / (1.0 + float(lam_L))
            old_logp = item.get("logp")
            if torch.is_tensor(old_logp):
                old_logp = float(old_logp.detach().cpu())
            else:
                old_logp = float(old_logp or 0.0)

            raw.append({
                "sp_obs": sp_obs,
                "z_w": [float(x) for x in (z_w[0] if isinstance(z_w[0], list) else z_w)],
                "z_b": float(z_b),
                "old_logp": old_logp,
                "adv": mixed,
                "adv_r": float(a_r),
                "adv_c": float(a_c),
                "ret_r": float(tr),
                "ret_c": float(tc),
            })

    if not raw:
        return []

    if len(raw) > 1:
        adv_arr = np.asarray([x["adv"] for x in raw], dtype=np.float64)
        adv_std = float(adv_arr.std())
        if adv_std > 1e-6:
            adv_norm = (adv_arr - adv_arr.mean()) / adv_std
            for x, a in zip(raw, adv_norm):
                x["adv"] = float(a)

    return raw


def structured_priority_ppo_update(
    policy,
    optimizer: torch.optim.Optimizer,
    traces: List[List[dict]],
    lam_L: float,
    *,
    epochs: int = 4,
    minibatch_size: int = 32,
    clip_ratio: float = 0.2,
    value_coef: float = 0.1,
    cost_coef: float = 0.1,
    entropy_coef: float = 0.01,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> Dict[str, float]:
    steps = build_sp_steps(traces, lam_L, gamma=gamma, gae_lambda=gae_lambda)
    if not steps:
        return {}

    device = next(policy.parameters()).device
    policy.train()

    actor_losses = []
    value_losses = []
    cost_losses = []
    approx_kls = []

    grad_norms = []
    for _ in range(epochs):
        indices = list(range(len(steps)))
        random.shuffle(indices)

        for start in range(0, len(steps), minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batch = [steps[i] for i in batch_idx]
            if not batch:
                continue

            # Batch forward pass: individual forward passes per transition (variable candidate task counts)
            total_actor_loss = torch.tensor(0.0, device=device)
            total_value_loss = torch.tensor(0.0, device=device)
            total_cost_loss = torch.tensor(0.0, device=device)
            kl_accum = []

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
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2) - entropy_coef * ent

                pred_r = policy.value(task_t, hlp_t, radio_t, budget_t)
                pred_c = policy.cost_value(task_t, hlp_t, radio_t, budget_t)

                v_loss = 0.5 * (pred_r.squeeze() - target_r) ** 2
                c_loss = 0.5 * (pred_c.squeeze() - target_c) ** 2

                total_actor_loss = total_actor_loss + policy_loss
                total_value_loss = total_value_loss + v_loss
                total_cost_loss = total_cost_loss + c_loss

                with torch.no_grad():
                    kl = (old_logp - logp).detach().cpu().item()
                    kl_accum.append(kl)

            bsz = max(1, len(batch))
            loss = (
                (total_actor_loss / bsz)
                + value_coef * (total_value_loss / bsz)
                + cost_coef * (total_cost_loss / bsz)
            )

            optimizer.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            grad_norms.append(float(gnorm.item()))
            optimizer.step()

            actor_losses.append(float((total_actor_loss / bsz).detach().cpu()))
            value_losses.append(float((total_value_loss / bsz).detach().cpu()))
            cost_losses.append(float((total_cost_loss / bsz).detach().cpu()))
            approx_kls.append(float(np.mean(kl_accum)))

    costs_all = [float(x) for tr in traces for x in episode_violation_costs(tr)]
    adv_r_all = [s["adv_r"] for s in steps if "adv_r" in s]
    adv_c_all = [s["adv_c"] for s in steps if "adv_c" in s]
    return {
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "cost_loss": float(np.mean(cost_losses)) if cost_losses else 0.0,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "mean_epoch_violation_rate": float(np.mean(costs_all)) if costs_all else 0.0,
        "mean_adv_r": float(np.mean(adv_r_all)) if adv_r_all else 0.0,
        "mean_adv_c": float(np.mean(adv_c_all)) if adv_c_all else 0.0,
    }
