"""Lagrangian PPO for B5′ continuous (λ, β) heads — not pack-sequence logπ."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from controllers.rl.lagrangian_ppo import episode_violation_costs


def build_b5p_steps(traces, lam_L: float, gamma=0.995, gae_lambda=0.95):
    raw = []
    for trace in traces:
        if not trace:
            continue
        rewards = torch.tensor([float(x.get("reward", 0.0)) for x in trace])
        costs = torch.tensor(episode_violation_costs(trace), dtype=torch.float32)
        old_v = torch.tensor([
            float(x["value"].detach()) if torch.is_tensor(x.get("value")) else float(x.get("value") or 0.0)
            for x in trace
        ])
        old_c = torch.tensor([
            float(x["cost_value"].detach()) if torch.is_tensor(x.get("cost_value"))
            else float(x.get("cost_value") or 0.0)
            for x in trace
        ])
        adv_r = torch.zeros_like(rewards)
        adv_c = torch.zeros_like(costs)
        gae_r = gae_c = torch.tensor(0.0)
        next_v = next_c = torch.tensor(0.0)
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
            if not item.get("pb_state") or item.get("pb_z") is None:
                continue
            mixed = (float(a_r) - float(lam_L) * float(a_c)) / (1.0 + float(lam_L))
            old_logp = item["logp"]
            if torch.is_tensor(old_logp):
                old_logp = float(old_logp.detach())
            else:
                old_logp = float(old_logp or 0.0)
            raw.append({
                "item": item,
                "old_logp": old_logp,
                "adv": mixed,
                "ret_r": float(tr),
                "ret_c": float(tc),
            })
    if not raw:
        return []
    adv = np.asarray([x["adv"] for x in raw], dtype=np.float64)
    adv = (adv - adv.mean()) / max(float(adv.std()), 1e-6)
    for x, a in zip(raw, adv):
        x["adv"] = float(a)
    return raw


def b5_prime_ppo_update(
    policy, optimizer, traces, lam_L: float,
    *, epochs=4, minibatch_size=64, clip_ratio=0.2,
    value_coef=0.1, cost_coef=0.1, entropy_coef=0.01,
    grad_clip=1.0, target_kl=0.03, gamma=0.995, gae_lambda=0.95,
) -> Optional[Dict[str, Any]]:
    steps = build_b5p_steps(traces, lam_L, gamma=gamma, gae_lambda=gae_lambda)
    if not steps:
        return None
    losses = []
    early = False
    for _ in range(int(epochs)):
        if early:
            break
        random.shuffle(steps)
        for start in range(0, len(steps), int(minibatch_size)):
            batch = steps[start:start + int(minibatch_size)]
            pol_t, ent_t, v_t, c_t = [], [], [], []
            for st in batch:
                item = st["item"]
                z = item["pb_z"]
                new_logp, ent = policy.evaluate_actions(
                    item["pb_state"], float(z[0]), float(z[1]),
                )
                ratio = torch.exp(new_logp - torch.tensor(st["old_logp"]))
                adv = torch.tensor(st["adv"])
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
                pol_t.append(-torch.minimum(unclipped, clipped))
                ent_t.append(ent)
                obs = torch.tensor(item["pb_state"], dtype=torch.float32)
                v_t.append((policy.value(obs).reshape(()) - st["ret_r"]).square())
                c_t.append((policy.cost_value(obs).reshape(()) - st["ret_c"]).square())
            if not pol_t:
                continue
            loss = (
                torch.stack(pol_t).mean()
                + value_coef * torch.stack(v_t).mean()
                + cost_coef * torch.stack(c_t).mean()
                - entropy_coef * torch.stack(ent_t).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip))
            optimizer.step()
            with torch.no_grad():
                kl = []
                for st in batch:
                    z = st["item"]["pb_z"]
                    new_logp, _ = policy.evaluate_actions(
                        st["item"]["pb_state"], float(z[0]), float(z[1]),
                    )
                    kl.append(torch.tensor(st["old_logp"]) - new_logp)
                approx_kl = float(torch.stack(kl).mean()) if kl else 0.0
            losses.append((
                float(torch.stack(pol_t).mean().detach()),
                float(torch.stack(v_t).mean().detach()),
                float(torch.stack(ent_t).mean().detach()),
                grad, approx_kl,
            ))
            if target_kl is not None and approx_kl > float(target_kl):
                early = True
                break
    if not losses:
        return None
    arr = np.asarray(losses)
    mean_cost = float(np.mean([
        sum(episode_violation_costs(tr)) / max(1, len(tr)) for tr in traces if tr
    ]))
    return {
        "actor_loss": float(arr[:, 0].mean()),
        "value_loss": float(arr[:, 1].mean()),
        "entropy": float(arr[:, 2].mean()),
        "gradient_norm": float(arr[:, 3].mean()),
        "approx_kl": float(arr[:, 4].mean()),
        "mean_epoch_violation_rate": mean_cost,
        "action_steps": len(steps),
    }
