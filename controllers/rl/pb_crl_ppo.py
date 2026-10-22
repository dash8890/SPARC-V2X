"""PB-CRL 3-Head Lagrangian PPO Update Module.

Implements Proximal Policy Optimization with Lagrangian relaxation for the full 
3-head PBCRLPolicy architecture (DeepSets task encoder, helper encoder, radio GRU, budget block).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from controllers.rl.lagrangian_ppo import episode_violation_costs


def build_pb_crl_steps(traces: List[List[dict]], lam_L: float, gamma: float = 0.995, gae_lambda: float = 0.95) -> List[dict]:
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
            full_obs = item.get("pb_full_obs")
            pb_z = item.get("pb_z")
            if not full_obs or pb_z is None or len(pb_z) < 3:
                continue

            mixed = (float(a_r) - float(lam_L) * float(a_c)) / (1.0 + float(lam_L))
            old_logp = item.get("logp")
            if torch.is_tensor(old_logp):
                old_logp = float(old_logp.detach().cpu())
            else:
                old_logp = float(old_logp or 0.0)

            raw.append({
                "full_obs": full_obs,
                "pb_z": [float(z) for z in pb_z[:3]],
                "old_logp": old_logp,
                "adv": mixed,
                "ret_r": float(tr),
                "ret_c": float(tc),
            })

    if not raw:
        return []

    adv_arr = np.asarray([x["adv"] for x in raw], dtype=np.float64)
    adv_std = float(adv_arr.std())
    adv_norm = (adv_arr - adv_arr.mean()) / max(adv_std, 1e-6)
    for x, a in zip(raw, adv_norm):
        x["adv"] = float(a)

    return raw


def pb_crl_ppo_update(
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
    grad_clip: float = 1.0,
    target_kl: float = 0.03,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Executes PPO update on PBCRLPolicy over collected episode traces."""
    steps = build_pb_crl_steps(traces, lam_L, gamma=gamma, gae_lambda=gae_lambda)
    if not steps:
        return None

    policy.train()
    batch_stats = []
    early_stop = False

    for epoch in range(int(epochs)):
        if early_stop:
            break
        random.shuffle(steps)

        for start in range(0, len(steps), int(minibatch_size)):
            mb = steps[start:start + int(minibatch_size)]

            pol_losses = []
            ent_losses = []
            v_losses = []
            c_losses = []
            kl_terms = []

            for st in mb:
                fobs = st["full_obs"]
                z = st["pb_z"]
                tf = torch.tensor([fobs["task_feats"]], dtype=torch.float32)
                hf = torch.tensor([fobs["helper_feats"]], dtype=torch.float32)
                rh = torch.tensor([fobs["radio_hist"]], dtype=torch.float32)
                bb = torch.tensor([fobs["budget_block"]], dtype=torch.float32)

                new_logp, ent = policy.evaluate_actions(tf, hf, rh, bb, z[0], z[1], z[2])
                old_logp = torch.tensor(st["old_logp"], dtype=torch.float32)

                ratio = torch.exp(new_logp - old_logp)
                adv = torch.tensor(st["adv"], dtype=torch.float32)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                pol_loss = -torch.min(surr1, surr2)

                pol_losses.append(pol_loss)
                ent_losses.append(ent)

                v_pred = policy.value(tf, hf, rh, bb).reshape(())
                c_pred = policy.cost_value(tf, hf, rh, bb).reshape(())

                v_losses.append((v_pred - st["ret_r"]).square())
                c_losses.append((c_pred - st["ret_c"]).square())

                with torch.no_grad():
                    kl_terms.append(old_logp - new_logp)

            actor_loss = torch.stack(pol_losses).mean()
            entropy = torch.stack(ent_losses).mean()
            value_loss = torch.stack(v_losses).mean()
            cost_loss = torch.stack(c_losses).mean()

            total_loss = actor_loss + value_coef * value_loss + cost_coef * cost_loss - entropy_coef * entropy

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

            approx_kl = float(torch.stack(kl_terms).mean().item()) if kl_terms else 0.0
            batch_stats.append((
                float(actor_loss.detach().item()),
                float(value_loss.detach().item()),
                float(cost_loss.detach().item()),
                float(entropy.detach().item()),
                float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
                approx_kl,
            ))

            if target_kl is not None and approx_kl > float(target_kl):
                early_stop = True
                break

    if not batch_stats:
        return None

    arr = np.asarray(batch_stats)
    mean_ep_cost = float(np.mean([
        sum(episode_violation_costs(tr)) / max(1, len(tr)) for tr in traces if tr
    ]))

    return {
        "actor_loss": float(arr[:, 0].mean()),
        "value_loss": float(arr[:, 1].mean()),
        "cost_loss": float(arr[:, 2].mean()),
        "entropy": float(arr[:, 3].mean()),
        "grad_norm": float(arr[:, 4].mean()),
        "approx_kl": float(arr[:, 5].mean()),
        "mean_epoch_violation_rate": mean_ep_cost,
        "n_steps": len(steps),
        "n_updates": len(batch_stats),
    }
