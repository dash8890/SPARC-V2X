from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PPOStep:
    item: dict
    old_logp: float
    advantage: float
    target_return: float


def _sequence_logp(policy, item, envelope_bytes: int, unique_sources: bool = True):
    """Joint log π of the **proposed** pack (full sequential selection).

    Prefers ``proposed_selected_candidate_indices`` so governor rewrite of the
    executed pack cannot change the PPO likelihood identity (Phase-2b).
    """
    if getattr(policy, "uses_sequential_context", False):
        from controllers.rl.context_direct_policy import sequence_logp_from_item
        return sequence_logp_from_item(policy, item, envelope_bytes, unique_sources=unique_sources)
    features = item.get("candidate_features", [])
    analytic = item.get("candidate_analytic_scores", [])
    task_ids = item.get("candidate_task_ids", [])
    source_ids = item.get("candidate_source_ids", [])
    payloads = item.get("candidate_payload_bytes", [])
    selected = item.get("proposed_selected_candidate_indices")
    if selected is None:
        selected = item.get("selected_candidate_indices", [])
    if not features or not selected:
        return None
    x = torch.tensor(features, dtype=torch.float32)
    anchor = torch.tensor(analytic, dtype=torch.float32)
    if hasattr(policy, "score_logits"):
        logits = policy.score_logits(x, anchor)
    else:
        logits = anchor + policy.residual(x)
    remaining = list(range(len(features)))
    used_tasks, used_sources, used_bytes = set(), set(), 0
    logps, entropies = [], []
    for selected_index in selected:
        choices = [i for i in remaining
                   if int(task_ids[i]) not in used_tasks
                   and (not unique_sources or int(source_ids[i]) not in used_sources)
                   and used_bytes + int(payloads[i]) <= int(envelope_bytes)]
        if int(selected_index) not in choices:
            return None
        dist = torch.distributions.Categorical(logits=logits[choices])
        local = torch.tensor(choices.index(int(selected_index)))
        logps.append(dist.log_prob(local)); entropies.append(dist.entropy())
        remaining.remove(int(selected_index))
        used_tasks.add(int(task_ids[selected_index]))
        used_sources.add(int(source_ids[selected_index]))
        used_bytes += int(payloads[selected_index])
    return torch.stack(logps).sum(), torch.stack(entropies).mean()


def build_ppo_steps(traces, gamma=1.0, gae_lambda=0.95, va_relative=True,
                    va_deviation_eps=0.02, changed_only=False,
                    advantage_mode: str = "gae", w_blend: float = 0.5):
    """Build one batch and normalize advantages once across actionable steps.

    Default ``gamma=1.0`` matches SPARC Phase-0 utility lock (undiscounted
    deadline-valid completed value). Pass ``gamma=0.99`` for legacy A/B.

    ``advantage_mode`` (Phase-2d):
      - ``gae``: standard GAE (default)
      - ``w_minus_v``: use item[``w_estimate``] - V(s) when ``w_estimate`` is set
      - ``blend``: (1-w_blend)*GAE + w_blend*(W-V)

    When ``va_relative`` is set, each episode's GAE advantages are shifted by
    (episode_return - VA-like baseline) before normalization. The VA-like
    baseline is the mean return of near-zero-deviation episodes in the batch
    (fallback: batch mean). That ties learning to beating the anchor rather
    than just fitting noisy absolute returns.
    """
    mode = str(advantage_mode or "gae").lower()
    if mode not in ("gae", "w_minus_v", "blend"):
        raise ValueError(f"unknown advantage_mode={advantage_mode!r}")
    blend = float(np.clip(w_blend, 0.0, 1.0))

    raw = []
    value_targets = []
    episode_stats = []
    for trace in traces:
        if not trace:
            continue
        rewards = torch.tensor([float(x.get("normalized_reward", 0.0)) for x in trace])
        old_values = torch.tensor([float(x["value"].detach()) for x in trace])
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0)
        next_value = torch.tensor(0.0)
        for index in reversed(range(len(trace))):
            delta = rewards[index] + gamma * next_value - old_values[index]
            gae = delta + gamma * gae_lambda * gae
            advantages[index] = gae
            next_value = old_values[index]
        returns = advantages + old_values
        ep_return = float(rewards.sum())
        deviation = sum(bool(x.get("residual_changed_anchor")) for x in trace) / max(1, len(trace))
        episode_stats.append({
            "return": ep_return,
            "deviation": deviation,
            "advantages": advantages,
            "returns": returns,
            "trace": trace,
        })
        for item, target in zip(trace, returns):
            value_targets.append((item, float(target)))

    if not episode_stats:
        return [], value_targets

    va_like = [e["return"] for e in episode_stats if e["deviation"] <= float(va_deviation_eps)]
    baseline = float(np.mean(va_like)) if va_like else float(np.mean([e["return"] for e in episode_stats]))

    for episode in episode_stats:
        delta_vs_va = float(episode["return"] - baseline) if va_relative else 0.0
        for item, advantage, target, old_v in zip(
                episode["trace"], episode["advantages"], episode["returns"],
                [float(x["value"].detach()) for x in episode["trace"]]):
            if not item.get("selected_candidate_indices"):
                continue
            changed = bool(item.get("residual_changed_anchor"))
            if changed_only and not changed:
                continue
            # Shift GAE by trajectory improvement over the VA-like baseline.
            # Changed steps get the full shift; unchanged steps get a weaker
            # shift so near-VA behavior is not heavily reinforced.
            shift = delta_vs_va if changed else 0.25 * delta_vs_va
            gae_adv = float(advantage) + shift
            w_est = item.get("w_estimate", None)
            if w_est is not None and mode in ("w_minus_v", "blend"):
                w_adv = float(w_est) - float(old_v)
                if mode == "w_minus_v":
                    final_adv = w_adv
                else:
                    final_adv = (1.0 - blend) * gae_adv + blend * w_adv
            else:
                final_adv = gae_adv
            # old_logp is joint logπ(A_proposed); never substitute executed-pack likelihood.
            old_lp = item.get("logp")
            if torch.is_tensor(old_lp):
                old_lp_f = float(old_lp.detach())
            else:
                old_lp_f = float(old_lp)
            raw.append((item, old_lp_f,
                        final_adv, float(target), delta_vs_va, changed))

    if not raw:
        return [], value_targets
    adv = np.asarray([x[2] for x in raw], dtype=np.float64)
    adv = (adv - adv.mean()) / max(float(adv.std()), 1e-6)
    steps = [PPOStep(item=x[0], old_logp=x[1], advantage=float(a), target_return=x[3])
             for x, a in zip(raw, adv)]
    return steps, value_targets


def ppo_update(policy, optimizer, traces, envelope_bytes: int, epochs=4,
               minibatch_size=128, clip_ratio=0.2, value_coef=0.5,
               entropy_coef=0.02, grad_clip=1.0, unique_sources=True,
               target_kl=0.015, va_relative=True, changed_only=False,
               gamma: float = 1.0, gae_lambda: float = 0.95,
               advantage_mode: str = "gae", w_blend: float = 0.5):
    steps, value_targets = build_ppo_steps(
        traces, gamma=float(gamma), gae_lambda=float(gae_lambda),
        va_relative=va_relative, changed_only=changed_only,
        advantage_mode=advantage_mode, w_blend=w_blend)
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
            policy_terms, entropy_terms, value_terms = [], [], []
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
            if not policy_terms:
                continue
            actor_loss = torch.stack(policy_terms).mean()
            value_loss = torch.stack(value_terms).mean()
            entropy = torch.stack(entropy_terms).mean()
            loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
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
                           float(entropy.detach()), grad, approx_kl))
            if target_kl is not None and approx_kl > float(target_kl):
                early_stop = True
                break
    if not losses:
        return None
    arr = np.asarray(losses)
    return {
        "actor_loss": float(arr[:, 0].mean()),
        "value_loss": float(arr[:, 1].mean()),
        "entropy": float(arr[:, 2].mean()),
        "gradient_norm": float(arr[:, 3].mean()),
        "approx_kl": float(arr[:, 4].mean()),
        "action_steps": len(steps),
        "value_steps": len(value_targets),
        "ppo_minibatches": int(len(losses)),
    }
