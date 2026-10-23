"""Double DQN Training Step for SCQL with Corrected Zero-Time SMDP Bellman Targets."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn

from .q_network import SCQLDuelingQNetwork
from .replay_buffer import SCQLReplayBuffer, Transition


def compute_ddqn_target(
    transition: Transition,
    online_q: SCQLDuelingQNetwork,
    target_q: SCQLDuelingQNetwork,
    gamma: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Computes corrected two-scale Double DQN Bellman target under causal action masking.

    Invariants:
      1. Assignment transitions: reward = 0.0, discount = 1.0 (zero-time matching step)
      2. STOP transitions: reward = transition.completed_value (simulator deadline-valid completions), discount = gamma
      3. Terminal: omits bootstrap term
      4. Double DQN: argmax action from online_q under action_mask, evaluated by target_q
    """
    if transition.kind == "assignment":
        reward = 0.0
        discount = 1.0
    elif transition.kind == "stop":
        reward = float(transition.completed_value)
        discount = float(gamma)
    else:
        raise ValueError(f"Unknown transition kind: {transition.kind}")

    if transition.done:
        return torch.tensor(reward, dtype=torch.float32, device=device)

    # Next partial matching: empty for STOP transition, next_partial_matching for assignment
    next_pm = [] if transition.kind == "stop" else transition.next_partial_matching

    # If no next candidates (e.g. terminal or empty epoch)
    if not transition.next_candidates:
        return torch.tensor(reward, dtype=torch.float32, device=device)

    # Both networks evaluate only feasible next actions
    q_next_online = online_q(
        transition.next_candidates,
        transition.next_history,
        transition.next_b_rem,
        action_mask=transition.next_action_mask,
        partial_matching=next_pm,
    )
    best_a = int(torch.argmax(q_next_online).item())

    q_next_target = target_q(
        transition.next_candidates,
        transition.next_history,
        transition.next_b_rem,
        action_mask=transition.next_action_mask,
        partial_matching=next_pm,
    )
    next_val = q_next_target[best_a]

    return reward + discount * next_val


class SCQLTrainer:
    def __init__(
        self,
        q_net: SCQLDuelingQNetwork,
        optimizer: torch.optim.Optimizer,
        buffer: SCQLReplayBuffer,
        gamma: float = 0.995,
        tau: float = 0.005,
        max_grad_norm: float = 1.0,
        margin_delta: float = 0.8,
    ):
        self.q_net = q_net
        self.target_net = copy.deepcopy(q_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optimizer
        self.buffer = buffer
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.margin_delta = margin_delta
        self.device = next(q_net.parameters()).device

    def train_step(
        self,
        batch_size: int = 64,
        beta: float = 0.4,
        lambda_margin: float = 0.0,
    ) -> Dict[str, float]:
        if len(self.buffer) < batch_size:
            return {"loss": 0.0, "td_error_mean": 0.0, "margin_loss": 0.0, "grad_norm": 0.0}

        samples, is_weights, indices = self.buffer.sample(batch_size, beta=beta)
        is_weights = is_weights.to(self.device)

        self.q_net.train()
        td_errors = []
        losses = []
        margin_losses = []

        for trans, is_w in zip(samples, is_weights):
            # 1. Current Q-value
            q_cur_all = self.q_net(
                trans.candidates,
                trans.history,
                trans.b_rem,
                partial_matching=trans.partial_matching,
            )
            q_cur = q_cur_all[trans.action_idx]

            # 2. Target Q-value via Double DQN
            with torch.no_grad():
                y = compute_ddqn_target(
                    trans,
                    online_q=self.q_net,
                    target_q=self.target_net,
                    gamma=self.gamma,
                    device=self.device,
                )

            delta = y - q_cur
            td_errors.append(float(delta.abs().item()))

            # Huber loss on TD error
            huber = nn.functional.smooth_l1_loss(q_cur, y, reduction="none")

            # DQfD Large-Margin loss on demonstrations
            m_loss = torch.tensor(0.0, device=self.device)
            if lambda_margin > 0.0 and len(q_cur_all) > 1:
                expert_a = trans.action_idx
                q_adj = q_cur_all + self.margin_delta
                q_adj[expert_a] = q_cur_all[expert_a]
                max_adj = torch.max(q_adj)
                m_loss = nn.functional.relu(max_adj - q_cur_all[expert_a])
                margin_losses.append(float(m_loss.item()))

            total_sample_loss = is_w * huber + lambda_margin * m_loss
            losses.append(total_sample_loss)

        total_loss = torch.stack(losses).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Update priorities in buffer
        self.buffer.update_priorities(indices, np.array(td_errors))

        # Polyak update of target network
        with torch.no_grad():
            for p, p_targ in zip(self.q_net.parameters(), self.target_net.parameters()):
                p_targ.data.mul_(1.0 - self.tau)
                p_targ.data.add_(self.tau * p.data)

        mean_m_loss = float(np.mean(margin_losses)) if margin_losses else 0.0
        return {
            "loss": float(total_loss.item()),
            "td_error_mean": float(np.mean(td_errors)),
            "margin_loss": mean_m_loss,
            "grad_norm": float(gnorm.item()) if torch.is_tensor(gnorm) else float(gnorm),
        }
