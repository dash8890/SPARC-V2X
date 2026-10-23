"""Prioritized Experience Replay Buffer for Sequential Capacity-Aware Q-Learning (SCQL).

Stores both intra-epoch sub-step transitions (r=0, gamma=1.0) and inter-epoch STOP transitions (r=r_t, gamma=0.995).
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from .action_mask import CandidateEdge


@dataclass
class Transition:
    candidates: List[CandidateEdge]
    history: torch.Tensor
    b_rem: float
    partial_matching: List[CandidateEdge]  # represents the chosen matching M^*
    action_idx: int = 0  # index in [0, len(candidates)] where len(candidates) is STOP
    reward: float = 0.0    # actual deadline-valid completed value r_t
    next_candidates: Optional[List[CandidateEdge]] = None
    next_history: Optional[torch.Tensor] = None
    next_b_rem: float = 175000.0
    next_partial_matching: Optional[List[CandidateEdge]] = None
    next_action_mask: Optional[List[bool]] = None
    done: bool = False
    is_stop: bool = True
    kind: str = "assignment"  # "matching", "assignment", or "stop"
    completed_value: float = 0.0
    next_matchings: Optional[List[List[CandidateEdge]]] = None  # \mathcal{N}(s_{t+1})
    next_b_rems: Optional[List[float]] = None
    delta_adv: float = 0.0

    def __post_init__(self):
        if self.next_candidates is None:
            self.next_candidates = []
        if self.next_partial_matching is None:
            self.next_partial_matching = []
        if self.next_action_mask is None:
            self.next_action_mask = [True]
        if self.next_matchings is None:
            self.next_matchings = []
        if self.next_b_rems is None:
            self.next_b_rems = []

        if not self.is_stop:
            self.kind = "assignment"
            self.reward = 0.0
            self.completed_value = 0.0
        elif self.kind == "matching":
            if self.completed_value == 0.0 and self.reward != 0.0:
                self.completed_value = float(self.reward)
            else:
                self.reward = float(self.completed_value)
        elif self.is_stop or self.kind == "stop":
            self.kind = "stop"
            if self.completed_value == 0.0 and self.reward != 0.0:
                self.completed_value = float(self.reward)
            else:
                self.reward = float(self.completed_value)


class SCQLReplayBuffer:
    def __init__(self, capacity: int = 200000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, transition: Transition, priority: Optional[float] = None):
        max_p = self.priorities.max() if len(self.buffer) > 0 else 1.0
        p = priority if priority is not None else max_p
        p = max(float(p), 1e-4)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], torch.Tensor, List[int]]:
        n = len(self.buffer)
        if n == 0:
            return [], torch.empty(0), []

        batch_size = min(batch_size, n)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights
        total = n
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights_t = torch.tensor(weights, dtype=torch.float32)

        return samples, weights_t, indices.tolist()

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = max(float(abs(err)), 1e-4)

    def __len__(self) -> int:
        return len(self.buffer)
