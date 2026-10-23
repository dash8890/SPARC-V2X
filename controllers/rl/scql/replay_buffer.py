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
    partial_matching: List[CandidateEdge]
    action_idx: int  # index in [0, len(candidates)] where len(candidates) is STOP
    reward: float    # 0 for intra-epoch; r_t for inter-epoch STOP
    next_candidates: List[CandidateEdge]
    next_history: torch.Tensor
    next_b_rem: float
    next_partial_matching: List[CandidateEdge]
    next_action_mask: List[bool]
    done: bool
    is_stop: bool


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
