"""SCQL Sequential Matching Policy for WiLab closed loop."""
from __future__ import annotations

import collections
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn

from .action_mask import CandidateEdge, build_admissible_action_mask, candidate_from_dict
from .q_network import SCQLDuelingQNetwork
from .replay_buffer import SCQLReplayBuffer, Transition


class SCQLSequentialPolicy(nn.Module):
    """Sequential Matching Policy driven by Graph-Conditioned Dueling Q-Network."""

    is_scql_policy = True
    is_structured_priority_policy = False

    def __init__(
        self,
        b_max: int = 175000,
        r_thresh: float = 0.20,
        epsilon: float = 0.0,
        delta: float = 0.0,
        history_len: int = 10,
        buffer: Optional[SCQLReplayBuffer] = None,
        expert_mode: str = "rl",
        record_transitions: bool = False,
    ):
        super().__init__()
        self.b_max = int(b_max)
        self.r_thresh = float(r_thresh)
        self.epsilon = float(epsilon)
        self.delta = float(delta)  # Advantage threshold for overriding reference matching
        self.history_len = int(history_len)
        self.buffer = buffer
        self.expert_mode = expert_mode  # "rl" or "guarded_b4"
        self.record_transitions = record_transitions

        self.q_net = SCQLDuelingQNetwork()
        self.score_and_match_fn = self._sequential_match

        # Trailing history buffer: [CBR, sum_v, sum_b, mean_q]
        self._history = collections.deque(maxlen=history_len)
        self._init_history()

        # Finite-horizon matching transition tracking
        self._pending_trans: Optional[Transition] = None
        self._last_completed_val: float = 0.0

    def _init_history(self):
        for _ in range(self.history_len):
            self._history.append([0.04, 0.0, 0.0, 0.0])

    def update_history(self, cbr: float, offered_v: float = 0.0, offered_b: float = 0.0, mean_q: float = 0.0):
        self._history.append([float(cbr), float(offered_v), float(offered_b) / 1e5, float(mean_q)])

    def get_history_tensor(self) -> torch.Tensor:
        hist_list = list(self._history)
        while len(hist_list) < self.history_len:
            hist_list.insert(0, [0.04, 0.0, 0.0, 0.0])
        return torch.tensor(hist_list, dtype=torch.float32).unsqueeze(0)

    def finalize_episode(self, final_completed: Optional[float] = None):
        """Commits any trailing pending matching transition at episode boundary with actual completion credit."""
        if self._pending_trans is not None and self.buffer is not None:
            advance_r = 0.0
            if final_completed is not None:
                advance_r = max(0.0, float(final_completed) - self._last_completed_val)
            self._pending_trans.reward = advance_r
            self._pending_trans.completed_value = advance_r
            self._pending_trans.done = True
            self._pending_trans.next_candidates = []
            self._pending_trans.next_matchings = []
            self._pending_trans.next_b_rems = []
            self.buffer.push(self._pending_trans, priority=max(1.0, advance_r * 2.0))
        self._pending_trans = None
        self._last_completed_val = 0.0

    def _build_reference_matching(
        self,
        candidates: Sequence[CandidateEdge],
        b_max: float,
        unique_sources: bool = True,
        unique_helpers: bool = True,
    ) -> Tuple[List[CandidateEdge], float]:
        """Constructs a complete feasible reference matching M_B via greedy value-density knapsack."""
        # Sort candidates descending by value density v_i / b_i
        sorted_cands = sorted(
            candidates,
            key=lambda c: c.task_value / max(1.0, c.payload_bytes / 1000.0),
            reverse=True,
        )
        matching: List[CandidateEdge] = []
        b_rem = float(b_max)
        assigned_tasks = set()
        assigned_sources = set()
        assigned_helpers = set()

        for c in sorted_cands:
            if c.payload_bytes > b_rem:
                continue
            if c.task_id in assigned_tasks:
                continue
            if unique_sources and c.source_id in assigned_sources:
                continue
            if unique_helpers and c.helper_id in assigned_helpers:
                continue

            matching.append(c)
            b_rem -= float(c.payload_bytes)
            assigned_tasks.add(c.task_id)
            if unique_sources:
                assigned_sources.add(c.source_id)
            if unique_helpers:
                assigned_helpers.add(c.helper_id)

        return matching, b_rem

    def _build_alternative_matching(
        self,
        candidates: Sequence[CandidateEdge],
        b_max: float,
        history_t: torch.Tensor,
        unique_sources: bool = True,
        unique_helpers: bool = True,
    ) -> Tuple[List[CandidateEdge], float]:
        """Constructs a candidate alternative matching M_R, selecting the matching prefix

        that maximizes predicted long-term completed value Q_theta(s, M), including the
        option to leave resources unused if further assignments are predicted harmful.
        """
        # 1. Sequentially order candidate assignments using edge advantage
        ordered_edges: List[CandidateEdge] = []
        b_rem_track: List[float] = [float(b_max)]
        cur_brem = float(b_max)
        assigned_tasks = set()
        assigned_sources = set()
        assigned_helpers = set()

        while True:
            feasible = [
                c for c in candidates
                if c.payload_bytes <= cur_brem
                and c.task_id not in assigned_tasks
                and (not unique_sources or c.source_id not in assigned_sources)
                and (not unique_helpers or c.helper_id not in assigned_helpers)
            ]
            if not feasible:
                break

            with torch.no_grad():
                scores = self.q_net.score_candidate_edges(
                    feasible, history_t, partial_matching=ordered_edges, b_rem=cur_brem, b_max=b_max
                )
                if scores.numel() > 0:
                    best_idx = int(torch.argmax(scores).item())
                    best_edge = feasible[best_idx]
                else:
                    best_edge = feasible[0]

            ordered_edges.append(best_edge)
            cur_brem -= float(best_edge.payload_bytes)
            b_rem_track.append(cur_brem)
            assigned_tasks.add(best_edge.task_id)
            if unique_sources:
                assigned_sources.add(best_edge.source_id)
            if unique_helpers:
                assigned_helpers.add(best_edge.helper_id)

        # 2. Evaluate all candidate prefixes M^(0), M^(1), ..., M^(K)
        # Prefix k represents admitting the first k assignments and leaving remaining resources unused.
        prefix_q_vals = []
        for k in range(len(ordered_edges) + 1):
            sub_m = ordered_edges[:k]
            sub_brem = b_rem_track[k]
            with torch.no_grad():
                q_val = float(self.q_net.evaluate_matching(
                    sub_m, history_t, b_rem=sub_brem, b_max=b_max
                ).item())
            prefix_q_vals.append(q_val)

        # Select prefix that maximizes predicted long-term return Q_theta(s, M)
        best_k = int(np.argmax(prefix_q_vals))
        best_matching = ordered_edges[:best_k]
        best_b_rem = b_rem_track[best_k]

        return best_matching, best_b_rem

    def _sequential_match(
        self,
        candidates: List[Dict[str, Any]],
        weights: Any = None,
        beta_throttle: float = 1.0,
        b_max: int = 175000,
        unique_sources: bool = True,
        unique_helpers: bool = True,
        layer: Any = None,
        deterministic: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Constructs complete reference M_B and alternative M_R, selecting M* via learned advantage Delta_theta > delta."""
        # 1. Update reward from actual simulator completions
        if layer is not None and hasattr(layer, "completed_value"):
            cur_comp = float(layer.completed_value())
            advance_reward = max(0.0, cur_comp - self._last_completed_val)
            self._last_completed_val = cur_comp
        else:
            advance_reward = 0.0

        from controllers.rl.causal_safety_guard import filter_candidates_causal_guard

        # Causal safety guard applies identically to both matchings
        filtered, rej, eff_b = filter_candidates_causal_guard(
            candidates, layer=layer, r_thresh=self.r_thresh, b_max=b_max
        )
        eff_b_max = min(eff_b, int(self.b_max * float(beta_throttle)))
        parsed_candidates = [candidate_from_dict(c) for c in filtered]

        # Update arrival rate and trailing history
        tot_arr_v = sum(c.task_value for c in parsed_candidates) if parsed_candidates else 0.0
        tot_arr_b = sum(c.payload_bytes for c in parsed_candidates) if parsed_candidates else 0.0
        mean_q = float(np.mean([c.queue_cycles / 1e8 for c in parsed_candidates])) if parsed_candidates else 0.0
        self.update_history(cbr=0.04, offered_v=tot_arr_v, offered_b=tot_arr_b, mean_q=mean_q)
        history_t = self.get_history_tensor()

        if not parsed_candidates:
            # Empty candidate set
            if self._pending_trans is not None and self.buffer is not None:
                self._pending_trans.reward = advance_reward
                self._pending_trans.completed_value = advance_reward
                self._pending_trans.next_candidates = []
                self._pending_trans.next_matchings = [[]]
                self._pending_trans.next_b_rems = [float(eff_b_max)]
                self._pending_trans.next_history = history_t
                self.buffer.push(self._pending_trans, priority=max(1.0, advance_reward * 2.0))
                self._pending_trans = None

            return [], {
                "n_pack": 0, "bytes": 0, "b_limit": eff_b_max, "b_rem": float(eff_b_max),
                "q_diff": 0.0, "selected_mode": "empty", "policy_mode": "scql", "rejected_geometry": rej,
            }

        # 2. Build complete feasible reference matching M_B
        m_b, b_rem_b = self._build_reference_matching(
            parsed_candidates, b_max=float(eff_b_max), unique_sources=unique_sources, unique_helpers=unique_helpers
        )

        # 3. Build complete feasible alternative matching M_R without premature STOP
        m_r, b_rem_r = self._build_alternative_matching(
            parsed_candidates, b_max=float(eff_b_max), history_t=history_t, unique_sources=unique_sources, unique_helpers=unique_helpers
        )

        # 4. Evaluate complete matchings and compute advantage Delta_theta(s, M_R, M_B)
        self.q_net.eval()
        with torch.no_grad():
            q_b = float(self.q_net.evaluate_matching(m_b, history_t, b_rem=b_rem_b, b_max=float(eff_b_max)).item())
            q_r = float(self.q_net.evaluate_matching(m_r, history_t, b_rem=b_rem_r, b_max=float(eff_b_max)).item())
            delta_theta = q_r - q_b

        # 5. Selection rule: M* = M_R if Delta_theta > delta, else M_B
        if self.expert_mode == "guarded_b4":
            selected_matching = m_b
            selected_b_rem = b_rem_b
            sel_mode = "reference_b4"
        else:
            if delta_theta > self.delta:
                selected_matching = m_r
                selected_b_rem = b_rem_r
                sel_mode = "scql_alternative"
            else:
                selected_matching = m_b
                selected_b_rem = b_rem_b
                sel_mode = "reference_b4"

        # 6. Commit pending matching transition with actual completion reward
        if self._pending_trans is not None and self.buffer is not None:
            self._pending_trans.reward = advance_reward
            self._pending_trans.completed_value = advance_reward
            self._pending_trans.next_candidates = parsed_candidates
            self._pending_trans.next_matchings = [m_b, m_r]
            self._pending_trans.next_b_rems = [b_rem_b, b_rem_r]
            self._pending_trans.next_history = history_t
            self.buffer.push(self._pending_trans, priority=max(1.0, advance_reward * 2.0))
            self._pending_trans = None

        # 7. Start new pending transition for current decision
        if self.record_transitions and self.buffer is not None:
            self._pending_trans = Transition(
                candidates=parsed_candidates,
                history=history_t,
                b_rem=selected_b_rem,
                partial_matching=list(selected_matching),
                action_idx=0,
                reward=0.0,  # credited on next epoch advance
                next_candidates=[],
                next_history=history_t,
                next_b_rem=float(self.b_max),
                next_partial_matching=[],
                next_action_mask=[True],
                done=False,
                is_stop=True,
                kind="matching",
                next_matchings=[],
                next_b_rems=[],
                delta_adv=delta_theta,
            )

        pack = [m.raw_dict for m in selected_matching]
        total_bytes = sum(int(a.get("payload_bytes", 0)) for a in pack)

        meta = {
            "n_pack": len(pack),
            "bytes": total_bytes,
            "b_limit": eff_b_max,
            "b_rem": selected_b_rem,
            "q_b": q_b,
            "q_r": q_r,
            "delta_theta": delta_theta,
            "selected_mode": sel_mode,
            "policy_mode": "scql",
            "rejected_geometry": rej,
        }
        return pack, meta
