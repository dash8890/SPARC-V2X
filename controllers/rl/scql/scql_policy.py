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
        history_len: int = 10,
        buffer: Optional[SCQLReplayBuffer] = None,
        expert_mode: str = "rl",
        record_transitions: bool = False,
    ):
        super().__init__()
        self.b_max = int(b_max)
        self.r_thresh = float(r_thresh)
        self.epsilon = float(epsilon)
        self.history_len = int(history_len)
        self.buffer = buffer
        self.expert_mode = expert_mode  # "rl" or "guarded_b4"
        self.record_transitions = record_transitions

        self.q_net = SCQLDuelingQNetwork()
        self.score_and_match_fn = self._sequential_match

        # Trailing history buffer: [CBR, sum_v, sum_b, mean_q]
        self._history = collections.deque(maxlen=history_len)
        self._init_history()

        # SMDP Transition logging state
        self._pending_stop_trans: Optional[Transition] = None
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
        """Commits any trailing pending STOP transition at episode boundary."""
        if self._pending_stop_trans is not None and self.buffer is not None:
            self._pending_stop_trans.reward = 0.0
            self._pending_stop_trans.done = True
            self._pending_stop_trans.next_candidates = []
            self._pending_stop_trans.next_action_mask = [True]
            self.buffer.push(self._pending_stop_trans, priority=1.0)
        self._pending_stop_trans = None
        self._last_completed_val = 0.0

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
        """Constructs a matching sequentially one edge at a time using Q-values or expert heuristic."""
        # 1. Update reward for pending STOP transition from previous epoch
        if layer is not None and hasattr(layer, "completed_value"):
            cur_comp = float(layer.completed_value())
            advance_reward = max(0.0, cur_comp - self._last_completed_val)
            self._last_completed_val = cur_comp
        else:
            advance_reward = 0.0

        from controllers.rl.causal_safety_guard import filter_candidates_causal_guard

        filtered, rej, eff_b = filter_candidates_causal_guard(
            candidates, layer=layer, r_thresh=self.r_thresh, b_max=b_max
        )
        eff_b_max = min(eff_b, int(self.b_max * float(beta_throttle)))
        b_rem = float(eff_b_max)

        # Parse candidates
        parsed_candidates = [candidate_from_dict(c) for c in filtered]

        # Initial mask at epoch start
        init_mask, _ = build_admissible_action_mask(
            parsed_candidates,
            [],
            b_rem=b_rem,
            cbr=0.04,
            r_thresh=self.r_thresh,
        )

        # Commit pending STOP transition if active
        if self._pending_stop_trans is not None and self.buffer is not None:
            self._pending_stop_trans.reward = 0.0
            self._pending_stop_trans.next_candidates = parsed_candidates
            self._pending_stop_trans.next_action_mask = init_mask
            self.buffer.push(self._pending_stop_trans, priority=1.0)
            self._pending_stop_trans = None

        if not parsed_candidates:
            return [], {
                "n_pack": 0, "bytes": 0, "b_limit": eff_b_max, "b_rem": b_rem,
                "sub_steps": 0, "q_history": [], "policy_mode": "scql", "rejected_geometry": rej,
            }

        partial_matching: List[CandidateEdge] = []
        sub_steps = 0
        q_history = []
        max_substeps = min(len(parsed_candidates), 12)
        history_t = self.get_history_tensor()

        self.q_net.eval()
        with torch.no_grad():
            while sub_steps < max_substeps:
                sub_steps += 1
                mask, _ = build_admissible_action_mask(
                    parsed_candidates,
                    partial_matching,
                    b_rem=b_rem,
                    cbr=0.04,
                    r_thresh=self.r_thresh,
                )

                admissible_indices = [idx for idx, ok in enumerate(mask) if ok]
                stop_idx = len(parsed_candidates)

                if not admissible_indices or admissible_indices == [stop_idx]:
                    # Only STOP or nothing is admissible
                    chosen_action = stop_idx
                elif self.expert_mode == "guarded_b4":
                    # Guarded B4 expert heuristic: max value density
                    cand_admissible = [i for i in admissible_indices if i < stop_idx]
                    if cand_admissible:
                        chosen_action = max(
                            cand_admissible,
                            key=lambda i: parsed_candidates[i].task_value / (parsed_candidates[i].payload_bytes / 1000.0),
                        )
                    else:
                        chosen_action = stop_idx
                else:
                    # SCQL RL policy
                    q_vals = self.q_net(
                        parsed_candidates,
                        history_t,
                        b_rem=b_rem,
                        action_mask=mask,
                        partial_matching=partial_matching,
                        b_max=eff_b_max,
                    )
                    if not deterministic and np.random.rand() < self.epsilon:
                        chosen_action = int(np.random.choice(admissible_indices))
                    else:
                        chosen_action = int(torch.argmax(q_vals).item())

                # If STOP action selected
                if chosen_action == stop_idx:
                    q_history.append(("STOP", 0.0))
                    if self.record_transitions and self.buffer is not None:
                        self._pending_stop_trans = Transition(
                            candidates=parsed_candidates,
                            history=history_t,
                            b_rem=b_rem,
                            partial_matching=list(partial_matching),
                            action_idx=stop_idx,
                            reward=0.0,  # will be populated on next epoch advance
                            next_candidates=[],
                            next_history=history_t,
                            next_b_rem=float(self.b_max),
                            next_partial_matching=[],
                            next_action_mask=[True],
                            done=False,
                            is_stop=True,
                        )
                    break

                # Valid candidate edge chosen
                selected_edge = parsed_candidates[chosen_action]
                next_partial = partial_matching + [selected_edge]
                next_b_rem = b_rem - float(selected_edge.payload_bytes)

                next_mask, _ = build_admissible_action_mask(
                    parsed_candidates,
                    next_partial,
                    b_rem=next_b_rem,
                    cbr=0.04,
                    r_thresh=self.r_thresh,
                )

                if self.record_transitions and self.buffer is not None:
                    trans_intra = Transition(
                        candidates=parsed_candidates,
                        history=history_t,
                        b_rem=b_rem,
                        partial_matching=list(partial_matching),
                        action_idx=chosen_action,
                        reward=float(selected_edge.task_value),
                        next_candidates=parsed_candidates,
                        next_history=history_t,
                        next_b_rem=next_b_rem,
                        next_partial_matching=list(next_partial),
                        next_action_mask=next_mask,
                        done=False,
                        is_stop=False,
                    )
                    self.buffer.push(trans_intra, priority=max(1.0, float(selected_edge.task_value)))

                partial_matching = next_partial
                b_rem = next_b_rem
                q_history.append((selected_edge.task_id, selected_edge.helper_id, 0.0))

                if b_rem <= 0:
                    # Budget filled; implicit STOP
                    if self.record_transitions and self.buffer is not None:
                        self._pending_stop_trans = Transition(
                            candidates=parsed_candidates,
                            history=history_t,
                            b_rem=b_rem,
                            partial_matching=list(partial_matching),
                            action_idx=stop_idx,
                            reward=0.0,
                            next_candidates=[],
                            next_history=history_t,
                            next_b_rem=float(self.b_max),
                            next_partial_matching=[],
                            next_action_mask=[True],
                            done=False,
                            is_stop=True,
                        )
                    break

        pack = [m.raw_dict for m in partial_matching]
        total_bytes = sum(int(a.get("payload_bytes", 0)) for a in pack)

        meta = {
            "n_pack": len(pack),
            "bytes": total_bytes,
            "b_limit": eff_b_max,
            "b_rem": b_rem,
            "sub_steps": sub_steps,
            "q_history": q_history,
            "policy_mode": "scql",
            "rejected_geometry": rej,
        }
        return pack, meta
