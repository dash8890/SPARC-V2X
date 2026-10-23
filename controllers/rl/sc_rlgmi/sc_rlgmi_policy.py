"""Safety-Constrained Reinforcement Learning-Guided Matching Improvement (SC-RLGMI) Policy.

Integrates advantage-guided beam search with the WiLab closed-loop environment,
maintaining the frozen causal safety guard, physical admission envelope,
and exact outcome-supervised credit assignment.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn

from controllers.rl.causal_safety_guard import filter_candidates_causal_guard
from controllers.rl.scql.action_mask import CandidateEdge, candidate_from_dict
from controllers.rl.sc_rlgmi.sc_rlgmi_core import (
    Edge,
    Matching,
    canonical,
    DecisionState,
    AdvantageModel,
    Selection,
    improve,
    feasible,
)


class SCRLGMIPolicy(nn.Module):
    """SC-RLGMI: Policy utilizing combinatorial beam search guided by learned advantage."""

    is_sc_rlgmi_policy = True
    is_scql_policy = True  # Backward-compatible hook for wilab closed loop
    is_structured_priority_policy = False

    def __init__(
        self,
        b_max: int = 175000,
        r_thresh: float = 0.20,
        delta: float = 0.0,
        depth: int = 2,
        beam: int = 8,
        state_dim: int = 6,
        edge_dim: int = 10,
        hidden: int = 64,
        advantage_model: Optional[AdvantageModel] = None,
        expert_mode: str = "rl",
    ):
        super().__init__()
        self.b_max = int(b_max)
        self.r_thresh = float(r_thresh)
        self.delta = float(delta)
        self.depth = int(depth)
        self.beam = int(beam)
        self.state_dim = int(state_dim)
        self.edge_dim = int(edge_dim)
        self.hidden = int(hidden)
        self.expert_mode = str(expert_mode)  # "rl" or "reference"

        if advantage_model is not None:
            self.advantage_model = advantage_model
        else:
            self.advantage_model = AdvantageModel(state_dim, edge_dim, hidden=hidden)

        # Performance tracking metrics
        self.total_decisions = 0
        self.override_decisions = 0
        self.reference_decisions = 0
        self.record_decisions = False
        self.recorded_decisions: Dict[int, Dict[str, Any]] = {}
        self._history = collections.deque(maxlen=10)
        for _ in range(10):
            self._history.append([0.04, 0.0, 0.0, 0.0])

    def update_history(self, cbr: float, offered_v: float = 0.0, offered_b: float = 0.0, mean_q: float = 0.0):
        self._history.append([float(cbr), float(offered_v), float(offered_b) / 1e5, float(mean_q)])

    def _build_reference_matching(
        self,
        candidates: Sequence[CandidateEdge],
        b_max: float,
        unique_sources: bool = True,
        unique_helpers: bool = True,
    ) -> Matching:
        """Constructs the deterministic reference matching π_B (greedy value-density knapsack)."""
        # Sort by value density v_k / b_k descending, break ties with deadline slack
        sorted_cands = sorted(
            candidates,
            key=lambda c: (
                c.task_value / max(1.0, float(c.payload_bytes)),
                -c.finish_ms,
            ),
            reverse=True,
        )

        matching_edges: List[Edge] = []
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

            matching_edges.append((c.task_id, c.helper_id))
            b_rem -= float(c.payload_bytes)
            assigned_tasks.add(c.task_id)
            if unique_sources:
                assigned_sources.add(c.source_id)
            if unique_helpers:
                assigned_helpers.add(c.helper_id)

        return canonical(matching_edges)

    def _extract_decision_state(
        self,
        candidates: Sequence[CandidateEdge],
        cbr: float,
        b_safe: float,
        current_epoch: int = 0,
        total_epochs: int = 100,
    ) -> Tuple[DecisionState, Dict[Edge, CandidateEdge]]:
        """Constructs causally observable DecisionState and index map."""
        epoch_frac = float(current_epoch) / max(1.0, float(total_epochs))
        tot_v = sum(c.task_value for c in candidates) if candidates else 0.0
        tot_b = sum(c.payload_bytes for c in candidates) if candidates else 0.0
        mean_q = float(np.mean([c.queue_cycles / 1e8 for c in candidates])) if candidates else 0.0
        n_cands = float(len(candidates))

        state_features = (
            float(cbr),
            float(epoch_frac),
            float(b_safe) / float(self.b_max),
            float(tot_v) / 100.0,
            float(tot_b) / 5e5,
            float(mean_q),
        )

        edge_features: Dict[Edge, Tuple[float, ...]] = {}
        edge_map: Dict[Edge, CandidateEdge] = {}

        for c in candidates:
            edge = (c.task_id, c.helper_id)
            # 10 causally observable edge features
            feat = (
                float(c.task_value) / 10.0,
                float(c.payload_bytes) / 100000.0,
                float(c.cycles) / 1e9,
                float(c.deadline_ms) / 100.0,
                float(c.finish_ms) / 100.0,
                float(c.risk),
                float(c.queue_cycles) / 1e9,
                float(c.cpu_cycles_s) / 1e9,
                float(c.t_release_s),
                float(c.t_contact_sec) / 100.0,
            )
            edge_features[edge] = feat
            edge_map[edge] = c

        return DecisionState(features=state_features, edge_features=edge_features), edge_map

    def _make_feasibility_oracle(
        self,
        edge_map: Dict[Edge, CandidateEdge],
        b_safe: float,
    ):
        """Creates full-matching feasibility checker satisfying constraints (1)-(5)."""
        def oracle(state: DecisionState, matching: Matching) -> bool:
            # 1. Total payload bytes must fit within safe radio admission budget
            tot_bytes = sum(edge_map[e].payload_bytes for e in matching if e in edge_map)
            if tot_bytes > b_safe:
                return False

            # 2. Each task-helper link must satisfy physical finish <= deadline and <= contact
            for e in matching:
                if e not in edge_map:
                    return False
                c = edge_map[e]
                if c.risk > self.r_thresh:
                    return False
                if c.finish_ms > c.deadline_ms:
                    return False
                if c.finish_ms / 1000.0 > c.t_contact_sec:
                    return False

            return True

        return oracle

    def select_matching(
        self,
        candidates: List[Dict[str, Any]],
        b_max: int = 175000,
        unique_sources: bool = True,
        unique_helpers: bool = True,
        layer: Any = None,
        state: Any = None,
        deterministic: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Executes SC-RLGMI decision: reference matching construction, beam search, and override check."""
        self.total_decisions += 1

        # 1. Apply invariant causal safety guard to filter candidate pool
        filtered, rej, eff_b = filter_candidates_causal_guard(
            candidates, layer=layer, r_thresh=self.r_thresh, b_max=b_max
        )
        parsed_cands = [candidate_from_dict(c) for c in filtered]

        if not parsed_cands or eff_b <= 0:
            return [], {
                "n_pack": 0,
                "bytes": 0,
                "b_limit": eff_b,
                "selected_mode": "empty",
                "policy_mode": "sc_rlgmi",
                "predicted_gain": 0.0,
                "rejected_geometry": rej,
            }

        # 2. Build complete feasible reference matching M_B = π_B(o_t)
        ref_matching = self._build_reference_matching(
            parsed_cands, b_max=float(eff_b), unique_sources=unique_sources, unique_helpers=unique_helpers
        )

        cbr_val = float(state.get("cbr", 0.04) or 0.04) if isinstance(state, dict) else 0.04
        cur_epoch = int(getattr(layer, "current_epoch", 0) or 0)
        tot_epochs = int(getattr(layer, "controller_epochs", 100) or 100)

        # 3. Construct DecisionState and Feasibility Oracle
        d_state, edge_map = self._extract_decision_state(
            parsed_cands, cbr=cbr_val, b_safe=float(eff_b), current_epoch=cur_epoch, total_epochs=tot_epochs
        )
        oracle = self._make_feasibility_oracle(edge_map, b_safe=float(eff_b))

        # Record pre-decision context for counterfactual dataset generation if enabled
        if self.record_decisions and len(parsed_cands) >= 2:
            self.recorded_decisions[cur_epoch] = {
                "epoch": cur_epoch,
                "d_state": d_state,
                "edge_map": edge_map,
                "ref_matching": ref_matching,
                "b_safe": float(eff_b),
                "candidates": parsed_cands,
            }

        # 4. Search or Reference Fallback
        if self.expert_mode == "reference":
            selected_matching = ref_matching
            sel_mode = "reference_b4"
            pred_gain = 0.0
            searched_count = 1
            self.reference_decisions += 1
        else:
            selection = improve(
                d_state,
                baseline=ref_matching,
                check=oracle,
                model=self.advantage_model,
                depth=self.depth,
                beam=self.beam,
                min_predicted_gain=self.delta,
            )
            selected_matching = selection.selected
            pred_gain = float(selection.predicted_gain)
            searched_count = int(selection.searched)

            if selected_matching != ref_matching and pred_gain > self.delta:
                sel_mode = "sc_rlgmi_alternative"
                self.override_decisions += 1
            else:
                sel_mode = "reference_b4"
                selected_matching = ref_matching
                self.reference_decisions += 1

        # 5. Convert canonical matching back to assignments list for simulator
        selected_pack: List[Dict[str, Any]] = []
        tot_pack_bytes = 0
        for edge in selected_matching:
            if edge in edge_map:
                cand = edge_map[edge]
                selected_pack.append(dict(cand.raw_dict))
                tot_pack_bytes += int(cand.payload_bytes)

        meta = {
            "n_pack": len(selected_pack),
            "bytes": tot_pack_bytes,
            "b_limit": eff_b,
            "selected_mode": sel_mode,
            "policy_mode": "sc_rlgmi",
            "predicted_gain": pred_gain,
            "searched_nodes": searched_count,
            "override_rate": self.override_decisions / max(1, self.total_decisions),
            "rejected_geometry": rej,
        }

        return selected_pack, meta

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
        """Bridge method matching the signature used by train_wilab_closed_loop."""
        eff_b = int(float(b_max) * float(beta_throttle))
        return self.select_matching(
            candidates,
            b_max=eff_b,
            unique_sources=unique_sources,
            unique_helpers=unique_helpers,
            layer=layer,
            state={"cbr": 0.04},
            deterministic=deterministic,
        )
