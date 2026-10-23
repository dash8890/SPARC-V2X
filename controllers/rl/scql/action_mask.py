"""Admissible Action Masking for Sequential Capacity-Aware Q-Learning (SCQL).

Enforces:
1. Refined Causal Safety Guard: Risk(i, j) <= r_thresh (0.20)
2. Channel Sensing State: CBR valid, Delay valid, CBR <= 0.16
3. Task Uniqueness: Task i assigned at most once per epoch
4. Helper Uniqueness: Helper j assigned at most once per epoch
5. Remaining Byte Envelope: payload b_i <= B_rem (initial B_max = 175 kB)
6. Physical Feasibility: finish time <= deadline and finish time <= contact duration
7. The STOP action is always admissible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import torch


@dataclass
class CandidateEdge:
    task_id: int
    source_id: int
    helper_id: int
    payload_bytes: int
    packet_count: int
    task_value: float
    cycles: float
    deadline_ms: float
    finish_ms: float
    t_contact_sec: float
    lat_pressure: float
    occ_cost: float
    rad_footprint: float
    risk: float
    features: List[float]
    raw_dict: Dict[str, Any]
    # Observable helper capacity opportunity features
    queue_cycles: float = 0.0
    cpu_cycles_s: float = 1e9
    t_release_s: float = 0.0
    post_slack_sec: float = 0.0
    competing_tasks: int = 1
    competing_value: float = 0.0


def candidate_from_dict(c: Dict[str, Any]) -> CandidateEdge:
    feat = c.get("features") or [0.0] * 10
    dist = float(c.get("_dist") or 0.0)
    bytes_val = int(c.get("payload_bytes", 0))
    # Risk calculation: (d / 250)^2 * (bytes / 175000)
    risk = float(c.get("_risk") or ((dist / 250.0) ** 2 * (bytes_val / 175000.0)))
    cycles = float(feat[1]) * 1e8 if len(feat) > 1 else float(bytes_val * 1000)

    # Causally observable helper kinematics and queue capacity
    f_j = float(feat[4]) * 1e9 if len(feat) > 4 and float(feat[4]) > 0 else 1e9
    q_cycles = float(feat[5]) * 1e8 if len(feat) > 5 else 0.0
    t_release = q_cycles / max(1e7, f_j)
    t_contact = float(c.get("t_contact_sec", 100.0))
    finish = float(c.get("finish_ms", 100.0))
    post_slack = max(0.0, t_contact - (finish / 1000.0))

    return CandidateEdge(
        task_id=int(c.get("task_id", -1)),
        source_id=int(c.get("source_id", -1)),
        helper_id=int(c.get("helper_id", -1)),
        payload_bytes=bytes_val,
        packet_count=int(c.get("packet_count", max(1, bytes_val // 1000))),
        task_value=float(c.get("task_value", c.get("analytic_score", 0.0))),
        cycles=cycles,
        deadline_ms=float(c.get("deadline_ms", 100.0)),
        finish_ms=finish,
        t_contact_sec=t_contact,
        lat_pressure=float(c.get("lat_pressure", c.get("L_kj", 0.0))),
        occ_cost=float(c.get("occ_cost", c.get("O_kj", 0.0))),
        rad_footprint=float(c.get("rad_footprint", c.get("R_kj", 0.0))),
        risk=risk,
        features=[float(x) for x in feat],
        raw_dict=c,
        queue_cycles=q_cycles,
        cpu_cycles_s=f_j,
        t_release_s=t_release,
        post_slack_sec=post_slack,
        competing_tasks=int(c.get("competing_tasks", 1)),
        competing_value=float(c.get("competing_value", c.get("task_value", 0.0))),
    )


def build_admissible_action_mask(
    candidates: Sequence[Dict[str, Any] | CandidateEdge],
    partial_matching: Sequence[Dict[str, Any] | CandidateEdge],
    b_rem: int,
    cbr: float = 0.04,
    cbr_valid: bool = True,
    delay_valid: bool = True,
    r_thresh: float = 0.20,
    cbr_thresh: float = 0.16,
) -> Tuple[List[bool], List[CandidateEdge]]:
    """Constructs a boolean mask over candidates + STOP.
    
    Returns:
      mask: List[bool] of length (N_candidates + 1), where mask[-1] is True (STOP is always valid).
      parsed_candidates: List[CandidateEdge] of length N_candidates.
    """
    assigned_tasks: Set[int] = set()
    assigned_helpers: Set[int] = set()
    for m in partial_matching:
        if isinstance(m, CandidateEdge):
            assigned_tasks.add(m.task_id)
            assigned_helpers.add(m.helper_id)
        elif isinstance(m, dict):
            assigned_tasks.add(int(m["task_id"]))
            assigned_helpers.add(int(m["helper_id"]))

    parsed: List[CandidateEdge] = []
    mask: List[bool] = []

    # If sensing startup is not ready or channel is critically congested, no transmissions allowed
    radio_admissible = bool(cbr_valid and delay_valid and cbr <= cbr_thresh)

    for c in candidates:
        edge = c if isinstance(c, CandidateEdge) else candidate_from_dict(c)
        parsed.append(edge)

        if not radio_admissible:
            mask.append(False)
            continue

        # 1. Refined Causal Safety Guard
        if edge.risk > r_thresh:
            mask.append(False)
            continue

        # 2. Task Uniqueness
        if edge.task_id in assigned_tasks:
            mask.append(False)
            continue

        # 3. Helper Uniqueness
        if edge.helper_id in assigned_helpers:
            mask.append(False)
            continue

        # 4. Remaining Byte Envelope
        if edge.payload_bytes > b_rem:
            mask.append(False)
            continue

        # 5. Physical Deadline & Contact Feasibility
        if edge.lat_pressure > 1.0 or edge.finish_ms > edge.deadline_ms:
            mask.append(False)
            continue

        if edge.finish_ms / 1000.0 > edge.t_contact_sec:
            mask.append(False)
            continue

        mask.append(True)

    # Annotate competing tasks and values for each helper in current candidate pool
    h_counts: Dict[int, int] = {}
    h_vals: Dict[int, float] = {}
    for edge in parsed:
        h_counts[edge.helper_id] = h_counts.get(edge.helper_id, 0) + 1
        h_vals[edge.helper_id] = h_vals.get(edge.helper_id, 0.0) + edge.task_value
    for edge in parsed:
        edge.competing_tasks = h_counts.get(edge.helper_id, 1)
        edge.competing_value = h_vals.get(edge.helper_id, edge.task_value)

    # STOP action (index -1) is always admissible
    mask.append(True)
    return mask, parsed
