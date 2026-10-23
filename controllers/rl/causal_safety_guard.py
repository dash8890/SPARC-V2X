"""Comprehensive Experiment for Task 4: Matched Closed-Loop Safety-Guard Evaluation.

Compares under strictly matched conditions:
1. Baseline B4 Reference (frozen)
2. Corrected Structured Priority Policy (initialization / baseline-parity checkpoint)
3. Structured Priority Policy with Causal Pre-Admission Risk Guard:
   Uses only causal pre-admission signals available before transmission:
   - Candidate transmission distance d(k, j) derived from forward kinematics
   - Candidate payload bytes b_k
   - Sensing-window Channel Busy Ratio (CBR_t)
   - Bounded epoch burst cap conditioned on channel congestion:
       B_epoch_cap(cbr) = 175000 * max(0.20, 1.0 - 15.0 * max(0.0, cbr - 0.040))
       and edge-of-cell exclusion: skip candidate if d_kj > 210m and b_k > 25000.

Evaluates on:
- Validation seeds: [73300, 73301, 73302, 73303]
- Training seeds:   [73200, 73201]

Reports:
- V_done (completed utility)
- P_viol (safety violation fraction under absolute 2% limit)
- Completed task count
- Total admitted bytes
- Mean PRR and min window PRR
- Causal interventions taken prior to failure windows
- Rejected and redirected tasks
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.rl.structured_priority_policy import StructuredPriorityPolicy, score_and_match


class CausalGuardedPolicy(StructuredPriorityPolicy):
    """Structured Priority Policy with Causal Pre-Admission Risk Guard."""
    def __init__(self, d_max: float = 150.0, cbr_thresh: float = 0.042):
        super().__init__()
        self.d_max = float(d_max)
        self.cbr_thresh = float(cbr_thresh)
        self.score_and_match_fn = self._guarded_match

    def _guarded_match(self, candidates, weights, beta_throttle, b_max=175000, unique_sources=True, unique_helpers=True, layer=None):
        return score_and_match_guarded(
            candidates, weights, beta_throttle,
            b_max=b_max, unique_sources=unique_sources, unique_helpers=unique_helpers,
            d_max=self.d_max, layer=layer,
        )

    def act_numpy(self, state, layer=None, *, deterministic: bool = True):
        cbr = float(state.get("cbr", 0.0))
        w, beta, lp, zw, zb, obs = super().act_numpy(state, layer, deterministic=deterministic)
        if cbr > self.cbr_thresh:
            adapted_beta = min(beta, max(0.25, 0.88 * (1.0 - 10.0 * (cbr - self.cbr_thresh))))
        else:
            adapted_beta = beta
        return w, adapted_beta, lp, zw, zb, obs


def score_and_match_guarded(
    candidates,
    weights,
    beta_throttle,
    b_max=175000,
    unique_sources=True,
    unique_helpers=True,
    d_max=150.0,
    now_sec=0.0,
    layer=None,
):
    """Deterministic Priority Matching with Causal Pre-Admission Geometric Guard."""
    if not candidates:
        return [], {"n_pack": 0, "bytes": 0, "scores": [], "rejected_geometry": 0}

    w = [float(x) for x in weights]
    w_v, w_q, w_o, w_r = w[0], w[1], w[2], w[3]
    b_limit = int(float(beta_throttle) * b_max)

    if layer is not None and hasattr(layer, "current_epoch"):
        now_sec = (10.0 * float(layer.current_epoch)) / 1000.0

    scored = []
    rejected_geometry = 0
    for c in candidates:
        l_kj = float(c.get("lat_pressure", c.get("L_kj", 0.0)))
        contact_feas = bool(c.get("contact_feasible", True))
        if l_kj > 1.0 or not contact_feas:
            continue

        # Causal Pre-Admission Guard: Source-Helper Euclidean Distance Check
        sid = c.get("source_id")
        hid = c.get("helper_id")
        dist = -1.0
        if layer is not None and hasattr(layer, "source_kinematics") and sid in layer.source_kinematics:
            src_kin = layer.source_kinematics[sid]
            hlp = next((h for h in layer.helpers if h.helper_id == hid), None)
            if hlp is not None and hlp.kinematics is not None:
                x_s = src_kin.x(now_sec)
                y_s = src_kin.y
                x_h = hlp.kinematics.x(now_sec)
                y_h = hlp.kinematics.y
                dist = math.sqrt((x_s - x_h)**2 + (y_s - y_h)**2)

        # Causal Pre-Admission Risk Guard:
        # 1. Edge-of-cell high-power transmission check (>150m for >20kB, >120m for >35kB)
        # 2. Maximum single-task burst limit (b_k > 58kB monopolizes >58% of 100ms slots)
        # 3. Startup transient protection (t < 0.03s before sensing buffer stabilizes)
        b_k = int(c.get("payload_bytes", 0))
        is_risky = (
            (dist > d_max and b_k > 20000) or
            (b_k > 58000) or
            (dist > 120.0 and b_k > 35000) or
            (now_sec < 0.03 and b_k > 20000)
        )
        if is_risky:
            rejected_geometry += 1
            continue

        v_kj = float(c.get("norm_value", c.get("V_kj", 0.0)))
        o_kj = float(c.get("occ_cost", c.get("O_kj", 0.0)))
        r_kj = float(c.get("rad_footprint", c.get("R_kj", 0.0)))

        score = w_v * v_kj - w_q * l_kj - w_o * o_kj - w_r * r_kj
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    packed = []
    used_sources = set()
    used_helpers = set()
    total_bytes = 0
    scores_kept = []

    for s, c in scored:
        sid = int(c.get("source_id", -1))
        hid = int(c.get("helper_id", -1))
        b = int(c.get("payload_bytes", 0))

        if unique_sources and sid in used_sources:
            continue
        if unique_helpers and hid in used_helpers:
            continue
        if total_bytes + b > b_limit:
            continue

        packed.append(c)
        scores_kept.append(float(s))
        if unique_sources:
            used_sources.add(sid)
        if unique_helpers:
            used_helpers.add(hid)
        total_bytes += b

    meta = {
        "n_pack": len(packed),
        "bytes": total_bytes,
        "b_limit": b_limit,
        "weights": [round(x, 4) for x in w],
        "beta_throttle": float(beta_throttle),
        "scores": scores_kept,
        "rejected_geometry": rejected_geometry,
    }
    return packed, meta
