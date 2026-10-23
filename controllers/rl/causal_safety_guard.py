"""SPARC-V2X Causal Pre-Admission Safety Guard.

Implements state- and distance-dependent sidelink safety filtering:
  Observed radio state -> Safety-feasible candidate set C_safe -> Task-helper priorities -> Assignments

Mathematical Formulation:
  The spatio-temporal interference risk of transmitting task k (payload b_k) to helper j (distance d_kj) is:
      Risk(k, j) = (d_kj / R_comm)^2 * (b_k / B_max)
  where R_comm = 250.0m (3GPP NR-V2X communication range) and B_max = 175,000 B (epoch airtime cap).

Pre-Admission Filter Criteria (C_safe):
  1. Risk(k, j) <= r_thresh (default 0.20)
  2. Cell-edge high-burst cut: reject if d_kj > 220m and b_k > 20,000 B
  3. Startup transient protection: reject if t < 0.025s and b_k > 10,000 B
  4. Monopolization cut: reject if b_k > 62,000 B and d_kj > 35m
  5. Dynamic epoch throttling: if any admitted candidate has d_kj > 180m,
     the aggregate epoch byte budget is throttled to 55,000 B to prevent co-channel half-duplex collisions.

Enables matched 4-way attribution:
  1. B4 Reference (no guard)
  2. Guarded B4 (B4 under C_safe)
  3. Causal Guarded Structured Priority (SP under C_safe)
  4. Zero Offload B_0 (Background safety floor)
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.rl.structured_priority_policy import StructuredPriorityPolicy, score_and_match
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.hq_spo_baselines import CeRiskWeights


def causal_sidelink_risk(
    dist: float,
    payload_bytes: int,
    r_comm: float = 250.0,
    b_max: float = 175000.0,
) -> float:
    """Computes spatio-temporal sidelink interference risk for candidate offload (k, j)."""
    if dist < 0.0:
        return 0.0
    return ((dist / r_comm) ** 2) * (float(payload_bytes) / b_max)


def filter_candidates_causal_guard(
    candidates: List[Dict[str, Any]],
    layer: Any = None,
    r_thresh: float = 0.20,
    r_comm: float = 250.0,
    b_max: int = 175000,
    startup_sec: float = 0.025,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Applies causal state- and distance-dependent sidelink safety filtering.

    Returns:
        (filtered_candidates, n_rejected, effective_b_max)
    """
    now_sec = 0.0
    if layer is not None and hasattr(layer, "current_epoch"):
        now_sec = (10.0 * float(layer.current_epoch)) / 1000.0

    filtered = []
    rejected = 0
    has_far = False

    for c in candidates:
        l_kj = float(c.get("lat_pressure", c.get("L_kj", 0.0)))
        contact_feas = bool(c.get("contact_feasible", True))
        if l_kj > 1.0 or not contact_feas:
            continue

        b_k = int(c.get("payload_bytes", 0))
        sid = c.get("source_id")
        hid = c.get("helper_id")

        dist = float(c.get("_dist") or 0.0)
        if layer is not None and hasattr(layer, "source_kinematics") and sid in layer.source_kinematics:
            src_kin = layer.source_kinematics[sid]
            hlp = next((h for h in getattr(layer, "helpers", []) if getattr(h, "helper_id", None) == hid), None)
            if hlp is not None and getattr(hlp, "kinematics", None) is not None:
                x_s = src_kin.x(now_sec)
                y_s = src_kin.y
                x_h = hlp.kinematics.x(now_sec)
                y_h = hlp.kinematics.y
                dist = math.hypot(x_s - x_h, y_s - y_h)

        risk = float(c.get("_risk") or causal_sidelink_risk(dist, b_k, r_comm=r_comm, b_max=float(b_max)))

        is_risky = (
            (risk > r_thresh) or
            (dist > 220.0 and b_k > 20000) or
            (now_sec < startup_sec and b_k > 10000) or
            (b_k > 62000 and dist > 35.0)
        )
        if is_risky:
            rejected += 1
            continue

        c_copy = dict(c)
        c_copy["_dist"] = dist
        c_copy["_risk"] = risk
        if dist > 180.0:
            has_far = True
        filtered.append(c_copy)

    eff_b_max = 55000 if has_far else b_max
    return filtered, rejected, eff_b_max


def score_and_match_causal_guard(
    candidates: List[Dict[str, Any]],
    weights: List[float],
    beta_throttle: float,
    b_max: int = 175000,
    unique_sources: bool = True,
    unique_helpers: bool = True,
    layer: Any = None,
    r_thresh: float = 0.20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministic Priority Matching under Causal Safety Guard."""
    filtered, rej, eff_b = filter_candidates_causal_guard(
        candidates, layer=layer, r_thresh=r_thresh, b_max=b_max,
    )
    b_limit = min(int(float(beta_throttle) * b_max), eff_b)
    pack, meta = score_and_match(
        filtered,
        weights=weights,
        beta_throttle=1.0,
        b_max=b_limit,
        unique_sources=unique_sources,
        unique_helpers=unique_helpers,
    )
    meta["rejected_geometry"] = rej
    return pack, meta


class CausalGuardedPolicy(StructuredPriorityPolicy):
    """Structured Priority Policy equipped with State-Dependent Causal Safety Guard."""
    def __init__(self, r_thresh: float = 0.20):
        super().__init__()
        self.r_thresh = float(r_thresh)
        self.score_and_match_fn = self._guarded_match

    def _guarded_match(
        self,
        candidates: List[Dict[str, Any]],
        weights: List[float],
        beta_throttle: float,
        b_max: int = 175000,
        unique_sources: bool = True,
        unique_helpers: bool = True,
        layer: Any = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return score_and_match_causal_guard(
            candidates,
            weights=weights,
            beta_throttle=beta_throttle,
            b_max=b_max,
            unique_sources=unique_sources,
            unique_helpers=unique_helpers,
            layer=layer,
            r_thresh=self.r_thresh,
        )


class GuardedB4Policy(ResidualTaskPolicy):
    """B4 Baseline Controller equipped with the identical State-Dependent Causal Safety Guard."""
    def __init__(self, r_thresh: float = 0.20):
        super().__init__(logit_mode="direct", logit_scale=100.0)
        self.ce_weights = CeRiskWeights.load()
        self.r_thresh = float(r_thresh)
        self.score_and_match_fn = self._guarded_b4_match

    def _guarded_b4_match(
        self,
        candidates: List[Dict[str, Any]],
        weights: List[float],
        beta_throttle: float,
        b_max: int = 175000,
        unique_sources: bool = True,
        unique_helpers: bool = True,
        layer: Any = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        filtered, rej, eff_b = filter_candidates_causal_guard(
            candidates, layer=layer, r_thresh=self.r_thresh, b_max=b_max,
        )
        b_limit = min(b_max, eff_b)
        # B4 greedy knapsack on value density
        pack, meta = score_and_match(
            filtered,
            weights=[1.0, 0.0, 0.0, 0.0],
            beta_throttle=1.0,
            b_max=b_limit,
            unique_sources=unique_sources,
            unique_helpers=unique_helpers,
        )
        meta["rejected_geometry"] = rej
        return pack, meta
