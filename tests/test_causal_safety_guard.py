"""Regression Tests for Causal Pre-Admission Safety Guard (SPARC-V2X Redesign).

Validates:
1. Candidate distance calculation obeys forward Euclidean kinematics.
2. Spatio-temporal interference risk formula: Risk(k, j) = (d_kj / R_comm)^2 * (b_k / B_max).
3. Long-range large bursts (high risk) are safely rejected while local bursts are preserved.
4. Early startup transients (t < 0.025s) reject premature bursts before sensing buffer stabilization.
5. CausalGuardedPolicy and GuardedB4Policy both produce deterministic safe packings under C_safe.
"""
from __future__ import annotations

import math
import pytest
import torch

from controllers.rl.causal_safety_guard import (
    causal_sidelink_risk,
    filter_candidates_causal_guard,
    score_and_match_causal_guard,
    CausalGuardedPolicy,
    GuardedB4Policy,
)


class DummyKinematics:
    def __init__(self, x0: float, y: float, v: float):
        self.x0 = x0
        self.y = y
        self.velocity = v

    def x(self, t: float) -> float:
        return self.x0 + self.velocity * t


class DummyHelper:
    def __init__(self, helper_id: int, x0: float, y: float, v: float):
        self.helper_id = helper_id
        self.kinematics = DummyKinematics(x0, y, v)


class DummyLayer:
    def __init__(self, current_epoch: int = 10):
        self.current_epoch = current_epoch
        # Source 0 at x=100, y=4
        self.source_kinematics = {
            0: DummyKinematics(100.0, 4.0, 20.0),
        }
        # Helper 0 close (dist ~ 20m), Helper 1 far (dist ~ 200m)
        self.helpers = [
            DummyHelper(0, 120.0, 4.0, 20.0),
            DummyHelper(1, 300.0, 4.0, 20.0),
        ]


def test_causal_sidelink_risk_calculation():
    # d = 125m (0.5 of 250m), payload = 87500B (0.5 of 175000B)
    # risk = (0.5)^2 * 0.5 = 0.25 * 0.5 = 0.125
    r = causal_sidelink_risk(125.0, 87500, r_comm=250.0, b_max=175000.0)
    assert math.isclose(r, 0.125, rel_tol=1e-5)

    # d = 250m, payload = 175000B -> risk = 1.0
    r_max = causal_sidelink_risk(250.0, 175000, r_comm=250.0, b_max=175000.0)
    assert math.isclose(r_max, 1.0, rel_tol=1e-5)


def test_causal_guard_distance_and_burst_rejection():
    layer = DummyLayer(current_epoch=10)
    # now_sec = 10 * 10 / 1000 = 0.1s
    # Dist to Helper 0: 20m. Dist to Helper 1: 200m.

    cands = [
        # Candidate 1: Close helper (20m), moderate burst (35kB) -> ACCEPT (risk = (20/250)^2 * (35/175) = 0.0064 * 0.2 = 0.00128)
        {"task_id": 1, "source_id": 0, "helper_id": 0, "payload_bytes": 35000,
         "norm_value": 0.8, "lat_pressure": 0.3, "occ_cost": 0.1, "rad_footprint": 0.2, "contact_feasible": True},
        # Candidate 2: Far helper (200m), moderate burst (60kB) -> REJECT (risk = (200/250)^2 * (60/175) = 0.64 * 0.342 = 0.219 > 0.20)
        {"task_id": 2, "source_id": 0, "helper_id": 1, "payload_bytes": 60000,
         "norm_value": 0.9, "lat_pressure": 0.2, "occ_cost": 0.1, "rad_footprint": 0.2, "contact_feasible": True},
    ]

    packed, meta = score_and_match_causal_guard(
        cands, weights=[2.5, 1.0, 1.0, 0.5], beta_throttle=0.95,
        b_max=175000, unique_sources=True, unique_helpers=True,
        layer=layer, r_thresh=0.20,
    )

    # Candidate 2 rejected by risk threshold
    assert meta["rejected_geometry"] == 1
    assert len(packed) == 1
    assert packed[0]["task_id"] == 1
    assert packed[0]["helper_id"] == 0


def test_causal_guard_startup_transient_protection():
    # At epoch 1 (t = 0.01s < 0.025s), sensing buffer not stabilized
    layer = DummyLayer(current_epoch=1)
    cands = [
        {"task_id": 10, "source_id": 0, "helper_id": 0, "payload_bytes": 28000,
         "norm_value": 0.9, "lat_pressure": 0.2, "occ_cost": 0.1, "rad_footprint": 0.2, "contact_feasible": True},
        {"task_id": 11, "source_id": 0, "helper_id": 0, "payload_bytes": 8000,
         "norm_value": 0.7, "lat_pressure": 0.2, "occ_cost": 0.1, "rad_footprint": 0.1, "contact_feasible": True},
    ]

    packed, meta = score_and_match_causal_guard(
        cands, weights=[2.5, 1.0, 1.0, 0.5], beta_throttle=0.95,
        b_max=175000, unique_sources=True, unique_helpers=True,
        layer=layer, r_thresh=0.20,
    )

    # 28kB burst at t=0.01s rejected; 8kB accepted
    assert meta["rejected_geometry"] == 1
    assert len(packed) == 1
    assert packed[0]["task_id"] == 11


def test_guarded_b4_policy_execution():
    layer = DummyLayer(current_epoch=10)
    cands = [
        {"task_id": 101, "source_id": 0, "helper_id": 0, "payload_bytes": 20000,
         "norm_value": 0.5, "lat_pressure": 0.1, "occ_cost": 0.1, "rad_footprint": 0.1, "contact_feasible": True},
        {"task_id": 102, "source_id": 0, "helper_id": 1, "payload_bytes": 70000,
         "norm_value": 0.9, "lat_pressure": 0.1, "occ_cost": 0.1, "rad_footprint": 0.1, "contact_feasible": True},
    ]
    b4_guarded = GuardedB4Policy(r_thresh=0.20)
    b4_guarded.eval()

    pack, meta = b4_guarded.score_and_match_fn(
        cands, weights=[1.0, 0.0, 0.0, 0.0], beta_throttle=1.0,
        b_max=175000, unique_sources=True, unique_helpers=True, layer=layer,
    )
    assert meta["rejected_geometry"] == 1
    assert len(pack) == 1
    assert pack[0]["task_id"] == 101
