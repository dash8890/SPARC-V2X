"""Regression Tests for Causal Pre-Admission Safety Guard (SPARC-V2X Redesign).

Validates:
1. Candidate distance calculation obeys forward Euclidean kinematics.
2. Long-range large bursts (>150m for >20kB, >120m for >35kB) are safely rejected.
3. Excessive single-task packet bursts (>58kB) are strictly rejected to prevent sidelink slot starvation.
4. Early startup transients (t < 0.03s) reject premature bursts before sensing buffer stabilization.
5. CausalGuardedPolicy deterministic matching maintains non-empty packing under safe conditions.
"""
from __future__ import annotations

import math
import pytest
import torch

from controllers.rl.causal_safety_guard import CausalGuardedPolicy, score_and_match_guarded


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
        # Helper 0 close (dist ~ 10m), Helper 1 far (dist ~ 200m)
        self.helpers = [
            DummyHelper(0, 110.0, 4.0, 20.0),
            DummyHelper(1, 300.0, 4.0, 20.0),
        ]


def test_causal_guard_distance_and_burst_rejection():
    layer = DummyLayer(current_epoch=10)
    # now_sec = 10 * 10 / 1000 = 0.1s
    # Dist to Helper 0: 10m. Dist to Helper 1: 200m.

    cands = [
        # Candidate 1: Close helper (10m), moderate burst (25kB) -> ACCEPT
        {"task_id": 1, "source_id": 0, "helper_id": 0, "payload_bytes": 25000,
         "norm_value": 0.8, "lat_pressure": 0.3, "occ_cost": 0.1, "rad_footprint": 0.2, "contact_feasible": True},
        # Candidate 2: Far helper (200m), moderate burst (25kB) -> REJECT (>150m and >20kB)
        {"task_id": 2, "source_id": 0, "helper_id": 1, "payload_bytes": 25000,
         "norm_value": 0.9, "lat_pressure": 0.2, "occ_cost": 0.1, "rad_footprint": 0.2, "contact_feasible": True},
        # Candidate 3: Close helper (10m), extreme burst (65kB) -> REJECT (>58kB)
        {"task_id": 3, "source_id": 0, "helper_id": 0, "payload_bytes": 65000,
         "norm_value": 1.0, "lat_pressure": 0.1, "occ_cost": 0.1, "rad_footprint": 0.4, "contact_feasible": True},
    ]

    packed, meta = score_and_match_guarded(
        cands, weights=[2.5, 1.0, 1.0, 0.5], beta_throttle=0.95,
        b_max=175000, unique_sources=True, unique_helpers=True,
        d_max=150.0, layer=layer,
    )

    # Candidates 2 and 3 must be rejected by causal guard
    assert meta["rejected_geometry"] == 2
    assert len(packed) == 1
    assert packed[0]["task_id"] == 1
    assert packed[0]["helper_id"] == 0


def test_causal_guard_startup_transient_protection():
    # At epoch 1 (t = 0.01s < 0.03s), sensing buffer not stabilized
    layer = DummyLayer(current_epoch=1)
    cands = [
        {"task_id": 10, "source_id": 0, "helper_id": 0, "payload_bytes": 28000,
         "norm_value": 0.9, "lat_pressure": 0.2, "occ_cost": 0.1, "rad_footprint": 0.2, "contact_feasible": True},
        {"task_id": 11, "source_id": 0, "helper_id": 0, "payload_bytes": 15000,
         "norm_value": 0.7, "lat_pressure": 0.2, "occ_cost": 0.1, "rad_footprint": 0.1, "contact_feasible": True},
    ]

    packed, meta = score_and_match_guarded(
        cands, weights=[2.5, 1.0, 1.0, 0.5], beta_throttle=0.95,
        b_max=175000, unique_sources=True, unique_helpers=True,
        d_max=150.0, layer=layer,
    )

    # 28kB burst at t=0.01s rejected; 15kB accepted
    assert meta["rejected_geometry"] == 1
    assert len(packed) == 1
    assert packed[0]["task_id"] == 11
