"""Integrity and Unit Tests for Redesigned SPARC-V2X Formulation."""
import math
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from experiments.wilab_closed_loop.task_layer import TaskLayer, VehicleKinematics, compute_contact_duration
from controllers.rl.structured_priority_policy import StructuredPriorityPolicy, score_and_match


def test_contact_duration_kinematics():
    """Verify analytical contact calculation across approaching, receding, and parallel trajectories."""
    # Case 1: Two vehicles traveling at identical speeds in range
    v1 = VehicleKinematics(0, is_source=True, x_0=100.0, y=4.0, velocity=20.0)
    v2 = VehicleKinematics(1, is_source=False, x_0=150.0, y=4.0, velocity=20.0)
    # Distance is 50m, r_comm=250m -> constant contact
    dur = compute_contact_duration(v1, v2, t_sec=0.0, r_comm=250.0)
    assert dur > 500.0, f"Expected long contact, got {dur}"

    # Case 2: Source traveling faster, currently approaching helper ahead
    # v1 at x=0 (25 m/s), v2 at x=200 (15 m/s). Delta_v = 10 m/s. Delta_x = -200 m.
    # r_comm = 250m. They are in range! Distance = 200m <= 250m.
    # They will pass each other and separate until distance exceeds 250m.
    # At t=0, delta_x = -200. Boundary is delta_x = +250.
    # Time to boundary = (250 - (-200)) / 10 = 450 / 10 = 45.0 seconds.
    v1_fast = VehicleKinematics(0, is_source=True, x_0=0.0, y=4.0, velocity=25.0)
    v2_slow = VehicleKinematics(1, is_source=False, x_0=200.0, y=4.0, velocity=15.0)
    dur_app = compute_contact_duration(v1_fast, v2_slow, t_sec=0.0, r_comm=250.0)
    assert abs(dur_app - 45.0) < 1e-3, f"Expected 45.0s, got {dur_app}"

    # Case 3: Out of range (cross-lane separation exceeds r_comm)
    v_far = VehicleKinematics(2, is_source=False, x_0=0.0, y=300.0, velocity=20.0)
    dur_far = compute_contact_duration(v1, v_far, t_sec=0.0, r_comm=250.0)
    assert dur_far == 0.0


def test_priority_ranking_breaks_invariance():
    """Verify that varying candidate-dependent weights w_v, w_q, w_o, w_r genuinely alters task rankings."""
    # Create two candidates competing for the same helper:
    # Candidate 1: High value, but high latency pressure & high occupancy
    # Candidate 2: Moderate value, but low latency pressure & low occupancy
    cand1 = {
        "task_id": 1,
        "source_id": 0,
        "helper_id": 0,
        "payload_bytes": 30000,
        "norm_value": 0.9,       # V_kj = 0.9
        "lat_pressure": 0.85,    # L_kj = 0.85 (tight deadline)
        "occ_cost": 0.80,        # O_kj = 0.80 (heavy helper queue load)
        "rad_footprint": 0.20,
        "contact_feasible": True,
    }
    cand2 = {
        "task_id": 2,
        "source_id": 1,
        "helper_id": 0,
        "payload_bytes": 25000,
        "norm_value": 0.6,       # V_kj = 0.6
        "lat_pressure": 0.20,    # L_kj = 0.20 (slack deadline)
        "occ_cost": 0.15,        # O_kj = 0.15 (light load)
        "rad_footprint": 0.15,
        "contact_feasible": True,
    }
    candidates = [cand1, cand2]

    # Policy Action A: Value-heavy policy (w_v=4.0, w_q=0.5, w_o=0.5, w_r=0.0)
    # Score 1: 4.0*0.9 - 0.5*0.85 - 0.5*0.80 = 3.6 - 0.425 - 0.40 = 2.775
    # Score 2: 4.0*0.6 - 0.5*0.20 - 0.5*0.15 = 2.4 - 0.10 - 0.075 = 2.225
    # Cand 1 ranks FIRST
    packed_a, meta_a = score_and_match(candidates, weights=[4.0, 0.5, 0.5, 0.0], beta_throttle=1.0)
    assert packed_a[0]["task_id"] == 1, "Expected Cand 1 to rank first under value-heavy weights"

    # Policy Action B: Latency/Occupancy-averse policy (w_v=1.0, w_q=3.0, w_o=3.0, w_r=0.0)
    # Score 1: 1.0*0.9 - 3.0*0.85 - 3.0*0.80 = 0.9 - 2.55 - 2.40 = -4.05
    # Score 2: 1.0*0.6 - 3.0*0.20 - 3.0*0.15 = 0.6 - 0.60 - 0.45 = -0.45
    # Cand 2 ranks FIRST!
    packed_b, meta_b = score_and_match(candidates, weights=[1.0, 3.0, 3.0, 0.0], beta_throttle=1.0)
    assert packed_b[0]["task_id"] == 2, "Expected Cand 2 to rank first under occupancy-averse weights"

    print("Success: Verified that varying weights genuinely reorders candidates (ranking invariance broken)!")


def test_structured_priority_policy_forward():
    """Verify neural network forward pass, shapes, and twin critics."""
    policy = StructuredPriorityPolicy()
    task_feats = torch.randn(2, 6, 5)     # 2 batches, 6 tasks, 5 features
    helper_feats = torch.randn(2, 8, 4)   # 2 batches, 8 helpers, 4 features
    radio_hist = torch.randn(2, 10, 4)    # 2 batches, 10 epochs, 4 features
    budget_block = torch.randn(2, 3)      # 2 batches, 3 features

    w_vec, beta, logp, z_w, z_b, ent = policy.forward_actions(
        task_feats, helper_feats, radio_hist, budget_block, deterministic=True
    )

    assert w_vec.shape == (2, 4), f"Expected (2, 4), got {w_vec.shape}"
    assert beta.shape == (2,), f"Expected (2,), got {beta.shape}"
    assert (w_vec >= 0.0).all(), "Weights must be non-negative"
    assert (beta >= 0.0).all() and (beta <= 1.0).all(), "Beta must be in [0, 1]"

    v_r = policy.value(task_feats, helper_feats, radio_hist, budget_block)
    v_c = policy.cost_value(task_feats, helper_feats, radio_hist, budget_block)
    assert v_r.shape == (2,), f"Expected (2,), got {v_r.shape}"
    assert v_c.shape == (2,), f"Expected (2,), got {v_c.shape}"


def test_task_layer_kinematics_and_queue_draining():
    """Verify TaskLayer candidate generation, contact filtering, and queue execution."""
    tl = TaskLayer(density=15, seed=73300, tasks_per_source=8, arrival_mode="streaming")
    assert len(tl.helpers) == 12
    assert len(tl.source_kinematics) == 3

    # Step through 5 epochs
    for ep in range(5):
        tl.set_epoch(ep)
        cands = tl.candidates(cbr=0.04)
        for c in cands:
            assert "norm_value" in c
            assert "lat_pressure" in c
            assert "occ_cost" in c
            assert "rad_footprint" in c
            assert c["lat_pressure"] <= 1.0
            assert c["contact_feasible"] is True

    print("Success: TaskLayer candidates generate all specification features and respect physical feasibility.")
