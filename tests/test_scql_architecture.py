"""Unit tests for SCQL (Sequential Capacity-Aware Q-Learning) architecture & action masking."""
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.rl.scql.action_mask import CandidateEdge, build_admissible_action_mask
from controllers.rl.scql.q_network import SCQLDuelingQNetwork
from controllers.rl.scql.scql_policy import SCQLSequentialPolicy


def mock_candidate(task_id: int, helper_id: int, payload_bytes: int, risk: float, value: float = 3.0) -> CandidateEdge:
    raw = {
        "task_id": task_id,
        "source_id": 0,
        "helper_id": helper_id,
        "payload_bytes": payload_bytes,
        "packet_count": max(1, payload_bytes // 1000),
        "task_value": value,
        "analytic_score": value / (payload_bytes / 1000.0),
        "deadline_ms": 100.0,
        "finish_ms": 50.0,
        "t_contact_sec": 50.0,
        "lat_pressure": 0.5,
        "occ_cost": 0.001,
        "rad_footprint": 0.1,
        "_dist": 100.0,
        "_risk": risk,
        "features": [0.03, 0.5, 0.1, value, 0.7, 0.0, 0.0, 0.04, 0.1, 1.0],
    }
    return CandidateEdge(
        task_id=task_id,
        source_id=0,
        helper_id=helper_id,
        payload_bytes=payload_bytes,
        packet_count=max(1, payload_bytes // 1000),
        task_value=value,
        cycles=1e8,
        deadline_ms=100.0,
        finish_ms=50.0,
        t_contact_sec=50.0,
        lat_pressure=0.5,
        occ_cost=0.001,
        rad_footprint=0.1,
        risk=risk,
        features=raw["features"],
        raw_dict=raw,
    )


def test_action_mask_invariants():
    c1 = mock_candidate(task_id=1, helper_id=1, payload_bytes=30000, risk=0.10)
    c2_risky = mock_candidate(task_id=2, helper_id=2, payload_bytes=80000, risk=0.25)  # Violates guard (0.25 > 0.20)
    c3_large = mock_candidate(task_id=3, helper_id=3, payload_bytes=100000, risk=0.15) # Exceeds b_rem (100k > 50k)

    candidates = [c1, c2_risky, c3_large]
    partial_matching = []
    b_rem = 50000

    mask, parsed = build_admissible_action_mask(candidates, partial_matching, b_rem=b_rem, r_thresh=0.20)

    assert len(mask) == 4  # 3 candidates + 1 STOP
    assert mask[0] is True   # c1 is safe and fits
    assert mask[1] is False  # c2_risky rejected by guard (0.25 > 0.20)
    assert mask[2] is False  # c3_large rejected by byte budget (100k > 50k)
    assert mask[3] is True   # STOP is always admissible


def test_helper_and_task_uniqueness():
    c1 = mock_candidate(task_id=1, helper_id=1, payload_bytes=20000, risk=0.05)
    c2 = mock_candidate(task_id=1, helper_id=2, payload_bytes=20000, risk=0.05)  # Same task 1
    c3 = mock_candidate(task_id=2, helper_id=1, payload_bytes=20000, risk=0.05)  # Same helper 1
    c4 = mock_candidate(task_id=3, helper_id=3, payload_bytes=20000, risk=0.05)  # Distinct

    candidates = [c1, c2, c3, c4]
    # Commit c1 (task 1 -> helper 1) to partial matching
    partial_matching = [c1]
    b_rem = 150000

    mask, _ = build_admissible_action_mask(candidates, partial_matching, b_rem=b_rem)

    assert mask[0] is False  # Task 1 & Helper 1 already assigned
    assert mask[1] is False  # Task 1 already assigned
    assert mask[2] is False  # Helper 1 already assigned
    assert mask[3] is True   # Distinct task 3 and helper 3 is admissible
    assert mask[4] is True   # STOP action is admissible


def test_scql_dueling_q_network_forward():
    q_net = SCQLDuelingQNetwork()
    q_net.eval()

    c1 = mock_candidate(task_id=1, helper_id=1, payload_bytes=25000, risk=0.08)
    c2 = mock_candidate(task_id=2, helper_id=2, payload_bytes=35000, risk=0.12)
    candidates = [c1, c2]

    history = torch.randn(1, 10, 4)
    b_rem = 120000.0
    action_mask = [True, False, True]  # c1 admissible, c2 masked, STOP admissible

    with torch.no_grad():
        q_vals = q_net(candidates, history, b_rem=b_rem, action_mask=action_mask)

    assert q_vals.shape == (3,)  # 2 candidates + 1 STOP
    assert torch.isfinite(q_vals[0]), "Admissible candidate 1 must have finite Q-value"
    assert q_vals[1].item() <= -1e8, "Inadmissible candidate 2 must be masked with large negative value"
    assert torch.isfinite(q_vals[2]), "STOP action must have finite Q-value"


def test_scql_sequential_policy_execution():
    policy = SCQLSequentialPolicy(b_max=175000, r_thresh=0.20)
    c1 = mock_candidate(task_id=1, helper_id=1, payload_bytes=25000, risk=0.08).raw_dict
    c2 = mock_candidate(task_id=2, helper_id=2, payload_bytes=35000, risk=0.12).raw_dict
    c3_risky = mock_candidate(task_id=3, helper_id=3, payload_bytes=90000, risk=0.35).raw_dict

    candidates = [c1, c2, c3_risky]
    pack, meta = policy._sequential_match(candidates, deterministic=True)

    assert isinstance(pack, list)
    assert meta["policy_mode"] == "scql"
    # c3_risky must never be admitted
    admitted_tasks = [a["task_id"] for a in pack]
    assert 3 not in admitted_tasks, "Risky candidate must be rejected by causal guard"
    assert meta["bytes"] <= 175000, "Pack must respect byte limit"


def test_scql_trainer_step():
    from controllers.rl.scql.trainer import SCQLTrainer
    from controllers.rl.scql.replay_buffer import SCQLReplayBuffer, Transition

    q_net = SCQLDuelingQNetwork()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = SCQLReplayBuffer(capacity=100)

    c1 = mock_candidate(task_id=1, helper_id=1, payload_bytes=20000, risk=0.05)
    c2 = mock_candidate(task_id=2, helper_id=2, payload_bytes=25000, risk=0.08)
    candidates = [c1, c2]
    history = torch.randn(1, 10, 4)

    # 1. Intra-epoch transition: choose c1, remaining b_rem decreases, partial matching gains c1
    trans_intra = Transition(
        candidates=candidates,
        history=history,
        b_rem=100000.0,
        partial_matching=[],
        action_idx=0,
        reward=0.0,
        next_candidates=candidates,
        next_history=history,
        next_b_rem=80000.0,
        next_partial_matching=[c1],
        next_action_mask=[False, True, True],
        done=False,
        is_stop=False,
    )
    # 2. Inter-epoch transition: choose STOP, simulation advances, receives reward
    trans_stop = Transition(
        candidates=candidates,
        history=history,
        b_rem=80000.0,
        partial_matching=[c1],
        action_idx=2,  # STOP
        reward=3.0,
        next_candidates=[c2],
        next_history=history,
        next_b_rem=175000.0,
        next_partial_matching=[],
        next_action_mask=[True, True],
        done=False,
        is_stop=True,
    )

    buffer.push(trans_intra)
    buffer.push(trans_stop)

    trainer = SCQLTrainer(q_net, optimizer, buffer, gamma=0.995, tau=0.05)
    metrics = trainer.train_step(batch_size=2, beta=0.4)

    assert "loss" in metrics
    assert "td_error_mean" in metrics
    assert "grad_norm" in metrics
    assert metrics["loss"] >= 0.0
    assert metrics["td_error_mean"] >= 0.0

