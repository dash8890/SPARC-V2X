import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.rl.sc_rlgmi.sc_rlgmi_core import (
    Edge,
    Matching,
    canonical,
    DecisionState,
    feasible,
    neighbors,
    AdvantageModel,
    Selection,
    improve,
    PairExample,
    paired_outcome,
    train_pairs,
)

S = DecisionState(
    features=(0.2, 0.5),
    edge_features={
        (1, 1): (1.0, 0.0),
        (1, 2): (1.0, 1.0),
        (2, 1): (2.0, 0.0),
        (2, 2): (2.0, 1.0),
        (3, 1): (3.0, 0.0),
    },
)


def frozen_guard(s, m):
    return len(m) <= 2 and sum(s.edge_features[e][0] for e in m) <= 3.0


def test_candidate_operations_are_feasible_and_diverse():
    b = canonical([(1, 1), (2, 2)])
    ns = neighbors(S, b, frozen_guard)
    assert canonical([(1, 2), (2, 1)]) in ns  # two-way exchange
    assert canonical([(1, 1)]) in ns  # remove / drop
    assert all(feasible(n, S, frozen_guard) for n in ns)
    assert all(len(set(j for _, j in n)) == len(n) for n in ns)


def test_swapped_advantage_exact_negative():
    model = AdvantageModel(state_dim=2, edge_dim=2, hidden=16)
    a = canonical([(1, 1)])
    b = canonical([(2, 2)])
    assert torch.allclose(model(S, a, b), -model(S, b, a), atol=1e-7)
    assert model(S, a, a).item() == 0.0


def test_guard_cannot_be_overridden_by_model():
    class FakeModel(torch.nn.Module):
        def forward(self, s, alt, ref):
            return torch.tensor(float(len(alt)) * 100.0)

    choice = improve(S, canonical([(1, 1)]), frozen_guard, FakeModel(), depth=2)
    assert feasible(choice.selected, S, frozen_guard)
    assert (3, 1) not in choice.selected or len(choice.selected) == 1


def test_reference_fallback_when_no_benefit():
    class AntiChange(torch.nn.Module):
        def forward(self, s, alt, ref):
            return torch.tensor(-1.0)

    b = canonical([(1, 1)])
    choice = improve(S, b, frozen_guard, AntiChange())
    assert choice.selected == b and choice.predicted_gain == 0.0


def test_learning_targets_are_paired_realized_differences():
    seen = []

    def replay(s, m, key, continuation):
        seen.append((m, key, continuation))
        return (12.0 if (1, 2) in m else 5.0, 0.01)

    a = canonical([(1, 2)])
    b = canonical([(1, 1)])
    p = paired_outcome(S, a, b, frozen_guard, replay, 42)
    assert p.advantage == 7.0
    assert len(seen) == 2 and seen[0][1:] == seen[1][1:]
    assert p.alt_violation == p.ref_violation == 0.01


def test_no_infeasible_counterfactuals():
    def replay(*args):
        return 0.0, 0.0

    with pytest.raises(ValueError):
        paired_outcome(S, canonical([(3, 1), (2, 2)]), (), frozen_guard, replay, 3)


def test_pair_training_improves_finite_loss():
    torch.manual_seed(0)
    model = AdvantageModel(state_dim=2, edge_dim=2, hidden=16)
    a = canonical([(1, 2)])
    b = canonical([(1, 1)])
    ex = [PairExample(S, a, b, 7.0, 0.01, 0.01)]
    before = abs(model(S, a, b).item() - 7.0)
    train_pairs(model, ex, epochs=60, lr=0.01)
    after = abs(model(S, a, b).item() - 7.0)
    assert after < before


def test_sc_rlgmi_policy_execution():
    from controllers.rl.sc_rlgmi.sc_rlgmi_policy import SCRLGMIPolicy

    policy = SCRLGMIPolicy(b_max=175000, r_thresh=0.20, delta=0.0)

    # Simulated candidate dictionaries matching WiLab bridge format
    candidates = [
        {
            "task_id": 1,
            "source_id": 10,
            "helper_id": 20,
            "payload_bytes": 25000,
            "packet_count": 1,
            "task_value": 4.5,
            "finish_ms": 25.0,
            "deadline_ms": 50.0,
            "t_contact_sec": 5.0,
            "features": [0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            "_dist": 50.0,
        },
        {
            "task_id": 2,
            "source_id": 11,
            "helper_id": 21,
            "payload_bytes": 35000,
            "packet_count": 1,
            "task_value": 6.0,
            "finish_ms": 30.0,
            "deadline_ms": 60.0,
            "t_contact_sec": 4.0,
            "features": [0.0, 1.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            "_dist": 60.0,
        },
    ]

    class MockLayer:
        current_epoch = 10
        controller_epochs = 100

    pack, meta = policy.select_matching(
        candidates, b_max=175000, unique_sources=True, unique_helpers=True, layer=MockLayer()
    )

    assert isinstance(pack, list)
    assert len(pack) >= 1
    assert "policy_mode" in meta
    assert meta["policy_mode"] == "sc_rlgmi"
    assert "selected_mode" in meta
    assert meta["bytes"] <= 175000

