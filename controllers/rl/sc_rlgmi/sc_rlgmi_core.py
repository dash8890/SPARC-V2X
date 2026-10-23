"""RL-guided improvement of complete, safety-feasible V2X matchings (SC-RLGMI Core).

Integration boundary: caller supplies *current* observable state, feasible-edge oracle,
physical matching validator (including unchanged causal guard), a baseline matching,
and matched-continuation replay when training. Nothing here alters the simulator.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence
import random
import torch
from torch import nn

Edge = tuple[int, int]  # (task ID, helper ID)
Matching = tuple[Edge, ...]


def canonical(edges: Iterable[Edge]) -> Matching:
    out = tuple(sorted(set(edges)))
    assert len({k for k, _ in out}) == len(out), "task assigned twice"
    assert len({j for _, j in out}) == len(out), "helper assigned twice this epoch"
    return out


@dataclass(frozen=True)
class DecisionState:
    """All feature vectors must contain information observable *at decision time*."""
    features: tuple[float, ...]
    # Map exact (task,helper) edges to causal feature vectors, aligned with the model.
    edge_features: dict[Edge, tuple[float, ...]]


class Feasibility(Protocol):
    def __call__(self, state: DecisionState, matching: Matching) -> bool:
        """Use existing guard, byte budget, radio eligibility and task constraints."""


def feasible(m: Matching, state: DecisionState, check: Feasibility) -> bool:
    if len(set(k for k, _ in m)) != len(m):
        return False
    if len(set(j for _, j in m)) != len(m):
        return False
    return all(edge in state.edge_features for edge in m) and bool(check(state, m))


def neighbors(state: DecisionState, incumbent: Matching, check: Feasibility) -> list[Matching]:
    """Insert, delete, reassign, exchange and replace complete matchings.

    All feasibility is checked against the FULL matching, not inferred from edge masks.
    Swapping two assigned helpers is included; single replacement handles task swaps.
    """
    candidates: set[Matching] = set()
    incumbent_set = set(incumbent)
    for edge in state.edge_features:
        try:
            candidates.add(canonical([*incumbent, edge]))  # insertion (Add)
        except AssertionError:
            pass
        for old in incumbent:  # replace task, or reassign helper
            try:
                candidates.add(canonical((incumbent_set - {old}) | {edge}))
            except AssertionError:
                pass
    for old in incumbent:
        candidates.add(canonical(incumbent_set - {old}))  # defer/remove (Drop)
    for i in range(len(incumbent)):
        for j in range(i + 1, len(incumbent)):
            (k1, h1), (k2, h2) = incumbent[i], incumbent[j]
            candidates.add(canonical((incumbent_set - {(k1, h1), (k2, h2)}) | {(k1, h2), (k2, h1)}))  # Exchange
    return sorted(m for m in candidates if m != incumbent and feasible(m, state, check))


class MatchingEncoder(nn.Module):
    def __init__(self, state_dim: int, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2 * hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.hidden = hidden

    def forward(self, state: DecisionState, matching: Matching) -> torch.Tensor:
        device = next(self.parameters()).device
        s = torch.tensor(state.features, dtype=torch.float32, device=device)
        if matching:
            x = torch.tensor([state.edge_features[e] for e in matching], dtype=torch.float32, device=device)
            z = self.edge(x)
            agg = torch.cat((z.sum(dim=0), z.max(dim=0).values))
        else:
            agg = torch.zeros(2 * self.hidden, device=device)
        return self.net(torch.cat((s, agg, torch.tensor([float(len(matching))], device=device))))[0]


class AdvantageModel(nn.Module):
    """Shared-value difference enforces A(s,alt,ref) = -A(s,ref,alt).

    Pairwise training labels are matched differences in REAL completed value;
    this is not independently supervised with assignment or admission value.
    """
    def __init__(self, state_dim: int, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.value = MatchingEncoder(state_dim, edge_dim, hidden)

    def forward(self, s: DecisionState, alt: Matching, ref: Matching) -> torch.Tensor:
        return self.value(s, alt) - self.value(s, ref)


@dataclass(frozen=True)
class Selection:
    reference: Matching
    selected: Matching
    predicted_gain: float
    searched: int


def improve(
    state: DecisionState,
    baseline: Matching,
    check: Feasibility,
    model: AdvantageModel,
    *,
    depth: int = 2,
    beam: int = 8,
    min_predicted_gain: float = 0.0,
) -> Selection:
    """RL-guided BEAM SEARCH, not a post-hoc filter of one greedy alternative.

    Allows negative intermediate steps along a search path; chooses the best
    complete matching only if predicted gain exceeds min_predicted_gain.
    This margin is NOT a safety or performance guarantee.
    """
    baseline = canonical(baseline)
    if not feasible(baseline, state, check):
        raise ValueError("Baseline matching fails frozen feasibility oracle")
    if depth < 1 or beam < 1:
        raise ValueError("depth and beam must be positive")
    visited: set[Matching] = {baseline}
    frontier = [baseline]
    best, best_gain = baseline, 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(depth):
            expanded: list[tuple[float, Matching]] = []
            for m in frontier:
                for c in neighbors(state, m, check):
                    if c in visited:
                        continue
                    visited.add(c)
                    gain = float(model(state, c, baseline).item())
                    expanded.append((gain, c))
                    if gain > best_gain:
                        best, best_gain = c, gain
            if not expanded:
                break
            expanded.sort(key=lambda x: (-x[0], x[1]))
            frontier = [m for _, m in expanded[:beam]]
    model.train(was_training)
    chosen = best if best_gain > min_predicted_gain else baseline
    return Selection(baseline, chosen, best_gain if chosen != baseline else 0.0, len(visited))


class MatchedReplay(Protocol):
    def __call__(
        self,
        state: DecisionState,
        matching: Matching,
        exogenous_key: int,
        continuation_policy: str,
    ) -> tuple[float, float]:
        """Return (sum of REAL deadline-valid completed value, absolute safety violation rate).
        Caller must restore exactly the same predecision snapshot, random draws,
        future exogenous arrivals and continuation policy for both branches.
        """


@dataclass(frozen=True)
class PairExample:
    state: DecisionState
    alternative: Matching
    reference: Matching
    advantage: float
    alt_violation: float
    ref_violation: float


def paired_outcome(
    state: DecisionState,
    alt: Matching,
    ref: Matching,
    check: Feasibility,
    replay: MatchedReplay,
    exogenous_key: int,
    continuation_policy: str = "reference",
) -> PairExample:
    """Call only on TRAINING snapshots. Safety rates are RECORDED, not asserted safe."""
    if not feasible(alt, state, check) or not feasible(ref, state, check):
        raise ValueError("Attempted to replay an infeasible matching")
    v_ref, p_ref = replay(state, ref, exogenous_key, continuation_policy)
    v_alt, p_alt = replay(state, alt, exogenous_key, continuation_policy)
    return PairExample(state, alt, ref, float(v_alt - v_ref), float(p_alt), float(p_ref))


def train_pairs(
    model: AdvantageModel,
    examples: Sequence[PairExample],
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 10,
) -> list[float]:
    if not examples:
        raise ValueError("Need paired training outcomes")
    random.seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        indices = list(range(len(examples)))
        random.shuffle(indices)
        for index in indices:
            item = examples[index]
            pred = model(item.state, item.alternative, item.reference)
            target = torch.tensor(item.advantage, dtype=torch.float32, device=pred.device)
            loss = nn.functional.smooth_l1_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return losses
