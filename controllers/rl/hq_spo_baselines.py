"""HQ-SPO official baselines 1–4 (user suite)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CE_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "sparc_performance_evaluation"
    / "results"
    / "raw"
    / "hq_spo_ce_risk"
    / "ce_weights.json"
)


def best_helper_edges(candidates: List[dict]) -> List[dict]:
    """E_t: one edge per task — helper with min finish_ms among edge_ok candidates."""
    best: Dict[int, dict] = {}
    for c in candidates:
        tid = int(c["task_id"])
        fin = float(c.get("finish_ms") or 1e18)
        prev = best.get(tid)
        if prev is None or fin < float(prev.get("finish_ms") or 1e18):
            best[tid] = c
    return list(best.values())


def greedy_value_density(
    candidates: List[dict],
    *,
    b_max: int,
    unique_sources: bool = True,
) -> List[dict]:
    """Baseline 1: helper-aware sort by v_i/b_i descending; pack under byte, source, and helper queue constraints."""
    from controllers.rl.pb_crl_packer import price_budget_pack
    return price_budget_pack(candidates, lam=0.0, beta_exec=1.0, b_max=b_max, unique_sources=unique_sources)


def cbr_effective_bytes(
    cbr: float,
    resel: float,
    *,
    b_max: int = 175000,
    th_lo: float = 0.3,
    th_hi: float = 0.8,
    th_resel: float = 0.5,
) -> int:
    """Baseline 2 byte cap from CBR (+ reselection kill switch)."""
    if float(resel) > float(th_resel):
        return 0
    denom = max(float(th_hi) - float(th_lo), 1e-9)
    scale = (float(th_hi) - float(cbr)) / denom
    scale = max(0.0, min(1.0, scale))
    return int(max(0, min(int(b_max), round(int(b_max) * scale))))


def myopic_dual_knapsack(
    candidates: List[dict],
    *,
    b_max: int,
    lam: float,
    bin_bytes: int = 1000,
) -> List[dict]:
    """Baseline 3: helper-aware multiple-choice dual knapsack.
    
    Edge utility u_i = v_i - lam * (b_i / B_max).
    """
    from controllers.rl.pb_crl_packer import price_budget_pack
    return price_budget_pack(candidates, lam=float(lam), beta_exec=1.0, b_max=b_max, unique_sources=True)


class MyopicDualBaseline:
    """Stateful λ for baseline 3 dual ascent on violation rate."""

    def __init__(self, lam: float = 0.0, eta: float = 0.05, alpha: float = 0.02):
        self.lam = float(lam)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.is_myopic_dual = True

    def update_dual(self, viol_rate_window: float):
        self.lam = float(max(0.0, self.lam + self.eta * (float(viol_rate_window) - self.alpha)))

    def note_window(self, epoch: dict):
        from controllers.rl.pb_crl_c2 import window_violated
        viol = 1.0 if window_violated(epoch) else 0.0
        self.update_dual(viol)

    def value(self, obs):
        import torch
        return torch.tensor(0.0)

    def cost_value(self, obs):
        import torch
        return torch.tensor(0.0)


@dataclass
class CeRiskWeights:
    """Baseline 4: P(viol) = sigmoid(w0 + w1*CBR + w2*resel + w3*bytes).

    Defaults keep empty-pack risk well below alpha=0.02 so B_eff is not
    vacuously zero (naive fits on mixed logs often push w0 too high).
    """

    w0: float = -5.0
    w1: float = 4.0
    w2: float = 2.0
    w3: float = 2e-5  # per byte
    alpha: float = 0.02

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Optional[Path] = None) -> "CeRiskWeights":
        path = Path(path) if path is not None else DEFAULT_CE_WEIGHTS_PATH
        if path.is_file():
            d = json.loads(path.read_text(encoding="utf-8"))
            return CeRiskWeights(**{k: d[k] for k in ("w0", "w1", "w2", "w3", "alpha") if k in d})
        return CeRiskWeights()


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def ce_byte_cap(
    cbr: float,
    resel: float,
    weights: CeRiskWeights,
    *,
    b_max: int = 175000,
) -> int:
    """Certainty-equivalent byte cut from logistic risk model.

    Model uses bytes_norm = bytes / B_max, so w3 is per full-envelope.
    """
    w = weights
    if abs(float(w.w3)) < 1e-15:
        return int(b_max)
    # sigmoid(w0+w1*cbr+w2*resel+w3*(B/Bmax)) <= alpha
    # => B/Bmax <= (logit(a)-w0-w1*cbr-w2*resel)/w3
    numer = logit(w.alpha) - float(w.w0) - float(w.w1) * float(cbr) - float(w.w2) * float(resel)
    if w.w3 < 0:
        return int(b_max)
    frac = numer / float(w.w3)
    return int(max(0, min(int(b_max), math.floor(frac * float(b_max)))))


def fit_ce_logistic(
    rows: List[Dict[str, float]],
    *,
    alpha: float = 0.02,
    b_max: int = 175000,
    steps: int = 600,
    lr: float = 0.1,
) -> CeRiskWeights:
    """GD on Bernoulli NLL with bytes/B_max feature; project usable CE cut."""
    if not rows:
        return CeRiskWeights(alpha=alpha)
    # w3 on normalized bytes in [0,1]
    w = [-4.0, 3.0, 1.0, 2.0]
    bm = max(float(b_max), 1.0)
    cbrs = sorted(float(r["cbr"]) for r in rows)
    med_cbr = cbrs[len(cbrs) // 2] if cbrs else 0.2
    for _ in range(int(steps)):
        g = [0.0, 0.0, 0.0, 0.0]
        for r in rows:
            x = [1.0, float(r["cbr"]), float(r["resel"]), float(r["bytes"]) / bm]
            z = sum(wi * xi for wi, xi in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            err = p - float(r["viol"])
            for j in range(4):
                g[j] += err * x[j]
        n = max(len(rows), 1)
        for j in range(4):
            w[j] -= lr * g[j] / n
        w[3] = max(w[3], 0.5)
    # Hard project: empty-pack risk at med CBR must be < alpha/2
    for _ in range(200):
        z0 = w[0] + w[1] * med_cbr
        p0 = 1.0 / (1.0 + math.exp(-max(min(z0, 30.0), -30.0)))
        if p0 <= alpha * 0.5:
            break
        w[0] -= 0.1
    # Ensure some positive Beff at med CBR
    for _ in range(200):
        be = ce_byte_cap(med_cbr, 0.0, CeRiskWeights(w[0], w[1], w[2], w[3], alpha), b_max=b_max)
        if be >= int(0.1 * b_max):
            break
        w[0] -= 0.1
    return CeRiskWeights(w0=w[0], w1=w[1], w2=w[2], w3=w[3], alpha=alpha)


def build_baseline_pack(
    candidates: List[dict],
    state: dict,
    *,
    mode: str,
    b_max: int = 175000,
    unique_sources: bool = True,
    th_lo: float = 0.3,
    th_hi: float = 0.8,
    th_resel: float = 0.5,
    lam: float = 0.0,
    ce_weights: Optional[CeRiskWeights] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """mode: greedy_vb | cbr_threshold | myopic_dual | ce_milp"""
    cbr = float(state.get("cbr") or 0.0)
    # resel_t: use instrument if present; else 0 (no false kill). Instrument later.
    resel = float(state.get("resel_t") or state.get("reselection_rate") or 0.0)
    meta: Dict[str, Any] = {"mode": mode, "cbr": cbr, "resel_t": resel}
    if mode == "cbr_threshold":
        b_eff = cbr_effective_bytes(
            cbr, resel, b_max=b_max, th_lo=th_lo, th_hi=th_hi, th_resel=th_resel,
        )
        meta.update({"b_eff": b_eff, "th_lo": th_lo, "th_hi": th_hi, "th_resel": th_resel})
        pack = greedy_value_density(candidates, b_max=b_eff, unique_sources=unique_sources)
    elif mode == "greedy_vb":
        meta["b_eff"] = int(b_max)
        pack = greedy_value_density(candidates, b_max=b_max, unique_sources=unique_sources)
    elif mode == "myopic_dual":
        meta["lam"] = float(lam)
        meta["b_eff"] = int(b_max)
        pack = myopic_dual_knapsack(candidates, b_max=b_max, lam=float(lam))
    elif mode == "ce_milp":
        w = ce_weights if ce_weights is not None else CeRiskWeights.load()
        b_eff = ce_byte_cap(cbr, resel, w, b_max=b_max)
        meta.update({"b_eff": b_eff, "ce": asdict(w)})
        # CE solve: knapsack with lam=0 under B_eff
        pack = myopic_dual_knapsack(candidates, b_max=b_eff, lam=0.0)
    else:
        raise ValueError(f"unknown baseline mode {mode}")
    meta["n_pack"] = len(pack)
    meta["bytes"] = int(sum(int(a.get("payload_bytes") or 0) for a in pack))
    return pack, meta


def assignments_to_action_fields(pack: List[dict], layer, budget: int, safe_actions: list) -> dict:
    packet_counts: Dict[int, int] = {}
    payload_bytes: Dict[int, int] = {}
    for a in pack:
        sid = int(a["source_id"])
        packet_counts[sid] = packet_counts.get(sid, 0) + int(a["packet_count"])
        payload_bytes[sid] = payload_bytes.get(sid, 0) + int(a["payload_bytes"])
    return {
        "budget": budget,
        "K_t": len(safe_actions),
        "safe_action_set": safe_actions,
        "assignments": pack,
        "allowed_sources": [int(a["source_id"]) for a in pack],
        "source_packet_ids": list(packet_counts.keys()),
        "source_packet_counts": list(packet_counts.values()),
        "source_payload_bytes": list(payload_bytes.values()),
        "computation_packet_bytes": int(layer.computation_packet_bytes),
        "computation_packets_offered": int(sum(packet_counts.values())),
        "computation_bytes_offered": int(sum(payload_bytes.values())),
        "n_candidates": 0,
    }
