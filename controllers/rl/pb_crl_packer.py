"""Shared price-and-budget packer for PB-CRL v3.1 / B5′.

Deterministic given (lambda, beta_exec, optional delta_i).

Score matches suite B3 units (not raw v/b − λ):
  score_i = v_i - λ * (b_i / B_max) [+ delta_i]
v/b − λ is broken when λ~O(1) and v/b~O(1e-2): init λ≈0.09 → empty packs → V≡0.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from controllers.rl.hq_spo_baselines import best_helper_edges


def price_budget_pack(
    candidates: List[dict],
    *,
    lam: float,
    beta_exec: float,
    b_max: int = 175000,
    unique_sources: bool = True,
    delta_by_task: Optional[Dict[int, float]] = None,
) -> List[dict]:
    """Shared helper-aware packer (PB-CRL core / B5′).
    
    Evaluates all (task, helper) candidate edges jointly, dynamically balancing
    helper compute queues and enforcing per-helper deadline feasibility.
    """
    cap = int(max(0, min(int(b_max), round(float(beta_exec) * int(b_max)))))
    bmax_f = max(float(b_max), 1.0)
    scored: List[tuple] = []
    for c in candidates:
        tid = int(c["task_id"])
        b = float(c.get("payload_bytes") or 0.0)
        v = float(c.get("task_value") or 0.0)
        fin = float(c.get("finish_ms") or 1e18)
        if b <= 0:
            continue
        dlt = 0.0
        if delta_by_task is not None:
            dlt = float(delta_by_task.get(tid, 0.0))
        score = v - float(lam) * (b / bmax_f) + dlt
        if score <= 0.0:
            continue
        surplus_dens = score / b
        scored.append((surplus_dens, score, fin, c))
    # Sort by surplus density descending, then score descending, then finish_ms ascending (faster helper first)
    scored.sort(key=lambda t: (-t[0], -t[1], t[2], int(t[3]["task_id"])))

    pack: List[dict] = []
    used_bytes = 0
    used_sources = set()
    used_tasks = set()
    helper_queued_ms: Dict[int, float] = {}

    for surplus_dens, score, fin, c in scored:
        tid = int(c["task_id"])
        sid = int(c["source_id"])
        hid = int(c["helper_id"])
        b = int(c.get("payload_bytes") or 0)
        dl = float(c.get("deadline_ms") or 1e18)

        if tid in used_tasks:
            continue
        if unique_sources and sid in used_sources:
            continue
        if used_bytes + b > cap:
            continue

        accum = helper_queued_ms.get(hid, 0.0)
        if fin + accum > dl:
            continue

        feats = c.get("features")
        if feats and len(feats) >= 5:
            cycles = float(feats[1]) * 1e8
            cpu = float(feats[4]) * 1e9
            compute_ms = 1000.0 * cycles / max(cpu, 1e6)
        else:
            compute_ms = max(0.0, fin - 10.0)

        pack.append({
            "task_id": c["task_id"],
            "helper_id": c["helper_id"],
            "source_id": c["source_id"],
            "payload_bytes": c["payload_bytes"],
            "packet_count": c["packet_count"],
        })
        used_bytes += b
        used_tasks.add(tid)
        used_sources.add(sid)
        helper_queued_ms[hid] = accum + compute_ms

    return pack


def pack_meta(pack: List[dict], *, lam: float, beta_exec: float, b_max: int) -> Dict[str, Any]:
    return {
        "mode": "price_budget",
        "lam": float(lam),
        "beta_exec": float(beta_exec),
        "b_eff": int(max(0, min(int(b_max), round(float(beta_exec) * int(b_max))))),
        "n_pack": len(pack),
        "bytes": int(sum(int(a.get("payload_bytes") or 0) for a in pack)),
    }
