"""Pathwise C2 metrics for PB-CRL v3.1 (HQ-SPO / WiLab audits).

C2: P(Z > K) ≤ δ where Z = # violated valid safety windows in an episode.
This is NOT the same as mean P_viol ≤ α; both are reported.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


PRR_FLOOR = 0.95
P95_CEILING = 50.0
DEFAULT_ALPHA = 0.02
# Fixed pathwise budget for ~100-epoch / 12s WiLab runs (calibrate later).
DEFAULT_K = 2
DEFAULT_DELTA = 0.05


def window_violated(epoch: dict, *, prr_floor=PRR_FLOOR, p95_ceiling=P95_CEILING) -> bool:
    if not (epoch.get("cbr_valid") and epoch.get("delay_valid")):
        return False
    prr = epoch.get("prr")
    p95 = epoch.get("p95_delay_ms")
    if prr is not None and float(prr) < prr_floor:
        return True
    if p95 is not None and float(p95) > p95_ceiling:
        return True
    return False


def episode_c2_from_audit(
    audit: Sequence[dict],
    *,
    K: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Compute Z, rate, and C2 feasibility flags from a bridge_audit list."""
    n = len(audit)
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    n_valid = len(valid)
    z_flags = [1.0 if window_violated(e) else 0.0 for e in valid]
    Z = int(sum(z_flags))
    p_viol = (Z / n_valid) if n_valid else None
    k_rate = int(max(0, math.floor(float(alpha) * n_valid))) if n_valid else 0
    return {
        "n_epochs": n,
        "n_valid": n_valid,
        "Z": Z,
        "K": int(K),
        "K_from_alpha_nvalid": k_rate,
        "C2_ok": (Z <= int(K)) if n_valid else False,
        "P_viol": p_viol,
        "z_flags": z_flags,
    }


def clopper_pearson_upper(k: int, n: int, *, conf: float = 0.95) -> Optional[float]:
    """One-sided Clopper–Pearson upper bound on Bernoulli p (k failures in n trials)."""
    if n <= 0:
        return None
    # Beta quantile: P(p <= u) = conf for Bin(n,p) observing k → use scipy if present
    try:
        from scipy.stats import beta
        if k >= n:
            return 1.0
        # upper (1-alpha) bound: Beta(k+1, n-k).ppf(conf) for two-sided differs;
        # one-sided upper: Beta.ppf(conf; k+1, n-k)
        return float(beta.ppf(conf, k + 1, n - k))
    except Exception:
        # Conservative fallback: rule-of-three when k=0
        if k == 0:
            return float(1.0 - (1.0 - conf) ** (1.0 / n))
        # Wilson-ish rough upper
        phat = k / n
        z = 1.96 if conf >= 0.95 else 1.64
        denom = 1 + z * z / n
        center = phat + z * z / (2 * n)
        rad = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
        return float(min(1.0, (center + rad) / denom))


def c2_cell_summary(
    episode_rows: List[Dict[str, Any]],
    *,
    K: int = DEFAULT_K,
    delta: float = DEFAULT_DELTA,
    conf: float = 0.95,
) -> Dict[str, Any]:
    """Across episodes in one scenario cell: fraction with Z>K and CP upper bound."""
    n = len(episode_rows)
    exceed = sum(1 for r in episode_rows if int(r.get("Z", 0)) > int(K))
    ub = clopper_pearson_upper(exceed, n, conf=conf)
    mean_u = None
    us = [float(r["U"]) for r in episode_rows if r.get("U") is not None]
    if us:
        mean_u = sum(us) / len(us)
    return {
        "n_episodes": n,
        "n_Z_gt_K": exceed,
        "frac_Z_gt_K": exceed / n if n else None,
        "cp_upper_P_Z_gt_K": ub,
        "feasible_cell": (ub is not None and ub <= float(delta)),
        "delta": float(delta),
        "K": int(K),
        "mean_U": mean_u,
    }
