"""Fit B4 CE logistic risk model from existing bridge_audit.json logs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.rl.hq_spo_baselines import CeRiskWeights, DEFAULT_CE_WEIGHTS_PATH, fit_ce_logistic

SEARCH_ROOTS = [
    ROOT / "sparc_performance_evaluation" / "results" / "raw" / "hq_spo_b1_b2_smoke" / "runs",
    ROOT / "sparc_performance_evaluation" / "results" / "raw" / "hq_spo_premise" / "runs",
    ROOT / "sparc_performance_evaluation" / "results" / "raw" / "hq_spo_method_eval" / "runs",
]


def rows_from_audit(path: Path):
    audit = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for e in audit:
        if not (e.get("cbr_valid") and e.get("delay_valid")):
            continue
        cbr = float(e.get("cbr") or 0.0)
        resel = float(e.get("resel_t") or 0.0)
        bytes_a = float(e.get("computation_bytes_offered") or 0.0)
        prr = e.get("prr")
        p95 = e.get("p95_delay_ms")
        viol = 0.0
        if prr is not None and float(prr) < 0.95:
            viol = 1.0
        if p95 is not None and float(p95) > 50.0:
            viol = 1.0
        out.append({"cbr": cbr, "resel": resel, "bytes": bytes_a, "viol": viol})
    return out


def main():
    rows = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for p in root.glob("episode_*/bridge_audit.json"):
            rows.extend(rows_from_audit(p))
    print(json.dumps({"n_rows": len(rows), "n_viol": sum(r["viol"] for r in rows)}), flush=True)
    w = fit_ce_logistic(rows, alpha=0.02)
    w.save(DEFAULT_CE_WEIGHTS_PATH)
    print(json.dumps({"weights": w.__dict__, "path": str(DEFAULT_CE_WEIGHTS_PATH)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
