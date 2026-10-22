from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

raw = Path("sparc_performance_evaluation/results/raw")
KS = [8, 16, 32, 64]
METHODS = [
    ("VAS", "catchup_fig10d_k{k}_va", "value_aware"),
    ("LAS", "catchup_fig10d_k{k}_mf", "minfinish"),
    ("LVM", "catchup_fig10d_k{k}_b4", "residual"),
    ("VLRO", "catchup_fig10d_k{k}_itora", "itora_inspired"),
    ("SPARC-V2X", "catchup_fig10d_k{k}_direct", "residual"),
]


def prr_of(r: dict) -> float | None:
    for k in ("safety_mean_prr_valid", "safety_PRR_cum", "mean_prr", "prr"):
        if r.get(k) is not None:
            return float(r[k])
    return None


def agg(folder: str, meth: str):
    rows = json.loads((raw / folder / "rows.json").read_text(encoding="utf-8-sig"))
    sel = [r for r in rows if r.get("method") == meth]
    R = mean(float(r["R_value"]) for r in sel)
    P = mean(float(r.get("safety_P_viol") or 0) for r in sel)
    prrs = [x for x in (prr_of(r) for r in sel) if x is not None]
    PRR = mean(prrs) if prrs else float("nan")
    return R, P, PRR


for lab, tmpl, meth in METHODS:
    print(f"=== {lab} ===")
    print(f"{'|C_t|':>6s} {'R':>8s} {'P_viol':>8s} {'PRR':>8s}")
    for k in KS:
        R, P, PRR = agg(tmpl.format(k=k), meth)
        print(f"{k:6d} {R:8.4f} {P:8.4f} {PRR:8.4f}")
    print()
