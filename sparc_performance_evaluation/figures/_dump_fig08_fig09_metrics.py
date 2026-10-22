"""Dump Fig.8 / Fig.9 metrics including P95 for canvas tables."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

raw = Path("sparc_performance_evaluation/results/raw")
OUT = Path("sparc_performance_evaluation/figures")


def prr_of(r: dict) -> float | None:
    for k in ("safety_mean_prr_valid", "safety_PRR_cum", "mean_prr", "prr"):
        if r.get(k) is not None:
            return float(r[k])
    return None


def p95_of(r: dict) -> float | None:
    for k in ("safety_mean_p95_valid", "safety_P95_cum", "mean_p95", "p95"):
        if r.get(k) is not None:
            return float(r[k])
    return None


def agg(folder: str, meth: str) -> dict:
    rows = json.loads((raw / folder / "rows.json").read_text(encoding="utf-8-sig"))
    sel = [r for r in rows if r.get("method") == meth]
    prrs = [x for x in (prr_of(r) for r in sel) if x is not None]
    p95s = [x for x in (p95_of(r) for r in sel) if x is not None]
    return {
        "R": mean(float(r["R_value"]) for r in sel),
        "P_viol": mean(float(r.get("safety_P_viol") or 0) for r in sel),
        "PRR": mean(prrs) if prrs else float("nan"),
        "P95": mean(p95s) if p95s else float("nan"),
        "n": len(sel),
        "source": folder,
    }


# --- Fig.8 ---
fig8_methods = [
    ("VAS", "catchup_fig10d_k{k}_va", "value_aware"),
    ("LAS", "catchup_fig10d_k{k}_mf", "minfinish"),
    ("LVM", "catchup_fig10d_k{k}_b4", "residual"),
    ("VLRO", "catchup_fig10d_k{k}_itora", "itora_inspired"),
    ("SPARC-V2X", "catchup_fig10d_k{k}_direct", "residual"),
]
fig8 = []
for lab, tmpl, meth in fig8_methods:
    for k in (8, 16, 32, 64):
        st = agg(tmpl.format(k=k), meth)
        fig8.append({"method": lab, "C_t": k, **st})

# --- Fig.9 ---
fig9_sources = {
    35: {
        "VAS": ("catchup_gov_on_d35_va_mf", "value_aware"),
        "LAS": ("catchup_gov_on_d35_va_mf", "minfinish"),
        "LVM": ("catchup_b4_density_d35", "residual"),
        "VLRO": ("catchup_envelope_itora_b175000", "itora_inspired"),
        "SPARC-N": ("catchup_gov_on_d35_direct", "residual"),
        "SPARC-S": ("catchup_gov_on_d35_direct", "residual"),
    },
    50: {
        "VAS": ("density_d50", "value_aware"),
        "LAS": ("density_d50", "minfinish"),
        "LVM": ("catchup_b4_density_d50", "residual"),
        "VLRO": ("density_d50", "itora_inspired"),
        "SPARC-N": ("catchup_fig11a_nominal_direct_on_d50", "residual"),
        "SPARC-S": ("catchup_fig11a_direct_d50_ft", "residual"),
    },
    100: {
        "VAS": ("density_d100", "value_aware"),
        "LAS": ("density_d100", "minfinish"),
        "LVM": ("catchup_b4_density_d100", "residual"),
        "VLRO": ("density_d100", "itora_inspired"),
        "SPARC-N": ("catchup_fig11a_nominal_direct_on_d100", "residual"),
        "SPARC-S": ("catchup_fig11a_direct_d100_ft", "residual"),
    },
}
fig9 = []
for dens, methods in fig9_sources.items():
    for lab, (folder, meth) in methods.items():
        st = agg(folder, meth)
        fig9.append({"density": dens, "method": lab, **st})

payload = {"fig8": fig8, "fig9": fig9}
(OUT / "fig08_fig09_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

# compact print for canvas regen
print("FIG8")
for row in fig8:
    print(
        f"{row['method']:10s} {row['C_t']:3d} "
        f"R={row['R']:.4f} P={row['P_viol']:.4f} PRR={row['PRR']:.4f} P95={row['P95']:.2f}"
    )
print("FIG9")
for row in fig9:
    print(
        f"d{row['density']:3d} {row['method']:8s} "
        f"R={row['R']:.4f} P={row['P_viol']:.4f} PRR={row['PRR']:.4f} P95={row['P95']:.2f}"
    )
