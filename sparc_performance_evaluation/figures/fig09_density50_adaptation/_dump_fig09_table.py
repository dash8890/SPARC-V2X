from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

raw = Path("sparc_performance_evaluation/results/raw")
sources = {
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
    prrs = [prr_of(r) for r in sel]
    prrs = [x for x in prrs if x is not None]
    PRR = mean(prrs) if prrs else float("nan")
    return R, P, PRR, len(sel)


rows_out = []
for dens in (35, 50, 100):
    print(f"=== dens {dens} ===")
    print(f"{'method':8s} {'R':>8s} {'P_viol':>8s} {'PRR':>8s}")
    for lab, (folder, meth) in sources[dens].items():
        R, P, PRR, n = agg(folder, meth)
        print(f"{lab:8s} {R:8.4f} {P:8.4f} {PRR:8.4f}")
        rows_out.append(
            {"density": dens, "method": lab, "R_value": R, "P_viol": P, "PRR": PRR, "n": n, "source": folder}
        )

Path("sparc_performance_evaluation/figures/fig09_density50_adaptation/fig09_baselines_table.json").write_text(
    json.dumps(rows_out, indent=2) + "\n", encoding="utf-8"
)
