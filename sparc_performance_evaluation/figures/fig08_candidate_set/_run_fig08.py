"""Fig.8 — Candidate-set sensitivity: two versions (SPARC-only + all baselines)."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS, LW = 50, 8
LW_SPARC = LW + 2
OUT = Path("sparc_performance_evaluation/figures/fig08_candidate_set")
RAW = Path("sparc_performance_evaluation/results/raw")
OUT.mkdir(parents=True, exist_ok=True)

KS = [8, 16, 32, 64]
METHODS = [
    ("VAS", "catchup_fig10d_k{k}_va", "value_aware", "#5b8db8", "^", (0, (6, 3)), False),
    ("LAS", "catchup_fig10d_k{k}_mf", "minfinish", "#7eb8a8", "v", (0, (2, 2)), False),
    ("LVM", "catchup_fig10d_k{k}_b4", "residual", "#e07a3d", "o", (0, (10, 3, 2, 3)), False),
    ("VLRO", "catchup_fig10d_k{k}_itora", "itora_inspired", "#3d8c5a", "s", (0, (1, 2)), False),
    ("SPARC-V2X", "catchup_fig10d_k{k}_direct", "residual", "#1f4e79", "D", "solid", True),
]


def mean_R(folder: str, method: str) -> float | None:
    p = RAW / folder / "rows.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8-sig"))
    xs = [
        float(r["R_value"])
        for r in rows
        if r.get("method") == method and r.get("R_value") is not None
    ]
    return mean(xs) if xs else None


series: dict[str, list[float]] = {}
missing: list[str] = []
for lab, folder_tmpl, meth, *_ in METHODS:
    ys = []
    for k in KS:
        val = mean_R(folder_tmpl.format(k=k), meth)
        if val is None:
            missing.append(f"{lab}@k={k}")
            break
        ys.append(val)
    else:
        series[lab] = ys

(OUT / "fig08_data.json").write_text(
    json.dumps(
        {
            "figure": "Fig.8",
            "k": KS,
            "R_value": series,
            "missing": missing,
            "note": "No inference-latency / runtime claims (latency not logged).",
            "versions": ["fig08_R_vs_k_sparc", "fig08_R_vs_k_all"],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
if missing:
    raise SystemExit(f"MISSING data: {missing}")


def plot_lines(labels: list[str], stem: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(20, 18))
    for lab, _folder, _meth, col, marker, ls, is_sparc in METHODS:
        if lab not in labels or lab not in series:
            continue
        ax.plot(
            KS,
            series[lab],
            color=col,
            linewidth=LW_SPARC if is_sparc else LW,
            linestyle=ls,
            marker=marker,
            markersize=18 if is_sparc else 16,
            label=lab,
            zorder=5 if is_sparc else 2,
        )
    ax.set_title(title, fontsize=FS)
    ax.set_xlabel(r"$|\mathcal{C}_t|$", fontsize=FS)
    ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
    ax.set_xticks(KS)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)


def plot_sparc_bars(stem: str, title: str) -> None:
    ys = series["SPARC-V2X"]
    # One light named color + hatch per |C_t| bar
    bar_colors = ["azure", "ivory", "honeydew", "seashell"]
    bar_hatches = ["/", "\\", "*", "+"]
    fig, ax = plt.subplots(figsize=(20, 18))
    x = list(range(len(KS)))
    for i, (xi, yi, k) in enumerate(zip(x, ys, KS)):
        ax.bar(
            xi,
            yi,
            color=bar_colors[i],
            hatch=bar_hatches[i],
            width=0.55,
            edgecolor="black",
            linewidth=2,
            zorder=3,
            label=rf"$|\mathcal{{C}}_t|={k}$",
        )
    ax.set_title(title, fontsize=FS)
    ax.set_xlabel(r"$|\mathcal{C}_t|$", fontsize=FS)
    ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
    ax.set_xticks([])
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ymin = min(ys) - 0.05
    ax.set_ylim(max(0.0, ymin), max(ys) * 1.08)
    fig.tight_layout()
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)


plot_sparc_bars(
    "fig08_R_vs_k_sparc",
    r"$R_{\mathrm{value}}$ vs $|\mathcal{C}_t|$",
)
plot_lines(
    ["VAS", "LAS", "LVM", "VLRO", "SPARC-V2X"],
    "fig08_R_vs_k_all",
    r"$R_{\mathrm{value}}$ vs $|\mathcal{C}_t|$",
)
# Keep legacy filename as the all-baselines line version
plot_lines(
    ["VAS", "LAS", "LVM", "VLRO", "SPARC-V2X"],
    "fig08_R_vs_k",
    r"$R_{\mathrm{value}}$ vs $|\mathcal{C}_t|$",
)
print("series", {k: [round(v, 4) for v in vs] for k, vs in series.items()})
