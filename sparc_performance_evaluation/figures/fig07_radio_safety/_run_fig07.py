"""Generate Fig.7(a–c) safety-message performance (no delay-histogram panel)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS, LW = 50, 8
LW_SPARC = LW + 2
OUT = Path("sparc_performance_evaluation/figures/fig07_radio_safety")
DATA = Path("sparc_performance_evaluation/figures/fig07_radio_safety/fig07_data.json")
FALLBACK = Path(
    "sparc_performance_evaluation/results/stage5_no_residual/figures/fig07_data.json"
)
OUT.mkdir(parents=True, exist_ok=True)
d = json.loads((DATA if DATA.exists() else FALLBACK).read_text(encoding="utf-8"))

# Main operating-region densities (clean; matches Fig.4)
DENS = [20, 25, 30, 35, 40, 45, 50]
METHODS = [
    ("VAS", "value_aware", "#5b8db8", "^", (0, (6, 3)), False),
    ("LAS", "minfinish", "#7eb8a8", "v", (0, (2, 2)), False),
    ("LVM", "b4_lagrangian", "#e07a3d", "o", (0, (10, 3, 2, 3)), False),
    ("VLRO", "itora_inspired", "#3d8c5a", "s", (0, (1, 2)), False),
    ("SPARC-V2X", "direct", "#1f4e79", "D", "solid", True),
]


def save(fig, stem: str):
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)


def plot_density(ax, panel: dict):
    for lab, key, col, marker, ls, is_sparc in METHODS:
        y = [float(panel[str(rho)][key]) for rho in DENS]
        ax.plot(
            DENS,
            y,
            color=col,
            linewidth=LW_SPARC if is_sparc else LW,
            linestyle=ls,
            marker=marker,
            markersize=18 if is_sparc else 16,
            label=lab,
            zorder=5 if is_sparc else 2,
        )


# --- 7(a) PRR vs density ---
fig, ax = plt.subplots(figsize=(20, 18))
plot_density(ax, d["panel_a_PRR_vs_density"])
ax.set_title("PRR", fontsize=FS)
ax.set_xlabel(r"Density $\rho$", fontsize=FS)
ax.set_ylabel("PRR", fontsize=FS)
ax.set_xticks(DENS)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig07a_PRR_vs_density")

# --- 7(b) P95 latency vs density ---
fig, ax = plt.subplots(figsize=(20, 18))
plot_density(ax, d["panel_b_P95_vs_density"])
ax.set_title("P95 latency", fontsize=FS)
ax.set_xlabel(r"Density $\rho$", fontsize=FS)
ax.set_ylabel("P95 latency (ms)", fontsize=FS)
ax.set_xticks(DENS)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig07b_P95_vs_density")

# --- 7(c) PRR vs distance (all baselines @ density 35; delay hist omitted) ---
panel_c = d["panel_d_PRR_vs_distance"]
curves = panel_c.get("curves") or {"direct": panel_c["curve"]}

# Line version (kept)
fig, ax = plt.subplots(figsize=(20, 18))
for lab, key, col, marker, ls, is_sparc in METHODS:
    curve = curves.get(key) or []
    if not curve:
        continue
    xs = [float(r["distance_m"]) for r in curve]
    ys = [float(r["mean_prr"]) for r in curve]
    ax.plot(
        xs,
        ys,
        color=col,
        linewidth=LW_SPARC if is_sparc else LW,
        linestyle=ls,
        marker=marker,
        markersize=18 if is_sparc else 16,
        markevery=max(1, len(xs) // 12),
        label=lab,
        zorder=5 if is_sparc else 2,
    )
ax.set_title("PRR vs distance", fontsize=FS)
ax.set_xlabel("Distance (m)", fontsize=FS)
ax.set_ylabel("PRR", fontsize=FS)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig07c_PRR_vs_distance_line")

# Bar version (main 7c): 5 representative distances
DIST_REQ = [50, 100, 200, 300, 400]


def nearest_prr(curve: list, target: float) -> tuple[float, float]:
    best = min(curve, key=lambda r: abs(float(r["distance_m"]) - target))
    return float(best["distance_m"]), float(best["mean_prr"])


n_grp = len(DIST_REQ)
n_m = len(METHODS)
x = list(range(n_grp))
width = 0.15
fig, ax = plt.subplots(figsize=(20, 18))
tick_labels = []
used_bins = []
for i, target in enumerate(DIST_REQ):
    # resolve display label from SPARC curve (same bins for all methods)
    ref = curves.get("direct") or next(iter(curves.values()))
    used, _ = nearest_prr(ref, float(target))
    used_bins.append(used)
    tick_labels.append(str(int(used)))

for j, (lab, key, col, _marker, _ls, is_sparc) in enumerate(METHODS):
    curve = curves.get(key) or []
    ys = []
    for target in DIST_REQ:
        _used, prr = nearest_prr(curve, float(target))
        ys.append(prr)
    offsets = (j - (n_m - 1) / 2) * width
    ax.bar(
        [xi + offsets for xi in x],
        ys,
        width=width,
        color=col,
        edgecolor="black" if is_sparc else "none",
        linewidth=2 if is_sparc else 0,
        label=lab,
        zorder=5 if is_sparc else 2,
    )

ax.set_title("PRR vs distance", fontsize=FS)
ax.set_xlabel("Distance (m)", fontsize=FS)
ax.set_ylabel("PRR", fontsize=FS)
ax.set_xticks(x)
ax.set_xticklabels(tick_labels)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, axis="y", alpha=0.3)
# Zoom y so small method gaps are visible
ymin = min(
    nearest_prr(curves[k], float(t))[1]
    for _, k, *_ in METHODS
    for t in DIST_REQ
    if curves.get(k)
)
ax.set_ylim(max(0.0, ymin - 0.05), 1.02)
fig.tight_layout()
save(fig, "fig07c_PRR_vs_distance")
print("7c bar bins used:", used_bins, "(requested", DIST_REQ, ")")
