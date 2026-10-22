"""Generate Fig.4(a–d) PNG/PDF from stage5 fig04_data.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS, LW = 50, 8
LW_SPARC = LW + 2  # SPARC-V2X two points thicker
OUT = Path("sparc_performance_evaluation/figures/fig04_density")
DATA = Path("sparc_performance_evaluation/results/stage5_no_residual/figures/fig04_data.json")
d = json.loads(DATA.read_text(encoding="utf-8"))

# Baselines first (dashed, under), SPARC-V2X last (solid, on top).
# linestyle dash tuples = distinct intervals; marker unique per method.
METHODS = [
    # key, label, color, marker, linestyle, is_sparc
    ("value_aware", "VAS", "#5b8db8", "^", (0, (6, 3)), False),
    ("minfinish", "LAS", "#7eb8a8", "v", (0, (2, 2)), False),
    ("b4_lagrangian", "LVM", "#e07a3d", "o", (0, (10, 3, 2, 3)), False),
    ("itora_inspired", "VLRO", "#3d8c5a", "s", (0, (1, 2)), False),
    ("direct", "SPARC-V2X", "#1f4e79", "D", "solid", True),
]
DENS_MAIN = [20, 25, 30, 35, 40, 45, 50]
DENS_UPPER = [35, 40, 45, 50]


def save(fig, stem: str):
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)


def plot_methods(ax, dens, y_of):
    """Plot all methods; SPARC-V2X drawn last with higher zorder/linewidth."""
    for key, lab, col, marker, ls, is_sparc in METHODS:
        y = y_of(key)
        ax.plot(
            dens,
            y,
            color=col,
            linewidth=LW_SPARC if is_sparc else LW,
            linestyle=ls,
            marker=marker,
            markersize=18 if is_sparc else 16,
            label=lab,
            zorder=5 if is_sparc else 2,
            solid_capstyle="round",
        )


def style_ax(ax, title, xlabel, ylabel, xticks, ylim=None, legend_loc="best", legend_ncol=2):
    ax.set_title(title, fontsize=FS)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_xticks(xticks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc=legend_loc, ncol=legend_ncol)
    ax.grid(True, alpha=0.3)


# --- 4(a) R_value ---
panel_a = d["panel_a_R_value_vs_density"]
fig, ax = plt.subplots(figsize=(20, 18))
plot_methods(ax, DENS_MAIN, lambda key: [panel_a[str(rho)][key] for rho in DENS_MAIN])
style_ax(ax, r"$R_{\mathrm{value}}$", r"Density $\rho$", r"$R_{\mathrm{value}}$", DENS_MAIN)
fig.tight_layout()
save(fig, "fig04a_R_vs_density")

# --- 4(b) P_viol ---
panel_b = d["panel_b_P_viol_vs_density"]
fig, ax = plt.subplots(figsize=(20, 18))
plot_methods(ax, DENS_MAIN, lambda key: [panel_b[str(rho)][key] for rho in DENS_MAIN])
style_ax(ax, r"$P_{\mathrm{viol}}$", r"Density $\rho$", r"$P_{\mathrm{viol}}$", DENS_MAIN,
         legend_loc="lower right", legend_ncol=1)
fig.tight_layout()
save(fig, "fig04b_Pviol_vs_density")

# --- 4(c) PRR ---
full = d["full"]
fig, ax = plt.subplots(figsize=(20, 18))
plot_methods(ax, DENS_UPPER, lambda key: [full[str(rho)][key]["mean_prr"] for rho in DENS_UPPER])
style_ax(ax, "PRR", r"Density $\rho$", "PRR", DENS_UPPER, legend_loc="upper right", legend_ncol=1)
fig.tight_layout()
save(fig, "fig04c_prr_upper")

# --- 4(d) P95 ---
fig, ax = plt.subplots(figsize=(20, 18))
plot_methods(ax, DENS_UPPER, lambda key: [full[str(rho)][key]["mean_p95"] for rho in DENS_UPPER])
style_ax(ax, "P95 latency", r"Density $\rho$", r"P95 latency (ms)", DENS_UPPER,
         legend_loc="upper left", legend_ncol=1)
ys = [full[str(rho)][key]["mean_p95"] for rho in DENS_UPPER for key, *_ in METHODS]
ax.set_ylim(min(ys) - 1.0, 30)
fig.tight_layout()
save(fig, "fig04d_p95_upper")
