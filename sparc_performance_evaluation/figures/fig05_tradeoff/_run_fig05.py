"""Generate Fig.5(a–b) admission-envelope / operating-point plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS, LW = 50, 8
LW_SPARC = LW + 2
OUT = Path("sparc_performance_evaluation/figures/fig05_tradeoff")
DATA = Path("sparc_performance_evaluation/results/stage5_no_residual/figures/fig05_data.json")
FIG04 = Path("sparc_performance_evaluation/results/stage5_no_residual/figures/fig04_data.json")
OUT.mkdir(parents=True, exist_ok=True)
d = json.loads(DATA.read_text(encoding="utf-8"))
d4 = json.loads(FIG04.read_text(encoding="utf-8"))

# Baselines dashed + distinct markers; SPARC-V2X solid, thicker, on top
METHODS = [
    ("value_aware", "VAS", "#5b8db8", "^", (0, (6, 3)), False),
    ("minfinish", "LAS", "#7eb8a8", "v", (0, (2, 2)), False),
    ("b4_lagrangian", "LVM", "#e07a3d", "o", (0, (10, 3, 2, 3)), False),
    ("itora_inspired", "VLRO", "#3d8c5a", "s", (0, (1, 2)), False),
    ("direct", "SPARC-V2X", "#1f4e79", "D", "solid", True),
]
# Operating-point labels in fig05 JSON (legacy B* keys)
OP_KEYS = [
    ("B1_VA", "VAS", "#5b8db8", "o"),
    ("B2_MF", "LAS", "#7eb8a8", "s"),
    ("B4_Lagrangian", "LVM", "#e07a3d", "^"),
    ("B5_ITORA", "VLRO", "#3d8c5a", "v"),
    ("B3_Direct", "SPARC-V2X", "#1f4e79", "D"),  # last = on top in scatter draw order
]

ENVELOPES_KB = [100, 175, 250, 350]
ENVELOPES_B = [e * 1000 for e in ENVELOPES_KB]


def save(fig, stem: str):
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)


# --- 5(a) R_value vs envelope ---
env = d["panel_envelope_sensitivity"]
fig, ax = plt.subplots(figsize=(20, 18))
for key, lab, col, marker, ls, is_sparc in METHODS:
    y = [env[str(b)][key]["R"] for b in ENVELOPES_B]
    ax.plot(
        ENVELOPES_KB,
        y,
        color=col,
        linewidth=LW_SPARC if is_sparc else LW,
        linestyle=ls,
        marker=marker,
        markersize=18 if is_sparc else 16,
        label=lab,
        zorder=5 if is_sparc else 2,
    )
ax.set_title(r"$R_{\mathrm{value}}$ vs envelope", fontsize=FS)
ax.set_xlabel(r"Envelope $E$ (kB)", fontsize=FS)
ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
ax.set_xticks(ENVELOPES_KB)
ax.tick_params(labelsize=FS)
ax.grid(True, alpha=0.3)
fig.tight_layout()
# Place legend, then nudge so its bottom edge sits on y=0.525 (data).
leg = ax.legend(
    fontsize=FS,
    frameon=False,
    ncol=1,
    loc="lower right",
    bbox_to_anchor=(350, 0.525),
    bbox_transform=ax.transData,
    borderaxespad=0.0,
    borderpad=0.0,
)
fig.canvas.draw()
bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
leg.set_bbox_to_anchor((350, 0.525 - (bb.y0 - 0.525)), transform=ax.transData)
save(fig, "fig05a_R_vs_envelope")

# --- 5(b) R_value vs P95 operating points (nominal dens=35; not a Pareto claim) ---
# P95 from density-35 sealed series (fig04 full), R from fig05 operating points
full35 = d4["full"]["35"]
prr_p95 = {
    "VAS": full35["value_aware"],
    "LAS": full35["minfinish"],
    "SPARC-V2X": full35["direct"],
    "LVM": full35["b4_lagrangian"],
    "VLRO": full35["itora_inspired"],
}
ops = {row["method"]: row for row in d["panel_operating_points"]}
# map legacy -> acronym + R
op_R = {
    "VAS": ops["B1_VA"]["R_value"],
    "LAS": ops["B2_MF"]["R_value"],
    "SPARC-V2X": ops["B3_Direct"]["R_value"],
    "LVM": ops["B4_Lagrangian"]["R_value"],
    "VLRO": ops["B5_ITORA"]["R_value"],
}

fig, ax = plt.subplots(figsize=(20, 18))
# draw baselines first, SPARC last
order = ["VAS", "LAS", "LVM", "VLRO", "SPARC-V2X"]
style = {lab: (col, marker) for _, lab, col, marker, *_ in [
    (*m[:4],) for m in METHODS
]}
# rebuild style map cleanly
style = {
    "VAS": ("#5b8db8", "^"),
    "LAS": ("#7eb8a8", "v"),
    "LVM": ("#e07a3d", "o"),
    "VLRO": ("#3d8c5a", "s"),
    "SPARC-V2X": ("#1f4e79", "D"),
}
# Same marker size for all; +3 pt vs prior SPARC, then doubled
MS = 2 * ((600 ** 0.5) + 3)
SS = MS ** 2
for lab in order:
    x = prr_p95[lab]["mean_p95"]
    y = op_R[lab]
    col, marker = style[lab]
    is_sparc = lab == "SPARC-V2X"
    ax.scatter(
        [x],
        [y],
        s=SS,
        color=col,
        marker=marker,
        label=lab,
        zorder=5 if is_sparc else 2,
        edgecolors="white",
        linewidths=2,
    )

ax.set_title(r"$R_{\mathrm{value}}$ vs P95 latency", fontsize=FS)
ax.set_xlabel(r"P95 latency (ms)", fontsize=FS)
ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
ax.set_xlim(21, 25)
ax.set_xticks([21, 22, 23, 24, 25])
ax.set_ylim(None, 0.8)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig05b_R_vs_P95")
