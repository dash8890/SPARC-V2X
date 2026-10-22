"""Generate Fig.6(a–b) with all five baselines (VAS, LAS, SPARC-V2X, LVM, VLRO)."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS, LW = 50, 8
LW_SPARC = LW + 2
OUT = Path("sparc_performance_evaluation/figures/fig06_workload")
RAW = Path("sparc_performance_evaluation/results/raw")
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("VAS", "#5b8db8", "^", (0, (6, 3)), False),
    ("LAS", "#7eb8a8", "v", (0, (2, 2)), False),
    ("LVM", "#e07a3d", "o", (0, (10, 3, 2, 3)), False),
    ("VLRO", "#3d8c5a", "s", (0, (1, 2)), False),
    ("SPARC-V2X", "#1f4e79", "D", "solid", True),
]


def save(fig, stem: str):
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)


def mean_R(folder: str, method: str) -> float:
    p = RAW / folder / "rows.json"
    if not p.exists():
        raise FileNotFoundError(p)
    rows = json.loads(p.read_text(encoding="utf-8-sig"))
    xs = [
        float(r["R_value"])
        for r in rows
        if r.get("method") == method and r.get("R_value") is not None
    ]
    if not xs:
        raise ValueError(f"no R_value for method={method} in {folder}")
    return mean(xs)


def plot_series(ax, xs, series: dict[str, list[float]]):
    for lab, col, marker, ls, is_sparc in METHODS:
        ax.plot(
            xs,
            series[lab],
            color=col,
            linewidth=LW_SPARC if is_sparc else LW,
            linestyle=ls,
            marker=marker,
            markersize=18 if is_sparc else 16,
            label=lab,
            zorder=5 if is_sparc else 2,
        )


def place_legend_bottom_on_y(ax, fig, x_anchor: float, y_line: float = 0.5):
    """Place legend so its bottom edge sits on the given horizontal data line."""
    leg = ax.legend(
        fontsize=FS,
        frameon=False,
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(x_anchor, y_line),
        bbox_transform=ax.transData,
        borderaxespad=0.0,
        borderpad=0.0,
    )
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    leg.set_bbox_to_anchor((x_anchor, y_line - (bb.y0 - y_line)), transform=ax.transData)
    return leg


# --- series: load ---
LOADS = [2, 3, 4, 5, 6]
series_a = {
    "VAS": [mean_R(f"catchup_e4_load_t{k}_va_mf", "value_aware") for k in LOADS],
    "LAS": [mean_R(f"catchup_e4_load_t{k}_va_mf", "minfinish") for k in LOADS],
    "SPARC-V2X": [mean_R(f"catchup_e4_load_t{k}_direct", "residual") for k in LOADS],
    "VLRO": [mean_R(f"catchup_e4_load_t{k}_itora", "itora_inspired") for k in LOADS],
    "LVM": [mean_R(f"catchup_e4_load_t{k}_b4", "residual") for k in LOADS],
}

# --- series: deadline ---
DMS = [0.5, 1.0, 1.5, 2.0]
TAGS = {0.5: "0p5", 1.0: "1p0", 1.5: "1p5", 2.0: "2p0"}
series_b = {
    "VAS": [mean_R(f"catchup_e4_deadline_m{TAGS[k]}_va_mf", "value_aware") for k in DMS],
    "LAS": [mean_R(f"catchup_e4_deadline_m{TAGS[k]}_va_mf", "minfinish") for k in DMS],
    "SPARC-V2X": [mean_R(f"catchup_e4_deadline_m{TAGS[k]}_direct", "residual") for k in DMS],
    "VLRO": [mean_R(f"catchup_e4_deadline_m{TAGS[k]}_itora", "itora_inspired") for k in DMS],
    "LVM": [mean_R(f"catchup_e4_deadline_m{TAGS[k]}_b4", "residual") for k in DMS],
}

# --- 6(a) ---
fig, ax = plt.subplots(figsize=(20, 18))
plot_series(ax, LOADS, series_a)
ax.set_title(r"$R_{\mathrm{value}}$ vs tasks/source", fontsize=FS)
ax.set_xlabel("Tasks per source", fontsize=FS)
ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
ax.set_xticks(LOADS)
ax.set_ylim(bottom=0.4)
ax.tick_params(labelsize=FS)
ax.legend(
    fontsize=FS,
    frameon=False,
    loc="lower left",
    ncol=2,
    bbox_to_anchor=(0.0, 0.0),
    borderaxespad=0.17,  # ~1/3 of matplotlib default (0.5)
    borderpad=0.13,  # ~1/3 of default (0.4)
    handletextpad=0.4,
    columnspacing=0.8,
)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig06a_R_vs_load")

# --- 6(b) ---
fig, ax = plt.subplots(figsize=(20, 18))
plot_series(ax, DMS, series_b)
ax.set_title(r"$R_{\mathrm{value}}$ vs deadline multiplier", fontsize=FS)
ax.set_xlabel("Deadline multiplier", fontsize=FS)
ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
ax.set_xticks(DMS)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig06b_R_vs_deadline")

(OUT / "fig06_all_baselines_series.json").write_text(
    json.dumps({"load": {"x": LOADS, "R": series_a}, "deadline": {"x": DMS, "R": series_b}}, indent=2),
    encoding="utf-8",
)
