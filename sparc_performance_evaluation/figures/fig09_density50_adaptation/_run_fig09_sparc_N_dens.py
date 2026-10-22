"""Fresh plot: SPARC-N R_value at dens 35 / 50 / 100. Does not overwrite existing Fig.9 files."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 50
OUT = Path("sparc_performance_evaluation/figures/fig09_density50_adaptation")
RAW = Path("sparc_performance_evaluation/results/raw")
OUT.mkdir(parents=True, exist_ok=True)

# SPARC-N = sealed dens-35 Direct checkpoint evaluated at each density
SOURCES = {
    35: "catchup_gov_on_d35_direct",
    50: "catchup_fig11a_nominal_direct_on_d50",
    100: "catchup_fig11a_nominal_direct_on_d100",
}
BAR_COLORS = ["azure", "ivory", "honeydew"]
BAR_HATCHES = ["/", "\\", "*"]


def stats(folder: str) -> dict:
    rows = json.loads((RAW / folder / "rows.json").read_text(encoding="utf-8-sig"))
    sel = [r for r in rows if r.get("method") == "residual"]
    return {
        "R_value": mean(float(r["R_value"]) for r in sel if r.get("R_value") is not None),
        "P_viol": mean(float(r.get("safety_P_viol") or 0.0) for r in sel),
        "n": len(sel),
        "source": folder,
    }


dens_list = [35, 50, 100]
data = {d: stats(SOURCES[d]) for d in dens_list}
(OUT / "fig09_sparc_N_dens_35_50_100_data.json").write_text(
    json.dumps({"label": "SPARC-N", "densities": dens_list, "bars": {str(d): data[d] for d in dens_list}}, indent=2)
    + "\n",
    encoding="utf-8",
)

ys = [data[d]["R_value"] for d in dens_list]
fig, ax = plt.subplots(figsize=(20, 18))
x = list(range(len(dens_list)))
for i, (xi, yi, d) in enumerate(zip(x, ys, dens_list)):
    ax.bar(
        xi,
        yi,
        color=BAR_COLORS[i],
        hatch=BAR_HATCHES[i],
        width=0.55,
        edgecolor="black",
        linewidth=2,
        zorder=3,
        label=rf"$\rho={d}$",
    )
ax.set_title(r"SPARC-N ($R_{\mathrm{value}}$)", fontsize=FS)
ax.set_xlabel(r"Density $\rho$", fontsize=FS)
ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
ax.set_xticks([])
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best")
ax.grid(True, axis="y", alpha=0.3, zorder=0)
ax.set_ylim(0.0, max(ys) * 1.2)
for i, d in enumerate(dens_list):
    ax.text(
        i,
        ys[i] + 0.015,
        rf"$P_{{\mathrm{{viol}}}}={data[d]['P_viol']:.2f}$",
        ha="center",
        va="bottom",
        fontsize=FS - 8,
    )
fig.tight_layout()
stem = "fig09_sparc_N_dens_35_50_100"
png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)
print("wrote", png)
print({d: round(data[d]["R_value"], 4) for d in dens_list})
