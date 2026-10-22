"""Fig.9 — SPARC-N vs SPARC-S at dens 35 / 50 / 100."""
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

# SPARC-N = train@35 eval@dens; SPARC-S = train@dens eval@dens
# At dens 35, N and S are the same sealed policy.
SOURCES = {
    35: {
        "SPARC-N": "catchup_gov_on_d35_direct",
        "SPARC-S": "catchup_gov_on_d35_direct",
    },
    50: {
        "SPARC-N": "catchup_fig11a_nominal_direct_on_d50",
        "SPARC-S": "catchup_fig11a_direct_d50_ft",
    },
    100: {
        "SPARC-N": "catchup_fig11a_nominal_direct_on_d100",
        "SPARC-S": "catchup_fig11a_direct_d100_ft",
    },
}
DENS = [35, 50, 100]
COLORS = {"SPARC-N": "#5b8db8", "SPARC-S": "#1f4e79"}


def stats(folder: str) -> dict:
    rows = json.loads((RAW / folder / "rows.json").read_text(encoding="utf-8-sig"))
    sel = [r for r in rows if r.get("method") == "residual"]
    if not sel:
        raise ValueError(f"no residual rows in {folder}")
    return {
        "R_value": mean(float(r["R_value"]) for r in sel if r.get("R_value") is not None),
        "P_viol": mean(float(r.get("safety_P_viol") or 0.0) for r in sel),
        "n": len(sel),
        "source": folder,
    }


panel = {}
for dens in DENS:
    panel[str(dens)] = {
        "SPARC-N": stats(SOURCES[dens]["SPARC-N"]),
        "SPARC-S": stats(SOURCES[dens]["SPARC-S"]),
    }

(OUT / "fig09_data.json").write_text(
    json.dumps(
        {
            "figure": "Fig.9",
            "title": "SPARC-N vs SPARC-S across density",
            "note": "At dens 35, SPARC-N and SPARC-S use the same sealed train@35 policy.",
            "panels": panel,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

# Grouped bars
fig, ax = plt.subplots(figsize=(20, 18))
x = list(range(len(DENS)))
width = 0.32
for j, lab in enumerate(("SPARC-N", "SPARC-S")):
    ys = [panel[str(d)][lab]["R_value"] for d in DENS]
    offsets = (j - 0.5) * width
    ax.bar(
        [xi + offsets for xi in x],
        ys,
        width=width,
        color=COLORS[lab],
        edgecolor="black" if lab == "SPARC-S" else "none",
        linewidth=2 if lab == "SPARC-S" else 0,
        label=lab,
        zorder=3,
    )

ax.set_title(r"Density adaptation ($R_{\mathrm{value}}$)", fontsize=FS)
ax.set_xlabel(r"Density $\rho$", fontsize=FS)
ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in DENS])
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best")
ax.grid(True, axis="y", alpha=0.3, zorder=0)
ymax = max(panel[str(d)][lab]["R_value"] for d in DENS for lab in ("SPARC-N", "SPARC-S"))
ax.set_ylim(0.0, ymax * 1.2)

# Annotate P_viol above each SPARC-N / SPARC-S pair cluster
for i, d in enumerate(DENS):
    for j, lab in enumerate(("SPARC-N", "SPARC-S")):
        yi = panel[str(d)][lab]["R_value"]
        pviol = panel[str(d)][lab]["P_viol"]
        ax.text(
            i + (j - 0.5) * width,
            yi + 0.012,
            rf"{pviol:.2f}",
            ha="center",
            va="bottom",
            fontsize=FS - 14,
        )
ax.text(
    0.02,
    0.98,
    r"$P_{\mathrm{viol}}$ above bars",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=FS - 16,
)

fig.tight_layout()
for stem in ("fig09_density_adaptation", "fig09_density50_adaptation"):
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", png)
plt.close(fig)

print(
    {
        str(d): {
            "N": round(panel[str(d)]["SPARC-N"]["R_value"], 4),
            "S": round(panel[str(d)]["SPARC-S"]["R_value"], 4),
        }
        for d in DENS
    }
)
