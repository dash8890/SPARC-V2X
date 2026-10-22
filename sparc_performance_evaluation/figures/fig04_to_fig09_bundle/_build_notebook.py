"""Build and execute fig04_to_fig09.ipynb into this folder only (no overwrite of existing figs)."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repo root
DATA = json.loads(
    (HERE / "_fig04_09_bundle_data.json").read_text(encoding="utf-8")
)
NB_PATH = HERE / "fig04_to_fig09.ipynb"

# Round floats lightly for readability in cells; keep enough precision for plots.
def R(x, nd=10):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [R(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: R(v, nd) for k, v in x.items()}
    return x


def lit(obj) -> str:
    return json.dumps(R(obj), indent=2)


METHODS_LINE = """\
METHODS = [
    ("VAS", "#5b8db8", "^", (0, (6, 3)), False),
    ("LAS", "#7eb8a8", "v", (0, (2, 2)), False),
    ("LVM", "#e07a3d", "o", (0, (10, 3, 2, 3)), False),
    ("VLRO", "#3d8c5a", "s", (0, (1, 2)), False),
    ("SPARC-V2X", "#1f4e79", "D", "solid", True),
]
"""

HEADER = f'''\
# Fig.4–9 reproduction notebook
# One cell → display figure + write PNG/PDF. Outputs ONLY under this folder.
# Existing figures elsewhere are never modified.
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display

OUT = Path(r"{HERE.as_posix()}")
OUT.mkdir(parents=True, exist_ok=True)

def save(fig, stem: str):
    """Save png+pdf to OUT, then print the figure in the notebook cell output."""
    png, pdf = OUT / f"{{stem}}.png", OUT / f"{{stem}}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(png)))
    print("wrote", png.name, "+", pdf.name)
'''


def line_plot_cell(stem, title, xlabel, ylabel, data_key, xticks=None, ylim=None, legend_ncol=1, legend_loc="best", extra=""):
    d = DATA[data_key]
    xs = d["x"]
    series = {k: v for k, v in d.items() if k != "x"}
    body = f'''\
# {stem}
FS, LW = 50, 8
LW_SPARC = LW + 2
{METHODS_LINE}
x = {lit(xs)}
series = {lit(series)}

fig, ax = plt.subplots(figsize=(20, 18))
for lab, col, marker, ls, is_sparc in METHODS:
    ax.plot(
        x, series[lab],
        color=col,
        linewidth=LW_SPARC if is_sparc else LW,
        linestyle=ls,
        marker=marker,
        markersize=18 if is_sparc else 16,
        label=lab,
        zorder=5 if is_sparc else 2,
        solid_capstyle="round",
    )
ax.set_title({title!r}, fontsize=FS)
ax.set_xlabel({xlabel!r}, fontsize=FS)
ax.set_ylabel({ylabel!r}, fontsize=FS)
'''
    if xticks is not None:
        body += f"ax.set_xticks({lit(xticks)})\n"
    else:
        body += "ax.set_xticks(x)\n"
    if ylim is not None:
        body += f"ax.set_ylim({ylim[0]}, {ylim[1]})\n"
    body += f'''\
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc={legend_loc!r}, ncol={legend_ncol})
ax.grid(True, alpha=0.3)
{extra}
fig.tight_layout()
save(fig, {stem!r})
'''
    return body


def cells():
    out = []
    out.append(new_markdown_cell(
        "# Fig.4–9 — all versions\n\n"
        "Each code cell is self-contained (local `FS`/`LW`, embedded plot-point data) "
        "and writes **one** `.png` + `.pdf` into this folder only.\n\n"
        "**Outputs:** 21 figures (Fig.4–9, all on-disk versions)."
    ))
    out.append(new_code_cell(HEADER))

    # --- Fig.4 ---
    out.append(new_markdown_cell("## Fig.4 — Density sweep"))
    out.append(new_code_cell(line_plot_cell(
        "fig04a_R_vs_density", r"$R_{\mathrm{value}}$", r"Density $\rho$", r"$R_{\mathrm{value}}$",
        "fig04a", legend_ncol=2,
    )))
    out.append(new_code_cell(line_plot_cell(
        "fig04b_Pviol_vs_density", r"$P_{\mathrm{viol}}$", r"Density $\rho$", r"$P_{\mathrm{viol}}$",
        "fig04b", legend_loc="lower right",
    )))
    out.append(new_code_cell(line_plot_cell(
        "fig04c_prr_upper", "PRR", r"Density $\rho$", "PRR",
        "fig04c", legend_loc="upper right",
    )))
    # fig04d with custom ylim
    d4d = DATA["fig04d"]
    ys_all = [v for k, vs in d4d.items() if k != "x" for v in vs]
    out.append(new_code_cell(line_plot_cell(
        "fig04d_p95_upper", "P95 latency", r"Density $\rho$", r"P95 latency (ms)",
        "fig04d", legend_loc="upper left",
        ylim=(min(ys_all) - 1.0, 30),
    )))

    # --- Fig.5 ---
    out.append(new_markdown_cell("## Fig.5 — Envelope / operating points"))
    # 5a with special legend placement
    d5a = DATA["fig05a"]
    out.append(new_code_cell(f'''\
# fig05a_R_vs_envelope
FS, LW = 50, 8
LW_SPARC = LW + 2
{METHODS_LINE}
x = {lit(d5a["x"])}
series = {lit({k: v for k, v in d5a.items() if k != "x"})}

fig, ax = plt.subplots(figsize=(20, 18))
for lab, col, marker, ls, is_sparc in METHODS:
    ax.plot(
        x, series[lab],
        color=col,
        linewidth=LW_SPARC if is_sparc else LW,
        linestyle=ls,
        marker=marker,
        markersize=18 if is_sparc else 16,
        label=lab,
        zorder=5 if is_sparc else 2,
    )
ax.set_title(r"$R_{{\\mathrm{{value}}}}$ vs envelope", fontsize=FS)
ax.set_xlabel(r"Envelope $E$ (kB)", fontsize=FS)
ax.set_ylabel(r"$R_{{\\mathrm{{value}}}}$", fontsize=FS)
ax.set_xticks(x)
ax.tick_params(labelsize=FS)
ax.grid(True, alpha=0.3)
fig.tight_layout()
leg = ax.legend(
    fontsize=FS, frameon=False, ncol=1, loc="lower right",
    bbox_to_anchor=(350, 0.525), bbox_transform=ax.transData,
    borderaxespad=0.0, borderpad=0.0,
)
fig.canvas.draw()
bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
leg.set_bbox_to_anchor((350, 0.525 - (bb.y0 - 0.525)), transform=ax.transData)
save(fig, "fig05a_R_vs_envelope")
'''))

    d5b = DATA["fig05b"]
    out.append(new_code_cell(f'''\
# fig05b_R_vs_P95
FS = 50
ops = {lit(d5b)}
order = ["VAS", "LAS", "LVM", "VLRO", "SPARC-V2X"]
style = {{
    "VAS": ("#5b8db8", "^"),
    "LAS": ("#7eb8a8", "v"),
    "LVM": ("#e07a3d", "o"),
    "VLRO": ("#3d8c5a", "s"),
    "SPARC-V2X": ("#1f4e79", "D"),
}}
MS = 2 * ((600 ** 0.5) + 3)
SS = MS ** 2
fig, ax = plt.subplots(figsize=(20, 18))
for lab in order:
    col, marker = style[lab]
    is_sparc = lab == "SPARC-V2X"
    ax.scatter(
        [ops[lab]["P95"]], [ops[lab]["R"]],
        s=SS, color=col, marker=marker, label=lab,
        zorder=5 if is_sparc else 2, edgecolors="white", linewidths=2,
    )
ax.set_title(r"$R_{{\\mathrm{{value}}}}$ vs P95 latency", fontsize=FS)
ax.set_xlabel(r"P95 latency (ms)", fontsize=FS)
ax.set_ylabel(r"$R_{{\\mathrm{{value}}}}$", fontsize=FS)
ax.set_xlim(21, 25)
ax.set_xticks([21, 22, 23, 24, 25])
ax.set_ylim(None, 0.8)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig05b_R_vs_P95")
'''))

    # --- Fig.6 ---
    out.append(new_markdown_cell("## Fig.6 — Workload"))
    d6a = DATA["fig06a"]
    out.append(new_code_cell(f'''\
# fig06a_R_vs_load
FS, LW = 50, 8
LW_SPARC = LW + 2
{METHODS_LINE}
x = {lit(d6a["x"])}
series = {lit({k: v for k, v in d6a.items() if k != "x"})}
fig, ax = plt.subplots(figsize=(20, 18))
for lab, col, marker, ls, is_sparc in METHODS:
    ax.plot(
        x, series[lab], color=col,
        linewidth=LW_SPARC if is_sparc else LW, linestyle=ls,
        marker=marker, markersize=18 if is_sparc else 16,
        label=lab, zorder=5 if is_sparc else 2,
    )
ax.set_title(r"$R_{{\\mathrm{{value}}}}$ vs tasks/source", fontsize=FS)
ax.set_xlabel("Tasks per source", fontsize=FS)
ax.set_ylabel(r"$R_{{\\mathrm{{value}}}}$", fontsize=FS)
ax.set_xticks(x)
ax.set_ylim(bottom=0.4)
ax.tick_params(labelsize=FS)
ax.legend(
    fontsize=FS, frameon=False, loc="lower left", ncol=2,
    bbox_to_anchor=(0.0, 0.0), borderaxespad=0.17, borderpad=0.13,
    handletextpad=0.4, columnspacing=0.8,
)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig06a_R_vs_load")
'''))
    out.append(new_code_cell(line_plot_cell(
        "fig06b_R_vs_deadline",
        r"$R_{\mathrm{value}}$ vs deadline multiplier",
        "Deadline multiplier",
        r"$R_{\mathrm{value}}$",
        "fig06b",
    )))

    # --- Fig.7 ---
    out.append(new_markdown_cell("## Fig.7 — Radio / safety"))
    out.append(new_code_cell(line_plot_cell(
        "fig07a_PRR_vs_density", "PRR", r"Density $\rho$", "PRR", "fig07a",
    )))
    out.append(new_code_cell(line_plot_cell(
        "fig07b_P95_vs_density", "P95 latency", r"Density $\rho$", "P95 latency (ms)", "fig07b",
    )))

    # 7c bar
    d7b = DATA["fig07c_bar"]
    out.append(new_code_cell(f'''\
# fig07c_PRR_vs_distance (bar)
FS = 50
{METHODS_LINE}
DIST = {lit(d7b["x"])}
bins = {lit(d7b["bins"])}
series = {lit({k: v for k, v in d7b.items() if k not in ("x", "bins")})}
n_grp, n_m = len(DIST), len(METHODS)
x = list(range(n_grp))
width = 0.15
fig, ax = plt.subplots(figsize=(20, 18))
for j, (lab, col, _m, _ls, is_sparc) in enumerate(METHODS):
    offsets = (j - (n_m - 1) / 2) * width
    ax.bar(
        [xi + offsets for xi in x], series[lab], width=width, color=col,
        edgecolor="black" if is_sparc else "none",
        linewidth=2 if is_sparc else 0,
        label=lab, zorder=5 if is_sparc else 2,
    )
ax.set_title("PRR vs distance", fontsize=FS)
ax.set_xlabel("Distance (m)", fontsize=FS)
ax.set_ylabel("PRR", fontsize=FS)
ax.set_xticks(x)
ax.set_xticklabels([str(int(b)) for b in bins])
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, axis="y", alpha=0.3)
ymin = min(v for ys in series.values() for v in ys)
ax.set_ylim(max(0.0, ymin - 0.05), 1.02)
fig.tight_layout()
save(fig, "fig07c_PRR_vs_distance")
'''))

    # 7c line
    d7l = DATA["fig07c_line"]
    out.append(new_code_cell(f'''\
# fig07c_PRR_vs_distance_line
FS, LW = 50, 8
LW_SPARC = LW + 2
{METHODS_LINE}
curves = {lit(d7l)}
fig, ax = plt.subplots(figsize=(20, 18))
for lab, col, marker, ls, is_sparc in METHODS:
    pts = curves[lab]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(
        xs, ys, color=col,
        linewidth=LW_SPARC if is_sparc else LW, linestyle=ls,
        marker=marker, markersize=18 if is_sparc else 16,
        markevery=max(1, len(xs) // 12),
        label=lab, zorder=5 if is_sparc else 2,
    )
ax.set_title("PRR vs distance", fontsize=FS)
ax.set_xlabel("Distance (m)", fontsize=FS)
ax.set_ylabel("PRR", fontsize=FS)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig07c_PRR_vs_distance_line")
'''))

    # --- Fig.8 ---
    out.append(new_markdown_cell("## Fig.8 — Candidate set $|C_t|$"))
    d8 = DATA["fig08"]
    # sparc bars
    out.append(new_code_cell(
        "# fig08_R_vs_k_sparc\n"
        "FS = 50\n"
        f"KS = {lit(d8['x'])}\n"
        f"ys = {lit(d8['SPARC-V2X'])}\n"
        "bar_colors = ['azure', 'ivory', 'honeydew', 'seashell']\n"
        "bar_hatches = ['/', '\\\\', '*', '+']\n"
        "fig, ax = plt.subplots(figsize=(20, 18))\n"
        "x = list(range(len(KS)))\n"
        "for i, (xi, yi, k) in enumerate(zip(x, ys, KS)):\n"
        "    ax.bar(\n"
        "        xi, yi, color=bar_colors[i], hatch=bar_hatches[i], width=0.55,\n"
        "        edgecolor='black', linewidth=2, zorder=3,\n"
        "        label=(r'$|\\mathcal{C}_t|=%d$' % k),\n"
        "    )\n"
        "ax.set_title(r'$R_{\\mathrm{value}}$ vs $|\\mathcal{C}_t|$', fontsize=FS)\n"
        "ax.set_xlabel(r'$|\\mathcal{C}_t|$', fontsize=FS)\n"
        "ax.set_ylabel(r'$R_{\\mathrm{value}}$', fontsize=FS)\n"
        "ax.set_xticks([])\n"
        "ax.tick_params(labelsize=FS)\n"
        "ax.legend(fontsize=FS, frameon=False, loc='best', ncol=1)\n"
        "ax.grid(True, axis='y', alpha=0.3, zorder=0)\n"
        "ax.set_ylim(max(0.0, min(ys) - 0.05), max(ys) * 1.08)\n"
        "fig.tight_layout()\n"
        "save(fig, 'fig08_R_vs_k_sparc')\n"
    ))

    # shared line cell template for fig08_R_vs_k_all and fig08_R_vs_k
    for stem in ("fig08_R_vs_k_all", "fig08_R_vs_k"):
        out.append(new_code_cell(
            f"# {stem}\n"
            "FS, LW = 50, 8\n"
            "LW_SPARC = LW + 2\n"
            f"{METHODS_LINE}"
            f"KS = {lit(d8['x'])}\n"
            f"series = {lit({k: v for k, v in d8.items() if k != 'x'})}\n"
            "fig, ax = plt.subplots(figsize=(20, 18))\n"
            "for lab, col, marker, ls, is_sparc in METHODS:\n"
            "    ax.plot(\n"
            "        KS, series[lab], color=col,\n"
            "        linewidth=LW_SPARC if is_sparc else LW, linestyle=ls,\n"
            "        marker=marker, markersize=18 if is_sparc else 16,\n"
            "        label=lab, zorder=5 if is_sparc else 2,\n"
            "    )\n"
            "ax.set_title(r'$R_{\\mathrm{value}}$ vs $|\\mathcal{C}_t|$', fontsize=FS)\n"
            "ax.set_xlabel(r'$|\\mathcal{C}_t|$', fontsize=FS)\n"
            "ax.set_ylabel(r'$R_{\\mathrm{value}}$', fontsize=FS)\n"
            "ax.set_xticks(KS)\n"
            "ax.tick_params(labelsize=FS)\n"
            "ax.legend(fontsize=FS, frameon=False, loc='best', ncol=1)\n"
            "ax.grid(True, alpha=0.3)\n"
            "fig.tight_layout()\n"
            f"save(fig, {stem!r})\n"
        ))

    # --- Fig.9 ---
    out.append(new_markdown_cell("## Fig.9 — Density adaptation"))
    # N vs S @ 35/50/100 → fig09_density_adaptation
    main = DATA["fig09_main"]
    out.append(new_code_cell(
        "# fig09_density_adaptation (SPARC-N vs SPARC-S @ 35/50/100)\n"
        "FS = 50\n"
        "DENS = [35, 50, 100]\n"
        f"panel = {lit(main)}\n"
        "COLORS = {'SPARC-N': '#5b8db8', 'SPARC-S': '#1f4e79'}\n"
        "fig, ax = plt.subplots(figsize=(20, 18))\n"
        "x = list(range(len(DENS)))\n"
        "width = 0.32\n"
        "for j, lab in enumerate(('SPARC-N', 'SPARC-S')):\n"
        "    ys = [panel[str(d)][lab]['R_value'] for d in DENS]\n"
        "    offsets = (j - 0.5) * width\n"
        "    ax.bar(\n"
        "        [xi + offsets for xi in x], ys, width=width, color=COLORS[lab],\n"
        "        edgecolor='black' if lab == 'SPARC-S' else 'none',\n"
        "        linewidth=2 if lab == 'SPARC-S' else 0,\n"
        "        label=lab, zorder=3,\n"
        "    )\n"
        "ax.set_title(r'Density adaptation ($R_{\\mathrm{value}}$)', fontsize=FS)\n"
        "ax.set_xlabel(r'Density $\\rho$', fontsize=FS)\n"
        "ax.set_ylabel(r'$R_{\\mathrm{value}}$', fontsize=FS)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels([str(d) for d in DENS])\n"
        "ax.tick_params(labelsize=FS)\n"
        "ax.legend(fontsize=FS, frameon=False, loc='best')\n"
        "ax.grid(True, axis='y', alpha=0.3, zorder=0)\n"
        "ymax = max(panel[str(d)][lab]['R_value'] for d in DENS for lab in ('SPARC-N', 'SPARC-S'))\n"
        "ax.set_ylim(0.0, ymax * 1.2)\n"
        "for i, d in enumerate(DENS):\n"
        "    for j, lab in enumerate(('SPARC-N', 'SPARC-S')):\n"
        "        yi = panel[str(d)][lab]['R_value']\n"
        "        pviol = panel[str(d)][lab]['P_viol']\n"
        "        ax.text(i + (j - 0.5) * width, yi + 0.012, '%.2f' % pviol,\n"
        "                ha='center', va='bottom', fontsize=FS - 14)\n"
        "ax.text(0.02, 0.98, r'$P_{\\mathrm{viol}}$ above bars', transform=ax.transAxes,\n"
        "        ha='left', va='top', fontsize=FS - 16)\n"
        "fig.tight_layout()\n"
        "save(fig, 'fig09_density_adaptation')\n"
    ))

    # dens50 three-bar
    d50 = DATA["fig09_d50"]
    out.append(new_code_cell(
        "# fig09_density50_adaptation (SPARC-N / SPARC-S50 / VLRO)\n"
        "FS = 50\n"
        f"bars = {lit(d50)}\n"
        "labels = ['SPARC-N', 'SPARC-S50', 'VLRO']\n"
        "colors = ['#5b8db8', '#1f4e79', '#3d8c5a']\n"
        "ys = [bars[lab][0] for lab in labels]\n"
        "pviol = bars['SPARC-N'][1]\n"
        "fig, ax = plt.subplots(figsize=(20, 18))\n"
        "x = list(range(len(labels)))\n"
        "ax.bar(x, ys, color=colors, width=0.55, zorder=3)\n"
        "ax.bar([1], [ys[1]], color=colors[1], width=0.55, edgecolor='black', linewidth=3, zorder=4)\n"
        "ax.set_title(r'Density-50 adaptation ($R_{\\mathrm{value}}$)', fontsize=FS)\n"
        "ax.set_ylabel(r'$R_{\\mathrm{value}}$', fontsize=FS)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(labels)\n"
        "ax.tick_params(labelsize=FS)\n"
        "ax.grid(True, axis='y', alpha=0.3, zorder=0)\n"
        "ax.set_ylim(0.0, max(ys) * 1.25)\n"
        "for i in (0, 1):\n"
        "    ax.text(i, ys[i] + 0.012, r'$P_{\\mathrm{viol}}=%.2f$' % pviol,\n"
        "            ha='center', va='bottom', fontsize=FS - 8)\n"
        "fig.tight_layout()\n"
        "save(fig, 'fig09_density50_adaptation')\n"
    ))

    # dens100 three-bar
    d100 = DATA["fig09_d100"]
    out.append(new_code_cell(
        "# fig09_density100_adaptation (SPARC-N / SPARC-S100 / VLRO)\n"
        "FS = 50\n"
        f"bars = {lit(d100)}\n"
        "labels = ['SPARC-N', 'SPARC-S100', 'VLRO']\n"
        "colors = ['#5b8db8', '#1f4e79', '#3d8c5a']\n"
        "ys = [bars[lab][0] for lab in labels]\n"
        "pviol = bars['SPARC-N'][1]\n"
        "fig, ax = plt.subplots(figsize=(20, 18))\n"
        "x = list(range(len(labels)))\n"
        "ax.bar(x, ys, color=colors, width=0.55, zorder=3)\n"
        "ax.bar([1], [ys[1]], color=colors[1], width=0.55, edgecolor='black', linewidth=3, zorder=4)\n"
        "ax.set_title(r'Density-100 adaptation ($R_{\\mathrm{value}}$)', fontsize=FS)\n"
        "ax.set_ylabel(r'$R_{\\mathrm{value}}$', fontsize=FS)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(labels)\n"
        "ax.tick_params(labelsize=FS)\n"
        "ax.grid(True, axis='y', alpha=0.3, zorder=0)\n"
        "ax.set_ylim(0.0, max(ys) * 1.28)\n"
        "for i in (0, 1):\n"
        "    ax.text(i, ys[i] + 0.012, r'$P_{\\mathrm{viol}}=%.2f$' % pviol,\n"
        "            ha='center', va='bottom', fontsize=FS - 8)\n"
        "fig.tight_layout()\n"
        "save(fig, 'fig09_density100_adaptation')\n"
    ))

    # combined
    comb = DATA["fig09_combined"]
    out.append(new_code_cell(
        "# fig09_density_adaptation_combined\n"
        "FS = 50\n"
        f"panels = {lit(comb)}\n"
        "dens_list = [50, 100]\n"
        "COLORS = {'SPARC-N': '#5b8db8', 'SPARC-S': '#1f4e79', 'VLRO': '#3d8c5a'}\n"
        "fig, ax = plt.subplots(figsize=(20, 18))\n"
        "n_grp = len(dens_list)\n"
        "width = 0.22\n"
        "x = list(range(n_grp))\n"
        "series_keys = [\n"
        "    ('SPARC-N', COLORS['SPARC-N'], lambda d: panels[str(d)]['SPARC-N'][0]),\n"
        "    ('SPARC-S', COLORS['SPARC-S'], lambda d: panels[str(d)]['SPARC-S'][0]),\n"
        "    ('VLRO', COLORS['VLRO'], lambda d: panels[str(d)]['VLRO'][0]),\n"
        "]\n"
        "for j, (lab, col, getter) in enumerate(series_keys):\n"
        "    ys = [getter(d) for d in dens_list]\n"
        "    offsets = (j - 1) * width\n"
        "    ax.bar(\n"
        "        [xi + offsets for xi in x], ys, width=width, color=col,\n"
        "        edgecolor='black' if lab == 'SPARC-S' else 'none',\n"
        "        linewidth=2 if lab == 'SPARC-S' else 0,\n"
        "        label=lab, zorder=3,\n"
        "    )\n"
        "ax.set_title(r'High-density adaptation ($R_{\\mathrm{value}}$)', fontsize=FS)\n"
        "ax.set_xlabel(r'Density $\\rho$', fontsize=FS)\n"
        "ax.set_ylabel(r'$R_{\\mathrm{value}}$', fontsize=FS)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels([str(d) for d in dens_list])\n"
        "ax.tick_params(labelsize=FS)\n"
        "ax.legend(fontsize=FS, frameon=False, loc='best')\n"
        "ax.grid(True, axis='y', alpha=0.3)\n"
        "fig.tight_layout()\n"
        "save(fig, 'fig09_density_adaptation_combined')\n"
    ))

    # SPARC-N only bars
    n_only = DATA["fig09_N_only"]
    out.append(new_code_cell(
        "# fig09_sparc_N_dens_35_50_100\n"
        "FS = 50\n"
        f"data = {lit(n_only)}\n"
        "dens_list = [35, 50, 100]\n"
        "BAR_COLORS = ['azure', 'ivory', 'honeydew']\n"
        "BAR_HATCHES = ['/', '\\\\', '*']\n"
        "ys = [data[str(d)][0] for d in dens_list]\n"
        "fig, ax = plt.subplots(figsize=(20, 18))\n"
        "x = list(range(len(dens_list)))\n"
        "for i, (xi, yi, d) in enumerate(zip(x, ys, dens_list)):\n"
        "    ax.bar(\n"
        "        xi, yi, color=BAR_COLORS[i], hatch=BAR_HATCHES[i], width=0.55,\n"
        "        edgecolor='black', linewidth=2, zorder=3, label=(r'$\\rho=%d$' % d),\n"
        "    )\n"
        "ax.set_title(r'SPARC-N ($R_{\\mathrm{value}}$)', fontsize=FS)\n"
        "ax.set_xlabel(r'Density $\\rho$', fontsize=FS)\n"
        "ax.set_ylabel(r'$R_{\\mathrm{value}}$', fontsize=FS)\n"
        "ax.set_xticks([])\n"
        "ax.tick_params(labelsize=FS)\n"
        "ax.legend(fontsize=FS, frameon=False, loc='best')\n"
        "ax.grid(True, axis='y', alpha=0.3, zorder=0)\n"
        "ax.set_ylim(0.0, max(ys) * 1.2)\n"
        "for i, d in enumerate(dens_list):\n"
        "    pviol = data[str(d)][1]\n"
        "    ax.text(i, ys[i] + 0.015, r'$P_{\\mathrm{viol}}=%.2f$' % pviol,\n"
        "            ha='center', va='bottom', fontsize=FS - 8)\n"
        "fig.tight_layout()\n"
        "save(fig, 'fig09_sparc_N_dens_35_50_100')\n"
    ))

    # S100 episode return
    s100 = DATA["fig09_S100_return"]
    out.append(new_code_cell(f'''\
# fig09_sparc_S100_episode_return
FS, LW = 50, 8
ep = np.array({lit(s100["episode"])}, dtype=float)
ret = np.array({lit(s100["return"])}, dtype=float)
w = 32
kernel = np.ones(w) / w
smooth = np.convolve(ret, kernel, mode="same")
half = w // 2
for i in range(half):
    smooth[i] = ret[: i + half + 1].mean()
    smooth[-(i + 1)] = ret[-(i + half + 1) :].mean()
fig, ax = plt.subplots(figsize=(20, 18))
ax.plot(ep, smooth, color="#c45c26", linewidth=LW, label="Return")
ax.set_xlabel("Episode", fontsize=FS)
ax.set_ylabel("Return", fontsize=FS)
ax.set_title(r"SPARC-S100 episode return", fontsize=FS)
xmax = float(ep.max())
ax.set_xlim(0, xmax)
ticks = list(np.arange(0, xmax, 16))
if not ticks or ticks[-1] != xmax:
    ticks.append(xmax)
ax.set_xticks(ticks)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "fig09_sparc_S100_episode_return")
'''))

    return out


def main():
    nb = new_notebook(
        cells=cells(),
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    NB_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
    print("wrote", NB_PATH)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    NB_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
    print("executed", NB_PATH)

    pngs = sorted(HERE.glob("fig*.png"))
    pdfs = sorted(HERE.glob("fig*.pdf"))
    print(f"png={len(pngs)} pdf={len(pdfs)}")
    for p in pngs:
        print(" ", p.name)


if __name__ == "__main__":
    main()
