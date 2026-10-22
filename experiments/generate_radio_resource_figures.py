
import json, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "task_aware_full_kpis_10seed.json"
OUT = ROOT / "results" / "radio_resource_figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = 55
LW = 5.0
MS = 16
COLORS = {"Final SPARC-V2X": "#1f77b4", "Minimum-Finish": "#d62728"}
MARKERS = {"Final SPARC-V2X": "o", "Minimum-Finish": "s"}
DENSITIES = [10, 25, 50, 80, 100, 125, 150, 175, 200]

rows = json.loads(DATA.read_text(encoding="utf-8"))["summary"]
lookup = {(r["method"], r["density"]): r for r in rows}
series = {
    "Final SPARC-V2X": {d: lookup[("residual_rl", d)] for d in DENSITIES},
    "Minimum-Finish": {d: lookup[("minfinish", d)] for d in DENSITIES},
}
for _vals in series.values():
    for _d, _row in _vals.items():
        _row["admitted_computation_mbit"] = _row["comp_resource_bits"] / 1e6
        _row["value_per_computation_bit"] = _row["eta_value"]


with (OUT / "radio_resource_figure_data.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["density", "method", "shield_intervention_rate", "mean_safe_action_set", "admitted_computation_mbit", "value_per_computation_bit"])
    for method, vals in series.items():
        for d, r in vals.items():
            w.writerow([d, method, r["shield_intervention_rate"], r["mean_safe_action_set"], r["comp_resource_bits"] / 1e6, r["eta_value"]])

def style(ax, ylabel, font=FONT):
    ax.set_xlabel("Vehicle density", fontsize=font)
    ax.set_ylabel(ylabel, fontsize=font)
    ax.tick_params(axis="both", labelsize=font - 5, width=2.5, length=10)
    ax.grid(True, alpha=0.28, linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    ax.set_xticks(DENSITIES)
    ax.legend(fontsize=font - 8, frameon=False, loc="best")

def draw(metric, ylabel, stem, panel):
    fig, ax = plt.subplots(figsize=(20, 18))
    for method, vals in series.items():
        ax.plot(DENSITIES, [vals[d][metric] for d in DENSITIES], color=COLORS[method], marker=MARKERS[method], markersize=MS, linewidth=LW, label=method)
    style(ax, ylabel)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

draw("shield_intervention_rate", "Shield intervention rate", "panel_a_intervention_rate", "(a)")
draw("mean_safe_action_set", r"Mean safe-action-set size $|\mathcal{A}^{safe}_t|$", "panel_b_safe_action_set", "(b)")
draw("admitted_computation_mbit", "Admitted computation load (Mbit)", "panel_c_admitted_computation", "(c)")
draw("value_per_computation_bit", "Deadline-valid value / admitted computation bit", "panel_d_value_efficiency", "(d)")

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
plots = [
    ("shield_intervention_rate", "Intervention rate", "(a)"),
    ("mean_safe_action_set", "Mean safe-set size", "(b)"),
    ("admitted_computation_mbit", "Admitted load (Mbit)", "(c)"),
    ("value_per_computation_bit", "Value / admitted bit", "(d)"),
]
for ax, (metric, ylabel, panel) in zip(axes.flat, plots):
    for method, vals in series.items():
        ax.plot(DENSITIES, [vals[d][metric] for d in DENSITIES], color=COLORS[method], marker=MARKERS[method], markersize=MS, linewidth=LW, label=method)
    style(ax, ylabel, font=24)

fig.suptitle("Radio-feasibility and computation-resource behavior of the final SPARC-V2X controller", fontsize=28, y=0.995)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=26, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.94], w_pad=3.0, h_pad=3.0)
fig.savefig(OUT / "radio_feasibility_computation_resources_4panel.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "radio_feasibility_computation_resources_4panel.pdf", bbox_inches="tight")
plt.close(fig)







