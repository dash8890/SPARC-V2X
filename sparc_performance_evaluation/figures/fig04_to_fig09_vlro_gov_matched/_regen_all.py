"""Regenerate Fig.4–9 with VLRO under governor-matched conditions.

Keeps VAS/LAS/LVM/SPARC-V2X from sealed Stage-5 / catchup artifacts.
Substitutes VLRO from results/raw/vlro_gov_matched_*.
Writes ONLY into this folder (png+pdf). Does not touch existing figure dirs.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
OUT = Path(__file__).resolve().parent
STAGE5 = ROOT / "results" / "stage5_no_residual" / "figures"
FS, LW = 50, 8
LW_SPARC = LW + 2

METHODS = [
    ("VAS", "#5b8db8", "^", (0, (6, 3)), False),
    ("LAS", "#7eb8a8", "v", (0, (2, 2)), False),
    ("LVM", "#e07a3d", "o", (0, (10, 3, 2, 3)), False),
    ("VLRO", "#3d8c5a", "s", (0, (1, 2)), False),
    ("SPARC-V2X", "#1f4e79", "D", "solid", True),
]
KEY = {
    "VAS": "value_aware",
    "LAS": "minfinish",
    "LVM": "b4_lagrangian",
    "VLRO": "itora_inspired",
    "SPARC-V2X": "direct",
}
DENS_MAIN = [20, 25, 30, 35, 40, 45, 50]
DENS_UPPER = [35, 40, 45, 50]


def mean_metric(folder: str, method: str, metric: str = "R_value") -> float:
    rows = json.loads((RAW / folder / "rows.json").read_text(encoding="utf-8-sig"))
    xs = [float(r[metric]) for r in rows if r.get("method") == method and r.get(metric) is not None]
    if not xs:
        raise ValueError(f"no {metric} for {method} in {folder}")
    return mean(xs)


def mean_pviol(folder: str, method: str) -> float:
    rows = json.loads((RAW / folder / "rows.json").read_text(encoding="utf-8-sig"))
    sel = [r for r in rows if r.get("method") == method]
    return mean(float(r.get("safety_P_viol") or 0.0) for r in sel)


def vlro_R(folder: str) -> float:
    return mean_metric(folder, "itora_inspired", "R_value")


def vlro_stats(folder: str) -> dict:
    rows = json.loads((RAW / folder / "rows.json").read_text(encoding="utf-8-sig"))
    sel = [r for r in rows if r.get("method") == "itora_inspired"]
    return {
        "R_value": mean(float(r["R_value"]) for r in sel),
        "P_viol": mean(float(r.get("safety_P_viol") or 0.0) for r in sel),
        "mean_prr": mean(float(r["safety_mean_prr_valid"]) for r in sel if r.get("safety_mean_prr_valid") is not None),
        "mean_p95": mean(float(r["safety_mean_p95_valid"]) for r in sel if r.get("safety_mean_p95_valid") is not None),
        "n": len(sel),
        "gov_intervened_mean": mean(float(r.get("governor_interventions") or 0) for r in sel),
        "source": folder,
    }


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png.name)


def plot_lines(xs, series: dict[str, list[float]], title, xlabel, ylabel, stem, ylim=None, legend_loc="best", legend_ncol=1):
    fig, ax = plt.subplots(figsize=(20, 18))
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
            solid_capstyle="round",
        )
    ax.set_title(title, fontsize=FS)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_xticks(xs)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc=legend_loc, ncol=legend_ncol)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, stem)


def main():
    # --- load sealed Stage-5 for non-VLRO baselines ---
    d4 = json.loads((STAGE5 / "fig04_data.json").read_text(encoding="utf-8"))
    d5 = json.loads((STAGE5 / "fig05_data.json").read_text(encoding="utf-8"))
    d7 = json.loads(
        (ROOT / "figures" / "fig07_radio_safety" / "fig07_data.json").read_text(encoding="utf-8")
        if (ROOT / "figures" / "fig07_radio_safety" / "fig07_data.json").exists()
        else (STAGE5 / "fig07_data.json").read_text(encoding="utf-8")
    )
    d6_old = json.loads(
        (ROOT / "figures" / "fig06_workload" / "fig06_all_baselines_series.json").read_text(encoding="utf-8")
    )
    d8_old = json.loads(
        (ROOT / "figures" / "fig08_candidate_set" / "fig08_data.json").read_text(encoding="utf-8")
    )

    # VLRO gov-matched substitutions
    vlro_dens = {d: vlro_stats(f"vlro_gov_matched_density_d{d}") for d in DENS_MAIN + [100]}
    vlro_env = {b: vlro_R(f"vlro_gov_matched_envelope_b{b}") for b in (100000, 175000, 250000, 350000)}
    vlro_load = {t: vlro_R(f"vlro_gov_matched_load_t{t}") for t in (2, 3, 4, 5, 6)}
    tags = {0.5: "0p5", 1.0: "1p0", 1.5: "1p5", 2.0: "2p0"}
    vlro_deadline = {dm: vlro_R(f"vlro_gov_matched_deadline_m{tags[dm]}") for dm in (0.5, 1.0, 1.5, 2.0)}
    vlro_k = {k: vlro_R(f"vlro_gov_matched_k{k}") for k in (8, 16, 32, 64)}

    meta = {
        "protocol": "VLRO governor-matched (same SafetyGovernor + B_max as SPARC-V2X)",
        "vlro_density": {str(d): vlro_dens[d] for d in sorted(vlro_dens)},
        "vlro_envelope_R": {str(b): vlro_env[b] for b in sorted(vlro_env)},
        "vlro_load_R": vlro_load,
        "vlro_deadline_R": {str(k): v for k, v in vlro_deadline.items()},
        "vlro_k_R": vlro_k,
    }
    (OUT / "vlro_gov_matched_plot_data.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # ========== Fig.4 ==========
    pa = d4["panel_a_R_value_vs_density"]
    pb = d4["panel_b_P_viol_vs_density"]
    full = d4["full"]

    def series_from_panel(panel, dens, metric_fn=None):
        out = {}
        for lab in ("VAS", "LAS", "LVM", "VLRO", "SPARC-V2X"):
            if lab == "VLRO" and metric_fn is not None:
                out[lab] = [metric_fn(d) for d in dens]
            else:
                out[lab] = [float(panel[str(d)][KEY[lab]]) for d in dens]
        return out

    plot_lines(
        DENS_MAIN,
        series_from_panel(pa, DENS_MAIN, lambda d: vlro_dens[d]["R_value"]),
        r"$R_{\mathrm{value}}$",
        r"Density $\rho$",
        r"$R_{\mathrm{value}}$",
        "fig04a_R_vs_density",
        legend_ncol=2,
    )
    plot_lines(
        DENS_MAIN,
        series_from_panel(pb, DENS_MAIN, lambda d: vlro_dens[d]["P_viol"]),
        r"$P_{\mathrm{viol}}$",
        r"Density $\rho$",
        r"$P_{\mathrm{viol}}$",
        "fig04b_Pviol_vs_density",
        legend_loc="lower right",
    )
    # 4c/4d from full + VLRO override
    s_prr = {}
    s_p95 = {}
    for lab in ("VAS", "LAS", "LVM", "VLRO", "SPARC-V2X"):
        if lab == "VLRO":
            s_prr[lab] = [vlro_dens[d]["mean_prr"] for d in DENS_UPPER]
            s_p95[lab] = [vlro_dens[d]["mean_p95"] for d in DENS_UPPER]
        else:
            s_prr[lab] = [full[str(d)][KEY[lab]]["mean_prr"] for d in DENS_UPPER]
            s_p95[lab] = [full[str(d)][KEY[lab]]["mean_p95"] for d in DENS_UPPER]
    plot_lines(DENS_UPPER, s_prr, "PRR", r"Density $\rho$", "PRR", "fig04c_prr_upper", legend_loc="upper right")
    ys = [v for vs in s_p95.values() for v in vs]
    plot_lines(
        DENS_UPPER,
        s_p95,
        "P95 latency",
        r"Density $\rho$",
        r"P95 latency (ms)",
        "fig04d_p95_upper",
        ylim=(min(ys) - 1.0, 30),
        legend_loc="upper left",
    )

    # ========== Fig.5 ==========
    env = d5["panel_envelope_sensitivity"]
    EB = [100000, 175000, 250000, 350000]
    EKB = [100, 175, 250, 350]
    s5 = {}
    for lab in ("VAS", "LAS", "LVM", "VLRO", "SPARC-V2X"):
        if lab == "VLRO":
            s5[lab] = [vlro_env[b] for b in EB]
        else:
            s5[lab] = [float(env[str(b)][KEY[lab]]["R"]) for b in EB]
    fig, ax = plt.subplots(figsize=(20, 18))
    for lab, col, marker, ls, is_sparc in METHODS:
        ax.plot(
            EKB, s5[lab], color=col,
            linewidth=LW_SPARC if is_sparc else LW, linestyle=ls,
            marker=marker, markersize=18 if is_sparc else 16,
            label=lab, zorder=5 if is_sparc else 2,
        )
    ax.set_title(r"$R_{\mathrm{value}}$ vs envelope", fontsize=FS)
    ax.set_xlabel(r"Envelope $E$ (kB)", fontsize=FS)
    ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
    ax.set_xticks(EKB)
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

    # 5b operating points: replace VLRO R with gov-matched @175kB dens35
    full35 = full["35"]
    ops = {row["method"]: row for row in d5["panel_operating_points"]}
    op_R = {
        "VAS": ops["B1_VA"]["R_value"],
        "LAS": ops["B2_MF"]["R_value"],
        "SPARC-V2X": ops["B3_Direct"]["R_value"],
        "LVM": ops["B4_Lagrangian"]["R_value"],
        "VLRO": vlro_dens[35]["R_value"],
    }
    p95 = {
        "VAS": full35["value_aware"]["mean_p95"],
        "LAS": full35["minfinish"]["mean_p95"],
        "SPARC-V2X": full35["direct"]["mean_p95"],
        "LVM": full35["b4_lagrangian"]["mean_p95"],
        "VLRO": vlro_dens[35]["mean_p95"],
    }
    style = {
        "VAS": ("#5b8db8", "^"),
        "LAS": ("#7eb8a8", "v"),
        "LVM": ("#e07a3d", "o"),
        "VLRO": ("#3d8c5a", "s"),
        "SPARC-V2X": ("#1f4e79", "D"),
    }
    MS = 2 * ((600 ** 0.5) + 3)
    SS = MS ** 2
    fig, ax = plt.subplots(figsize=(20, 18))
    for lab in ("VAS", "LAS", "LVM", "VLRO", "SPARC-V2X"):
        col, marker = style[lab]
        ax.scatter(
            [p95[lab]], [op_R[lab]], s=SS, color=col, marker=marker, label=lab,
            zorder=5 if lab == "SPARC-V2X" else 2, edgecolors="white", linewidths=2,
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

    # ========== Fig.6 ==========
    loads = d6_old["load"]["x"]
    s6a = {lab: list(d6_old["load"]["R"][lab]) for lab in ("VAS", "LAS", "LVM", "VLRO", "SPARC-V2X")}
    s6a["VLRO"] = [vlro_load[t] for t in loads]
    fig, ax = plt.subplots(figsize=(20, 18))
    for lab, col, marker, ls, is_sparc in METHODS:
        ax.plot(
            loads, s6a[lab], color=col,
            linewidth=LW_SPARC if is_sparc else LW, linestyle=ls,
            marker=marker, markersize=18 if is_sparc else 16,
            label=lab, zorder=5 if is_sparc else 2,
        )
    ax.set_title(r"$R_{\mathrm{value}}$ vs tasks/source", fontsize=FS)
    ax.set_xlabel("Tasks per source", fontsize=FS)
    ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
    ax.set_xticks(loads)
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

    dms = d6_old["deadline"]["x"]
    s6b = {lab: list(d6_old["deadline"]["R"][lab]) for lab in ("VAS", "LAS", "LVM", "VLRO", "SPARC-V2X")}
    s6b["VLRO"] = [vlro_deadline[dm] for dm in dms]
    plot_lines(
        dms, s6b,
        r"$R_{\mathrm{value}}$ vs deadline multiplier",
        "Deadline multiplier",
        r"$R_{\mathrm{value}}$",
        "fig06b_R_vs_deadline",
    )

    # ========== Fig.7 ==========
    pa7 = d7["panel_a_PRR_vs_density"]
    pb7 = d7["panel_b_P95_vs_density"]
    s7a = series_from_panel(pa7, DENS_MAIN, lambda d: vlro_dens[d]["mean_prr"])
    s7b = series_from_panel(pb7, DENS_MAIN, lambda d: vlro_dens[d]["mean_p95"])
    plot_lines(DENS_MAIN, s7a, "PRR", r"Density $\rho$", "PRR", "fig07a_PRR_vs_density")
    plot_lines(DENS_MAIN, s7b, "P95 latency", r"Density $\rho$", "P95 latency (ms)", "fig07b_P95_vs_density")

    # 7c: keep sealed curves for non-VLRO; aggregate VLRO dens35 gov-matched PRR-vs-distance
    curves = d7["panel_d_PRR_vs_distance"]["curves"]
    dens35_runs = RAW / "vlro_gov_matched_density_d35" / "runs"
    bucket: dict[float, list[float]] = {}
    if dens35_runs.exists():
        for ep in sorted(dens35_runs.glob("episode_*/bridge_audit.json")):
            audit = json.loads(ep.read_text(encoding="utf-8"))
            raw_curve = None
            for x in reversed(audit):
                if x.get("prr_vs_distance"):
                    raw_curve = x["prr_vs_distance"]
                    break
            if not raw_curve:
                continue
            for pt in raw_curve:
                if isinstance(pt, dict):
                    dist = float(pt.get("distance_m", pt.get("distance", 0)))
                    prr = float(pt.get("mean_prr", pt.get("prr", 0)))
                else:
                    continue
                bucket.setdefault(dist, []).append(prr)
    if bucket:
        vlro_curve = [
            {"distance_m": d, "mean_prr": mean(vs)}
            for d, vs in sorted(bucket.items())
        ]
    else:
        vlro_curve = curves.get("itora_inspired")  # fallback sealed

    # line
    fig, ax = plt.subplots(figsize=(20, 18))
    for lab, col, marker, ls, is_sparc in METHODS:
        key = KEY[lab]
        curve = vlro_curve if lab == "VLRO" else curves.get(key)
        if not curve:
            continue
        xs = [float(r["distance_m"]) for r in curve]
        ys = [float(r["mean_prr"]) for r in curve]
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

    DIST_REQ = [50, 100, 200, 300, 400]

    def nearest_prr(curve, target):
        best = min(curve, key=lambda r: abs(float(r["distance_m"]) - target))
        return float(best["distance_m"]), float(best["mean_prr"])

    n_grp, n_m = len(DIST_REQ), len(METHODS)
    x = list(range(n_grp))
    width = 0.15
    fig, ax = plt.subplots(figsize=(20, 18))
    tick_labels = []
    ref = curves.get("direct") or next(iter(curves.values()))
    for target in DIST_REQ:
        used, _ = nearest_prr(ref, float(target))
        tick_labels.append(str(int(used)))
    all_ys = []
    for j, (lab, col, _m, _ls, is_sparc) in enumerate(METHODS):
        key = KEY[lab]
        curve = vlro_curve if lab == "VLRO" else curves.get(key)
        ys = [nearest_prr(curve, float(t))[1] for t in DIST_REQ]
        all_ys.extend(ys)
        offsets = (j - (n_m - 1) / 2) * width
        ax.bar(
            [xi + offsets for xi in x], ys, width=width, color=col,
            edgecolor="black" if is_sparc else "none",
            linewidth=2 if is_sparc else 0,
            label=lab, zorder=5 if is_sparc else 2,
        )
    ax.set_title("PRR vs distance", fontsize=FS)
    ax.set_xlabel("Distance (m)", fontsize=FS)
    ax.set_ylabel("PRR", fontsize=FS)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(max(0.0, min(all_ys) - 0.05), 1.02)
    fig.tight_layout()
    save(fig, "fig07c_PRR_vs_distance")

    # ========== Fig.8 ==========
    KS = [8, 16, 32, 64]
    series8 = {lab: list(d8_old["R_value"][lab]) for lab in ("VAS", "LAS", "LVM", "VLRO", "SPARC-V2X")}
    series8["VLRO"] = [vlro_k[k] for k in KS]

    # sparc bars
    ys = series8["SPARC-V2X"]
    bar_colors = ["azure", "ivory", "honeydew", "seashell"]
    bar_hatches = ["/", "\\", "*", "+"]
    fig, ax = plt.subplots(figsize=(20, 18))
    for i, (yi, k) in enumerate(zip(ys, KS)):
        ax.bar(
            i, yi, color=bar_colors[i], hatch=bar_hatches[i], width=0.55,
            edgecolor="black", linewidth=2, zorder=3,
            label=r"$|\mathcal{C}_t|=%d$" % k,
        )
    ax.set_title(r"$R_{\mathrm{value}}$ vs $|\mathcal{C}_t|$", fontsize=FS)
    ax.set_xlabel(r"$|\mathcal{C}_t|$", fontsize=FS)
    ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
    ax.set_xticks([])
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="best", ncol=1)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(max(0.0, min(ys) - 0.05), max(ys) * 1.08)
    fig.tight_layout()
    save(fig, "fig08_R_vs_k_sparc")

    for stem in ("fig08_R_vs_k_all", "fig08_R_vs_k"):
        plot_lines(
            KS, series8,
            r"$R_{\mathrm{value}}$ vs $|\mathcal{C}_t|$",
            r"$|\mathcal{C}_t|$",
            r"$R_{\mathrm{value}}$",
            stem,
        )

    # ========== Fig.9 ==========
    # N vs S @ 35/50/100 (SPARC unchanged) + VLRO gov-matched overlays where applicable
    SOURCES = {
        35: ("catchup_gov_on_d35_direct", "catchup_gov_on_d35_direct"),
        50: ("catchup_fig11a_nominal_direct_on_d50", "catchup_fig11a_direct_d50_ft"),
        100: ("catchup_fig11a_nominal_direct_on_d100", "catchup_fig11a_direct_d100_ft"),
    }

    def sparc_stats(folder):
        rows = json.loads((RAW / folder / "rows.json").read_text(encoding="utf-8-sig"))
        sel = [r for r in rows if r.get("method") == "residual"]
        return {
            "R_value": mean(float(r["R_value"]) for r in sel),
            "P_viol": mean(float(r.get("safety_P_viol") or 0.0) for r in sel),
        }

    panel = {}
    for dens, (n_f, s_f) in SOURCES.items():
        panel[dens] = {"SPARC-N": sparc_stats(n_f), "SPARC-S": sparc_stats(s_f)}

    DENS = [35, 50, 100]
    COLORS = {"SPARC-N": "#5b8db8", "SPARC-S": "#1f4e79"}
    fig, ax = plt.subplots(figsize=(20, 18))
    x = list(range(len(DENS)))
    width = 0.32
    for j, lab in enumerate(("SPARC-N", "SPARC-S")):
        ys = [panel[d][lab]["R_value"] for d in DENS]
        offsets = (j - 0.5) * width
        ax.bar(
            [xi + offsets for xi in x], ys, width=width, color=COLORS[lab],
            edgecolor="black" if lab == "SPARC-S" else "none",
            linewidth=2 if lab == "SPARC-S" else 0,
            label=lab, zorder=3,
        )
    ax.set_title(r"Density adaptation ($R_{\mathrm{value}}$)", fontsize=FS)
    ax.set_xlabel(r"Density $\rho$", fontsize=FS)
    ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in DENS])
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="best")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ymax = max(panel[d][lab]["R_value"] for d in DENS for lab in ("SPARC-N", "SPARC-S"))
    ax.set_ylim(0.0, ymax * 1.2)
    for i, d in enumerate(DENS):
        for j, lab in enumerate(("SPARC-N", "SPARC-S")):
            yi = panel[d][lab]["R_value"]
            pviol = panel[d][lab]["P_viol"]
            ax.text(i + (j - 0.5) * width, yi + 0.012, "%.2f" % pviol,
                    ha="center", va="bottom", fontsize=FS - 14)
    ax.text(0.02, 0.98, r"$P_{\mathrm{viol}}$ above bars", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS - 16)
    fig.tight_layout()
    save(fig, "fig09_density_adaptation")

    # dens50 / dens100 three-bar with gov-matched VLRO
    for dens, s_lab, s_folder in (
        (50, "SPARC-S50", "catchup_fig11a_direct_d50_ft"),
        (100, "SPARC-S100", "catchup_fig11a_direct_d100_ft"),
    ):
        bars = {
            "SPARC-N": panel[dens]["SPARC-N"],
            s_lab: sparc_stats(s_folder),
            "VLRO": vlro_dens[dens],
        }
        labels = ["SPARC-N", s_lab, "VLRO"]
        colors = ["#5b8db8", "#1f4e79", "#3d8c5a"]
        ys = [bars[lab]["R_value"] for lab in labels]
        pviol = bars["SPARC-N"]["P_viol"]
        fig, ax = plt.subplots(figsize=(20, 18))
        xx = list(range(len(labels)))
        ax.bar(xx, ys, color=colors, width=0.55, zorder=3)
        ax.bar([1], [ys[1]], color=colors[1], width=0.55, edgecolor="black", linewidth=3, zorder=4)
        ax.set_title(rf"Density-{dens} adaptation ($R_{{\mathrm{{value}}}}$)", fontsize=FS)
        ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
        ax.set_xticks(xx)
        ax.set_xticklabels(labels)
        ax.tick_params(labelsize=FS)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0.0, max(ys) * 1.28)
        for i in (0, 1):
            ax.text(i, ys[i] + 0.012, r"$P_{\mathrm{viol}}=%.2f$" % pviol,
                    ha="center", va="bottom", fontsize=FS - 8)
        fig.tight_layout()
        stem = f"fig09_density{dens}_adaptation" if dens != 50 else "fig09_density50_adaptation"
        save(fig, stem)

    # combined dens 50/100 with VLRO gov-matched
    dens_list = [50, 100]
    COLORS3 = {"SPARC-N": "#5b8db8", "SPARC-S": "#1f4e79", "VLRO": "#3d8c5a"}
    fig, ax = plt.subplots(figsize=(20, 18))
    width = 0.22
    xx = list(range(len(dens_list)))
    series_keys = [
        ("SPARC-N", lambda d: panel[d]["SPARC-N"]["R_value"]),
        ("SPARC-S", lambda d: panel[d]["SPARC-S"]["R_value"]),
        ("VLRO", lambda d: vlro_dens[d]["R_value"]),
    ]
    for j, (lab, getter) in enumerate(series_keys):
        ys = [getter(d) for d in dens_list]
        offsets = (j - 1) * width
        ax.bar(
            [xi + offsets for xi in xx], ys, width=width, color=COLORS3[lab],
            edgecolor="black" if lab == "SPARC-S" else "none",
            linewidth=2 if lab == "SPARC-S" else 0,
            label=lab, zorder=3,
        )
    ax.set_title(r"High-density adaptation ($R_{\mathrm{value}}$)", fontsize=FS)
    ax.set_xlabel(r"Density $\rho$", fontsize=FS)
    ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
    ax.set_xticks(xx)
    ax.set_xticklabels([str(d) for d in dens_list])
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig09_density_adaptation_combined")

    # SPARC-N only
    BAR_COLORS = ["azure", "ivory", "honeydew"]
    BAR_HATCHES = ["/", "\\", "*"]
    ys = [panel[d]["SPARC-N"]["R_value"] for d in DENS]
    fig, ax = plt.subplots(figsize=(20, 18))
    for i, (yi, d) in enumerate(zip(ys, DENS)):
        ax.bar(
            i, yi, color=BAR_COLORS[i], hatch=BAR_HATCHES[i], width=0.55,
            edgecolor="black", linewidth=2, zorder=3, label=r"$\rho=%d$" % d,
        )
    ax.set_title(r"SPARC-N ($R_{\mathrm{value}}$)", fontsize=FS)
    ax.set_xlabel(r"Density $\rho$", fontsize=FS)
    ax.set_ylabel(r"$R_{\mathrm{value}}$", fontsize=FS)
    ax.set_xticks([])
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="best")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0.0, max(ys) * 1.2)
    for i, d in enumerate(DENS):
        ax.text(i, ys[i] + 0.015, r"$P_{\mathrm{viol}}=%.2f$" % panel[d]["SPARC-N"]["P_viol"],
                ha="center", va="bottom", fontsize=FS - 8)
    fig.tight_layout()
    save(fig, "fig09_sparc_N_dens_35_50_100")

    # S100 episode return (unchanged SPARC training curve)
    repo = ROOT.parent
    met_path = repo / "results" / "wilab_catchup_direct_d100" / "training_metrics.json"
    rows = json.loads(met_path.read_text(encoding="utf-8"))
    ep = np.array([r["episode"] for r in rows], dtype=float)
    ret = np.array([r["return"] for r in rows], dtype=float)
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

    pngs = sorted(OUT.glob("fig*.png"))
    print(f"done png={len(pngs)}")
    for p in pngs:
        print(" ", p.name)


if __name__ == "__main__":
    main()
