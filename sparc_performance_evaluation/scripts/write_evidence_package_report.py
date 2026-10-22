"""Generate a faithful SPARC-V2X evidence package report from locked + improve artifacts.

Does not mutate raw episode outcomes. Safe to re-run.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "sparc_performance_evaluation"
RAW = PKG / "results" / "raw"
AGG = PKG / "results" / "aggregates"
OUT_MD = PKG / "reports" / "EVIDENCE_PACKAGE_REPORT.md"
OUT_JSON = AGG / "evidence_package_report.json"
IMPROVE_DONE = AGG / "improve_aggregate_DONE"
CAMPAIGN_DONE = AGG / "campaign_aggregate_DONE"
EPS = 1e-6

IMPROVE_LABELS = {
    "improve_i1_high_beta_d35": "I1 high-beta (gain=3)",
    "improve_i2_residual_only_d35": "I2 residual_only",
    "improve_i3_changed_only_d35": "I3 changed-only",
    "improve_i4_high_beta_changed_d35": "I4 high-beta + changed-only",
    "improve_c1_slow_decay_d35": "C1 slow LR/entropy decay",
    "improve_c2_more_rollouts_entropy_d35": "C2 more rollouts + entropy",
    "improve_c3_ema_selection_d35": "C3 EMA delta-VA selection",
    "improve_c4_mean_v_selection_d35": "C4 mean-V selection",
    "improve_x1_high_beta_ema_d35": "X1 high-beta + EMA + slow decay",
    "improve_x2_residual_only_ema_d35": "X2 residual_only + EMA + slow decay",
}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=2):
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def load_rows(name: str):
    p = RAW / name / "rows.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def method_stats(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    out = {}
    for m, vals in by.items():
        out[m] = {
            "mean_V": mean([float(v["completed_value"]) for v in vals]),
            "mean_R": mean([float(v["R_value"]) for v in vals]),
            "mean_P_viol": mean([float(v.get("safety_P_viol") or 0) for v in vals]),
            "mean_P_prr_fail": mean([float(v.get("safety_P_prr_fail") or 0) for v in vals]),
            "mean_P_p95_fail": mean([float(v.get("safety_P_p95_fail") or 0) for v in vals]),
            "mean_invalid_frac": mean([float(v.get("safety_invalid_frac") or 0) for v in vals]),
            "mean_prr": mean([
                float(v["safety_mean_prr_valid"]) for v in vals
                if v.get("safety_mean_prr_valid") is not None
            ]),
            "mean_p95": mean([
                float(v["safety_mean_p95_valid"]) for v in vals
                if v.get("safety_mean_p95_valid") is not None
            ]),
            "frac_compliant": mean([1.0 if v.get("safety_compliant") else 0.0 for v in vals]),
            "mean_offered_bytes_epoch": mean([float(v.get("mean_offered_bytes") or 0) for v in vals]),
            "mean_sum_offered_bytes": mean([float(v.get("sum_offered_bytes") or 0) for v in vals]),
            "mean_gate_blocks": mean([float(v.get("governor_gate_block_epochs") or 0) for v in vals]),
            "mean_gov_interventions": mean([float(v.get("governor_interventions") or 0) for v in vals]),
            "n": len(vals),
        }
    return out


def paired_delta(a_by_seed, b_by_seed):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if not seeds:
        return None
    diffs = [a_by_seed[s] - b_by_seed[s] for s in seeds]
    return {
        "n": len(seeds),
        "mean_delta": mean(diffs),
        "min": min(diffs),
        "max": max(diffs),
        "wins": sum(1 for d in diffs if d > EPS),
        "ties": sum(1 for d in diffs if abs(d) < EPS),
        "losses": sum(1 for d in diffs if d < -EPS),
    }


def by_seed_V(rows, method=None):
    out = {}
    for r in rows:
        if method is not None and r["method"] != method:
            continue
        out[r["sim_seed"]] = float(r["completed_value"])
    return out


def train_curve_stats(name: str):
    p = ROOT / "results" / name / "training_metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    ret = [float(r["return"]) for r in m]
    if not ret:
        return None
    roll16 = [mean(ret[max(0, i - 15): i + 1]) for i in range(len(ret))]
    return {
        "n": len(ret),
        "mean_return": mean(ret),
        "roll16_end": roll16[-1],
        "half1": mean(ret[: len(ret) // 2]),
        "half2": mean(ret[len(ret) // 2:]),
        "zero_frac": sum(1 for x in ret if abs(x) < EPS) / len(ret),
    }


def improve_entry(folder: Path, va_by_seed: dict, frozen: dict):
    rows_path = folder / "rows.json"
    if not rows_path.exists():
        return None
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    vals = [float(r["completed_value"]) for r in rows]
    m = mean(vals)
    collapse = 0
    per_seed = []
    for r in rows:
        seed = r["sim_seed"]
        v = float(r["completed_value"])
        va = va_by_seed.get(seed)
        eq = va is not None and abs(v - va) < EPS
        if eq:
            collapse += 1
        per_seed.append({
            "sim_seed": seed, "V": v,
            "vs_va": None if va is None else v - va,
            "equals_va": eq,
        })
    beats_direct = m is not None and m > frozen["direct"]["mean_V"]
    stage = None
    done_path = folder / "DONE"
    if done_path.exists():
        try:
            stage = json.loads(done_path.read_text(encoding="utf-8")).get("stage")
        except Exception:
            stage = "done"
    return {
        "folder": folder.name,
        "label": IMPROVE_LABELS.get(folder.name, folder.name),
        "complete": done_path.exists() and len(vals) >= 5,
        "stage": stage,
        "mean_V": m,
        "n": len(vals),
        "vs_sparc": None if m is None else m - frozen["sparc"]["mean_V"],
        "vs_direct": None if m is None else m - frozen["direct"]["mean_V"],
        "va_collapse_seeds": collapse,
        "success": bool(beats_direct and (len(vals) - collapse) >= 4),
        "per_seed": per_seed,
        "mean_P_viol": mean([float(r.get("safety_P_viol") or 0) for r in rows]),
    }


def build_payload():
    nom = load_rows("final_test_nominal_d35") or []
    direct_rows = load_rows("final_test_direct_d35") or []
    b4_rows = load_rows("final_test_b4_d35") or []
    itora_rows = load_rows("final_test_itora_d35") or []

    nom_s = method_stats(nom) if nom else {}
    direct_s = method_stats(direct_rows) if direct_rows else {}
    b4_s = method_stats(b4_rows) if b4_rows else {}
    itora_s = method_stats(itora_rows) if itora_rows else {}

    va_by = by_seed_V(nom, "value_aware")
    mf_by = by_seed_V(nom, "minfinish")
    sparc_by = by_seed_V(nom, "residual")
    direct_by = by_seed_V(direct_rows)
    lag_by = by_seed_V(b4_rows)
    itora_by = by_seed_V(itora_rows)

    frozen = {
        "va": nom_s.get("value_aware", {}),
        "mf": nom_s.get("minfinish", {}),
        "sparc": nom_s.get("residual", {}),
        "direct": direct_s.get("residual", {}),
        "lagrangian": b4_s.get("residual", {}),
        "itora": itora_s.get("itora_inspired", {}),
        "sparc_va_collapse_seeds": sum(
            1 for s in sparc_by if s in va_by and abs(sparc_by[s] - va_by[s]) < EPS
        ),
    }

    per_seed = []
    for s in sorted(set(va_by) | set(sparc_by) | set(direct_by)):
        per_seed.append({
            "sim_seed": s,
            "VA": va_by.get(s), "MF": mf_by.get(s), "SPARC": sparc_by.get(s),
            "Direct": direct_by.get(s), "Lagrangian": lag_by.get(s), "ITORA": itora_by.get(s),
        })

    contrasts = {
        "sparc_minus_va": paired_delta(sparc_by, va_by),
        "sparc_minus_mf": paired_delta(sparc_by, mf_by),
        "sparc_minus_direct": paired_delta(sparc_by, direct_by),
        "direct_minus_sparc": paired_delta(direct_by, sparc_by),
        "direct_minus_va": paired_delta(direct_by, va_by),
        "lag_minus_direct": paired_delta(lag_by, direct_by),
        "lag_minus_sparc": paired_delta(lag_by, sparc_by),
        "itora_minus_sparc": paired_delta(itora_by, sparc_by),
    }

    ablations = {}
    for name, label, admission in [
        ("final_test_a1_d35", "A1 without PRR/P95 gates (byte envelope retained)", "envelope_no_gates"),
        ("final_test_a3_d35", "A3 without VA-relative PPO", "common_governor"),
        ("final_test_a5_d35", "A5 residual-only logits + VA-relative advantages", "common_governor"),
        ("final_test_a6_d35", "A6 random residual init", "common_governor"),
    ]:
        rows = load_rows(name)
        if rows:
            st = method_stats(rows).get("residual", {})
            ablations[name] = {"label": label, "admission": admission, **st}

    density = {}
    for d in (25, 35, 50, 80, 100):
        rows = load_rows(f"density_d{d}")
        drows = load_rows(f"density_direct_d{d}")
        entry = {}
        if rows:
            entry.update(method_stats(rows))
        if drows:
            entry["direct"] = method_stats(drows).get("residual", {})
        if entry:
            density[str(d)] = entry

    envelope = {}
    for b in (100000, 175000, 250000, 350000):
        rows = load_rows(f"envelope_b{b}")
        if rows:
            envelope[str(b)] = method_stats(rows)

    improve = {}
    for p in sorted(RAW.glob("improve_*_d35")):
        entry = improve_entry(p, va_by, frozen)
        if entry:
            improve[p.name] = entry

    train_curves = {}
    for name, label in [
        ("wilab_residual_ppo_96ep_va_relative", "SPARC locked"),
        ("wilab_direct_ppo_96ep", "Direct locked"),
        ("wilab_lagrangian_ppo_96ep", "B4 Lagrangian"),
        ("wilab_improve_i1_high_beta", "I1"),
        ("wilab_improve_i2_residual_only", "I2"),
        ("wilab_improve_i3_changed_only", "I3"),
        ("wilab_improve_i4_high_beta_changed", "I4"),
    ]:
        s = train_curve_stats(name)
        if s:
            train_curves[label] = s

    table2 = load_json(PKG / "tables" / "table_ii_parameters" / "table_ii.json") or {}
    table4 = load_json(PKG / "tables" / "table_iv_complexity" / "table_iv.json") or {}
    gates = load_json(PKG / "verification" / "gate_results.json") or []
    wilab_audit = load_json(PKG / "verification" / "wilab_trace_audit.json") or []
    status = load_json(RAW / "improve_status.json") or []

    winners = [k for k, v in improve.items() if v.get("success")]
    best_improve = None
    complete = [v for v in improve.values() if v.get("n", 0) >= 5]
    if complete:
        best_improve = max(complete, key=lambda v: v["mean_V"] or -1e9)

    # Honest lock / figure gap inventory from LOCK_CRITERIA + what exists
    gaps = {
        "present": [
            "B1/B2/B3/B4/B5 nominal final-test (n=5)",
            "Ablations A0-A6 (A2/A4 reuse)",
            "Density E2 for VA/MF/SPARC/ITORA + separate Direct density_*",
            "Envelope E3 for VA/MF/SPARC",
            "Table III campaign merge + Table IV stub",
            "Seed seal n=5; campaign_aggregate_DONE",
            "P0 offered-byte backfill from bridge_audit",
        ],
        "missing_or_partial": [
            "E2 density does NOT include B4 Lagrangian at any density",
            "E2 main job excludes Direct (exists only as density_direct_d*)",
            "E4 workload/relative-deadline sensitivity - not run",
            "E5 Figs 7-8 radio diagnostics package - not assembled",
            "E6 Fig 9 ablation CIs / formal panels - numbers only",
            "E7 Fig 10 governor enforcement cost - not assembled",
            "E8 Fig 11 generalization - not assembled",
            "Simultaneous multi-contrast inference - not sealed",
            "protocol_hashes lock.status=LOCKED - not set",
            "G1/G4/G8/G9 still require/partial runtime audit entries in gate_results.json",
        ],
        "interpretation_holds": [
            "B4/B5 are unshielded or differently admitted; not governor-matched to SPARC/Direct",
            "At nominal d=35, PRR/P95 gates never fired; 175 kB envelope is binding (see P0_P3 audit)",
            "ITORA density V wins at rho=80 coincide with P_viol~1 (governor disabled)",
            "B4 eligibility audit: INCONCLUSIVE for replacing Direct (see B4 report)",
        ],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_status": "PARTIAL_NOT_LOCKABLE",
        "campaign_aggregate_done": CAMPAIGN_DONE.exists(),
        "improve_aggregate_done": IMPROVE_DONE.exists(),
        "frozen_nominal": frozen,
        "per_seed_nominal": per_seed,
        "paired_contrasts": contrasts,
        "ablations": ablations,
        "density": density,
        "envelope": envelope,
        "table_ii": table2,
        "table_iv": table4,
        "gates": gates,
        "wilab_trace_audit": wilab_audit,
        "train_curves": train_curves,
        "improve": improve,
        "improve_winners": winners,
        "best_improve": best_improve,
        "improve_queue_status": status,
        "gaps": gaps,
        "success_criteria_improve": {
            "mean_V_gt_frozen_direct": frozen["direct"].get("mean_V"),
            "non_collapse_seeds_ge": 4,
        },
        "cross_reports": [
            "reports/P0_P3_EVIDENCE_AUDIT.md",
            "reports/B4_LAGRANGIAN_ELIGIBILITY_AUDIT.md",
            "protocol/b5_itora_mapping_audit.md",
            "protocol/LOCK_CRITERIA.md",
        ],
    }


def render_md(p: dict) -> str:
    f = p["frozen_nominal"]
    L = []

    def h(n, t):
        L.append("")
        L.append("#" * n + " " + t)
        L.append("")

    L.append("# SPARC-V2X Evidence Package Report")
    L.append("")
    L.append(f"**Generated (UTC):** `{p['generated_at']}`")
    L.append(f"**Package status:** `{p['package_status']}` "
             f"(campaign aggregate `{'DONE' if p['campaign_aggregate_done'] else 'PENDING'}`; "
             f"improve aggregate `{'DONE' if p['improve_aggregate_done'] else 'IN_PROGRESS/PARTIAL'}`)")
    L.append("")
    L.append(
        "This report is assembled only from sealed/locked raw rows and audits. "
        "It separates **governor-matched** methods (SPARC, B1-B3, A0/A2-A6 under common 175 kB+PRR/P95) "
        "from **unshielded / differently admitted** methods (B4 Lagrangian 500 kB no governor; "
        "B5/ITORA governor disabled). Ranking claims must respect that split."
    )
    L.append("")
    L.append("Companion audits: " + ", ".join(f"`{x}`" for x in p["cross_reports"]))

    # ---- 1 Claim hierarchy ----
    h(2, "1. Claim hierarchy (faithful reading)")
    L.append("| Tier | Methods | Admission contract | How to use in manuscript |")
    L.append("|---|---|---|---|")
    L.append("| Governor-matched | SPARC A0, B1 VA, B2 MF, B3 Direct, A3/A5/A6 | Common SafetyGovernor: PRR>=0.95, P95<=50 ms, 175 kB envelope | Primary value/safety comparisons |")
    L.append("| Gate ablation | A1 | Envelope retained; PRR/P95/validity gates open | Isolates gate effect |")
    L.append("| Unshielded learned | B4 Lagrangian | No governor; physical_byte_cap=500 kB; soft dual kappa=0.05 | System-level alternative; **not** ranked vs Direct under SPARC shield |")
    L.append("| Unshielded heuristic | B5 ITORA-inspired | `gov=None` in eval (byte budget only) | Literature-inspired; report with P_viol |")
    L.append("")
    gov_v = {
        "SPARC (A0)": f["sparc"].get("mean_V"),
        "B1 VA": f["va"].get("mean_V"),
        "B2 MF": f["mf"].get("mean_V"),
        "B3 Direct": f["direct"].get("mean_V"),
    }
    L.append("**Governor-matched nominal ranking by mean V_done:**")
    for name, v in sorted(gov_v.items(), key=lambda kv: -(kv[1] or -1)):
        L.append(f"- {name}: **{fmt(v)}**")
    L.append("")
    L.append(
        f"Direct leads governor-matched learned methods "
        f"(Direct {fmt(f['direct'].get('mean_V'))} vs SPARC {fmt(f['sparc'].get('mean_V'))}, "
        f"gap {fmt((f['direct'].get('mean_V') or 0) - (f['sparc'].get('mean_V') or 0))}). "
        f"B4 mean V={fmt(f['lagrangian'].get('mean_V'))} and ITORA={fmt(f['itora'].get('mean_V'))} "
        "are **out of this ranking**."
    )
    L.append("")
    L.append(
        f"SPARC VA-collapse seeds (exact V equality): "
        f"**{f.get('sparc_va_collapse_seeds', '-')}**/5."
    )
    a5 = p["ablations"].get("final_test_a5_d35")
    a6 = p["ablations"].get("final_test_a6_d35")
    dv = f["direct"].get("mean_V")
    if a5 and dv is not None:
        L.append(
            f"Ablation A5 residual-only mean V={fmt(a5.get('mean_V'))} "
            f"({'beats' if (a5.get('mean_V') or 0) > dv else 'does not beat'} Direct) "
            "under the **common governor** - relevant to SPARC design, not a new baseline."
        )
    if a6 and dv is not None:
        L.append(
            f"Ablation A6 random-init mean V={fmt(a6.get('mean_V'))} "
            f"({'beats' if (a6.get('mean_V') or 0) > dv else 'does not beat'} Direct) "
            "under the common governor."
        )

    # ---- 2 Lock status ----
    h(2, "2. Lock status and completeness")
    L.append(f"Per `protocol/LOCK_CRITERIA.md`, status remains **PARTIAL / not lockable**.")
    L.append("")
    L.append("### Present")
    for x in p["gaps"]["present"]:
        L.append(f"- {x}")
    L.append("")
    L.append("### Missing or partial (blocks LOCKED)")
    for x in p["gaps"]["missing_or_partial"]:
        L.append(f"- {x}")
    L.append("")
    L.append("### Interpretation holds from audits (do not contradict in tables)")
    for x in p["gaps"]["interpretation_holds"]:
        L.append(f"- {x}")
    L.append("")
    L.append("### Verification gates (`verification/gate_results.json` + wilab audit)")
    L.append("")
    L.append("| Gate | pass/status |")
    L.append("|---|---|")
    for g in p.get("gates") or []:
        st = g.get("pass")
        if st is None:
            st = g.get("status")
        L.append(f"| {g.get('gate')} | {st} |")
    for g in p.get("wilab_trace_audit") or []:
        L.append(f"| {g.get('gate')} (wilab audit) | {g.get('pass')} / {g.get('status', '')} |")

    # ---- 3 Table II ----
    h(2, "3. Table II - parameters (sourced)")
    t2 = p.get("table_ii") or {}
    values = t2.get("values") or t2.get("parameters") or {}
    prefer = [
        "density", "controller_epochs", "tasks_per_source", "simulation_time",
        "envelope_bytes", "safety_payload_bytes", "episodes", "rollouts_per_update",
        "ppo_epochs", "minibatch_size", "lr", "logit_scale", "residual_gain",
        "entropy_coef", "target_kl", "va_relative",
    ]
    L.append("| Parameter | Value |")
    L.append("|---|---|")
    if isinstance(values, dict) and values:
        seen = set()
        for k in prefer:
            if k in values:
                L.append(f"| {k} | {values[k]} |")
                seen.add(k)
        for k, v in values.items():
            if k in seen or isinstance(v, (dict, list)):
                continue
            L.append(f"| {k} | {v} |")
    else:
        L.append("| density | 35 |")
        L.append("| envelope_bytes | 175000 |")
        L.append("| episodes | 96 |")
        L.append("| safety_payload_bytes | 300 |")
        L.append("| simulation_time | 12.0 |")
    L.append("")
    L.append(
        "Source: `tables/table_ii_parameters/table_ii.json`. "
        "Nominal envelope_bytes=175000; B4 FT uses physical_byte_cap=500000 (not this row)."
    )

    # ---- 4 Table III ----
    h(2, "4. Table III - nominal final-test (density=35, n=5 sealed seeds)")
    L.append(
        "Seeds: sim `93000`-`93004`. "
        "`R_value` = completed_value / offered_value (task-value ratio). "
        "Offered **bytes** are backfilled from `bridge_audit.json` "
        "(P0 fix: trace key mismatch had previously forced zeros)."
    )
    L.append("")
    L.append("### 4.1 Governor-matched methods")
    L.append("")
    L.append(
        "| Method | V_done | R_value | sum offered B | mean epoch offered B | "
        "P_viol | mean PRR | mean P95 ms | invalid frac | gate-block epochs |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, key in [
        ("SPARC-V2X (A0)", "sparc"),
        ("B1 Corrected VA", "va"),
        ("B2 Min-Finish", "mf"),
        ("B3 Direct PPO", "direct"),
    ]:
        m = f[key]
        if not m:
            continue
        L.append(
            f"| {label} | {fmt(m.get('mean_V'))} | {fmt(m.get('mean_R'), 4)} | "
            f"{fmt(m.get('mean_sum_offered_bytes'), 0)} | {fmt(m.get('mean_offered_bytes_epoch'), 0)} | "
            f"{fmt(m.get('mean_P_viol'), 4)} | {fmt(m.get('mean_prr'), 4)} | "
            f"{fmt(m.get('mean_p95'), 2)} | {fmt(m.get('mean_invalid_frac'), 4)} | "
            f"{fmt(m.get('mean_gate_blocks'), 1)} |"
        )
    L.append("")
    L.append("### 4.2 Unshielded / differently admitted (do not merge into matched ranking)")
    L.append("")
    L.append(
        "| Method | V_done | R_value | sum offered B | P_viol | Admission |"
    )
    L.append("|---|---:|---:|---:|---:|---|")
    L.append(
        f"| B4 Lagrangian | {fmt(f['lagrangian'].get('mean_V'))} | "
        f"{fmt(f['lagrangian'].get('mean_R'), 4)} | "
        f"{fmt(f['lagrangian'].get('mean_sum_offered_bytes'), 0)} | "
        f"{fmt(f['lagrangian'].get('mean_P_viol'), 4)} | "
        "no governor; 500 kB physical cap; soft dual |"
    )
    L.append(
        f"| B5 ITORA-inspired | {fmt(f['itora'].get('mean_V'))} | "
        f"{fmt(f['itora'].get('mean_R'), 4)} | "
        f"{fmt(f['itora'].get('mean_sum_offered_bytes'), 0)} | "
        f"{fmt(f['itora'].get('mean_P_viol'), 4)} | "
        "gov=None in run_paired_eval; see mapping audit |"
    )
    L.append("")
    L.append(
        "At nominal density, B4/B5 also report P_viol=0, but that does **not** make them "
        "governor-matched: B4 admits ~26% more computation bytes than Direct and exceeds "
        "175 kB on 40% of positive epochs (B4 eligibility audit)."
    )

    h(3, "4.3 Per-seed completed value")
    L.append("| sim_seed | VA | MF | SPARC | Direct | B4 Lag | ITORA |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in p["per_seed_nominal"]:
        L.append(
            f"| {row['sim_seed']} | {fmt(row.get('VA'))} | {fmt(row.get('MF'))} | "
            f"{fmt(row.get('SPARC'))} | {fmt(row.get('Direct'))} | "
            f"{fmt(row.get('Lagrangian'))} | {fmt(row.get('ITORA'))} |"
        )

    h(3, "4.4 Paired contrasts (completed value)")
    L.append("| Contrast | mean dV | min | max | wins | ties | losses | n | Tier |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for key, label, tier in [
        ("sparc_minus_va", "SPARC - VA", "matched"),
        ("sparc_minus_mf", "SPARC - MF", "matched"),
        ("sparc_minus_direct", "SPARC - Direct", "matched"),
        ("direct_minus_sparc", "Direct - SPARC", "matched"),
        ("direct_minus_va", "Direct - VA", "matched"),
        ("lag_minus_direct", "B4 - Direct", "cross-admission"),
        ("lag_minus_sparc", "B4 - SPARC", "cross-admission"),
        ("itora_minus_sparc", "ITORA - SPARC", "cross-admission"),
    ]:
        c = p["paired_contrasts"].get(key)
        if not c:
            continue
        L.append(
            f"| {label} | {fmt(c['mean_delta'])} | {fmt(c['min'])} | {fmt(c['max'])} | "
            f"{c['wins']} | {c['ties']} | {c['losses']} | {c['n']} | {tier} |"
        )

    # ---- 5 Ablations ----
    h(2, "5. Ablations (Fig. 9 family)")
    L.append("| ID | Configuration | mean V | mean R | P_viol | sum offered B | Admission |")
    L.append("|---|---|---:|---:|---:|---:|---|")
    L.append(
        f"| A0 | Full SPARC (reuse nominal) | {fmt(f['sparc'].get('mean_V'))} | "
        f"{fmt(f['sparc'].get('mean_R'), 4)} | {fmt(f['sparc'].get('mean_P_viol'), 4)} | "
        f"{fmt(f['sparc'].get('mean_sum_offered_bytes'), 0)} | common_governor |"
    )
    for _, a in p["ablations"].items():
        L.append(
            f"| | {a['label']} | {fmt(a.get('mean_V'))} | {fmt(a.get('mean_R'), 4)} | "
            f"{fmt(a.get('mean_P_viol'), 4)} | {fmt(a.get('mean_sum_offered_bytes'), 0)} | "
            f"{a.get('admission')} |"
        )
    L.append(
        f"| A2 | = B1 VA (reuse) | {fmt(f['va'].get('mean_V'))} | "
        f"{fmt(f['va'].get('mean_R'), 4)} | {fmt(f['va'].get('mean_P_viol'), 4)} | "
        f"{fmt(f['va'].get('mean_sum_offered_bytes'), 0)} | common_governor |"
    )
    L.append(
        f"| A4 | = B3 Direct (reuse) | {fmt(f['direct'].get('mean_V'))} | "
        f"{fmt(f['direct'].get('mean_R'), 4)} | {fmt(f['direct'].get('mean_P_viol'), 4)} | "
        f"{fmt(f['direct'].get('mean_sum_offered_bytes'), 0)} | common_governor |"
    )
    L.append("")
    L.append(
        "Note (P1): A1 equals A0 at nominal because PRR/P95 gates never fired at d=35; "
        "removing them is a no-op under these radio conditions."
    )

    # ---- 6 Density ----
    h(2, "6. Density sweep E2 (value + safety)")
    L.append(
        "Main job methods: VA, MF, SPARC, ITORA. Direct from `density_direct_d*`. "
        "**B4 Lagrangian has no density results.** "
        "ITORA runs with governor disabled even inside the density job."
    )
    L.append("")
    L.append(
        "| density | VA V / P_viol | MF V / P_viol | SPARC V / P_viol | "
        "Direct V / P_viol | ITORA V / P_viol |"
    )
    L.append("|---:|---:|---:|---:|---:|---:|")
    for d in sorted(p["density"], key=lambda x: int(x)):
        m = p["density"][d]

        def cell(key):
            st = m.get(key) or {}
            if not st:
                return "- / -"
            return f"{fmt(st.get('mean_V'))} / {fmt(st.get('mean_P_viol'), 3)}"

        L.append(
            f"| {d} | {cell('value_aware')} | {cell('minfinish')} | {cell('residual')} | "
            f"{cell('direct')} | {cell('itora_inspired')} |"
        )
    L.append("")
    L.append(
        "At rho=80, SPARC/VA/MF/Direct collapse (V~3-4) with P_viol~0.98 under PRR gating, "
        "while ITORA reports V=55.83 with P_viol~0.998 - not a feasible win "
        "(see `reports/P0_P3_EVIDENCE_AUDIT.md` P3)."
    )

    # ---- 7 Envelope ----
    h(2, "7. Byte-envelope sensitivity E3")
    L.append("Methods: VA, MF, SPARC only (Direct/B4/ITORA not in envelope sweep).")
    L.append("")
    L.append("| envelope B | VA V | MF V | SPARC V | SPARC P_viol |")
    L.append("|---:|---:|---:|---:|---:|")
    for b in sorted(p["envelope"], key=lambda x: int(x)):
        m = p["envelope"][b]
        L.append(
            f"| {b} | {fmt(m.get('value_aware', {}).get('mean_V'))} | "
            f"{fmt(m.get('minfinish', {}).get('mean_V'))} | "
            f"{fmt(m.get('residual', {}).get('mean_V'))} | "
            f"{fmt(m.get('residual', {}).get('mean_P_viol'), 4)} |"
        )

    # ---- 8 Safety / governor ----
    h(2, "8. Safety and governor operation (nominal)")
    L.append(
        "From P0/P1 audit of `bridge_audit.json`: at d=35, "
        "PRR_min stays above 0.95 and P95_max below 50 ms for governor-matched FT - "
        "**zero PRR/P95 interventions**. Binding restriction is the **175 kB envelope** "
        "(positive-action epochs reach ~174 kB; proposed==executed). "
        "`governor_interventions` undercounts gate activity when the actor already proposes empty "
        "after `admission_allowed=False`; use `governor_gate_block_epochs` from reason codes."
    )
    L.append("")
    L.append(
        "Offered-byte Table III zeros were a **reporting bug** (trace key `bytes` vs "
        "`computation_bytes_offered`), not phantom completions. Backfilled from audits; "
        "V_done unchanged."
    )

    # ---- 9 B4 / B5 ----
    h(2, "9. B4 / B5 eligibility notes")
    L.append(
        "- **B4:** Eligibility audit verdict **INCONCLUSIVE** for replacing Direct/SPARC. "
        "Implements soft dual (kappa=0.05), not hard SPARC constraints; "
        "+4.71 V vs Direct with +26% bytes and **lower** V/kB; no density FT. "
        "Details: `reports/B4_LAGRANGIAN_ELIGIBILITY_AUDIT.md`."
    )
    L.append(
        "- **B5:** Mapping audit at `protocol/b5_itora_mapping_audit.md`. "
        "Eval forces `gov=None` for `itora_inspired`. Report value only with P_viol."
    )

    # ---- 10 Improve ----
    h(2, "10. Post-improvement SPARC (isolated; does not mutate lock)")
    L.append(
        "Success: mean V > frozen Direct "
        f"**{fmt(p['success_criteria_improve']['mean_V_gt_frozen_direct'])}** "
        "and non-collapse on >=4/5 seeds; P_viol=0. "
        "Escape I* = fixed 96; Convergence C*/X* = 96 -> auto +48 -> optional last +48 with gate."
    )
    L.append("")
    if not p["improve"]:
        L.append("_No improve FT rows yet._")
    else:
        L.append("| Variant | stage | n | mean V | vs SPARC | vs Direct | VA-collapse | P_viol | success |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for key in sorted(p["improve"]):
            e = p["improve"][key]
            L.append(
                f"| {e['label']} | {e.get('stage') or ('done' if e['complete'] else 'partial')} | "
                f"{e['n']} | {fmt(e.get('mean_V'))} | {fmt(e.get('vs_sparc'))} | "
                f"{fmt(e.get('vs_direct'))} | {e.get('va_collapse_seeds')}/{e['n']} | "
                f"{fmt(e.get('mean_P_viol'), 4)} | {'YES' if e.get('success') else 'no'} |"
            )
    running = [e for e in p["improve_queue_status"] if e.get("status") == "RUNNING"]
    if running:
        L.append("")
        L.append("Live queue: " + ", ".join(f"`{e.get('name')}`" for e in running))
    if p.get("best_improve") and p["best_improve"].get("n", 0) >= 5:
        b = p["best_improve"]
        L.append("")
        L.append(
            f"Best completed improve so far: **{b['label']}** "
            f"V={fmt(b['mean_V'])} (vs Direct {fmt(b['vs_direct'])}; success={b['success']})."
        )

    # ---- 11 Train curves ----
    h(2, "11. Training-curve diagnostics")
    L.append("| Run | n | mean return | roll16 end | half1 | half2 | zero-ep frac |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, s in p["train_curves"].items():
        L.append(
            f"| {label} | {s['n']} | {fmt(s['mean_return'])} | {fmt(s['roll16_end'])} | "
            f"{fmt(s['half1'])} | {fmt(s['half2'])} | {fmt(s['zero_frac'], 2)} |"
        )

    # ---- 12 Table IV ----
    h(2, "12. Table IV - complexity / admission")
    L.append("| Method | trainable params | admission |")
    L.append("|---|---:|---|")
    for k, v in (p.get("table_iv") or {}).items():
        L.append(f"| {k} | {v.get('trainable', '-')} | {v.get('admission', '-')} |")

    # ---- 13 Artifacts ----
    h(2, "13. Artifact index")
    L.append("| Family | Path |")
    L.append("|---|---|")
    L.append("| Nominal VA/MF/SPARC | `results/raw/final_test_nominal_d35/` |")
    L.append("| Direct | `results/raw/final_test_direct_d35/` |")
    L.append("| B4 Lagrangian | `results/raw/final_test_b4_d35/` |")
    L.append("| B5 ITORA | `results/raw/final_test_itora_d35/` |")
    L.append("| Ablations | `results/raw/final_test_a*_d35/` |")
    L.append("| Density (+ Direct) | `results/raw/density_d*` / `density_direct_d*` |")
    L.append("| Envelope | `results/raw/envelope_b*` |")
    L.append("| Improve | `results/raw/improve_*_d35/` + `results/wilab_improve_*` |")
    L.append("| This report JSON | `results/aggregates/evidence_package_report.json` |")
    L.append("| Offered-byte backfill log | `results/aggregates/offered_bytes_backfill.json` |")

    h(2, "14. Regeneration")
    L.append("```text")
    L.append("python -u sparc_performance_evaluation/scripts/write_evidence_package_report.py")
    L.append("python -u sparc_performance_evaluation/scripts/write_evidence_package_report.py --watch")
    L.append("```")
    L.append("")
    L.append(
        "Locked V_done outcomes are never rewritten by this script. "
        "Section 10 grows as improve FTs finish; "
        "`aggregate_improve_vs_frozen.py` also refreshes this report."
    )
    L.append("")
    return "\n".join(L)


def main():
    payload = build_payload()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print("status", payload["package_status"],
          "improve", len(payload["improve"]),
          "winners", payload["improve_winners"])
    return 0


def watch_until_improve_done(poll_s: float = 60.0):
    print(f"Watching for {IMPROVE_DONE} (refresh every {poll_s}s)", flush=True)
    last_sig = None
    while True:
        main()
        sig_parts = []
        st = RAW / "improve_status.json"
        if st.exists():
            sig_parts.append(str(st.stat().st_mtime))
        for p in sorted(RAW.glob("improve_*_d35")):
            for marker in ("DONE", "rows.json"):
                f = p / marker
                if f.exists():
                    sig_parts.append(f"{p.name}:{marker}:{f.stat().st_mtime}")
        sig = "|".join(sig_parts)
        if sig != last_sig:
            last_sig = sig
            print(f"refresh @ {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        if IMPROVE_DONE.exists():
            main()
            print("improve_aggregate_DONE - final report written.", flush=True)
            return 0
        time.sleep(poll_s)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        raise SystemExit(watch_until_improve_done())
    raise SystemExit(main())
