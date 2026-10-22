"""HQ-SPO baseline panel: B1–B5 on fresh test seeds. Defaults, no retune in this run."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.hq_spo_baselines import CeRiskWeights, MyopicDualBaseline
from sparc_performance_evaluation.scripts.run_paired_eval import safety_from_audit

RAW = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "hq_spo_baseline_panel"
REPORTS = ROOT / "sparc_performance_evaluation" / "reports"
ENVELOPE = 175000
SEEDS = list(range(73100, 73108))
B5_CKPT = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "hq_spo_b4_b5_smoke" / "b5_smoke_checkpoint.pt"

METHODS = [
    ("B1_greedy_vb", "greedy_vb"),
    ("B2_cbr_threshold", "cbr_threshold"),
    ("B3_myopic_dual", "myopic_dual"),
    ("B4_ce_milp", "ce_milp"),
    ("B5_ppo_lag", "residual"),
]


def make_policy(name: str):
    if name == "B3_myopic_dual":
        return MyopicDualBaseline(lam=0.0, eta=0.05, alpha=0.02)
    pol = ResidualTaskPolicy(logit_mode="direct", logit_scale=100.0)
    if name == "B4_ce_milp":
        pol.ce_weights = CeRiskWeights.load()
    if name == "B5_ppo_lag" and B5_CKPT.is_file():
        ck = torch.load(B5_CKPT, map_location="cpu")
        pol.load_state_dict(ck["policy"], strict=False)
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    return pol


def run_one(policy, mode, *, episode_id, sim_seed, task_seed, runs_dir):
    bridge = Path(tempfile.gettempdir()) / f"hqpan_{os.getpid()}_{episode_id}"
    if bridge.exists():
        shutil.rmtree(bridge)
    os.environ["WILAB_OUTPUT_ROOT"] = str(runs_dir)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)
    t0 = time.perf_counter()
    _, _, completed, _ = run_episode(
        WILAB_ROOT, bridge, episode_id, 35, policy, 900.0,
        False, False, 100.0, 20.0, sim_seed, 100,
        None, 4, False, ENVELOPE,
        300, 12.0, mode, None,
        True, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
    )
    wall = time.perf_counter() - t0
    ep = runs_dir / f"episode_{episode_id:04d}"
    safety = safety_from_audit(ep / "bridge_audit.json")
    return {
        "V_done": float(completed),
        "P_viol": safety.get("P_viol"),
        "wall_s": wall,
        "feasible": bool(safety.get("P_viol") is not None and float(safety["P_viol"]) <= 0.02),
    }


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    runs = RAW / "runs"
    runs.mkdir(exist_ok=True)
    rows_path = RAW / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    done = {(r["method"], int(r["sim_seed"])) for r in rows}
    eid = 1 + len(list(runs.glob("episode_*")))
    t0 = time.time()
    print(json.dumps({"event": "start", "seeds": SEEDS, "methods": [m[0] for m in METHODS], "eta_min": 35}), flush=True)

    for sim_seed in SEEDS:
        task_seed = 186100 + (sim_seed - 73100)
        for name, mode in METHODS:
            if (name, sim_seed) in done:
                continue
            print(json.dumps({"event": "run", "method": name, "sim_seed": sim_seed}), flush=True)
            pol = make_policy(name)
            out = run_one(pol, mode, episode_id=eid, sim_seed=sim_seed, task_seed=task_seed, runs_dir=runs)
            eid += 1
            row = {"method": name, "sim_seed": sim_seed, **out}
            rows.append(row)
            done.add((name, sim_seed))
            rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps({"event": "done", **row}, default=str), flush=True)

    by_m = {}
    for r in rows:
        by_m.setdefault(r["method"], []).append(r)

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    summary = {"n_seeds": len(SEEDS), "elapsed_s": time.time() - t0, "methods": {}}
    for name, _ in METHODS:
        rs = by_m.get(name, [])
        feas = [r for r in rs if r.get("feasible")]
        summary["methods"][name] = {
            "mean_V": mean([r["V_done"] for r in rs]),
            "mean_V_feasible_only": mean([r["V_done"] for r in feas]),
            "mean_P_viol": mean([float(r["P_viol"]) for r in rs if r.get("P_viol") is not None]),
            "frac_feasible": mean([1.0 if r.get("feasible") else 0.0 for r in rs]),
            "mean_wall_s": mean([r["wall_s"] for r in rs]),
        }
    # best feasible by mean_V among methods with frac_feasible==1 (or highest frac then V)
    ranked = sorted(
        summary["methods"].items(),
        key=lambda kv: (
            kv[1]["frac_feasible"] if kv[1]["frac_feasible"] is not None else 0,
            kv[1]["mean_V"] if kv[1]["mean_V"] is not None else -1e9,
        ),
        reverse=True,
    )
    summary["best_feasible"] = ranked[0][0] if ranked else None
    summary["note_B5"] = "B5 uses 1-ep smoke checkpoint — undertrained vs full protocol budget"
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# HQ-SPO baseline panel (B1–B5)",
        "",
        f"**Seeds:** `{SEEDS[0]}–{SEEDS[-1]}` (tps4, test)",
        f"**Elapsed:** {summary['elapsed_s']:.0f}s",
        f"**Best (feasibility then U):** `{summary['best_feasible']}`",
        "",
        f"Note: {summary['note_B5']}",
        "",
        "| method | mean V | mean P_viol | frac P_viol≤0.02 | mean wall_s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _ in METHODS:
        st = summary["methods"][name]
        lines.append(
            f"| {name} | {st['mean_V']:.2f} | {st['mean_P_viol']:.3f} | "
            f"{st['frac_feasible']:.2f} | {st['mean_wall_s']:.1f} |"
        )
    lines += ["", "| seed | " + " | ".join(m[0] for m in METHODS) + " |", "|---:|" + "|".join(["---:"] * len(METHODS)) + "|"]
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["sim_seed"], {})[r["method"]] = r["V_done"]
    for s in SEEDS:
        vals = by_seed.get(s, {})
        lines.append("| " + str(s) + " | " + " | ".join(f"{vals.get(m[0], float('nan')):.2f}" for m in METHODS) + " |")
    (REPORTS / "SPARC_HQ_SPO_BASELINE_PANEL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
