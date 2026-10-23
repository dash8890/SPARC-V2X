"""Matched Closed-Loop Comparison of Causal Pre-Admission Safety Guard.

Evaluates under identical frozen environment conditions:
- Controllers (4-way attribution panel + reference):
  1. B4 Reference Baseline (frozen heuristic knapsack, no guard)
  2. Structured Priority Parity (baseline parity initialization, no guard)
  3. Guarded B4 Policy (B4 heuristic under C_safe)
  4. Causal Guarded SP Policy (Structured Priority under C_safe)
  5. Zero Offload B_0 (Background safety floor)
- Scenarios:
  - Validation seeds: [73300, 73301, 73302, 73303]
  - Training seeds:   [73200, 73201]
- Metrics:
  - V_done (completed utility)
  - P_viol (fraction of windows with PRR < 0.95, evaluated against absolute 2% limit)
  - Completed task count
  - Admitted bytes
  - Min PRR and Mean PRR
  - Rejected candidate tasks (via pre-admission filter)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.structured_priority_policy import StructuredPriorityPolicy
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.hq_spo_baselines import CeRiskWeights
from controllers.rl.causal_safety_guard import (
    CausalGuardedPolicy,
    GuardedB4Policy,
)


class ZeroOffloadPolicy(StructuredPriorityPolicy):
    """Zero-offload baseline policy B_0 that rejects all computation traffic."""
    def __init__(self):
        super().__init__()
        self.score_and_match_fn = lambda candidates, weights, beta_throttle, **kwargs: (
            [], {"n_pack": 0, "bytes": 0, "scores": [], "rejected_geometry": len(candidates)}
        )


def run_one_eval(
    policy,
    sim_seed: int,
    eid: int,
    runs_dir: Path,
    is_training_seed: bool = False,
    policy_mode: str = "structured_priority",
):
    bridge = Path(tempfile.gettempdir()) / f"eval_cguard_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    
    if is_training_seed:
        t_seed = 191000 + (sim_seed - 73200)
    else:
        t_seed = 186100 + (sim_seed - 73300)

    t0 = time.perf_counter()
    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, 15, policy, 900.0,
        False, False, 100.0, 20.0, sim_seed, 100,
        None, 12, False, 175000,
        300, 12.0, policy_mode, None,
        True, "off", "", True, t_seed, gov,
        175000, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
        arrival_mode="streaming",
    )
    wall = time.perf_counter() - t0

    ep_dir = runs_dir / f"episode_{eid:04d}"
    audit_file = ep_dir / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []

    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    cbrs = [float(e.get("cbr", 0.0)) for e in valid]
    viols = sum(1 for p in prrs if p < 0.95)
    pv = viols / len(valid) if valid else 1.0

    admitted_bytes = sum(int(e.get("computation_bytes_offered", 0)) for e in audit)
    tl_file = ep_dir / "task_layer_state.json"
    completed_tasks = 0
    if tl_file.is_file():
        try:
            tl_data = json.loads(tl_file.read_text(encoding="utf-8"))
            completed_tasks = sum(1 for t in tl_data.get("tasks", []) if t.get("done"))
        except Exception:
            completed_tasks = 0

    # Geometry rejections count
    rejected_geom = 0
    for e in audit:
        meta = e.get("structured_priority_meta") or {}
        rejected_geom += meta.get("rejected_geometry", 0)

    return {
        "seed": sim_seed,
        "V_done": comp_val,
        "P_viol": pv,
        "n_viols": viols,
        "mean_prr": float(np.mean(prrs)) if prrs else 1.0,
        "min_prr": float(np.min(prrs)) if prrs else 1.0,
        "mean_cbr": float(np.mean(cbrs)) if cbrs else 0.0,
        "admitted_bytes": admitted_bytes,
        "completed_tasks": completed_tasks,
        "rejected_geom_candidates": rejected_geom,
        "wall_s": wall,
        "audit": audit,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Matched Causal Safety Guard")
    parser.add_argument("--controllers", nargs="+", default=["Guarded_B4", "Causal_Guarded_SP"],
                        help="Controllers to evaluate: B4_Reference, SP_Parity, Guarded_B4, Causal_Guarded_SP, Zero_Offload")
    args = parser.parse_args()

    print("=" * 80)
    print("MATCHED CLOSED-LOOP EVALUATION: CAUSAL PRE-ADMISSION SAFETY GUARD")
    print("=" * 80)

    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "causal_guard_eval"
    runs_dir.mkdir(parents=True, exist_ok=True)

    val_seeds = [73300, 73301, 73302, 73303]
    train_seeds = [73200, 73201]

    b4_ref = ResidualTaskPolicy(logit_mode="direct", logit_scale=100.0)
    b4_ref.ce_weights = CeRiskWeights.load()
    b4_ref.eval()

    sp_parity = StructuredPriorityPolicy()
    sp_parity.eval()

    b4_guarded = GuardedB4Policy(r_thresh=0.20)
    b4_guarded.eval()

    sp_guarded = CausalGuardedPolicy(r_thresh=0.20)
    sp_guarded.eval()

    zero_offload = ZeroOffloadPolicy()
    zero_offload.eval()

    controller_map = {
        "B4_Reference": (b4_ref, "ce_milp"),
        "SP_Parity": (sp_parity, "structured_priority"),
        "Guarded_B4": (b4_guarded, "structured_priority"),
        "Causal_Guarded_SP": (sp_guarded, "structured_priority"),
        "Zero_Offload": (zero_offload, "structured_priority"),
    }

    selected = [(name, controller_map[name][0], controller_map[name][1]) for name in args.controllers if name in controller_map]

    results = {}
    base_eid = 85000

    for cname, pol, pmode in selected:
        print(f"\n================ Evaluating Controller: {cname} (mode={pmode}) ================")
        results[cname] = {"validation": [], "training": []}

        # Validation seeds
        print("  Evaluating Validation Seeds [73300-73303]...")
        for i, s in enumerate(val_seeds):
            eid = base_eid + 1
            base_eid += 1
            res = run_one_eval(pol, s, eid, runs_dir, is_training_seed=False, policy_mode=pmode)
            results[cname]["validation"].append(res)
            print(f"    Val Seed {res['seed']}: V={res['V_done']:.2f} | P_viol={res['P_viol']*100:.2f}% ({res['n_viols']} viols) | PRR={res['mean_prr']:.4f} (min={res['min_prr']:.4f}) | Bytes={res['admitted_bytes']} | Tasks={res['completed_tasks']} | RejGeom={res['rejected_geom_candidates']}")

        val_v = np.mean([r["V_done"] for r in results[cname]["validation"]])
        val_pv = np.mean([r["P_viol"] for r in results[cname]["validation"]])
        print(f"\n  >>> {cname} Validation Mean: V = {val_v:.2f} | P_viol = {val_pv*100:.2f}% | Absolute Safe (<=2.0%): {val_pv <= 0.02} <<<")

    # Save summary
    summary_file = runs_dir / "matched_guard_comparison.json"
    existing = {}
    if summary_file.is_file():
        try:
            existing = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    for cname in results:
        existing[cname] = {
            "validation": [{k: v for k, v in r.items() if k != "audit"} for r in results[cname]["validation"]],
            "val_mean_v": float(np.mean([r["V_done"] for r in results[cname]["validation"]])),
            "val_mean_p_viol": float(np.mean([r["P_viol"] for r in results[cname]["validation"]])),
            "val_safe": bool(np.mean([r["P_viol"] for r in results[cname]["validation"]]) <= 0.02),
        }

    summary_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nSaved updated results to {summary_file}")


if __name__ == "__main__":
    main()
