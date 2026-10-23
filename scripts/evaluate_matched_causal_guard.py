"""Matched Closed-Loop Comparison of Causal Pre-Admission Safety Guard.

Evaluates under identical frozen environment conditions:
- Controllers:
  1. B4 Reference Baseline (frozen)
  2. Structured Priority Policy (repaired baseline-parity initialization)
  3. Structured Priority with Causal Pre-Admission Risk Guard (CausalGuardedPolicy)
- Scenarios:
  - Validation seeds: [73300, 73301, 73302, 73303]
  - Training seeds:   [73200, 73201]
- Metrics:
  - V_done (completed utility)
  - P_viol (fraction of windows with PRR < 0.95, evaluated against absolute 2% limit)
  - Completed task count
  - Admitted bytes
  - Min PRR and Mean PRR
  - Rejected/redirected candidate tasks
"""
from __future__ import annotations

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
from controllers.rl.causal_safety_guard import CausalGuardedPolicy


def run_one_eval(policy, sim_seed: int, eid: int, runs_dir: Path, is_training_seed: bool = False, policy_mode: str = "structured_priority"):
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
    print("=" * 80)
    print("MATCHED CLOSED-LOOP EVALUATION: CAUSAL PRE-ADMISSION SAFETY GUARD")
    print("=" * 80)

    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "causal_guard_eval"
    runs_dir.mkdir(parents=True, exist_ok=True)

    val_seeds = [73300, 73301, 73302, 73303]
    train_seeds = [73200, 73201]

    from controllers.rl.residual_task_policy import ResidualTaskPolicy
    from controllers.rl.hq_spo_baselines import CeRiskWeights

    b4_pol = ResidualTaskPolicy(logit_mode="direct", logit_scale=100.0)
    b4_pol.ce_weights = CeRiskWeights.load()
    b4_pol.eval()

    sp_parity_pol = StructuredPriorityPolicy()
    sp_parity_pol.eval()

    cguard_pol = CausalGuardedPolicy(d_max=150.0, cbr_thresh=0.042)
    cguard_pol.eval()

    controllers = [
        ("B4_Reference", b4_pol, "ce_milp"),
        ("Structured_Priority_Parity", sp_parity_pol, "structured_priority"),
        ("Causal_Guarded_SP", cguard_pol, "structured_priority"),
    ]

    results = {}
    base_eid = 83000

    import concurrent.futures

    for cname, pol, pmode in controllers:
        print(f"\n================ Evaluating Controller: {cname} (mode={pmode}) ================")
        results[cname] = {"validation": [], "training": []}

        # Validation seeds in parallel
        print("  Evaluating Validation Seeds [73300-73303]...")
        val_tasks = [(pol, s, base_eid + i + 1, runs_dir, False, pmode) for i, s in enumerate(val_seeds)]
        base_eid += len(val_seeds)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futs = {executor.submit(run_one_eval, *t): t[1] for t in val_tasks}
            for fut in concurrent.futures.as_completed(futs):
                res = fut.result()
                results[cname]["validation"].append(res)
        results[cname]["validation"].sort(key=lambda x: x["seed"])
        for res in results[cname]["validation"]:
            print(f"    Val Seed {res['seed']}: V={res['V_done']:.2f} | P_viol={res['P_viol']*100:.2f}% ({res['n_viols']} viols) | PRR={res['mean_prr']:.4f} (min={res['min_prr']:.4f}) | Bytes={res['admitted_bytes']} | Tasks={res['completed_tasks']} | RejGeom={res['rejected_geom_candidates']}")

        # Training seeds in parallel
        print("  Evaluating Training Seeds [73200-73201]...")
        train_tasks = [(pol, s, base_eid + i + 1, runs_dir, True, pmode) for i, s in enumerate(train_seeds)]
        base_eid += len(train_seeds)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futs = {executor.submit(run_one_eval, *t): t[1] for t in train_tasks}
            for fut in concurrent.futures.as_completed(futs):
                res = fut.result()
                results[cname]["training"].append(res)
        results[cname]["training"].sort(key=lambda x: x["seed"])
        for res in results[cname]["training"]:
            print(f"    Train Seed {res['seed']}: V={res['V_done']:.2f} | P_viol={res['P_viol']*100:.2f}% ({res['n_viols']} viols) | PRR={res['mean_prr']:.4f} (min={res['min_prr']:.4f}) | Bytes={res['admitted_bytes']} | Tasks={res['completed_tasks']} | RejGeom={res['rejected_geom_candidates']}")

        val_v = np.mean([r["V_done"] for r in results[cname]["validation"]])
        val_pv = np.mean([r["P_viol"] for r in results[cname]["validation"]])
        print(f"\n  >>> {cname} Validation Mean: V = {val_v:.2f} | P_viol = {val_pv*100:.2f}% | Absolute Safe (<=2.0%): {val_pv <= 0.02} <<<")

    # Serialize summary
    summary_data = {}
    for cname, _, _ in controllers:
        summary_data[cname] = {
            "validation": [{k: v for k, v in r.items() if k != "audit"} for r in results[cname]["validation"]],
            "training": [{k: v for k, v in r.items() if k != "audit"} for r in results[cname]["training"]],
            "val_mean_v": float(np.mean([r["V_done"] for r in results[cname]["validation"]])),
            "val_mean_p_viol": float(np.mean([r["P_viol"] for r in results[cname]["validation"]])),
            "val_safe": bool(np.mean([r["P_viol"] for r in results[cname]["validation"]]) <= 0.02),
        }

    out_file = runs_dir / "matched_guard_comparison.json"
    out_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    main()
