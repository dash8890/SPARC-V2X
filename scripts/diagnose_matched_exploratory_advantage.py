"""Diagnose Matched Exploratory Advantage on Training Seeds.

Directly compares:
1. Baseline B4 (CE-MILP)
2. Structured Priority (Deterministic: w_vec = mu_w, beta = mu_beta)
3. Structured Priority (Sampled: w_vec ~ Normal, beta ~ Normal)

Across identical training seeds [73200, 73201, 73202, 73203] under matched streaming arrivals.
Measures:
- Completed value (V_done)
- Safety violations (P_viol over valid sensing epochs)
- PRR and CBR
- Decision divergence epochs between B4 and Structured Priority
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from metrics_logger import safety_from_audit_list
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.hq_spo_baselines import CeRiskWeights
from controllers.rl.structured_priority_policy import StructuredPriorityPolicy

ENVELOPE = 175000


def run_episode_matched(
    mode_name: str,
    policy: Any,
    policy_mode: str,
    deterministic: bool,
    seed: int,
    eid: int,
    runs_dir: Path,
    density: int = 15,
    tps: int = 12,
    sim_time: float = 12.0,
) -> dict:
    bridge = Path(tempfile.gettempdir()) / f"diag_adv_{mode_name}_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)
    rnd = (seed - 73200) // 2
    i = (seed - 73200) % 2
    t_seed = 191000 + rnd * 100 + i

    t0 = time.perf_counter()
    _, _, comp_val, trace = run_episode(
        WILAB_ROOT, bridge, eid, density, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, tps, False, ENVELOPE,
        300, sim_time, policy_mode, None,
        deterministic, "off", "", True, t_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
        arrival_mode="streaming",
    )
    wall = time.perf_counter() - t0

    ep = runs_dir / f"episode_{eid:04d}"
    audit_file = ep / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    safety = safety_from_audit_list(audit) if audit else {}
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    mean_prr = float(np.mean([float(e.get("window_prr", e.get("prr", 1.0))) for e in valid])) if valid else 1.0
    mean_cbr = float(np.mean([float(e.get("cbr", 0.0)) for e in valid])) if valid else 0.0

    return {
        "mode": mode_name,
        "seed": seed,
        "V_done": float(comp_val),
        "P_viol": safety.get("P_viol", 0.0),
        "mean_prr": mean_prr,
        "mean_cbr": mean_cbr,
        "wall_s": wall,
        "audit": audit,
        "trace": trace,
    }


def main():
    seeds = [73200, 73201, 73202, 73203]
    out_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "diagnose_advantage"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("DIAGNOSING MATCHED EXPLORATORY ADVANTAGE ON TRAINING SEEDS")
    print(f"Seeds: {seeds} | Workload: Density 15, 12 tasks/source (Streaming)")
    print("=" * 90)

    # 1. Setup policies
    b4_pol = ResidualTaskPolicy(logit_mode="direct", logit_scale=100.0)
    b4_pol.ce_weights = CeRiskWeights.load()
    b4_pol.eval()

    sp_pol = StructuredPriorityPolicy()
    sp_pol.eval()

    eid = 80000
    results = {}

    for s in seeds:
        print(f"\n>>> Running Seed {s}...")
        # B4
        res_b4 = run_episode_matched("B4_ce_milp", b4_pol, "ce_milp", True, s, eid, runs_dir)
        eid += 1
        print(f"  [B4]          V_done = {res_b4['V_done']:.2f} | P_viol = {res_b4['P_viol']*100:.2f}% | PRR = {res_b4['mean_prr']:.4f}")

        # SP Deterministic
        res_sp_det = run_episode_matched("SP_Deterministic", sp_pol, "structured_priority", True, s, eid, runs_dir)
        eid += 1
        print(f"  [SP Det]      V_done = {res_sp_det['V_done']:.2f} | P_viol = {res_sp_det['P_viol']*100:.2f}% | PRR = {res_sp_det['mean_prr']:.4f}")

        # SP Sampled (Exploratory)
        res_sp_sample = run_episode_matched("SP_Sampled", sp_pol, "structured_priority", False, s, eid, runs_dir)
        eid += 1
        print(f"  [SP Sampled]  V_done = {res_sp_sample['V_done']:.2f} | P_viol = {res_sp_sample['P_viol']*100:.2f}% | PRR = {res_sp_sample['mean_prr']:.4f}")

        # Compute differences in assignments across epochs
        diff_det_b4 = 0
        diff_sample_b4 = 0
        diff_sample_det = 0
        n_epochs = min(len(res_b4["audit"]), len(res_sp_det["audit"]), len(res_sp_sample["audit"]))
        for ep in range(n_epochs):
            assn_b4 = [(a["task_id"], a["helper_id"]) for a in res_b4["audit"][ep].get("selected_assignments", [])]
            assn_det = [(a["task_id"], a["helper_id"]) for a in res_sp_det["audit"][ep].get("selected_assignments", [])]
            assn_sample = [(a["task_id"], a["helper_id"]) for a in res_sp_sample["audit"][ep].get("selected_assignments", [])]
            if sorted(assn_b4) != sorted(assn_det):
                diff_det_b4 += 1
            if sorted(assn_b4) != sorted(assn_sample):
                diff_sample_b4 += 1
            if sorted(assn_det) != sorted(assn_sample):
                diff_sample_det += 1

        print(f"  Divergence Epochs vs B4: Det={diff_det_b4}, Sampled={diff_sample_b4} (Sampled vs Det={diff_sample_det})")
        results[s] = {
            "B4": {"V": res_b4["V_done"], "P_viol": res_b4["P_viol"], "PRR": res_b4["mean_prr"]},
            "SP_Det": {"V": res_sp_det["V_done"], "P_viol": res_sp_det["P_viol"], "PRR": res_sp_det["mean_prr"]},
            "SP_Sampled": {"V": res_sp_sample["V_done"], "P_viol": res_sp_sample["P_viol"], "PRR": res_sp_sample["mean_prr"]},
            "diff_det_b4": diff_det_b4,
            "diff_sample_b4": diff_sample_b4,
            "diff_sample_det": diff_sample_det,
        }

    # Summary Table
    print("\n" + "=" * 90)
    print("MATCHED COMPARISON SUMMARY ACROSS TRAINING SEEDS")
    print("=" * 90)
    print(f"{'Seed':<8} | {'B4 V_done':<11} | {'SP Det V':<11} | {'SP Sample V':<12} | {'d_(Sample-B4)':<15} | {'B4 P_viol':<10} | {'SP Samp P_viol':<14}")
    print("-" * 90)
    b4_vals, det_vals, samp_vals = [], [], []
    for s in seeds:
        r = results[s]
        b4_vals.append(r["B4"]["V"])
        det_vals.append(r["SP_Det"]["V"])
        samp_vals.append(r["SP_Sampled"]["V"])
        delta = r["SP_Sampled"]["V"] - r["B4"]["V"]
        print(f"{s:<8} | {r['B4']['V']:<11.2f} | {r['SP_Det']['V']:<11.2f} | {r['SP_Sampled']['V']:<12.2f} | {delta:<+15.2f} | {r['B4']['P_viol']*100:<9.2f}% | {r['SP_Sampled']['P_viol']*100:<13.2f}%")

    print("-" * 90)
    print(f"{'Mean':<8} | {np.mean(b4_vals):<11.2f} | {np.mean(det_vals):<11.2f} | {np.mean(samp_vals):<12.2f} | {np.mean(samp_vals)-np.mean(b4_vals):<+15.2f} |")
    print("=" * 90)

    # Save summary
    out_file = out_dir / "matched_advantage_summary.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
