"""Evaluate remaining baselines (B0, B1, B2, B3) on Density 15.

Then consolidate with B4, B5', and PB-CRL to provide a complete comparison.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.hq_spo_baselines import MyopicDualBaseline

ENVELOPE = 175000


class ZeroOffloadPolicy:
    def __init__(self):
        self.beta = 0.0
        self.lam = 0.0
        self.is_pb_crl_b5_prime = True

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        return 0.0, 0.0, torch.tensor(0.0), 0.0, 0.0


def make_policy_mode(method: str):
    if method == "B0_zero_offload":
        return ZeroOffloadPolicy(), "b5_prime"
    elif method == "B1_greedy_vb":
        return None, "greedy_vb"
    elif method == "B2_cbr_threshold":
        return None, "cbr_threshold"
    elif method == "B3_myopic_dual":
        return MyopicDualBaseline(lam=0.0, eta=0.05, alpha=0.02), "myopic_dual"
    else:
        raise ValueError(f"Unknown method: {method}")


def audit_task_opportunity(csv_file: Path, max_cpu: float = 1e9) -> dict:
    if not csv_file.exists():
        return {}
    df = pd.read_csv(csv_file)
    total_val = float(df["task_value"].sum())
    done_df = df[df["done"] == True]
    done_val = float(done_df["task_value"].sum())
    incomp = df[df["done"] == False]

    infeasible_count = 0
    infeasible_val = 0.0
    feasible_count = 0
    feasible_val = 0.0

    for _, row in incomp.iterrows():
        min_exec_ms = (float(row["cycles"]) / max_cpu) * 1000.0
        val = float(row["task_value"])
        if min_exec_ms > float(row["deadline_ms"]):
            infeasible_count += 1
            infeasible_val += val
        else:
            feasible_count += 1
            feasible_val += val

    return {
        "n_total_tasks": int(len(df)),
        "total_offered_val": total_val,
        "n_completed_tasks": int(len(done_df)),
        "completed_val": done_val,
        "n_rejected_tasks": int(len(incomp)),
        "rejected_val": float(incomp["task_value"].sum()),
        "physically_impossible_rejections": infeasible_count,
        "physically_impossible_val": infeasible_val,
        "potentially_feasible_rejections": feasible_count,
        "potentially_feasible_val": feasible_val,
    }


def run_single(method: str, seed: int, eid: int, runs_dir: Path):
    bridge = Path(tempfile.gettempdir()) / f"b15_{method}_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)
    task_seed = 186100 + (seed - 73300)

    policy, mode = make_policy_mode(method)

    t0 = time.perf_counter()
    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, 15, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 4, False, ENVELOPE,
        300, 12.0, mode, None,
        True, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
    )
    wall = time.perf_counter() - t0

    ep_dir = runs_dir / f"episode_{eid:04d}"
    audit_file = ep_dir / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8"))

    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    cbr = [float(e.get("cbr", 0.0)) for e in valid]
    delay = [float(e.get("p95_delay_ms", 0.0)) for e in valid]
    offered_b = sum(int(e.get("computation_bytes_offered", 0)) for e in audit)
    viols = sum(1 for p in prr if p < 0.95)
    p_viol = viols / max(len(valid), 1)

    task_csv = ep_dir / "task_trace.csv"
    opp = audit_task_opportunity(task_csv)

    return {
        "density": 15,
        "method": method,
        "seed": seed,
        "episode_id": eid,
        "V_done": float(comp_val),
        "offered_bytes": offered_b,
        "mean_prr": float(np.mean(prr)) if prr else 1.0,
        "min_prr": float(np.min(prr)) if prr else 1.0,
        "mean_cbr": float(np.mean(cbr)) if cbr else 0.0,
        "p95_delay": float(np.mean(delay)) if delay else 0.0,
        "n_viols": viols,
        "p_viol": p_viol,
        "feasible_alpha": bool(p_viol <= 0.02),
        "wall_s": wall,
        "opportunity": opp,
    }


def main():
    base_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "within_regime_benchmark" / "density_15"
    base_dir.mkdir(parents=True, exist_ok=True)
    methods = ["B0_zero_offload", "B1_greedy_vb", "B2_cbr_threshold", "B3_myopic_dual"]
    seeds = [73300, 73301, 73302, 73303]

    print("=" * 95)
    print("RUNNING ADDITIONAL BASELINES (B0, B1, B2, B3) FOR DENSITY 15")
    print("=" * 95)

    tasks = []
    eid_cur = 95600
    for s in seeds:
        for m in methods:
            eid_cur += 1
            runs_m = base_dir / m / "runs"
            runs_m.mkdir(parents=True, exist_ok=True)
            tasks.append((m, s, eid_cur, runs_m))

    raw_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futs = {executor.submit(run_single, *t): (t[0], t[1]) for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            raw_results.append(res)
            print(f"  [15 veh/km] {res['method']:<18} | Seed {res['seed']} | V_done={res['V_done']:6.2f} | PRR={res['mean_prr']:.4f} | Viols={res['n_viols']:2d}/100 | Feas={str(res['feasible_alpha']):<5} ({res['wall_s']:.1f}s)")

    # Load existing summary for B4, B5', PB-CRL
    sum_file = base_dir / "density_15_benchmark_summary.json"
    existing = json.loads(sum_file.read_text(encoding="utf-8")) if sum_file.exists() else {}

    # Merge new methods
    for m in methods:
        m_rows = [r for r in raw_results if r["method"] == m]
        mV = float(np.mean([r["V_done"] for r in m_rows]))
        mB = float(np.mean([r["offered_bytes"] for r in m_rows]))
        mPRR = float(np.mean([r["mean_prr"] for r in m_rows]))
        mCBR = float(np.mean([r["mean_cbr"] for r in m_rows]))
        mPviol = float(np.mean([r["p_viol"] for r in m_rows]))
        mFeas = float(np.mean([1.0 if r["feasible_alpha"] else 0.0 for r in m_rows]))
        existing[m] = {
            "density": 15,
            "mean_v_done": mV,
            "mean_bytes": mB,
            "mean_prr": mPRR,
            "mean_cbr": mCBR,
            "mean_p_viol": mPviol,
            "feasible_fraction": mFeas,
            "seed_details": m_rows,
        }

    # Save comprehensive 7-method summary
    full_out = base_dir / "density_15_full_baseline_comparison.json"
    full_out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nSaved comprehensive baseline comparison to: {full_out}")

    # Print final grand table
    order = [
        "B0_zero_offload",
        "B1_greedy_vb",
        "B2_cbr_threshold",
        "B3_myopic_dual",
        "B4_ce_milp",
        "B5_prime",
        "PB_CRL_calibrated",
        "PB_CRL_trained",
    ]
    print("\n" + "=" * 95)
    print(f"{'Method':<20} | {'Mean V_done':<11} | {'Mean Bytes':<10} | {'Mean PRR':<10} | {'Mean CBR':<10} | {'P_viol':<8} | {'Feas Frac'}")
    print("-" * 95)
    for m in order:
        if m in existing:
            rec = existing[m]
            print(f"{m:<20} | {rec['mean_v_done']:<11.2f} | {rec['mean_bytes']:<10.0f} | {rec['mean_prr']:<10.4f} | {rec['mean_cbr']:<10.4f} | {rec['mean_p_viol']:<8.4f} | {rec['feasible_fraction']*100:5.1f}%")
    print("=" * 95)


if __name__ == "__main__":
    main()
