"""Within-Regime Comparative Benchmark across Density 12 and Density 20.

Evaluates 4 methods:
  1. B4_ce_milp: Certainty-Equivalent MILP
  2. B5_prime: Preserved two-head policy checkpoint (43p86)
  3. PB_CRL_calibrated: Initial calibrated 3-head PB-CRL
  4. PB_CRL_trained: Trained 3-head PB-CRL (checkpoints/pb_crl_trained_best.pt)
Across validation seeds [73300, 73301, 73302, 73303] under matched workloads.
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

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.pb_crl_b5_prime import B5PrimePolicy
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.hq_spo_baselines import CeRiskWeights
from controllers.rl.pb_crl_policy import PBCRLPolicy, PBCRLCoreAgent, EnsembleRiskModel

ENVELOPE = 175000
B5P_CKPT = ROOT / "checkpoints" / "b5_prime_best_preserved_43p86.pt"
PB_TRAINED_CKPT = ROOT / "checkpoints" / "pb_crl_trained_best.pt"


def make_policy(method: str):
    if method == "B4_ce_milp":
        pol = ResidualTaskPolicy(logit_mode="direct", logit_scale=100.0)
        pol.ce_weights = CeRiskWeights.load()
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        return pol, "ce_milp"
    elif method == "B5_prime":
        if not B5P_CKPT.is_file():
            raise FileNotFoundError(f"Missing B5' checkpoint: {B5P_CKPT}")
        pol = B5PrimePolicy(state_dim=16, hidden=64)
        ck = torch.load(B5P_CKPT, map_location="cpu")
        pol.load_state_dict(ck["policy"], strict=False)
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        return pol, "b5_prime"
    elif method == "PB_CRL_calibrated":
        policy_net = PBCRLPolicy()
        policy_net.eval()
        for p in policy_net.parameters():
            p.requires_grad_(False)
        risk_model = EnsembleRiskModel()
        pol = PBCRLCoreAgent(policy_net, risk_model, K=1, total_epochs=100, delta_s=0.05, shield_enabled=False)
        return pol, "b5_prime"
    elif method == "PB_CRL_trained":
        if not PB_TRAINED_CKPT.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint: {PB_TRAINED_CKPT}")
        policy_net = PBCRLPolicy()
        ck = torch.load(PB_TRAINED_CKPT, map_location="cpu")
        policy_net.load_state_dict(ck["policy"])
        policy_net.eval()
        for p in policy_net.parameters():
            p.requires_grad_(False)
        risk_model = EnsembleRiskModel()
        pol = PBCRLCoreAgent(policy_net, risk_model, K=1, total_epochs=100, delta_s=0.05, shield_enabled=False)
        return pol, "b5_prime"
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


def run_benchmark_episode(density: int, method: str, seed: int, eid: int, runs_dir: Path):
    bridge = Path(tempfile.gettempdir()) / f"bench_d{density}_{method}_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)
    task_seed = 186100 + (seed - 73300)

    policy, mode = make_policy(method)

    t0 = time.perf_counter()
    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, density, policy, 900.0,
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
    if not audit_file.is_file():
        raise FileNotFoundError(f"Missing audit file: {audit_file}")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))

    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    if not valid:
        raise RuntimeError(f"No valid sensing windows in {audit_file}")
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    cbr = [float(e.get("cbr", 0.0)) for e in valid]
    delay = [float(e.get("p95_delay_ms", 0.0)) for e in valid]
    offered_b = sum(int(e.get("computation_bytes_offered", 0)) for e in audit)
    viols = sum(1 for p in prr if p < 0.95)
    p_viol = viols / max(len(valid), 1)

    # Opportunity audit
    task_csv = ep_dir / "task_trace.csv"
    opp = audit_task_opportunity(task_csv)

    return {
        "density": density,
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


def run_regime(density: int, max_workers: int = 2):
    base_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "within_regime_benchmark" / f"density_{density}"
    base_dir.mkdir(parents=True, exist_ok=True)
    methods = ["B4_ce_milp", "B5_prime", "PB_CRL_calibrated", "PB_CRL_trained"]
    seeds = [73300, 73301, 73302, 73303]

    print("=" * 95)
    print(f"BENCHMARK: DENSITY {density} veh/km | CAM: 300B @ 10Hz | Envelope: 175 kB | Horizon: 12.0s")
    print(f"Methods: {methods}")
    print(f"Seeds:   {seeds}")
    print("=" * 95)

    tasks = []
    eid_cur = 95000 if density == 12 else (95500 if density == 15 else 96000)
    for s in seeds:
        for m in methods:
            eid_cur += 1
            runs_m = base_dir / m / "runs"
            runs_m.mkdir(parents=True, exist_ok=True)
            tasks.append((density, m, s, eid_cur, runs_m))

    raw_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futs = {executor.submit(run_benchmark_episode, *t): (t[1], t[2]) for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            raw_results.append(res)
            print(f"  [{res['density']:2d} veh/km] {res['method']:<18} | Seed {res['seed']} | V_done={res['V_done']:6.2f} | PRR={res['mean_prr']:.4f} | Viols={res['n_viols']:2d}/100 | Feas={str(res['feasible_alpha']):<5} ({res['wall_s']:.1f}s)")

    # Aggregate by method
    print("\n" + "=" * 95)
    print(f"{'Method':<20} | {'Mean V_done':<11} | {'Mean Bytes':<10} | {'Mean PRR':<10} | {'Mean CBR':<10} | {'P_viol':<8} | {'Feas Frac'}")
    print("-" * 95)

    summary_records = {}
    for m in methods:
        m_rows = [r for r in raw_results if r["method"] == m]
        mV = float(np.mean([r["V_done"] for r in m_rows]))
        mB = float(np.mean([r["offered_bytes"] for r in m_rows]))
        mPRR = float(np.mean([r["mean_prr"] for r in m_rows]))
        mCBR = float(np.mean([r["mean_cbr"] for r in m_rows]))
        mPviol = float(np.mean([r["p_viol"] for r in m_rows]))
        mFeas = float(np.mean([1.0 if r["feasible_alpha"] else 0.0 for r in m_rows]))
        summary_records[m] = {
            "density": density,
            "mean_v_done": mV,
            "mean_bytes": mB,
            "mean_prr": mPRR,
            "mean_cbr": mCBR,
            "mean_p_viol": mPviol,
            "feasible_fraction": mFeas,
            "seed_details": m_rows,
        }
        print(f"{m:<20} | {mV:<11.2f} | {mB:<10.0f} | {mPRR:<10.4f} | {mCBR:<10.4f} | {mPviol:<8.4f} | {mFeas*100:5.1f}%")
    print("=" * 95)

    out_file = base_dir / f"density_{density}_benchmark_summary.json"
    out_file.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
    print(f"\nSaved summary to: {out_file}")
    return summary_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=int, default=12, choices=[12, 15, 20])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    run_regime(args.density, max_workers=args.workers)


if __name__ == "__main__":
    main()
