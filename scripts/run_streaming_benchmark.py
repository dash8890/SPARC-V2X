"""Definitive Benchmark Suite on Stochastic Streaming Workload (Density 15).

Evaluates the complete baseline suite:
  B0 (Zero Offload)
  B1 (Greedy Value Density)
  B2 (CBR Threshold)
  B3 (Myopic Dual Knapsack)
  B4 (CE-MILP)
  B5' (Two-Head PPO)
  PB-CRL Calibrated
  PB-CRL Trained

Across validation seeds 73300-73303 under the calibrated streaming workload.
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.pb_crl_b5_prime import B5PrimePolicy
from controllers.rl.pb_crl_policy import PBCRLPolicy, PBCRLCoreAgent, EnsembleRiskModel
from controllers.rl.hq_spo_baselines import CeRiskWeights, MyopicDualBaseline

ENVELOPE = 175000
DEFAULT_K = 1
B5P_CKPT = ROOT / "checkpoints" / "b5_prime_best_preserved_43p86.pt"
PB_TRAINED_CKPT = ROOT / "checkpoints" / "pb_crl_trained_best.pt"
PB_FEAS_CKPT = ROOT / "checkpoints" / "pb_crl_trained_best_feasible.pt"
SP_TRAINED_CKPT = ROOT / "checkpoints" / "structured_priority_best_feasible.pt"
SP_BEST_CKPT = ROOT / "checkpoints" / "structured_priority_best.pt"


class ZeroOffloadPolicy:
    def __init__(self):
        self.beta = 0.0
        self.lam = 0.0
        self.is_pb_crl_b5_prime = True

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        return 0.0, 0.0, torch.tensor(0.0), 0.0, 0.0


def make_policy(method: str):
    if method == "B0_zero_offload":
        return ZeroOffloadPolicy(), "b5_prime"
    elif method == "B1_greedy_vb":
        return None, "greedy_vb"
    elif method == "B2_cbr_threshold":
        return None, "cbr_threshold"
    elif method == "B3_myopic_dual":
        return MyopicDualBaseline(lam=0.0, eta=0.05, alpha=0.02), "myopic_dual"
    elif method == "B4_ce_milp":
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
    elif method in ("PB_CRL_trained", "PB_CRL_Preliminary_PPO"):
        ckpt_to_use = PB_FEAS_CKPT if PB_FEAS_CKPT.is_file() else PB_TRAINED_CKPT
        if not ckpt_to_use.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint: {ckpt_to_use}")
        policy_net = PBCRLPolicy()
        ck = torch.load(ckpt_to_use, map_location="cpu")
        policy_net.load_state_dict(ck["policy"])
        policy_net.eval()
        for p in policy_net.parameters():
            p.requires_grad_(False)
        risk_model = EnsembleRiskModel()
        pol = PBCRLCoreAgent(policy_net, risk_model, K=1, total_epochs=100, delta_s=0.05, shield_enabled=False)
        return pol, "b5_prime"
    elif method in ("Structured_Priority_Untrained", "Structured_Priority_Redesign"):
        from controllers.rl.structured_priority_policy import StructuredPriorityPolicy
        pol = StructuredPriorityPolicy()
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        return pol, "structured_priority"
    elif method in ("Structured_Priority_Trained", "Structured_Priority_Preliminary_PPO"):
        from controllers.rl.structured_priority_policy import StructuredPriorityPolicy
        ckpt = SP_TRAINED_CKPT if SP_TRAINED_CKPT.is_file() else SP_BEST_CKPT
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing structured priority checkpoint: {ckpt}")
        pol = StructuredPriorityPolicy()
        ck = torch.load(ckpt, map_location="cpu")
        pol.load_state_dict(ck["policy"])
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        return pol, "structured_priority"
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


def run_single(
    method: str,
    seed: int,
    eid: int,
    density: int,
    tasks_per_source: int,
    runs_dir: Path,
    sim_time: float = 12.0,
) -> dict:
    bridge = Path(tempfile.gettempdir()) / f"str_bench_{method}_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)
    task_seed = 186100 + (seed - 73300)

    policy, mode = make_policy(method)

    t0 = time.perf_counter()
    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, density, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, tasks_per_source, False, ENVELOPE,
        300, sim_time, mode, None,
        True, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
        arrival_mode="streaming",
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
        "feasible": bool(p_viol <= 0.02),
        "wall_s": wall,
        "opportunity": opp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=int, default=15)
    ap.add_argument("--tasks-per-source", type=int, default=12)
    ap.add_argument("--sim-time", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--methods", nargs="+", default=[
        "B0_zero_offload",
        "B1_greedy_vb",
        "B2_cbr_threshold",
        "B3_myopic_dual",
        "B4_ce_milp",
        "B5_prime",
        "PB_CRL_calibrated",
        "PB_CRL_trained",
        "Structured_Priority_Untrained",
        "Structured_Priority_Trained",
    ])
    args = ap.parse_args()

    seeds = [73300, 73301, 73302, 73303]
    out_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "streaming_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*90}")
    print(f"SPARC-V2X: STREAMING WORKLOAD BENCHMARK SUITE (DENSITY {args.density})")
    print(f"Tasks/Source: {args.tasks_per_source} | Seeds: {seeds} | Workers: {args.workers}")
    print(f"Methods: {args.methods}")
    print(f"{'='*90}\n")

    all_results = []
    eid = 50000

    for method in args.methods:
        print(f"\n>>> Running Method: {method} across {len(seeds)} validation seeds...")
        method_tasks = []
        for s in seeds:
            method_tasks.append((method, s, eid, args.density, args.tasks_per_source, runs_dir, args.sim_time))
            eid += 1

        method_res = []
        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(seeds))) as executor:
                futs = [executor.submit(run_single, *t) for t in method_tasks]
                for fut in concurrent.futures.as_completed(futs):
                    res = fut.result()
                    method_res.append(res)
                    print(f"    Seed {res['seed']}: V_done = {res['V_done']:.2f} | P_viol = {res['p_viol']*100:.1f}% | PRR = {res['mean_prr']:.4f} | CBR = {res['mean_cbr']:.4f}")
        else:
            for t in method_tasks:
                res = run_single(*t)
                method_res.append(res)
                print(f"    Seed {res['seed']}: V_done = {res['V_done']:.2f} | P_viol = {res['p_viol']*100:.1f}% | PRR = {res['mean_prr']:.4f} | CBR = {res['mean_cbr']:.4f}")

        all_results.extend(method_res)

    # Save raw records
    raw_path = out_dir / "streaming_raw_records.json"
    raw_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Aggregate by method
    summary = []
    for m in args.methods:
        m_rows = [r for r in all_results if r["method"] == m]
        if not m_rows:
            continue
        v_done_vals = [r["V_done"] for r in m_rows]
        p_viol_vals = [r["p_viol"] for r in m_rows]
        prr_vals = [r["mean_prr"] for r in m_rows]
        cbr_vals = [r["mean_cbr"] for r in m_rows]
        bytes_vals = [r["offered_bytes"] for r in m_rows]
        feas_count = sum(1 for r in m_rows if r["feasible"])

        summary.append({
            "method": m,
            "mean_V_done": float(np.mean(v_done_vals)),
            "std_V_done": float(np.std(v_done_vals)),
            "mean_P_viol": float(np.mean(p_viol_vals)),
            "std_P_viol": float(np.std(p_viol_vals)),
            "mean_PRR": float(np.mean(prr_vals)),
            "mean_CBR": float(np.mean(cbr_vals)),
            "mean_bytes": float(np.mean(bytes_vals)),
            "feasible_seeds": f"{feas_count}/{len(m_rows)}",
            "feasible_fraction": float(feas_count / len(m_rows)),
        })

    sum_path = out_dir / "streaming_benchmark_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*90}")
    print(f"FINAL STREAMING BENCHMARK RESULTS (DENSITY {args.density})")
    print(f"{'='*90}")
    print(f"{'Method':<20} | {'Mean V_done':<11} | {'Mean P_viol':<11} | {'Mean PRR':<10} | {'Mean CBR':<10} | {'Feasible':<8}")
    print("-" * 90)
    for s in summary:
        print(f"{s['method']:<20} | {s['mean_V_done']:<11.2f} | {s['mean_P_viol']*100:<10.2f}% | {s['mean_PRR']:<10.4f} | {s['mean_CBR']:<10.4f} | {s['feasible_seeds']:<8}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
