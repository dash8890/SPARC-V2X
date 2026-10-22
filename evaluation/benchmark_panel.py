"""Unified Benchmark Panel: B1–B5 and PB-CRL Foundation on Validated Test Seeds.

Runs side-by-side comparative evaluation at simulation_time = 12.0s (100 controller epochs,
tasks_per_source = 4, density = 35) to verify restored throughput and safety.
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
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.pb_crl_b5_prime import B5PrimePolicy
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.hq_spo_baselines import CeRiskWeights, MyopicDualBaseline

ENVELOPE = 175000
B5_CKPT = ROOT / "checkpoints" / "b5_baseline_panel_39p91.pt"
B5P_CKPT = ROOT / "checkpoints" / "b5_prime_best_preserved_43p86.pt"
PBCRL_TRAINED_CKPT = ROOT / "checkpoints" / "pb_crl_trained_best.pt"

METHODS = [
    ("B1_greedy_vb", "greedy_vb"),
    ("B2_cbr_threshold", "cbr_threshold"),
    ("B3_myopic_dual", "myopic_dual"),
    ("B4_ce_milp", "ce_milp"),
    ("B5_ppo_lag", "residual"),
    ("PB_CRL_foundation", "b5_prime"),
    ("PB_CRL_trained", "b5_prime"),
]


def make_policy(name: str):
    if name == "B3_myopic_dual":
        return MyopicDualBaseline(lam=0.0, eta=0.05, alpha=0.02)
    if name == "PB_CRL_foundation":
        pol = B5PrimePolicy(state_dim=16, hidden=64)
        if B5P_CKPT.is_file():
            ck = torch.load(B5P_CKPT, map_location="cpu")
            pol.load_state_dict(ck["policy"], strict=False)
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        return pol
    if name == "PB_CRL_trained":
        pol = B5PrimePolicy(state_dim=16, hidden=64)
        if PBCRL_TRAINED_CKPT.is_file():
            ck = torch.load(PBCRL_TRAINED_CKPT, map_location="cpu")
            pol.load_state_dict(ck["policy"], strict=False)
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        return pol

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


def run_one(name: str, mode: str, *, eid: int, sim_seed: int, task_seed: int, runs_dir: Path, sim_time: float = 12.0, density: int = 35):
    policy = make_policy(name)
    bridge = Path(tempfile.gettempdir()) / f"bench_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    os.environ["WILAB_OUTPUT_ROOT"] = str(runs_dir)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)

    t0 = time.perf_counter()
    _, _, completed, _ = run_episode(
        WILAB_ROOT, bridge, eid, density, policy, 900.0,
        False, False, 100.0, 20.0, sim_seed, 100,
        None, 4, False, ENVELOPE,
        300, sim_time, mode, None,
        True, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
    )
    wall = time.perf_counter() - t0

    ep_dir = runs_dir / f"episode_{eid:04d}"
    audit_file = ep_dir / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []

    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    cbr = [float(e.get("cbr", 0.0)) for e in valid]
    delay = [float(e.get("p95_delay_ms", 0.0)) for e in valid]
    viols = sum(1 for p in prr if p < 0.95)
    p_viol = viols / max(len(valid), 1)

    return {
        "method": name,
        "sim_seed": sim_seed,
        "V_done": float(completed),
        "P_viol": p_viol,
        "n_viols": viols,
        "mean_prr": float(np.mean(prr)) if prr else 1.0,
        "mean_cbr": float(np.mean(cbr)) if cbr else 0.0,
        "p95_delay": float(np.mean(delay)) if delay else 0.0,
        "wall_s": wall,
        "feasible": bool(p_viol <= 0.02),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[73100, 73101, 73102, 73103])
    ap.add_argument("--sim-time", type=float, default=12.0)
    ap.add_argument("--density", type=int, default=35)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "benchmark_panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"SPARC-V2X Restored Benchmark Panel (B1–B5 & PB-CRL)")
    print(f"Seeds: {args.seeds} | Sim Time: {args.sim_time}s | Density: {args.density} | TPS: 4 | Workers: {args.workers}")
    print(f"{'='*80}\n")

    existing_eids = [int(p.name.split("_")[1]) for p in runs_dir.glob("episode_*") if len(p.name.split("_")) > 1 and p.name.split("_")[1].isdigit()]
    eid_cur = (max(existing_eids) + 1) if existing_eids else 20000

    tasks = []
    for seed in args.seeds:
        task_seed = 186100 + (seed - 73100)
        for name, mode in METHODS:
            eid_cur += 1
            tasks.append((name, mode, eid_cur, seed, task_seed, runs_dir, args.sim_time, args.density))

    results = []
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    run_one, t[0], t[1], eid=t[2], sim_seed=t[3], task_seed=t[4], runs_dir=t[5], sim_time=t[6], density=t[7]
                ): t for t in tasks
            }
            for fut in concurrent.futures.as_completed(future_to_task):
                res = fut.result()
                results.append(res)
                print(f"  {res['method']:<20} (Seed {res['sim_seed']}) | V = {res['V_done']:<6.2f} | PRR = {res['mean_prr']:<6.4f} | CBR = {res['mean_cbr']:<6.4f} | P_viol = {res['P_viol']:<6.4f} | Feasible: {res['feasible']}")
    else:
        for t in tasks:
            res = run_one(t[0], t[1], eid=t[2], sim_seed=t[3], task_seed=t[4], runs_dir=t[5], sim_time=t[6], density=t[7])
            results.append(res)
            print(f"  {res['method']:<20} (Seed {res['sim_seed']}) | V = {res['V_done']:<6.2f} | PRR = {res['mean_prr']:<6.4f} | CBR = {res['mean_cbr']:<6.4f} | P_viol = {res['P_viol']:<6.4f} | Feasible: {res['feasible']}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY ACROSS ALL TEST SEEDS")
    print(f"{'='*80}")
    print(f"{'Method':<20} | {'Mean V':<10} | {'Mean PRR':<10} | {'Mean CBR':<10} | {'P_viol':<10} | {'Feasible Frac':<12}")
    print("-" * 80)
    for name, _ in METHODS:
        m_rows = [r for r in results if r["method"] == name]
        mV = float(np.mean([r["V_done"] for r in m_rows]))
        mPRR = float(np.mean([r["mean_prr"] for r in m_rows]))
        mCBR = float(np.mean([r["mean_cbr"] for r in m_rows]))
        mPviol = float(np.mean([r["P_viol"] for r in m_rows]))
        fFeas = float(np.mean([1.0 if r["feasible"] else 0.0 for r in m_rows]))
        print(f"{name:<20} | {mV:<10.2f} | {mPRR:<10.4f} | {mCBR:<10.4f} | {mPviol:<10.4f} | {fFeas:<12.1%}")
    print("=" * 80)

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
