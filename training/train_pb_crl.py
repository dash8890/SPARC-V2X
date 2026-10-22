"""PB-CRL End-to-End Policy Training (Phase 3).

Trains the policy on the shared helper-aware multiple-choice price-and-budget knapsack packer
under Lagrangian PPO in closed-loop WiLab simulation (12.0s horizon, active SPS reselection).
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.pb_crl_b5_prime import B5PrimePolicy
from controllers.rl.pb_crl_b5_prime_ppo import b5_prime_ppo_update
from controllers.rl.pb_crl_c2 import DEFAULT_K, episode_c2_from_audit
from controllers.rl.lagrangian_ppo import dual_ascent
from controllers.rl.safety_governor import SafetyGovernor


def safety_from_audit(audit_file: Path) -> dict:
    if not audit_file.is_file():
        return {}
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        cbr = [float(e.get("cbr", 0.0)) for e in valid]
        viols = sum(1 for p in prr if p < 0.95)
        return {
            "P_viol": viols / max(len(valid), 1),
            "n_valid": len(valid),
            "n_viols": viols,
            "mean_prr": float(np.mean(prr)) if prr else 1.0,
            "mean_cbr": float(np.mean(cbr)) if cbr else 0.0,
        }
    except Exception:
        return {}

ENVELOPE = 175000
ALPHA = 0.02
CHECKPOINTS_DIR = ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def run_single_ep(policy, eid: int, sim_seed: int, task_seed: int, runs_dir: Path, sim_time: float, density: int, return_trace: bool = False, deterministic: bool = False):
    bridge = Path(tempfile.gettempdir()) / f"pbcrl_tr_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    os.environ["WILAB_OUTPUT_ROOT"] = str(runs_dir)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)

    t0 = time.perf_counter()
    _, _, completed, trace = run_episode(
        WILAB_ROOT, bridge, eid, density, policy, 900.0,
        False, False, 100.0, 20.0, sim_seed, 100,
        None, 4, False, ENVELOPE,
        300, sim_time, "b5_prime", None,
        True, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
    )
    wall = time.perf_counter() - t0

    ep = runs_dir / f"episode_{eid:04d}"
    audit_file = ep / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    safety = safety_from_audit(audit_file) if audit_file.is_file() else {}
    c2 = episode_c2_from_audit(audit, K=DEFAULT_K)

    return float(completed), safety.get("P_viol"), c2, trace, wall, safety.get("mean_prr", 1.0), safety.get("mean_cbr", 0.0)


def prepare_trace(tr, pol):
    for item in tr:
        obs = item.get("pb_state") or item.get("observation") or [0.0] * 16
        if not torch.is_tensor(item.get("value")):
            item["value"] = pol.value(torch.tensor(obs, dtype=torch.float32)).detach()
        if not torch.is_tensor(item.get("cost_value")):
            item["cost_value"] = pol.cost_value(torch.tensor(obs, dtype=torch.float32)).detach()
        if not torch.is_tensor(item.get("logp")):
            item["logp"] = torch.tensor(float(item.get("logp") or 0.0))
    return tr


def eval_model(policy, seeds: List[int], runs_dir: Path, eid_start: int, sim_time: float, density: int, workers: int = 1):
    policy.eval()
    tasks = []
    for i, s in enumerate(seeds):
        eid = eid_start + i
        t_seed = 190000 + (s % 10000)
        tasks.append((policy, eid, s, t_seed, runs_dir, sim_time, density, False, True))

    ep_results = []
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            future_to_ep = {executor.submit(run_single_ep, *t): t for t in tasks}
            for fut in concurrent.futures.as_completed(future_to_ep):
                ep_results.append(fut.result())
    else:
        for t in tasks:
            ep_results.append(run_single_ep(*t))

    rows = []
    for (v, pv, c2, _, _, m_prr, m_cbr) in ep_results:
        rows.append({
            "V": v, "P_viol": pv, "Z": c2["Z"], "C2_ok": c2["C2_ok"],
            "feasible_alpha": (pv is not None and float(pv) <= ALPHA),
            "mean_prr": m_prr,
            "mean_cbr": m_cbr,
        })
    mean_v = float(np.mean([r["V"] for r in rows]))
    frac_a = float(np.mean([1.0 if r["feasible_alpha"] else 0.0 for r in rows]))
    frac_c2 = float(np.mean([1.0 if r["C2_ok"] else 0.0 for r in rows]))
    mean_prr = float(np.mean([r["mean_prr"] for r in rows]))
    mean_cbr = float(np.mean([r["mean_cbr"] for r in rows]))
    return mean_v, mean_prr, mean_cbr, frac_a, frac_c2, rows, eid_start + len(seeds)


def train(args):
    out_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "pb_crl_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Starting PB-CRL End-to-End Training (Phase 3)")
    print(f"Rounds: {args.rounds} | Eps/Round: {args.eps_per_round} | LR: {args.lr}")
    print(f"Sim Time: {args.sim_time}s | Density: {args.density} | TPS: 4 | Workers: {args.workers}")
    print(f"{'='*80}\n")

    # Initialize policy
    policy = B5PrimePolicy(log_std_lam=-1.0, log_std_beta=-1.0)
    if args.warm_start and Path(args.warm_start).is_file():
        ckpt = torch.load(args.warm_start, map_location="cpu")
        st = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
        policy.load_state_dict(st)
        lam_L = float(ckpt.get("lam_L", 0.0)) if isinstance(ckpt, dict) else 0.0
        print(f"Loaded warm-start policy from {args.warm_start} (initial lam_L={lam_L:.4f})")
    else:
        # Default unbiased start with high admission and low radio price
        torch.nn.init.zeros_(policy.mu_lam.weight)
        torch.nn.init.constant_(policy.mu_lam.bias, -3.5)
        torch.nn.init.zeros_(policy.mu_beta.weight)
        torch.nn.init.constant_(policy.mu_beta.bias, 3.5)
        lam_L = 0.0

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    existing_eids = [int(p.name.split("_")[1]) for p in runs_dir.glob("episode_*") if len(p.name.split("_")) > 1 and p.name.split("_")[1].isdigit()]
    eid = (max(existing_eids) + 1) if existing_eids else 30000

    train_pool = list(range(73200, 73248))
    curve_seeds = list(range(73300, 73304))
    best_v = -1e9
    best_path = CHECKPOINTS_DIR / "pb_crl_trained_best.pt"
    curve_history = []

    t_start = time.perf_counter()

    for rnd in range(args.rounds):
        round_t0 = time.perf_counter()
        seeds = [train_pool[(rnd * args.eps_per_round + i) % len(train_pool)] for i in range(args.eps_per_round)]
        print(f"\n--- Round {rnd+1}/{args.rounds} (Dual lam_L={lam_L:.4f}) ---")
        print(f"  Rollout Seeds: {seeds}")

        policy.train()
        tasks = []
        for i, s in enumerate(seeds):
            eid += 1
            t_seed = 191000 + rnd * 100 + i
            tasks.append((policy, eid, s, t_seed, runs_dir, args.sim_time, args.density, True, False))

        # Parallel episode rollouts
        ep_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_ep = {executor.submit(run_single_ep, *t): t for t in tasks}
            for fut in concurrent.futures.as_completed(future_to_ep):
                ep_results.append(fut.result())

        train_vs = [r[0] for r in ep_results]
        traces = [prepare_trace(r[3], policy) for r in ep_results if r[3]]

        # PPO policy and value updates
        stats = b5_prime_ppo_update(
            policy, optimizer, traces, lam_L,
            epochs=args.ppo_epochs, minibatch_size=64,
            entropy_coef=0.01, value_coef=0.1, gamma=0.995,
        )
        mean_cost = float(stats["mean_epoch_violation_rate"]) if stats else 0.0
        lam_L = min(0.04, dual_ascent(lam_L, mean_cost, cost_limit=0.08, lr_dual=0.01))

        round_el = time.perf_counter() - round_t0
        print(f"  Round {rnd+1} complete in {round_el:.1f}s | Mean Train V = {np.mean(train_vs):.2f} | Mean Cost = {mean_cost:.4f} | New lam_L = {lam_L:.4f}")

        # Evaluation every eval_every rounds
        if (rnd + 1) % args.eval_every == 0 or (rnd + 1) == args.rounds:
            print(f"  >>> Evaluating on validation seeds {curve_seeds}...")
            val_v, val_prr, val_cbr, val_fa, val_fc2, val_rows, eid = eval_model(
                policy, curve_seeds, runs_dir, eid, args.sim_time, args.density, workers=args.workers,
            )
            print(f"  >>> Validation Mean V: {val_v:.2f} | PRR: {val_prr:.4f} | CBR: {val_cbr:.4f} | Frac Alpha Safe: {val_fa*100:.1f}% | Frac C2 Safe: {val_fc2*100:.1f}%")

            entry = {
                "round": rnd + 1,
                "train_mean_v": float(np.mean(train_vs)),
                "val_mean_v": val_v,
                "val_mean_prr": val_prr,
                "val_mean_cbr": val_cbr,
                "frac_alpha": val_fa,
                "frac_c2": val_fc2,
                "lam_L": lam_L,
                "elapsed_s": time.perf_counter() - t_start,
            }
            curve_history.append(entry)
            (out_dir / "training_history.json").write_text(json.dumps(curve_history, indent=2), encoding="utf-8")

            if val_v > best_v:
                best_v = val_v
                torch.save({
                    "policy": policy.state_dict(),
                    "lam_L": lam_L,
                    "round": rnd + 1,
                    "val_mean_v": val_v,
                }, best_path)
                print(f"  *** NEW BEST CHECKPOINT SAVED: V = {val_v:.2f} -> {best_path} ***")

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*80}")
    print(f"PB-CRL Training Completed in {total_time/60:.2f} minutes!")
    print(f"Best Validation Score: V = {best_v:.2f}")
    print(f"Best Checkpoint: {best_path}")
    print(f"{'='*80}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6, help="Number of training rounds")
    ap.add_argument("--eps-per-round", type=int, default=3, help="Episodes per round")
    ap.add_argument("--eval-every", type=int, default=2, help="Evaluate every N rounds")
    ap.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    ap.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs")
    ap.add_argument("--sim-time", type=float, default=12.0, help="Simulation duration (s)")
    ap.add_argument("--density", type=int, default=35, help="Vehicle density")
    ap.add_argument("--workers", type=int, default=3, help="Parallel rollout workers")
    ap.add_argument("--warm-start", type=str, default="", help="Path to checkpoint to warm-start training")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
