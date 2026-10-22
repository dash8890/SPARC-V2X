"""PB-CRL End-to-End Policy Training (Phase 3).

Trains the 3-head PBCRLPolicy (DeepSets, helper encoder, radio GRU, budget block)
on the shared helper-aware multiple-choice price-and-budget knapsack packer
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
from controllers.rl.pb_crl_policy import PBCRLPolicy, PBCRLCoreAgent, EnsembleRiskModel
from controllers.rl.pb_crl_ppo import pb_crl_ppo_update
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


def run_single_ep(
    policy: PBCRLPolicy,
    eid: int,
    sim_seed: int,
    task_seed: int,
    runs_dir: Path,
    sim_time: float,
    density: int,
    return_trace: bool = False,
    deterministic: bool = False,
):
    bridge = Path(tempfile.gettempdir()) / f"pbcrl_tr_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)

    risk_model = EnsembleRiskModel()
    agent = PBCRLCoreAgent(policy, risk_model, K=1, total_epochs=100, delta_s=0.05, shield_enabled=False)

    t0 = time.perf_counter()
    _, _, completed, trace = run_episode(
        WILAB_ROOT, bridge, eid, density, agent, 900.0,
        False, False, 100.0, 20.0, sim_seed, 100,
        None, 12, False, ENVELOPE,
        300, sim_time, "b5_prime", None,
        deterministic, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
        arrival_mode="streaming",
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
        if not torch.is_tensor(item.get("logp")):
            item["logp"] = torch.tensor(float(item.get("logp") or 0.0))
        if not torch.is_tensor(item.get("value")):
            item["value"] = torch.tensor(float(item.get("value") or 0.0))
        if not torch.is_tensor(item.get("cost_value")):
            item["cost_value"] = torch.tensor(float(item.get("cost_value") or 0.0))
    return tr


def eval_model(
    policy: PBCRLPolicy,
    seeds: List[int],
    runs_dir: Path,
    eid_start: int,
    sim_time: float,
    density: int,
    workers: int = 1,
):
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
            "V": v,
            "P_viol": pv,
            "Z": c2["Z"],
            "C2_ok": c2["C2_ok"],
            "feasible_alpha": (pv is not None and float(pv) <= ALPHA),
            "mean_prr": m_prr,
            "mean_cbr": m_cbr,
        })
    mean_v = float(np.mean([r["V"] for r in rows]))
    frac_a = float(np.mean([1.0 if r["feasible_alpha"] else 0.0 for r in rows]))
    frac_c2 = float(np.mean([1.0 if r["C2_ok"] else 0.0 for r in rows]))
    mean_prr = float(np.mean([r["mean_prr"] for r in rows]))
    mean_cbr = float(np.mean([r["mean_cbr"] for r in rows]))
    mean_pviol = float(np.mean([r["P_viol"] for r in rows if r["P_viol"] is not None])) if rows else 1.0
    return mean_v, mean_prr, mean_cbr, mean_pviol, frac_a, frac_c2, rows, eid_start + len(seeds)


def train(args):
    out_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "pb_crl_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Starting PB-CRL 3-Head End-to-End Training (Phase 3)")
    print(f"Rounds: {args.rounds} | Eps/Round: {args.eps_per_round} | LR: {args.lr}")
    print(f"Sim Time: {args.sim_time}s | Density: {args.density} | TPS: 12 | Workers: {args.workers}")
    print(f"Safety Constraint Threshold ALPHA: {ALPHA:.4f}")
    print(f"{'='*80}\n")

    policy = PBCRLPolicy()
    if args.warm_start and Path(args.warm_start).is_file():
        ckpt = torch.load(args.warm_start, map_location="cpu")
        st = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
        policy.load_state_dict(st, strict=False)
        lam_L = float(ckpt.get("lam_L", 0.0)) if isinstance(ckpt, dict) else 0.0
        print(f"Loaded warm-start policy from {args.warm_start} (initial lam_L={lam_L:.4f})")
    else:
        lam_L = 0.0

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    existing_eids = [int(p.name.split("_")[1]) for p in runs_dir.glob("episode_*") if len(p.name.split("_")) > 1 and p.name.split("_")[1].isdigit()]
    eid = (max(existing_eids) + 1) if existing_eids else 30000

    train_pool = list(range(73200, 73248))
    curve_seeds = list(range(73300, 73304))
    best_v = -1e9
    best_feasible_v = -1e9
    best_path = CHECKPOINTS_DIR / "pb_crl_trained_best.pt"
    best_feas_path = CHECKPOINTS_DIR / "pb_crl_trained_best_feasible.pt"
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

        ep_results = []
        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
                future_to_ep = {executor.submit(run_single_ep, *t): t for t in tasks}
                for fut in concurrent.futures.as_completed(future_to_ep):
                    ep_results.append(fut.result())
        else:
            for t in tasks:
                ep_results.append(run_single_ep(*t))

        train_vs = [r[0] for r in ep_results]
        traces = [prepare_trace(r[3], policy) for r in ep_results if r[3]]

        stats = pb_crl_ppo_update(
            policy, optimizer, traces, lam_L,
            epochs=args.ppo_epochs, minibatch_size=32,
            entropy_coef=0.01, value_coef=0.1, cost_coef=0.1, gamma=0.995,
        )
        mean_cost = float(stats["mean_epoch_violation_rate"]) if stats else 0.0
        lam_L = dual_ascent(lam_L, mean_cost, cost_limit=ALPHA, lr_dual=args.lr_dual)

        round_el = time.perf_counter() - round_t0
        print(f"  Round {rnd+1} complete in {round_el:.1f}s | Mean Train V = {np.mean(train_vs):.2f} | Mean Cost = {mean_cost:.4f} | New lam_L = {lam_L:.4f}")
        if stats:
            print(f"  PPO Stats: Actor Loss = {stats['actor_loss']:.4f} | Value Loss = {stats['value_loss']:.4f} | KL = {stats['approx_kl']:.4f}")

        if (rnd + 1) % args.eval_every == 0 or (rnd + 1) == args.rounds:
            print(f"  >>> Evaluating on validation seeds {curve_seeds}...")
            val_v, val_prr, val_cbr, val_pv, val_fa, val_fc2, val_rows, eid = eval_model(
                policy, curve_seeds, runs_dir, eid, args.sim_time, args.density, workers=args.workers,
            )
            print(f"  >>> Validation Mean V: {val_v:.2f} | PRR: {val_prr:.4f} | CBR: {val_cbr:.4f} | Mean P_viol: {val_pv:.4f} | Frac Alpha Safe: {val_fa*100:.1f}%")

            entry = {
                "round": rnd + 1,
                "train_mean_v": float(np.mean(train_vs)),
                "val_mean_v": val_v,
                "val_mean_prr": val_prr,
                "val_mean_cbr": val_cbr,
                "val_mean_p_viol": val_pv,
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
                    "val_mean_p_viol": val_pv,
                    "feasible": bool(val_pv <= ALPHA),
                }, best_path)
                print(f"  *** NEW BEST UTILITY CHECKPOINT: V = {val_v:.2f} (P_viol = {val_pv:.4f}) -> {best_path} ***")

            if val_pv <= ALPHA and val_v > best_feasible_v:
                best_feasible_v = val_v
                torch.save({
                    "policy": policy.state_dict(),
                    "lam_L": lam_L,
                    "round": rnd + 1,
                    "val_mean_v": val_v,
                    "val_mean_p_viol": val_pv,
                    "feasible": True,
                }, best_feas_path)
                print(f"  *** NEW BEST FEASIBLE CHECKPOINT: V = {val_v:.2f} (P_viol = {val_pv:.4f}) -> {best_feas_path} ***")

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*80}")
    print(f"PB-CRL Training Completed in {total_time/60:.2f} minutes!")
    print(f"Best Validation Utility: V = {best_v:.2f}")
    print(f"Best Feasible Score:     V = {best_feasible_v:.2f}")
    print(f"{'='*80}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    ap.add_argument("--eps-per-round", type=int, default=3, help="Episodes per round")
    ap.add_argument("--eval-every", type=int, default=1, help="Evaluate every N rounds")
    ap.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    ap.add_argument("--lr-dual", type=float, default=0.01, help="Dual ascent learning rate")
    ap.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs")
    ap.add_argument("--sim-time", type=float, default=12.0, help="Simulation duration (s)")
    ap.add_argument("--density", type=int, default=15, help="Vehicle density")
    ap.add_argument("--workers", type=int, default=2, help="Parallel rollout workers")
    ap.add_argument("--warm-start", type=str, default="", help="Path to checkpoint to warm-start training")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
