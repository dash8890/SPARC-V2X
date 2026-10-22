"""B4: Unshielded Lagrangian PPO on WiLab (no SPARC PRR/P95 governor).

Physical interface capacity: forced_byte_budget (default 500 kB), unique sources.
Dual ascent on mean epoch violation rate vs cost_limit.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from train_wilab_residual_ppo import evaluate, mean_delta_vs_va, save_checkpoint, set_lr
from controllers.rl.lagrangian_ppo import dual_ascent, lagrangian_ppo_update
from controllers.rl.residual_task_policy import ResidualTaskPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--rollouts-per-update", type=int, default=4)
    ap.add_argument("--ppo-epochs", type=int, default=6)
    ap.add_argument("--minibatch-size", type=int, default=64)
    ap.add_argument("--controller-epochs", type=int, default=100)
    ap.add_argument("--density", type=int, default=35)
    ap.add_argument("--tasks-per-source", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=74000)
    ap.add_argument("--task-seed-base", type=int, default=184000)
    ap.add_argument("--physical-byte-cap", type=int, default=500000,
                    help="disclosed physical interface capacity (not SPARC safety envelope)")
    ap.add_argument("--safety-payload-bytes", type=int, default=300)
    ap.add_argument("--simulation-time", type=float, default=12.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final-frac", type=float, default=0.2)
    ap.add_argument("--logit-scale", type=float, default=100.0)
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--entropy-final-frac", type=float, default=0.25)
    ap.add_argument("--target-kl", type=float, default=0.015)
    ap.add_argument("--lambda0", type=float, default=0.1)
    ap.add_argument("--cost-limit", type=float, default=0.05)
    ap.add_argument("--lr-dual", type=float, default=0.05)
    ap.add_argument("--eval-every-updates", type=int, default=2)
    ap.add_argument("--eval-sim-seeds", type=int, nargs="+",
                    default=[82001, 82002, 82011, 82012])
    ap.add_argument("--eval-task-seeds", type=int, nargs="+",
                    default=[182001, 182002, 182011, 182012])
    ap.add_argument("--output-root", default="results/wilab_lagrangian_ppo_96ep")
    ap.add_argument("--checkpoint-out", default="checkpoints/wilab_lagrangian_ppo_96ep.pt")
    # evaluate() expects envelope_bytes attribute
    ap.add_argument("--envelope-bytes", type=int, default=175000,
                    help="used only for validation SPARC-style governor baselines in evaluate()")
    args = ap.parse_args()
    args.va_relative = False  # for evaluate compatibility if referenced

    random.seed(616161); np.random.seed(616161); torch.manual_seed(616161)
    root = Path(args.output_root)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(
            f"output-root already exists and is non-empty: {root}. "
            "Delete it or choose a new path to avoid clobbering a run."
        )
    root.mkdir(parents=True, exist_ok=True)
    train_root = root / "training_runs"
    train_root.mkdir(exist_ok=True)

    policy = ResidualTaskPolicy(logit_scale=args.logit_scale, residual_gain=1.0, logit_mode="direct")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    lam = float(args.lambda0)
    total_updates = max(1, args.episodes // max(1, args.rollouts_per_update))
    cfg = vars(args).copy()
    cfg.update({
        "algorithm": "Unshielded Lagrangian PPO (B4)",
        "admission": f"no SPARC governor; physical_byte_cap={args.physical_byte_cap}",
        "cost": "per-epoch union PRR/P95 violation on valid windows only; invalid→0",
        "dual": f"lambda0={args.lambda0}, cost_limit={args.cost_limit}, lr_dual={args.lr_dual}",
        "planned_updates": total_updates,
    })
    (root / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Validation still uses SPARC governor path for fair ΔVA reporting of the learned actor
    evaluations = evaluate(policy, root, 0, args)
    (root / "fixed_evaluation.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
    best = {"delta": mean_delta_vs_va(evaluations)[0], "update": 0,
            "path": root / "checkpoint_best_delta_va.pt"}
    save_checkpoint(best["path"], policy, optimizer, cfg, [], evaluations, 0, 0,
                    extra={"best_delta_completed_vs_va": best["delta"], "lambda": lam})

    rows, rollout_batch = [], []
    update = 0
    for episode in range(args.episodes):
        bridge = Path(tempfile.gettempdir()) / f"wilab_lag_ppo_{os.getpid()}_{episode}"
        if bridge.exists():
            shutil.rmtree(bridge)
        os.environ["WILAB_OUTPUT_ROOT"] = str(train_root)
        total, _, completed, trace = run_episode(
            WILAB_ROOT, bridge, episode, args.density, policy, 900.0,
            False, False, 100.0, 20.0, args.seed_base + episode,
            args.controller_epochs, None, args.tasks_per_source, False,
            args.physical_byte_cap, args.safety_payload_bytes, args.simulation_time,
            "residual", None, False, "off", "", True,
            args.task_seed_base + episode, None, args.physical_byte_cap,
            True, True,
        )
        rollout_batch.append(trace)
        offered = float(trace[0]["offered_value"]) if trace else 0.0
        row = {
            "episode": episode,
            "return": float(total),
            "completed_value": float(completed),
            "R_value": float(completed) / max(offered, 1e-9),
            "lambda": lam,
        }
        if len(rollout_batch) == args.rollouts_per_update:
            update += 1
            progress = (update - 1) / max(1, total_updates - 1)
            lr_now = args.lr * (1.0 - (1.0 - args.lr_final_frac) * progress)
            ent_now = args.entropy_coef * (1.0 - (1.0 - args.entropy_final_frac) * progress)
            set_lr(optimizer, lr_now)
            metrics = lagrangian_ppo_update(
                policy, optimizer, rollout_batch, args.physical_byte_cap, lam=lam,
                epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
                entropy_coef=ent_now, target_kl=args.target_kl,
            )
            if metrics:
                row.update(metrics)
                lam = dual_ascent(lam, metrics["mean_epoch_violation_rate"],
                                  cost_limit=args.cost_limit, lr_dual=args.lr_dual)
                row["lambda_after"] = lam
            row["update"] = update
            row["lr"] = lr_now
            rollout_batch = []
            if update % args.eval_every_updates == 0 or update == total_updates:
                evaluations.extend(evaluate(policy, root, update, args))
                (root / "fixed_evaluation.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
                delta, _ = mean_delta_vs_va([e for e in evaluations if e["update"] == update])
                if delta > best["delta"]:
                    best = {"delta": delta, "update": update, "path": root / "checkpoint_best_delta_va.pt"}
                    save_checkpoint(best["path"], policy, optimizer, cfg, rows, evaluations,
                                    episode, update, extra={"best_delta_completed_vs_va": delta, "lambda": lam})
                (root / "best_checkpoint.json").write_text(json.dumps({
                    "update": best["update"], "mean_delta_completed_vs_va": best["delta"],
                    "path": str(best["path"]), "lambda": lam,
                }, indent=2), encoding="utf-8")
        rows.append(row)
        (root / "training_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({k: row[k] for k in row if k in (
            "episode", "return", "completed_value", "update", "lambda", "mean_epoch_violation_rate"
        )} ), flush=True)

    save_checkpoint(Path(args.checkpoint_out), policy, optimizer, cfg, rows, evaluations,
                    args.episodes - 1, update, extra={"lambda": lam, "best": best})
    # also copy best into evidence package
    evid = ROOT / "sparc_performance_evaluation" / "checkpoints" / "lagrangian_ppo"
    evid.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["path"], evid / "checkpoint_best_delta_va.pt")
    best_public = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in best.items()
    }
    print(json.dumps({"done": True, "best": best_public, "lambda": lam}), flush=True)


if __name__ == "__main__":
    main()
