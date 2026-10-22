"""Residual PPO training on ordinary WiLab rollouts.

Uses scaled logits, VA-relative advantages, LR decay, and keeps the checkpoint
with the best mean completed-value delta vs corrected Value-Aware eval.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from train_wilab_closed_loop import ROOT, run_episode
from controllers.rl.residual_ppo import ppo_update
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.context_direct_policy import ContextConditionedDirectPolicy
from controllers.rl.safety_governor import SafetyGovernor


def save_checkpoint(path, policy, optimizer, cfg, rows, evaluations, episode, update, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "environment": cfg,
        "rows": rows,
        "evaluations": evaluations,
        "episode": int(episode),
        "update": int(update),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def final_safety(output_root: Path, episode: int):
    audit_path = output_root / f"episode_{episode:04d}" / "bridge_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    last = audit[-1] if audit else {}
    return last.get("safety_prr_cumulative"), last.get("safety_p95_delay_cumulative_ms")


def mean_delta_vs_va(eval_rows):
    """Mean residual-VA completed_value over seeds present in this eval round."""
    by_seed = {}
    for row in eval_rows:
        by_seed.setdefault(row["sim_seed"], {})[row["method"]] = row
    deltas = []
    for seed, methods in by_seed.items():
        if "residual" in methods and "value_aware" in methods:
            deltas.append(methods["residual"]["completed_value"] - methods["value_aware"]["completed_value"])
    if not deltas:
        return 0.0, 0
    return float(sum(deltas) / len(deltas)), len(deltas)


def mean_residual_v(eval_rows):
    vals = [float(r["completed_value"]) for r in eval_rows if r.get("method") == "residual"]
    if not vals:
        return 0.0, 0
    return float(sum(vals) / len(vals)), len(vals)


def selection_score(rule: str, delta: float, mean_v: float, delta_hist: list[float], ema_window: int):
    """Score used to decide whether to publish a checkpoint (higher is better)."""
    rule = str(rule)
    if rule == "best_mean_v":
        return float(mean_v)
    if rule == "best_delta_va_ema":
        hist = list(delta_hist) + [float(delta)]
        w = max(1, int(ema_window))
        window = hist[-w:]
        return float(sum(window) / len(window))
    # default: best_delta_va (and unknown → same)
    return float(delta)


def evaluate(policy, root: Path, update: int, args):
    eval_root = root / "fixed_evaluation" / f"update_{update:03d}"
    eval_root.mkdir(parents=True, exist_ok=True)
    output = []
    modes = ("residual", "value_aware", "minfinish")
    for scenario, (sim_seed, task_seed) in enumerate(zip(args.eval_sim_seeds, args.eval_task_seeds)):
        for method_offset, mode in enumerate(modes):
            episode_id = update * 100 + scenario * len(modes) + method_offset
            bridge = Path(tempfile.gettempdir()) / f"wilab_ppo_eval_{os.getpid()}_{episode_id}"
            if bridge.exists():
                shutil.rmtree(bridge)
            os.environ["WILAB_OUTPUT_ROOT"] = str(eval_root)
            governor = SafetyGovernor(args.envelope_bytes, horizon_envelope_only=True)
            total, _, completed, trace = run_episode(
                ROOT, bridge, episode_id, args.density, policy, 900.0,
                False, False, 100.0, 20.0, sim_seed, args.controller_epochs,
                None, args.tasks_per_source, False, args.envelope_bytes,
                args.safety_payload_bytes, args.simulation_time, mode, None,
                True, "off", "", True, task_seed, governor,
                args.envelope_bytes, True, True,
            )
            offered = float(trace[0]["offered_value"]) if trace else 0.0
            prr, p95 = final_safety(eval_root, episode_id)
            output.append({
                "update": update, "scenario": scenario, "method": mode,
                "sim_seed": sim_seed, "task_seed": task_seed,
                "completed_value": float(completed),
                "R_value": float(completed) / max(offered, 1e-9),
                "PRR": prr, "P95_ms": p95,
                "selected_epochs": sum(bool(x.get("selected_candidate_indices")) for x in trace),
                "deviation_fraction": sum(bool(x.get("residual_changed_anchor")) for x in trace) / max(1, len(trace)),
                "governor_interventions": sum(bool(x.get("governor_intervened")) for x in trace),
            })
    delta, n = mean_delta_vs_va(output)
    for row in output:
        row["mean_delta_completed_vs_va"] = delta
        row["n_paired_seeds"] = n
    (eval_root / "evaluation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def set_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--rollouts-per-update", type=int, default=4)
    ap.add_argument("--ppo-epochs", type=int, default=6)
    ap.add_argument("--minibatch-size", type=int, default=64)
    ap.add_argument("--controller-epochs", type=int, default=100)
    ap.add_argument("--density", type=int, default=35)
    ap.add_argument("--tasks-per-source", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=72000)
    ap.add_argument("--task-seed-base", type=int, default=182000)
    ap.add_argument("--envelope-bytes", type=int, default=175000)
    ap.add_argument("--safety-payload-bytes", type=int, default=300)
    ap.add_argument("--simulation-time", type=float, default=12.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final-frac", type=float, default=0.2,
                    help="final LR = lr * lr_final_frac after last update")
    ap.add_argument("--logit-scale", type=float, default=100.0)
    ap.add_argument("--residual-gain", type=float, default=1.0)
    ap.add_argument("--logit-mode", choices=("va_residual", "direct", "residual_only"),
                    default="va_residual")
    ap.add_argument(
        "--policy-arch",
        choices=("residual", "context_direct"),
        default="residual",
        help="residual=legacy ResidualTaskPolicy; context_direct=Stage-1 conditional Direct scorer",
    )
    ap.add_argument("--random-residual-init", action="store_true",
                    help="A6: keep random last-layer init instead of exact zero")
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--entropy-final-frac", type=float, default=0.25)
    ap.add_argument("--target-kl", type=float, default=0.015)
    ap.add_argument("--va-relative", action="store_true", default=True)
    ap.add_argument("--no-va-relative", action="store_false", dest="va_relative")
    ap.add_argument("--changed-only", action="store_true",
                    help="PPO only on steps that changed the VA anchor (escape-VA experiments)")
    ap.add_argument(
        "--gamma", type=float, default=1.0,
        help="PPO discount; default 1.0 matches SPARC Phase-0 utility lock (use 0.99 for legacy A/B)",
    )
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument(
        "--advantage-mode", choices=("gae", "w_minus_v", "blend"), default="gae",
        help="Phase-2d: gae | w_minus_v | blend (requires per-step w_estimate on traces)",
    )
    ap.add_argument("--w-blend", type=float, default=0.5,
                    help="blend weight on (W-V) when --advantage-mode=blend")
    ap.add_argument(
        "--selection-rule",
        choices=("best_delta_va", "best_mean_v", "best_delta_va_ema"),
        default="best_delta_va",
        help="Checkpoint publish rule (default preserves lock-package behavior)",
    )
    ap.add_argument("--ema-eval-window", type=int, default=3,
                    help="Window for best_delta_va_ema selection")
    ap.add_argument("--eval-every-updates", type=int, default=2)
    ap.add_argument("--eval-sim-seeds", type=int, nargs="+",
                    default=[82001, 82002, 82011, 82012])
    ap.add_argument("--eval-task-seeds", type=int, nargs="+",
                    default=[182001, 182002, 182011, 182012])
    ap.add_argument("--output-root", default="results/wilab_residual_ppo_96ep_va_relative")
    ap.add_argument("--checkpoint-out", default="checkpoints/wilab_residual_ppo_96ep_va_relative.pt")
    ap.add_argument("--continue-from", default="",
                    help="Resume from checkpoint_last/best; allows non-empty output-root")
    ap.add_argument("--extra-episodes", type=int, default=0,
                    help="With --continue-from, run this many additional episodes")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if len(args.eval_sim_seeds) != len(args.eval_task_seeds):
        raise ValueError("evaluation seed lists must have equal length")
    if args.smoke:
        args.episodes = 2
        args.rollouts_per_update = 2
        args.controller_epochs = 30
        args.eval_every_updates = 1
        args.eval_sim_seeds = args.eval_sim_seeds[:1]
        args.eval_task_seeds = args.eval_task_seeds[:1]

    random.seed(616161); np.random.seed(616161); torch.manual_seed(616161)
    root = Path(args.output_root)
    continue_from = Path(args.continue_from) if args.continue_from else None
    if continue_from is not None:
        if not continue_from.exists():
            raise SystemExit(f"--continue-from not found: {continue_from}")
        if int(args.extra_episodes) <= 0:
            raise SystemExit("--continue-from requires --extra-episodes > 0")
        root.mkdir(parents=True, exist_ok=True)
    else:
        if root.exists() and any(root.iterdir()):
            raise SystemExit(
                f"output-root already exists and is non-empty: {root}. "
                "Delete it or choose a new path to avoid clobbering a run."
            )
        root.mkdir(parents=True, exist_ok=True)
    train_root = root / "training_runs"
    train_root.mkdir(exist_ok=True)

    if args.policy_arch == "context_direct":
        policy = ContextConditionedDirectPolicy(logit_scale=args.logit_scale)
    else:
        policy = ResidualTaskPolicy(
            logit_scale=args.logit_scale,
            residual_gain=args.residual_gain,
            logit_mode=args.logit_mode,
        )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rows: list = []
    evaluations: list = []
    delta_hist: list[float] = []
    update = 0
    start_episode = 0

    if continue_from is not None:
        payload = torch.load(continue_from, map_location="cpu", weights_only=False)
        missing, unexpected = policy.load_state_dict(payload["policy"], strict=False)
        if unexpected:
            print(json.dumps({"continue_unexpected": list(unexpected)}), flush=True)
        if "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception as exc:
                print(json.dumps({"optimizer_load_warn": str(exc)}), flush=True)
        metrics_path = root / "training_metrics.json"
        if metrics_path.exists():
            rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        elif payload.get("rows"):
            rows = list(payload["rows"])
        fe_path = root / "fixed_evaluation.json"
        if fe_path.exists():
            evaluations = json.loads(fe_path.read_text(encoding="utf-8"))
        elif payload.get("evaluations"):
            evaluations = list(payload["evaluations"])
        start_episode = len(rows)
        update = int(payload.get("update", start_episode // max(1, args.rollouts_per_update)))
        for r in rows:
            if "eval_mean_delta_completed_vs_va" in r:
                delta_hist.append(float(r["eval_mean_delta_completed_vs_va"]))
        best_path = root / "checkpoint_best_delta_va.pt"
        best_meta = root / "best_checkpoint.json"
        if best_meta.exists():
            bm = json.loads(best_meta.read_text(encoding="utf-8"))
            best = {
                "score": float(bm.get("selection_score", bm.get("mean_delta_completed_vs_va", -1e9))),
                "delta": float(bm.get("mean_delta_completed_vs_va", 0.0)),
                "mean_v": float(bm.get("mean_residual_v", 0.0)),
                "update": int(bm.get("update", 0)),
                "path": best_path,
            }
        else:
            best = {
                "score": -1e9, "delta": 0.0, "mean_v": 0.0, "update": update, "path": best_path,
            }
        end_episode = start_episode + int(args.extra_episodes)
        print(json.dumps({
            "status": "continue",
            "from": str(continue_from),
            "start_episode": start_episode,
            "extra_episodes": int(args.extra_episodes),
            "end_episode": end_episode,
            "continue_missing": list(missing),
        }), flush=True)
    else:
        if args.policy_arch != "context_direct" and not args.random_residual_init:
            with torch.no_grad():
                policy.net[-1].weight.zero_()
                policy.net[-1].bias.zero_()
        end_episode = int(args.episodes)
        evaluations = evaluate(policy, root, 0, args)
        (root / "fixed_evaluation.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
        init_delta, _ = mean_delta_vs_va(evaluations)
        init_v, _ = mean_residual_v(evaluations)
        best = {
            "score": selection_score(args.selection_rule, init_delta, init_v, delta_hist, args.ema_eval_window),
            "delta": init_delta,
            "mean_v": init_v,
            "update": 0,
            "path": root / "checkpoint_best_delta_va.pt",
        }

    total_updates = max(1, end_episode // max(1, args.rollouts_per_update))
    if args.logit_mode == "va_residual":
        logit_form = (
            f"logits = {args.logit_scale:g} * (analytic + {args.residual_gain:g} * residual)"
        )
    else:
        logit_form = (
            f"logits = {args.logit_scale:g} * ({args.residual_gain:g} * residual) "
            f"[mode={args.logit_mode}; no additive VA]"
        )
    cfg = vars(args).copy()
    cfg.update({
        "algorithm": (
            f"PPO logit_mode={args.logit_mode}; va_relative={args.va_relative}; "
            "scaled logits + LR/entropy decay"
        ),
        "initialization": (
            f"continue-from {continue_from}" if continue_from is not None
            else ("random residual head" if args.random_residual_init
                  else "fresh; residual output exactly zero; seed 616161")
        ),
        "selection_rule": args.selection_rule,
        "ema_eval_window": int(args.ema_eval_window),
        "logit_forming": logit_form,
        "admission": "175-kB governor enforced before sampling; legacy action-count shield bypassed",
        "task_process": "three tasks/source spread uniformly across the episode; absolute deadlines",
        "attribution": "at most one selected task per source per epoch",
        "counterfactual_replay": False,
        "off_policy_replay": False,
        "planned_updates": total_updates,
        "start_episode": start_episode,
        "end_episode": end_episode,
    })
    (root / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if continue_from is None:
        save_checkpoint(best["path"], policy, optimizer, cfg, rows, evaluations, 0, 0,
                        extra={"best_delta_completed_vs_va": best["delta"],
                               "selection_rule": args.selection_rule,
                               "selection_score": best["score"]})
        (root / "best_checkpoint.json").write_text(json.dumps({
            "update": 0, "mean_delta_completed_vs_va": best["delta"],
            "mean_residual_v": best["mean_v"],
            "selection_rule": args.selection_rule,
            "selection_score": best["score"],
            "path": str(best["path"]),
        }, indent=2), encoding="utf-8")

    rollout_batch = []
    for episode in range(start_episode, end_episode):
        bridge = Path(tempfile.gettempdir()) / f"wilab_residual_ppo_{os.getpid()}_{episode}"
        if bridge.exists():
            shutil.rmtree(bridge)
        os.environ["WILAB_OUTPUT_ROOT"] = str(train_root)
        governor = SafetyGovernor(args.envelope_bytes, horizon_envelope_only=True)
        total, _, completed, trace = run_episode(
            ROOT, bridge, episode, args.density, policy, 900.0,
            False, False, 100.0, 20.0, args.seed_base + episode,
            args.controller_epochs, None, args.tasks_per_source, False,
            args.envelope_bytes, args.safety_payload_bytes, args.simulation_time,
            "residual", None, False, "off", "", True,
            args.task_seed_base + episode, governor, args.envelope_bytes,
            True, True,
        )
        rollout_batch.append(trace)
        offered = float(trace[0]["offered_value"]) if trace else 0.0
        row = {
            "episode": episode,
            "sim_seed": args.seed_base + episode,
            "task_seed": args.task_seed_base + episode,
            "return": float(total),
            "R_value": float(completed) / max(offered, 1e-9),
            "completed_value": float(completed),
            "offered_value": offered,
            "nonzero_reward_steps": sum(float(x.get("reward", 0.0)) != 0 for x in trace),
            "selected_epochs": sum(bool(x.get("selected_candidate_indices")) for x in trace),
            "deviation_fraction": sum(bool(x.get("residual_changed_anchor")) for x in trace) / max(1, len(trace)),
            "governor_interventions": sum(bool(x.get("governor_intervened")) for x in trace),
        }
        if len(rollout_batch) == args.rollouts_per_update:
            update += 1
            progress = (update - 1) / max(1, total_updates - 1)
            lr_now = args.lr * (1.0 - (1.0 - args.lr_final_frac) * progress)
            ent_now = args.entropy_coef * (1.0 - (1.0 - args.entropy_final_frac) * progress)
            set_lr(optimizer, lr_now)
            metrics = ppo_update(
                policy, optimizer, rollout_batch, args.envelope_bytes,
                epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
                entropy_coef=ent_now, target_kl=args.target_kl,
                unique_sources=True, va_relative=args.va_relative,
                changed_only=bool(args.changed_only),
                gamma=float(args.gamma), gae_lambda=float(args.gae_lambda),
                advantage_mode=str(args.advantage_mode),
                w_blend=float(args.w_blend),
            )
            if metrics:
                row.update(metrics)
            row["update"] = update
            row["lr"] = lr_now
            row["entropy_coef"] = ent_now
            row["gamma"] = float(args.gamma)
            row["gae_lambda"] = float(args.gae_lambda)
            row["advantage_mode"] = str(args.advantage_mode)
            row["w_blend"] = float(args.w_blend)
            rollout_batch = []
            checkpoint = root / f"checkpoint_update_{update:03d}.pt"
            save_checkpoint(checkpoint, policy, optimizer, cfg, rows + [row], evaluations, episode + 1, update)
            if update % args.eval_every_updates == 0:
                new_eval = evaluate(policy, root, update, args)
                evaluations.extend(new_eval)
                (root / "fixed_evaluation.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
                delta, n = mean_delta_vs_va(new_eval)
                mean_v, n_v = mean_residual_v(new_eval)
                row["eval_mean_delta_completed_vs_va"] = delta
                row["eval_mean_residual_v"] = mean_v
                row["eval_n_paired_seeds"] = n
                score = selection_score(
                    args.selection_rule, delta, mean_v, delta_hist, args.ema_eval_window)
                delta_hist.append(float(delta))
                if score > best["score"]:
                    best = {
                        "score": score,
                        "delta": delta,
                        "mean_v": mean_v,
                        "update": update,
                        "path": root / "checkpoint_best_delta_va.pt",
                    }
                    save_checkpoint(best["path"], policy, optimizer, cfg, rows + [row], evaluations,
                                    episode + 1, update,
                                    extra={"best_delta_completed_vs_va": delta,
                                           "mean_residual_v": mean_v,
                                           "selection_rule": args.selection_rule,
                                           "selection_score": score})
                    (root / "best_checkpoint.json").write_text(json.dumps({
                        "update": update,
                        "mean_delta_completed_vs_va": delta,
                        "mean_residual_v": mean_v,
                        "selection_rule": args.selection_rule,
                        "selection_score": score,
                        "n_paired_seeds": n,
                        "n_residual_seeds": n_v,
                        "path": str(best["path"]),
                    }, indent=2), encoding="utf-8")
        rows.append(row)
        (root / "training_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    # Prefer best-by-ΔVA as the published checkpoint; also keep last iterate.
    last_path = root / "checkpoint_last.pt"
    save_checkpoint(last_path, policy, optimizer, cfg, rows, evaluations, end_episode, update)
    if best["path"].exists():
        shutil.copy2(best["path"], Path(args.checkpoint_out))
        published = str(args.checkpoint_out)
        published_from = "best_delta_va"
    else:
        shutil.copy2(last_path, Path(args.checkpoint_out))
        published = str(args.checkpoint_out)
        published_from = "last"
    print(json.dumps({
        "status": "complete",
        "start_episode": start_episode,
        "end_episode": end_episode,
        "episodes_completed": len(rows),
        "updates": update,
        "output_root": str(root),
        "checkpoint": published,
        "published_from": published_from,
        "best_delta_completed_vs_va": best["delta"],
        "best_update": best["update"],
        "last_checkpoint": str(last_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
