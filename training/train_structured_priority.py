"""End-to-End PPO Training for Structured Priority Controller (SPARC-V2X Redesign).

Executes a bounded, disciplined training campaign on the nominal scenario:
  - Density: 15 veh/km
  - Workload: 12 tasks/source (Poisson streaming with bursts, continuous kinematics, persistent helper queues)
  - Objective: Deadline-valid completed utility
  - Constraint: 3GPP sidelink safety threshold alpha = 0.02
  - Action space: Continuous candidate-dependent priority weights w(t) in R_+^4 and volume throttle beta_t in [0, 1]
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
from typing import List, Tuple

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from metrics_logger import safety_from_audit_list
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.lagrangian_ppo import dual_ascent
from controllers.rl.pb_crl_c2 import episode_c2_from_audit, DEFAULT_K
from controllers.rl.structured_priority_policy import StructuredPriorityPolicy
from controllers.rl.structured_priority_ppo import structured_priority_ppo_update
from controllers.rl.causal_safety_guard import CausalGuardedPolicy

ENVELOPE = 175000
ALPHA = 0.02
GUARDED_B4_TARGET = 49.15
GUARDED_SP_INIT = 46.83
UNGUARDED_B4_REF = 59.79
CHECKPOINTS_DIR = ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def run_single_ep(
    policy: StructuredPriorityPolicy,
    eid: int,
    sim_seed: int,
    task_seed: int,
    runs_dir: Path,
    sim_time: float,
    density: int,
    tasks_per_source: int = 12,
    return_trace: bool = False,
    deterministic: bool = False,
):
    bridge = Path(tempfile.gettempdir()) / f"sp_tr_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)

    t0 = time.perf_counter()
    _, _, completed, trace = run_episode(
        WILAB_ROOT, bridge, eid, density, policy, 900.0,
        False, False, 100.0, 20.0, sim_seed, 100,
        None, tasks_per_source, False, ENVELOPE,
        300, sim_time, "structured_priority", None,
        deterministic, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
        arrival_mode="streaming",
    )
    wall = time.perf_counter() - t0

    ep = runs_dir / f"episode_{eid:04d}"
    audit_file = ep / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    safety = safety_from_audit_list(audit) if audit else {}
    c2 = episode_c2_from_audit(audit, K=DEFAULT_K)
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    mean_prr = float(np.mean([float(e.get("window_prr", e.get("prr", 1.0))) for e in valid])) if valid else 1.0
    mean_cbr = float(np.mean([float(e.get("cbr", 0.0)) for e in valid])) if valid else 0.0

    return float(completed), safety.get("P_viol"), c2, trace, wall, mean_prr, mean_cbr


def prepare_trace(tr):
    for item in tr:
        if not torch.is_tensor(item.get("logp")):
            item["logp"] = torch.tensor(float(item.get("logp") or 0.0))
        if not torch.is_tensor(item.get("value")):
            item["value"] = torch.tensor(float(item.get("value") or 0.0))
        if not torch.is_tensor(item.get("cost_value")):
            item["cost_value"] = torch.tensor(float(item.get("cost_value") or 0.0))
    return tr


def eval_model(
    policy: StructuredPriorityPolicy,
    seeds: List[int],
    runs_dir: Path,
    eid_start: int,
    sim_time: float,
    density: int,
    tasks_per_source: int = 12,
    workers: int = 1,
):
    policy.eval()
    tasks = []
    for i, s in enumerate(seeds):
        eid = eid_start + i + 1
        t_seed = 186100 + (s - 73300)
        tasks.append((policy, eid, s, t_seed, runs_dir, sim_time, density, tasks_per_source, False, True))

    rows = []
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            future_to_ep = {executor.submit(run_single_ep, *t): t for t in tasks}
            for fut in concurrent.futures.as_completed(future_to_ep):
                t = future_to_ep[fut]
                v, pv, c2, _, wall, prr, cbr = fut.result()
                rows.append({
                    "seed": t[2], "V_done": v, "P_viol": pv, "C2_ok": c2,
                    "mean_prr": prr, "mean_cbr": cbr, "wall_s": wall,
                })
    else:
        for t in tasks:
            v, pv, c2, _, wall, prr, cbr = run_single_ep(*t)
            rows.append({
                "seed": t[2], "V_done": v, "P_viol": pv, "C2_ok": c2,
                "mean_prr": prr, "mean_cbr": cbr, "wall_s": wall,
            })
    rows.sort(key=lambda r: r["seed"])

    mean_v = float(np.mean([r["V_done"] for r in rows]))
    mean_prr = float(np.mean([r["mean_prr"] for r in rows]))
    mean_cbr = float(np.mean([r["mean_cbr"] for r in rows]))
    mean_pviol = float(np.mean([r["P_viol"] for r in rows if r["P_viol"] is not None])) if rows else 1.0
    frac_safe = float(np.mean([1.0 if r["P_viol"] is not None and r["P_viol"] <= ALPHA else 0.0 for r in rows]))
    return mean_v, mean_prr, mean_cbr, mean_pviol, frac_safe, rows, eid_start + len(seeds)


def train(args):
    out_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "sp_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Starting Structured Priority PPO Training Campaign (Redesign)")
    print(f"Rounds: {args.rounds} | Eps/Round: {args.eps_per_round} | LR: {args.lr}")
    print(f"Sim Time: {args.sim_time}s | Density: {args.density} | TPS: {args.tasks_per_source} | Workers: {args.workers}")
    print(f"Safety Constraint Threshold ALPHA: {ALPHA:.4f}")
    print(f"{'='*80}\n")

    policy = CausalGuardedPolicy(r_thresh=0.20)
    lam_L = 0.0

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    eid = 60000
    curve_seeds = [73300, 73301, 73302, 73303]
    best_path = CHECKPOINTS_DIR / "structured_priority_best.pt"
    best_feas_path = CHECKPOINTS_DIR / "structured_priority_best_feasible.pt"

    # Task 2: Include initialized baseline-parity guarded policy as candidate checkpoint.
    # Initial Guarded SP performance on validation seeds [73300-73303] is 46.83 (P_viol = 1.25% <= 2.0%).
    # Target comparison reference is Guarded B4: 49.15 (P_viol = 1.25%).
    init_v = GUARDED_SP_INIT
    init_pv = 0.0125
    init_feasible = bool(init_pv <= ALPHA)
    best_v = init_v
    best_feasible_v = init_v if init_feasible else -1e9

    torch.save({
        "policy": policy.state_dict(),
        "lam_L": lam_L,
        "round": 0,
        "val_mean_v": init_v,
        "val_mean_p_viol": init_pv,
        "feasible": init_feasible,
    }, best_path)
    if init_feasible:
        torch.save({
            "policy": policy.state_dict(),
            "lam_L": lam_L,
            "round": 0,
            "val_mean_v": init_v,
            "val_mean_p_viol": init_pv,
            "feasible": True,
        }, best_feas_path)
    print(f"  *** REGISTERED ROUND 0 GUARDED BASELINE CANDIDATE: V = {init_v:.2f} (P_viol = {init_pv*100:.2f}%) -> {best_path} ***")
    print(f"  *** BENCHMARK TARGET: Guarded B4 = {GUARDED_B4_TARGET:.2f} | Gap to overcome: {GUARDED_B4_TARGET - init_v:+.2f} units ***\n")

    t_start = time.perf_counter()
    history = []

    for rnd in range(args.rounds):
        round_t0 = time.perf_counter()
        print(f"\n--- Round {rnd+1}/{args.rounds} (Dual lam_L={lam_L:.4f}) ---")

        # Rollout training episodes
        tasks = []
        for i in range(args.eps_per_round):
            eid += 1
            s = 73200 + rnd * args.eps_per_round + i
            t_seed = 191000 + rnd * 100 + i
            tasks.append((policy, eid, s, t_seed, runs_dir, args.sim_time, args.density, args.tasks_per_source, True, False))

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
        traces = [prepare_trace(r[3]) for r in ep_results if r[3]]

        # Decision experience & queue metrics
        informative_steps = 0
        active_states = 0
        distinct_matchings = set()
        executed_w_list = []
        executed_beta_list = []
        helper_queues = []

        for tr in traces:
            for item in tr:
                cands_cnt = item.get("n_candidates", 0)
                if cands_cnt > 0:
                    active_states += 1
                sp_obs = item.get("sp_obs")
                assn = tuple(sorted([(a["task_id"], a["helper_id"]) for a in (item.get("assignments") or item.get("executed_assignments") or [])]))
                if assn:
                    distinct_matchings.add(assn)
                if sp_obs is not None:
                    informative_steps += 1
                    if "weights" in item:
                        executed_w_list.append(item["weights"])
                    if "beta_throttle" in item:
                        executed_beta_list.append(float(item["beta_throttle"]))
                    if "hlp_t" in sp_obs:
                        q_vals = sp_obs["hlp_t"][0, :, 1].tolist()
                        helper_queues.extend(q_vals)

        # PPO update with target_kl early stopping and detached trunk for critics
        stats = structured_priority_ppo_update(
            policy, optimizer, traces, lam_L,
            epochs=args.ppo_epochs, minibatch_size=32,
            entropy_coef=0.01, value_coef=0.1, cost_coef=0.1, gamma=0.995,
            target_kl=0.02, detach_trunk=True,
        )
        mean_cost = float(stats.get("mean_epoch_violation_rate", 0.0))
        lam_L = dual_ascent(lam_L, mean_cost, cost_limit=ALPHA, lr_dual=args.lr_dual)

        round_el = time.perf_counter() - round_t0
        cur_w_unbiased = policy.base_z_w + policy.scale_w * torch.tanh(policy.head_w.bias)
        cur_b_unbiased = policy.base_z_beta + policy.scale_beta * torch.tanh(policy.head_beta.bias.squeeze(-1))
        cur_w = [float(f"{x:.3f}") for x in (torch.sigmoid(cur_w_unbiased) * policy.w_max).detach().cpu().numpy()]
        cur_beta = float(torch.sigmoid(cur_b_unbiased).detach().cpu().item())

        mean_q = float(np.mean(helper_queues)) if helper_queues else 0.0
        max_q = float(np.max(helper_queues)) if helper_queues else 0.0
        adv_r_m = float(stats.get("mean_adv_r", 0.0))
        adv_c_m = float(stats.get("mean_adv_c", 0.0))

        print(f"  Round {rnd+1} complete in {round_el:.1f}s | Mean Train V = {np.mean(train_vs):.2f} | Mean Cost = {mean_cost:.4f} | New lam_L = {lam_L:.4f}")
        print(f"  Decision Experience: Active States = {active_states} | Informative Transitions = {informative_steps} | Distinct Matchings = {len(distinct_matchings)}")
        print(f"  Queue Backlog: Mean = {mean_q:.4f} | Max = {max_q:.4f} | Adv_R = {adv_r_m:+.4f} | Adv_C = {adv_c_m:+.4f}")
        print(f"  Policy State: Priority Weights w = {cur_w} | Throttle beta = {cur_beta:.3f}")
        if stats:
            print(f"  PPO Stats: Actor Loss = {stats['actor_loss']:.4f} | Value Loss = {stats['value_loss']:.4f} | Cost Loss = {stats['cost_loss']:.4f} | Grad Norm = {stats.get('grad_norm', 0.0):.4f} | KL = {stats['approx_kl']:.4f} | Early Stopped = {stats.get('early_stopped', False)}")

        # Evaluation
        if (rnd + 1) % args.eval_every == 0 or (rnd + 1) == args.rounds:
            print(f"  >>> Evaluating on validation seeds {curve_seeds}...")
            val_v, val_prr, val_cbr, val_pv, val_safe, val_rows, eid = eval_model(
                policy, curve_seeds, runs_dir, eid, args.sim_time, args.density, args.tasks_per_source, workers=args.workers,
            )
            delta_guarded_b4 = val_v - GUARDED_B4_TARGET
            delta_guarded_init = val_v - GUARDED_SP_INIT
            delta_unguarded_b4 = val_v - UNGUARDED_B4_REF
            for r in val_rows:
                print(f"      Seed {r['seed']}: V_done = {r['V_done']:.2f} | P_viol = {r['P_viol']*100:.2f}% | PRR = {r['mean_prr']:.4f}")
            print(f"  >>> Validation Mean V: {val_v:.2f} (d_GuardedB4: {delta_guarded_b4:+.2f}, d_GuardedInit: {delta_guarded_init:+.2f}, d_UnguardedB4: {delta_unguarded_b4:+.2f})")
            print(f"      PRR: {val_prr:.4f} | CBR: {val_cbr:.4f} | Mean P_viol: {val_pv*100:.2f}% | Feasible Fraction: {val_safe*100:.1f}%")
            if val_v > GUARDED_B4_TARGET and val_pv <= ALPHA:
                print(f"  *** CONSTRAINED SUPERIORITY ACHIEVED: V = {val_v:.2f} > Guarded B4 ({GUARDED_B4_TARGET:.2f}) with P_viol = {val_pv*100:.2f}% <= 2.0% ***")

            round_entry = {
                "round": rnd + 1,
                "wall_s": round_el,
                "train_mean_v": float(np.mean(train_vs)),
                "active_states": active_states,
                "informative_steps": informative_steps,
                "distinct_matchings": len(distinct_matchings),
                "mean_queue": mean_q,
                "max_queue": max_q,
                "mean_adv_r": adv_r_m,
                "mean_adv_c": adv_c_m,
                "val_mean_v": val_v,
                "val_mean_p_viol": val_pv,
                "val_feasible_fraction": val_safe,
                "delta_guarded_b4": delta_guarded_b4,
                "delta_guarded_init": delta_guarded_init,
                "delta_unguarded_b4": delta_unguarded_b4,
                "lam_L": lam_L,
                "weights_bias": cur_w,
                "beta_bias": cur_beta,
                "actor_loss": stats.get("actor_loss", 0.0),
                "value_loss": stats.get("value_loss", 0.0),
                "cost_loss": stats.get("cost_loss", 0.0),
                "grad_norm": stats.get("grad_norm", 0.0),
                "approx_kl": stats.get("approx_kl", 0.0),
                "early_stopped": stats.get("early_stopped", False),
                "val_rows": val_rows,
            }
            history.append(round_entry)
            (out_dir / "sp_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

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
                print(f"  *** NEW BEST UTILITY CHECKPOINT: V = {val_v:.2f} (P_viol = {val_pv*100:.2f}%) -> {best_path} ***")
            else:
                print(f"  [Checkpoint Retained]: Current V = {val_v:.2f} <= best V = {best_v:.2f}; preserving reference candidate {best_path}.")

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
                print(f"  *** NEW BEST FEASIBLE CHECKPOINT: V = {val_v:.2f} (P_viol = {val_pv*100:.2f}%) -> {best_feas_path} ***")

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*80}")
    print(f"Training Completed in {total_time/60:.2f} minutes!")
    print(f"Best Validation Utility:  V = {best_v:.2f}")
    print(f"Best Feasible Utility:    V = {best_feasible_v:.2f}")
    print(f"{'='*80}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    ap.add_argument("--eps-per-round", type=int, default=5, help="Episodes per round")
    ap.add_argument("--eval-every", type=int, default=1, help="Evaluate every N rounds")
    ap.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    ap.add_argument("--lr-dual", type=float, default=0.01, help="Dual ascent learning rate")
    ap.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs")
    ap.add_argument("--sim-time", type=float, default=12.0, help="Simulation duration (s)")
    ap.add_argument("--density", type=int, default=15, help="Vehicle density")
    ap.add_argument("--tasks-per-source", type=int, default=12, help="Tasks per source")
    ap.add_argument("--workers", type=int, default=4, help="Parallel rollout workers")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
