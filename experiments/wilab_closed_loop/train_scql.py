"""Sequential Capacity-Aware Q-Learning (SCQL) Closed-Loop Training & Evaluation.

Replaces the 5-weight Structured Priority heuristic ranking with a value-based,
queue-aware sequential matching policy using Masked Double DQN.

Target to beat: Guarded B4 (V_done = 49.15, P_viol <= 2.0%) across development
validation seeds [73300, 73301, 73302, 73303].
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.scql.action_mask import (
    CandidateEdge,
    build_admissible_action_mask,
    candidate_from_dict,
)
from controllers.rl.scql.q_network import SCQLDuelingQNetwork
from controllers.rl.scql.scql_policy import SCQLSequentialPolicy
from controllers.rl.scql.replay_buffer import SCQLReplayBuffer, Transition
from controllers.rl.scql.trainer import SCQLTrainer

TRAIN_SEEDS = [73200, 73201]
DEV_SEEDS = [73300, 73301, 73302, 73303]
GUARDED_B4_TARGET = 49.15
SAFETY_LIMIT = 0.020


def run_single_episode(
    policy: SCQLSequentialPolicy,
    seed: int,
    eid: int,
    out_dir: Path,
    policy_mode: str = "scql",
    deterministic: bool = True,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Runs a single closed-loop simulation episode."""
    t_seed = 186100 + (seed - 73300) if seed >= 73300 else 191000 + (seed - 73200)
    bridge = Path(tempfile.gettempdir()) / f"bridge_scql_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    ep_dir = out_dir / f"episode_{eid:04d}"

    _, _, comp_val, _ = run_episode(
        WROOT, bridge, eid, 15, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 12, False, 175000,
        300, 12.0, policy_mode, None,
        deterministic, "off", "", True, t_seed, gov,
        175000, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=ep_dir,
        arrival_mode="streaming",
    )

    audit_file = ep_dir / f"episode_{eid:04d}" / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    viols = sum(1 for p in prrs if p < 0.95)
    pv = viols / len(valid) if valid else 1.0

    return comp_val, pv, audit


def evaluate_policy_on_dev(
    policy: SCQLSequentialPolicy,
    dev_seeds: List[int],
    out_dir: Path,
    base_eid: int = 91000,
) -> Dict[str, Any]:
    """Evaluates the SCQL policy on all 4 development validation seeds."""
    policy.eval()
    policy.expert_mode = "rl"
    policy.record_transitions = False
    policy.epsilon = 0.0

    results = []
    print("\n" + "=" * 70)
    print("  SCQL DEVELOPMENT VALIDATION EVALUATION (SEEDS 73300-73303)")
    print("=" * 70)

    for i, s in enumerate(dev_seeds):
        eid = base_eid + i + 1
        comp, pv, audit = run_single_episode(
            policy, s, eid, out_dir / f"val_s{s}", policy_mode="scql", deterministic=True
        )
        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        mean_prr = float(np.mean(prrs)) if prrs else 1.0
        results.append({
            "seed": s,
            "V_done": comp,
            "P_viol": pv,
            "mean_prr": mean_prr,
        })
        print(f"  Seed {s}: V_done = {comp:6.2f} | P_viol = {pv*100:5.2f}% | PRR = {mean_prr:.4f}")

    mean_v = float(np.mean([r["V_done"] for r in results]))
    mean_pv = float(np.mean([r["P_viol"] for r in results]))
    beats_b4 = (mean_v > GUARDED_B4_TARGET) and (mean_pv <= SAFETY_LIMIT)

    summary = {
        "mean_V_done": mean_v,
        "mean_P_viol": mean_pv,
        "results": results,
        "guarded_b4_target": GUARDED_B4_TARGET,
        "sp_target": 47.70,
        "delta_vs_b4": mean_v - GUARDED_B4_TARGET,
        "delta_vs_sp": mean_v - 47.70,
        "beats_b4": beats_b4,
        "safety_compliant": mean_pv <= SAFETY_LIMIT,
    }
    print("-" * 70)
    print(f"  Mean Completed Value: {mean_v:.2f} (Guarded B4: {GUARDED_B4_TARGET:.2f} | SP: 47.70)")
    print(f"  Delta vs Guarded B4 : {mean_v - GUARDED_B4_TARGET:+.2f}")
    print(f"  Mean Safety Violation: {mean_pv*100:.2f}% (Limit: {SAFETY_LIMIT*100:.1f}%)")
    print(f"  BEATS GUARDED B4    : {beats_b4}")
    print("=" * 70 + "\n")
    return summary


def train_scql(
    output_dir: Path,
    warm_start_steps: int = 300,
    online_episodes: int = 4,
    lr: float = 3e-4,
    batch_size: int = 32,
    tau: float = 0.01,
) -> SCQLSequentialPolicy:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    buffer = SCQLReplayBuffer(capacity=50000, alpha=0.6)
    policy = SCQLSequentialPolicy(
        b_max=175000,
        r_thresh=0.20,
        buffer=buffer,
        expert_mode="guarded_b4",
        record_transitions=True,
    )
    optimizer = torch.optim.Adam(policy.q_net.parameters(), lr=lr)
    trainer = SCQLTrainer(policy.q_net, optimizer, buffer, gamma=0.995, tau=tau)

    # Step 1: Warm-start collection via Guarded B4 on training seeds
    print("\n" + "=" * 70)
    print("[Step 1] Collecting Guarded B4 warm-start demonstrations on training seeds [73200, 73201]")
    print("=" * 70)

    for s_idx, seed in enumerate(TRAIN_SEEDS):
        eid = 85000 + s_idx
        comp, pv, audit = run_single_episode(
            policy, seed, eid, output_dir / f"warm_b4_s{seed}", policy_mode="scql", deterministic=True
        )
        policy.finalize_episode(final_completed=comp)
        print(f"  Guarded B4 Seed {seed}: V = {comp:.2f} | P_viol = {pv*100:.2f}% | Replay Buffer Size: {len(buffer)}")

    # Step 2: Warm-start pretraining on buffer
    print("\n" + "=" * 70)
    print(f"[Step 2] Pretraining Double Q-Network for {warm_start_steps} steps on warm-start buffer ({len(buffer)} transitions)")
    print("=" * 70)
    for step in range(1, warm_start_steps + 1):
        beta = 0.4 + 0.6 * (step / warm_start_steps)
        metrics = trainer.train_step(batch_size=batch_size, beta=beta)
        if step % 50 == 0 or step == warm_start_steps:
            print(f"  Step {step:03d}/{warm_start_steps}: Loss = {metrics['loss']:.4f} | TD Err = {metrics['td_error_mean']:.4f} | Grad = {metrics['grad_norm']:.4f}")

    # Save initial warm checkpoint
    warm_ckpt_path = checkpoints_dir / "scql_warm_start.pt"
    torch.save({"policy": policy.state_dict(), "buffer_len": len(buffer)}, warm_ckpt_path)
    print(f"  Saved warm-start checkpoint to {warm_ckpt_path}")

    # Step 3: Online exploration & training episodes
    print("\n" + "=" * 70)
    print(f"[Step 3] Running {online_episodes} closed-loop online training episodes on training seeds")
    print("=" * 70)
    policy.expert_mode = "rl"
    policy.record_transitions = True

    for ep in range(1, online_episodes + 1):
        seed = TRAIN_SEEDS[(ep - 1) % len(TRAIN_SEEDS)]
        eid = 86000 + ep
        eps = max(0.01, 0.08 * (1.0 - (ep - 1) / max(1, online_episodes)))
        policy.epsilon = eps

        comp, pv, audit = run_single_episode(
            policy, seed, eid, output_dir / f"train_ep{ep}_s{seed}", policy_mode="scql", deterministic=False
        )
        policy.finalize_episode(final_completed=comp)

        # Train multiple batches after episode
        losses = []
        for _ in range(25):
            m = trainer.train_step(batch_size=batch_size, beta=0.6)
            losses.append(m["loss"])

        print(f"  Episode {ep:02d} (Seed {seed}, eps={eps:.3f}): V = {comp:.2f} | P_viol = {pv*100:.2f}% | Buffer: {len(buffer)} | Mean Loss: {np.mean(losses):.4f}")

    # Save final trained checkpoint
    final_ckpt_path = checkpoints_dir / "scql_best.pt"
    torch.save({"policy": policy.state_dict()}, final_ckpt_path)
    print(f"\nSaved trained SCQL checkpoint to {final_ckpt_path}\n")

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate SCQL on WiLab Mode 2")
    parser.add_argument("--warm-steps", type=int, default=250, help="Warm-start training steps")
    parser.add_argument("--online-episodes", type=int, default=2, help="Online training episodes")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on dev seeds")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to evaluate")
    args = parser.parse_args()

    out_root = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "scql_training"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        policy = SCQLSequentialPolicy(b_max=175000, r_thresh=0.20)
        ckpt_path = Path(args.checkpoint) if args.checkpoint else ROOT / "checkpoints" / "scql_best.pt"
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            policy.load_state_dict(ckpt["policy"])
            print(f"Loaded checkpoint from {ckpt_path}")
        summary = evaluate_policy_on_dev(policy, DEV_SEEDS, out_root / "eval")
    else:
        trained_policy = train_scql(
            out_root,
            warm_start_steps=args.warm_steps,
            online_episodes=args.online_episodes,
        )
        summary = evaluate_policy_on_dev(trained_policy, DEV_SEEDS, out_root / "eval")

    summary_file = out_root / "scql_development_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_file}")
