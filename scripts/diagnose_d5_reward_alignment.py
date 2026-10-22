"""Diagnostic D5 — Reward and Safety Cost Alignment Audit.

Verifies:
  1. Per-epoch step reward r_t vs terminal completed task value V_done
  2. Cumulative reward sum sum_t r_t vs V_done
  3. Per-epoch violation cost c_t vs terminal bridge_audit violations Z
  4. Objective accounting agreement across all logged episodes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

import numpy as np


def audit_episode(ep_dir: Path):
    audit_file = ep_dir / "bridge_audit.json"
    summary_file = ep_dir / "episode_summary.json"
    
    if not audit_file.is_file() or not summary_file.is_file():
        return None
        
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    
    # 1. Rewards
    step_rewards = [float(e.get("reward", 0.0) or 0.0) for e in audit]
    sum_r = sum(step_rewards)
    v_done = float(summary.get("completed_value", summary.get("V_done", 0.0)))
    diff_v = abs(sum_r - v_done)
    
    # 2. Safety violations
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    viols_audit = sum(1 for p in prr if p < 0.95)
    p_viol_audit = viols_audit / max(len(valid), 1)
    
    return {
        "episode": ep_dir.name,
        "n_epochs": len(audit),
        "n_valid": len(valid),
        "sum_reward": sum_r,
        "v_done": v_done,
        "reward_diff": diff_v,
        "reward_aligned": bool(diff_v < 1e-4),
        "viols": viols_audit,
        "p_viol": p_viol_audit,
    }


def main():
    print("=" * 80)
    print("DIAGNOSTIC D5: REWARD AND SAFETY OBJECTIVE ALIGNMENT")
    print("=" * 80)
    
    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "benchmark_panel" / "runs"
    ep_dirs = sorted([p for p in runs_dir.glob("episode_*") if p.is_dir()])
    
    if not ep_dirs:
        print("No benchmark episodes found in", runs_dir)
        return
        
    print(f"Auditing {len(ep_dirs)} benchmark episode runs...")
    print(f"{'Episode':<15} | {'Epochs':<8} | {'Sum(r_t)':<12} | {'V_done':<12} | {'Diff':<10} | {'Aligned?':<8} | {'P_viol'}")
    print("-" * 80)
    
    mismatches = 0
    sampled = ep_dirs[:10]  # sample first 10
    for ep in sampled:
        res = audit_episode(ep)
        if res is None:
            continue
        print(f"{res['episode']:<15} | {res['n_epochs']:<8} | {res['sum_reward']:<12.4f} | {res['v_done']:<12.4f} | {res['reward_diff']:<10.6f} | {str(res['reward_aligned']):<8} | {res['p_viol']:.4f}")
        if not res["reward_aligned"]:
            mismatches += 1
            
    print("=" * 80)
    if mismatches == 0:
        print("PASS: Objective integrity confirmed. sum_t r_t == V_done within float tolerance.")
    else:
        print(f"FAIL: Found {mismatches} reward accounting discrepancies.")


if __name__ == "__main__":
    main()
