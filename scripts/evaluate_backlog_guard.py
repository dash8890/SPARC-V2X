"""Evaluate the backlog-aware safety-reserved governor (Track 1).

Runs the frozen SC-RLGMI checkpoint under the backlog-aware governor and reports
completed value, the legacy window metric, and the governor's own causal
diagnostics.  Nothing about the policy, workload, reward, simulator or the
B1-B5 baselines is modified.

Examples
--------
Calibrate on the training seeds:
    python scripts/evaluate_backlog_guard.py --seeds 73200,73201 \
        --horizon 2 --reserve-frac 0.8 --label train_h2_r08 --base-eid 96000

Evaluate the frozen configuration on the development seeds:
    python scripts/evaluate_backlog_guard.py --seeds 73300,73301,73302,73303 \
        --horizon 2 --reserve-frac 0.8 --label dev_h2_r08 --base-eid 98200

Same-guard greedy reference ablation (isolates the learning contribution):
    python scripts/evaluate_backlog_guard.py --seeds 73300,... --reference ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode  # noqa: E402
from controllers.rl.sc_rlgmi.sc_rlgmi_policy import SCRLGMIPolicy  # noqa: E402
from controllers.rl.backlog_governor import BacklogAwareGovernor  # noqa: E402


def load_policy(reference: bool = False) -> SCRLGMIPolicy:
    policy = SCRLGMIPolicy(b_max=175000, r_thresh=0.20, delta=0.0, depth=2, beam=8)
    ckpt = ROOT / "checkpoints" / "sc_rlgmi_best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing frozen checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    policy.advantage_model.load_state_dict(state["advantage_model"])
    policy.eval()
    policy.expert_mode = "reference" if reference else "rl"
    return policy


def run_one(policy, governor, seed: int, eid: int, ep_dir: Path) -> Dict[str, Any]:
    # Seed mapping is identical to the frozen evaluation driver.
    t_seed = 186100 + (seed - 73300) if seed >= 73300 else 191000 + (seed - 73200)
    bridge = Path(tempfile.gettempdir()) / f"bridge_backlog_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)

    _, _, comp_val, _ = run_episode(
        WROOT, bridge, eid, 15, policy, 900.0, False, False, 100.0, 20.0, seed,
        100, None, 12, False, 175000, 300, 12.0, "scql", None, True, "off", "",
        True, t_seed, governor, 175000, True, True, 1.0, None,
        helper_swap=None, force_pack=None, helper_cpu_lo=2e8, helper_cpu_hi=1e9,
        output_root=ep_dir, arrival_mode="streaming",
    )

    audit_file = ep_dir / f"episode_{eid:04d}" / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    viols = sum(1 for p in prrs if p < 0.95)
    pv = viols / len(valid) if valid else 1.0
    rx = sum(float(e.get("safety_rx_cumulative") or 0) for e in valid[-1:])
    return {
        "seed": seed,
        "episode": eid,
        "V_done": float(comp_val),
        "P_viol": float(pv),
        "n_viols": int(viols),
        "n_valid": len(valid),
        "mean_prr": float(np.mean(prrs)) if prrs else None,
        "pending_state_present": bool(audit and "wilab_pending_packets" in audit[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--reserve-frac", type=float, default=1.0)
    ap.add_argument("--ewma-alpha", type=float, default=0.3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-eid", type=int, default=96000)
    ap.add_argument("--reference", action="store_true",
                    help="Use the deterministic reference matching (no beam search).")
    ap.add_argument("--out-root", default="sparc_performance_evaluation/results/raw/backlog_guard")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_root = ROOT / args.out_root / args.label
    out_root.mkdir(parents=True, exist_ok=True)

    policy = load_policy(reference=args.reference)
    results: List[Dict[str, Any]] = []

    for i, seed in enumerate(seeds):
        eid = args.base_eid + i + 1
        gov = BacklogAwareGovernor(
            max_bytes=175000,
            horizon=args.horizon,
            reserve_frac=args.reserve_frac,
            ewma_alpha=args.ewma_alpha,
        )
        ep_dir = out_root / f"val_s{seed}"
        print(f"[{args.label}] seed {seed} (episode {eid}) ...", flush=True)
        r = run_one(policy, gov, seed, eid, ep_dir)
        r["governor"] = gov.summary()
        (out_root / f"governor_log_s{seed}.json").write_text(
            json.dumps(gov.log, indent=1), encoding="utf-8"
        )
        results.append(r)
        print(
            f"  V_done={r['V_done']:6.2f}  P_viol={r['P_viol']*100:5.2f}% "
            f"({r['n_viols']}/{r['n_valid']})  pending_state={r['pending_state_present']}",
            flush=True,
        )

    mean_v = float(np.mean([r["V_done"] for r in results]))
    mean_pv = float(np.mean([r["P_viol"] for r in results]))
    pooled_viols = sum(r["n_viols"] for r in results)
    pooled_windows = sum(r["n_valid"] for r in results)
    summary = {
        "label": args.label,
        "seeds": seeds,
        "horizon": args.horizon,
        "reserve_frac": args.reserve_frac,
        "ewma_alpha": args.ewma_alpha,
        "reference_mode": args.reference,
        "mean_V_done": mean_v,
        "mean_P_viol": mean_pv,
        "pooled_viols": pooled_viols,
        "pooled_windows": pooled_windows,
        "pooled_P_viol": (pooled_viols / pooled_windows) if pooled_windows else None,
        "beats_b1_39_20": mean_v > 39.20,
        "utility_target_44_96": mean_v >= 44.96,
        "results": results,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 60)
    print(f"{args.label}: mean V={mean_v:.2f}  mean P_viol={mean_pv*100:.2f}%  "
          f"pooled={pooled_viols}/{pooled_windows}")
    print(f"summary -> {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
