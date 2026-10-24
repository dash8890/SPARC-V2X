"""Smoke integration episode for the corrected payload-faithful contract.

Runs one real WiLab/Python closed-loop episode and reports how many tasks
completed with their *full* payload before the deadline.  This is the gate
that must pass before any retraining.

Corrected physical configuration (defaults):
  * 20 MHz NR-V2X carrier (106 PRBs, five 20-PRB frequency resources)
  * MCS 11, 300 B PSSCH transport blocks
  * 2 ms semi-persistent reservation, 20 ms controller clock
  * 2 kB application segments fragmented into 7 x 300 B TBs
  * 10-80 kB payloads, 100-500 ms deadlines, 5-50 Mcycles, 1-4 GHz helpers
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode  # noqa: E402
from controllers.rl.sc_rlgmi.sc_rlgmi_policy import SCRLGMIPolicy  # noqa: E402
from controllers.rl.residual_task_policy import ResidualTaskPolicy  # noqa: E402
from controllers.rl.safety_governor import SafetyGovernor  # noqa: E402
from controllers.rl.pb_crl_b5_prime import B5PrimePolicy  # noqa: E402


def load_policy(mode: str, reference: bool = False):
    if mode in ("greedy_vb", "cbr_threshold", "myopic_dual", "ce_milp"):
        return ResidualTaskPolicy()
    if mode == "b5_prime":
        policy = B5PrimePolicy(state_dim=16)
        ckpt = torch.load(
            ROOT / "checkpoints" / "b5_prime_best_preserved_43p86.pt",
            map_location="cpu",
        )
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        return policy
    policy = SCRLGMIPolicy(b_max=175000, r_thresh=0.20, delta=0.0, depth=2, beam=8)
    ckpt = ROOT / "checkpoints" / "sc_rlgmi_best.pt"
    state = torch.load(ckpt, map_location="cpu")
    policy.advantage_model.load_state_dict(state["advantage_model"])
    policy.eval()
    policy.expert_mode = "reference" if reference else "rl"
    return policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=73300)
    ap.add_argument("--eid", type=int, default=98001)
    ap.add_argument("--density", type=int, default=15)
    ap.add_argument("--mode", default="scql",
                    choices=["scql", "greedy_vb", "cbr_threshold", "myopic_dual", "ce_milp", "b5_prime"],
                    help="Policy/baseline mode")
    ap.add_argument("--reference", action="store_true",
                    help="Use the SC-RLGMI deterministic reference matching")
    ap.add_argument("--no-governor", action="store_true",
                    help="Disable the execution-time safety governor")
    ap.add_argument("--envelope", type=int, default=175000,
                    help="Per-epoch safety byte envelope (governor hard cap)")
    ap.add_argument("--budget", type=int, default=175000,
                    help="Policy/action byte budget (forced_byte_budget)")
    ap.add_argument("--safety-only", action="store_true",
                    help="Run the zero-offload safety baseline (no computation)")
    ap.add_argument("--task-seed", type=int, default=None,
                    help="Override the task-layer seed (default derived from --seed)")
    ap.add_argument("--packet-bytes", type=int, default=300,
                    help="PHY transport-block payload (300 B at MCS 11 / 17 PRBs)")
    ap.add_argument("--reservation-ms", type=float, default=2.0,
                    help="Semi-persistent PSSCH reservation interval")
    ap.add_argument("--bw-mhz", type=int, default=20)
    ap.add_argument("--payload-lo", type=int, default=10_000)
    ap.add_argument("--payload-hi", type=int, default=80_000)
    ap.add_argument("--deadline-lo", type=float, default=100.0)
    ap.add_argument("--deadline-hi", type=float, default=500.0)
    ap.add_argument("--cycles-lo", type=float, default=5e6)
    ap.add_argument("--cycles-hi", type=float, default=5e7)
    ap.add_argument("--helper-cpu-lo", type=float, default=1e9)
    ap.add_argument("--helper-cpu-hi", type=float, default=4e9)
    ap.add_argument("--tasks-per-source", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--sim-time", type=float, default=12.0)
    ap.add_argument("--out-root", default="sparc_performance_evaluation/results/raw/corrected_smoke")
    args = ap.parse_args()

    tag = f"{args.mode}_tb{args.packet_bytes}_rri{args.reservation_ms:g}ms_bw{args.bw_mhz}_p{args.payload_lo//1000}-{args.payload_hi//1000}k"
    out = ROOT / args.out_root / tag / f"val_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    bridge = Path(tempfile.gettempdir()) / f"bridge_corrected_smoke_{os.getpid()}_{args.eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    task_seed = args.task_seed if args.task_seed is not None else (186100 + (args.seed - 73300))
    governor = None if args.no_governor else SafetyGovernor(
        args.envelope, horizon_envelope_only=True,
    )
    policy_mode = "scql" if args.mode == "scql" else args.mode

    _, _, completed = run_episode(
        WROOT, bridge, args.eid, args.density, load_policy(args.mode, args.reference), 900.0,
        False, args.safety_only, 100.0, 20.0, args.seed, args.epochs,
        None, args.tasks_per_source, False, args.budget, 300, args.sim_time, policy_mode, None,
        True, "off", "", False, task_seed, governor, args.budget,
        True, True, 1.0, None,
        helper_swap=None, force_pack=None,
        helper_cpu_lo=args.helper_cpu_lo, helper_cpu_hi=args.helper_cpu_hi,
        output_root=out, arrival_mode="streaming", accounting="payload",
        computation_packet_bytes=args.packet_bytes,
        tb_bytes=args.packet_bytes,
        segment_bytes=2000,
        reservation_ms=args.reservation_ms,
        bw_mhz=args.bw_mhz,
        payload_bytes_lo=args.payload_lo,
        payload_bytes_hi=args.payload_hi,
        deadline_ms_lo=args.deadline_lo,
        deadline_ms_hi=args.deadline_hi,
        cycles_lo=args.cycles_lo,
        cycles_hi=args.cycles_hi,
        packets_per_epoch=max(1, int(round(20.0 / args.reservation_ms))),
    )

    ep_dir = out / f"episode_{args.eid:04d}"
    summary = ep_dir / "episode_summary.json"
    tasks_csv = ep_dir / "task_trace.csv"
    rows = []
    if tasks_csv.is_file():
        import csv
        with tasks_csv.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    done = [r for r in rows if str(r.get("done", "")).lower() in ("true", "1")]
    payload_done = [
        r for r in done
        if int(float(r.get("payload_required", 0) or 0)) > 0
        and int(float(r.get("payload_received", 0) or 0)) >= int(float(r.get("payload_required", 0) or 0))
    ]
    print(f"episode_dir={ep_dir}")
    print(f"completed_value={completed:.6f}")
    print(f"tasks={len(rows)} done={len(done)} payload_complete={len(payload_done)}")
    for r in payload_done[:5]:
        print(
            "  task", r.get("task_id"), "payload", r.get("payload_received"),
            "/", r.get("payload_required"), "completion_ms", r.get("completion_time_ms"),
            "deadline_abs_ms", r.get("deadline_abs_ms"),
        )
    if summary.is_file():
        print("summary:", json.dumps(json.loads(summary.read_text()), indent=2)[:2000])


if __name__ == "__main__":
    main()
