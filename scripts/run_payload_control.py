"""Run one controlled SC-RLGMI episode under a chosen completion accounting.

This is the driver used for the legacy-vs-payload comparison in
``docs/dual_track/CORRECTED_CONTRACT_RESULTS.md``.  It keeps the frozen
SC-RLGMI checkpoint and the frozen evaluation configuration (density 15,
100 controller epochs, streaming arrivals, unique sources) and changes only
``accounting``.

Examples
--------
    python scripts/run_payload_control.py --seed 73300 --eid 97002 --accounting payload
    python scripts/run_payload_control.py --seed 73300 --eid 97003 --accounting legacy

The episode is written to
``sparc_performance_evaluation/results/raw/payload_control/<accounting>/val_s<seed>/episode_<eid>``.
"""
from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=73300)
    ap.add_argument("--eid", type=int, required=True)
    ap.add_argument("--accounting", choices=["payload", "legacy"], default="payload")
    ap.add_argument("--density", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out-root", default="sparc_performance_evaluation/results/raw/payload_control")
    ap.add_argument("--reference", action="store_true",
                    help="Use the deterministic reference matching instead of the RL expert.")
    ap.add_argument("--computation-packet-bytes", type=int, default=1000,
                    help="Application payload bytes carried per Mode-2 packet")
    ap.add_argument("--packets-per-epoch", type=int, default=1,
                    help="Payload packets the Mode-2 queue can admit per 20 ms epoch")
    ap.add_argument("--r-comm", type=float, default=250.0,
                    help="Application contact radius (m) used for helper eligibility")
    args = ap.parse_args()

    policy = load_policy(reference=args.reference)
    if args.seed >= 73300:
        t_seed = 186100 + (args.seed - 73300)
    else:
        t_seed = 191000 + (args.seed - 73200)

    out = ROOT / args.out_root / args.accounting / f"val_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    bridge = Path(tempfile.gettempdir()) / f"bridge_payload_control_{os.getpid()}_{args.eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)

    _, _, completed, _ = run_episode(
        WROOT, bridge, args.eid, args.density, policy, 900.0,
        False, False, 100.0, 20.0, args.seed, args.epochs,
        None, 12, False, 175000, 300, 12.0, "scql", None,
        True, "off", "", True, t_seed, None, 175000,
        True, True, 1.0, None,
        helper_swap=None, force_pack=None, helper_cpu_lo=2e8, helper_cpu_hi=1e9,
        output_root=out, arrival_mode="streaming", accounting=args.accounting,
        computation_packet_bytes=args.computation_packet_bytes,
        packets_per_epoch=args.packets_per_epoch,
        r_comm=args.r_comm,
    )
    ep_dir = out / f"episode_{args.eid:04d}"
    print(f"{args.accounting} seed={args.seed} eid={args.eid} V_done={completed:.6f}")
    print(ep_dir)
    print("verify with:")
    print(f"  python scripts/verify_payload_accounting.py --episode-dir {ep_dir}")


if __name__ == "__main__":
    main()
