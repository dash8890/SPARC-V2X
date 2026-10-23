"""Diagnostic for Task 3: Identify Controllable Safety Violations.

For validation seeds 73300 and 73301:
Compares zero-offload (beta=0, B=0) and baseline B4 safety traces at the individual sensing-window level.
Identifies:
1. Where computation transmissions push PRR < 0.95 (offload-induced / controllable failures)
2. Where failures occur even with zero computation offload (exogenous uncoordinated CAM collisions)
3. Evaluates admission control / governor actions that suppress offload-induced violations
   under the frozen absolute safety definition (P_viol <= 0.02).
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.structured_priority_policy import StructuredPriorityPolicy


class ZeroOffloadPolicy:
    def __init__(self):
        self.weights = [2.5, 1.887, 1.887, 1.887]
        self.beta = 0.0
        self.is_structured_priority_policy = True

    def act_numpy(self, state, layer=None, *, deterministic: bool = True):
        return [2.5, 1.887, 1.887, 1.887], 0.0, torch.tensor(0.0), [0.0, -0.5, -0.5, -0.5], -10.0, None


def run_test_episode(policy, mode_name: str, sim_seed: int, eid: int, runs_dir: Path, gov=None):
    bridge = Path(tempfile.gettempdir()) / f"diag_s3_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    if gov is None:
        gov = SafetyGovernor(175000, horizon_envelope_only=True)
    t_seed = 186100 + (sim_seed - 73300)

    t0 = time.perf_counter()
    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, 15, policy, 900.0,
        False, False, 100.0, 20.0, sim_seed, 100,
        None, 12, False, 175000,
        300, 12.0, "structured_priority", None,
        True, "off", "", True, t_seed, gov,
        175000, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
        arrival_mode="streaming",
    )
    wall = time.perf_counter() - t0

    ep_dir = runs_dir / f"episode_{eid:04d}"
    audit_file = ep_dir / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    return comp_val, audit, wall


def main():
    print("=" * 80)
    print("TASK 3: CONTROLLABLE SAFETY VIOLATION AUDIT (SEEDS 73300 & 73301)")
    print("=" * 80)

    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "controllable_safety_audit"
    runs_dir.mkdir(parents=True, exist_ok=True)

    seeds = [73300, 73301]
    sp_b4_init = StructuredPriorityPolicy()
    zero_pol = ZeroOffloadPolicy()

    analysis = {}

    for s in seeds:
        print(f"\n--- Analyzing Seed {s} ---")
        # 1. Zero offload
        eid_zero = 81000 + (s - 73300) * 10 + 0
        v_zero, audit_zero, _ = run_test_episode(zero_pol, "ZeroOffload", s, eid_zero, runs_dir)

        # 2. Baseline B4 / SP parity init
        eid_b4 = 81000 + (s - 73300) * 10 + 1
        v_b4, audit_b4, _ = run_test_episode(sp_b4_init, "B4_Init", s, eid_b4, runs_dir)

        # Extract per-window PRR, CBR, and bytes
        n_epochs = min(len(audit_zero), len(audit_b4))
        window_diffs = []
        offload_induced_viols = []
        exogenous_viols = []

        for ep in range(n_epochs):
            az = audit_zero[ep]
            ab = audit_b4[ep]

            prr_z = float(az.get("window_prr", az.get("prr", 1.0)))
            prr_b = float(ab.get("window_prr", ab.get("prr", 1.0)))
            cbr_z = float(az.get("cbr", 0.0))
            cbr_b = float(ab.get("cbr", 0.0))
            bytes_b = int(ab.get("computation_bytes_offered", 0))
            pkts_b = int(ab.get("computation_packets_offered", 0))

            viol_z = bool(prr_z < 0.95)
            viol_b = bool(prr_b < 0.95)

            if viol_b and not viol_z:
                offload_induced_viols.append({
                    "epoch": ep,
                    "prr_zero": prr_z,
                    "prr_b4": prr_b,
                    "cbr_zero": cbr_z,
                    "cbr_b4": cbr_b,
                    "bytes_offered": bytes_b,
                    "packets_offered": pkts_b,
                    "classification": "CONTROLLABLE_OFFLOAD_INDUCED",
                })
            elif viol_b and viol_z:
                exogenous_viols.append({
                    "epoch": ep,
                    "prr_zero": prr_z,
                    "prr_b4": prr_b,
                    "cbr_zero": cbr_z,
                    "cbr_b4": cbr_b,
                    "bytes_offered": bytes_b,
                    "classification": "EXOGENOUS_CAM_COLLISION",
                })

        valid_z = [e for e in audit_zero if e.get("cbr_valid") and e.get("delay_valid")]
        valid_b = [e for e in audit_b4 if e.get("cbr_valid") and e.get("delay_valid")]

        pv_z = float(np.mean([1.0 if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95 else 0.0 for e in valid_z]))
        pv_b = float(np.mean([1.0 if float(e.get("window_prr", e.get("prr", 1.0))) < 0.95 else 0.0 for e in valid_b]))

        print(f"Seed {s} Results:")
        print(f"  Zero-Offload:  V = {v_zero:.2f} | P_viol = {pv_z*100:.2f}% ({sum(1 for e in valid_z if float(e.get('window_prr', e.get('prr', 1.0))) < 0.95)} violations)")
        print(f"  B4 Baseline:   V = {v_b4:.2f} | P_viol = {pv_b*100:.2f}% ({sum(1 for e in valid_b if float(e.get('window_prr', e.get('prr', 1.0))) < 0.95)} violations)")
        print(f"  Controllable Offload-Induced Violations: {len(offload_induced_viols)}")
        for v in offload_induced_viols:
            print(f"    -> Epoch {v['epoch']:02d}: PRR_zero={v['prr_zero']:.4f} -> PRR_B4={v['prr_b4']:.4f} | CBR: {v['cbr_zero']:.3f}->{v['cbr_b4']:.3f} | Bytes={v['bytes_offered']}")
        print(f"  Exogenous CAM Violations: {len(exogenous_viols)}")
        for v in exogenous_viols:
            print(f"    -> Epoch {v['epoch']:02d}: PRR_zero={v['prr_zero']:.4f}, PRR_B4={v['prr_b4']:.4f} | CBR: {v['cbr_zero']:.3f}")

        analysis[str(s)] = {
            "v_zero": v_zero,
            "v_b4": v_b4,
            "pv_zero": pv_z,
            "pv_b4": pv_b,
            "offload_induced_violations": offload_induced_viols,
            "exogenous_violations": exogenous_viols,
        }

    out_file = runs_dir / "controllable_safety_analysis.json"
    out_file.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nAnalysis saved to {out_file}")


if __name__ == "__main__":
    main()
