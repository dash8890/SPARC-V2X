"""Diagnostic D3 — Radio Sensitivity to Computation Offload Load.

Runs a controlled WiLab experiment comparing:
  1. Zero computation offload (beta = 0.0) — measures background safety traffic collision floor
  2. Intermediate computation load (beta = 0.5)
  3. Full computation load (beta = 1.0)
All on the exact same simulation seed (73100) and horizon (12.0s).
"""
from __future__ import annotations

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


class ConstantBetaPolicy:
    def __init__(self, beta: float, lam: float = 0.0):
        self.beta = float(beta)
        self.lam = float(lam)
        self.is_pb_crl_b5_prime = True

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        return self.lam, self.beta, torch.tensor(0.0), 0.0, 0.0


def run_test(beta: float, eid: int, seed: int = 73100):
    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "radio_sensitivity" / f"beta_{beta:.2f}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    bridge = Path(tempfile.gettempdir()) / f"d3_bench_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    os.environ["WILAB_OUTPUT_ROOT"] = str(runs_dir)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    policy = ConstantBetaPolicy(beta=beta)

    t0 = time.perf_counter()
    _, _, completed, _ = run_episode(
        WILAB_ROOT, bridge, eid, 35, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 4, False, 175000,
        300, 12.0, "b5_prime", None,
        True, "off", "", True, 186100, gov,
        175000, True, True, 1.0, None, None, None, 2e8, 1e9,
    )
    wall = time.perf_counter() - t0

    ep_dir = runs_dir / f"episode_{eid:04d}"
    audit_file = ep_dir / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []

    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    cbr = [float(e.get("cbr", 0.0)) for e in valid]
    delay = [float(e.get("p95_delay_ms", 0.0)) for e in valid]
    bytes_offered = sum(int(e.get("computation_bytes_offered", 0)) for e in audit)
    viols = sum(1 for p in prr if p < 0.95)
    p_viol = viols / max(len(valid), 1)

    return {
        "beta": beta,
        "V_done": float(completed),
        "total_bytes": bytes_offered,
        "mean_prr": float(np.mean(prr)) if prr else 1.0,
        "min_prr": float(np.min(prr)) if prr else 1.0,
        "mean_cbr": float(np.mean(cbr)) if cbr else 0.0,
        "p95_delay": float(np.mean(delay)) if delay else 0.0,
        "n_viols": viols,
        "P_viol": p_viol,
        "wall_s": wall,
    }


def main():
    print("=" * 80)
    print("DIAGNOSTIC D3: RADIO SENSITIVITY TO COMPUTATION LOAD")
    print("Seed: 73100 | Horizon: 12.0s (100 epochs) | Density: 35")
    print("=" * 80)
    
    eid = 50000
    results = []
    for beta in [0.0, 0.5, 1.0]:
        eid += 1
        print(f"\nRunning beta = {beta:.2f}...")
        res = run_test(beta, eid=eid)
        results.append(res)
        print(f"  beta={beta:.2f} | V={res['V_done']:.2f} | Bytes={res['total_bytes']} | PRR={res['mean_prr']:.4f} (min={res['min_prr']:.4f}) | CBR={res['mean_cbr']:.4f} | Viols={res['n_viols']} (P_viol={res['P_viol']:.4f})")

    print("\n" + "=" * 80)
    print(f"{'beta':<6} | {'Delivered V':<12} | {'Offered kB':<12} | {'Mean PRR':<10} | {'Min PRR':<10} | {'Mean CBR':<10} | {'P_viol':<8} | {'Viols'}")
    print("-" * 80)
    for r in results:
        print(f"{r['beta']:<6.2f} | {r['V_done']:<12.2f} | {r['total_bytes']/1000:<12.1f} | {r['mean_prr']:<10.4f} | {r['min_prr']:<10.4f} | {r['mean_cbr']:<10.4f} | {r['P_viol']:<8.4f} | {r['n_viols']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
