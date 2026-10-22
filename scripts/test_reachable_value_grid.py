"""Exhaustive Reachable Value Grid Sweep over (lambda, beta) vs B4.

Evaluates whether ANY action (lambda, beta) in PB-CRL's control interface
can outperform B4 on the validation episodes at Density 15.
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

ENVELOPE = 175000


class StaticPBCRLPolicy:
    def __init__(self, lam: float, beta: float):
        self.lam = float(lam)
        self.beta = float(beta)
        self.is_pb_crl_b5_prime = True

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        # returns lam, beta, logp, pb_z, obs_dict
        return self.lam, self.beta, torch.tensor(0.0), [0.0, 0.0], {}


def run_single_eval(density: int, seed: int, eid: int, lam: float, beta: float, runs_dir: Path):
    bridge = Path(tempfile.gettempdir()) / f"sweep_{seed}_{eid}_{os.getpid()}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)
    task_seed = 186100 + (seed - 73300)
    policy = StaticPBCRLPolicy(lam=lam, beta=beta)

    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, density, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 4, False, ENVELOPE,
        300, 12.0, "b5_prime", None,
        True, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
    )

    ep_dir = runs_dir / f"episode_{eid:04d}"
    audit_file = ep_dir / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    viols = sum(1 for p in prr if p < 0.95)
    p_viol = viols / max(len(valid), 1)

    return float(comp_val), float(np.mean(prr)) if prr else 1.0, p_viol


def main():
    print("=" * 80)
    print("REACHABLE VALUE ANALYSIS OVER (LAMBDA, BETA) AT DENSITY 15")
    print("B4 baseline reference: V_done = [28.42, 21.53, 22.44, 30.86] | Mean = 25.81")
    print("=" * 80)

    # Test on validation seed 73300 first to map response surface
    seed = 73300
    density = 15
    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "reachable_grid_sweep"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Sweep lambda and beta
    lambdas = [0.0, 5.0, 10.0, 15.0, 25.0, 40.0]
    betas = [1.0, 0.75, 0.50, 0.25, 0.0]

    eid = 99000
    results = []
    print(f"\n--- Testing Seed {seed} Grid ---")
    for b in betas:
        for l in lambdas:
            eid += 1
            v, prr, p_viol = run_single_eval(density, seed, eid, l, b, runs_dir)
            results.append({"lam": l, "beta": b, "V_done": v, "prr": prr, "p_viol": p_viol})
            print(f"  beta={b:4.2f}, lam={l:4.1f} => V_done={v:6.2f} | PRR={prr:.4f} | P_viol={p_viol:.4f}")

    v_b4 = 28.42
    max_v = max(r["V_done"] for r in results)
    best_configs = [r for r in results if r["V_done"] == max_v]
    print("\n" + "=" * 80)
    print(f"Seed {seed} B4 Utility: {v_b4:.2f}")
    print(f"Max Reachable Utility:  {max_v:.2f}")
    print(f"Reachable Gap (Max - B4): {max_v - v_b4:.2f}")
    print(f"Configs achieving max utility: {best_configs}")
    print("=" * 80)


if __name__ == "__main__":
    main()
