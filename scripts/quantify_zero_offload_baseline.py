"""Quantify Zero-Offload (beta = 0.0) Safety Behavior Across Prescribed Seeds.

Directly addresses Gate 3 / Section 2 of the Audit:
  Retains the exact frozen safety criterion (P_viol = fraction of windows with PRR < 0.95).
  Measures the baseline background safety behavior across test seeds [73100, 73101, 73102, 73103]
  under pure background CAM traffic (zero computation offloading).
"""
from __future__ import annotations

import concurrent.futures
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


class ZeroOffloadPolicy:
    def __init__(self):
        self.beta = 0.0
        self.lam = 0.0
        self.is_pb_crl_b5_prime = True

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        return 0.0, 0.0, torch.tensor(0.0), 0.0, 0.0


def run_one_seed(seed: int, eid: int, runs_dir: Path):
    bridge = Path(tempfile.gettempdir()) / f"zero_offload_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    os.environ["WILAB_OUTPUT_ROOT"] = str(runs_dir)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    policy = ZeroOffloadPolicy()

    t0 = time.perf_counter()
    _, _, completed, _ = run_episode(
        WILAB_ROOT, bridge, eid, 35, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 4, False, 175000,
        300, 12.0, "b5_prime", None,
        True, "off", "", True, 186100 + (seed - 73100), gov,
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
    viols = sum(1 for p in prr if p < 0.95)
    p_viol = viols / max(len(valid), 1)

    return {
        "seed": seed,
        "n_valid": len(valid),
        "mean_prr": float(np.mean(prr)) if prr else 1.0,
        "min_prr": float(np.min(prr)) if prr else 1.0,
        "mean_cbr": float(np.mean(cbr)) if cbr else 0.0,
        "p95_delay": float(np.mean(delay)) if delay else 0.0,
        "n_viols": viols,
        "p_viol": p_viol,
        "wall_s": wall,
    }


def main():
    print("=" * 80)
    print("GATE 3: ZERO-OFFLOAD BACKGROUND SAFETY CHARACTERIZATION")
    print("Seeds: [73100, 73101, 73102, 73103] | Density: 35 | Horizon: 12.0s | beta = 0.0")
    print("=" * 80)

    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "zero_offload_audit" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    seeds = [73100, 73101, 73102, 73103]
    tasks = [(s, 60000 + i, runs_dir) for i, s in enumerate(seeds)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futs = {executor.submit(run_one_seed, *t): t[0] for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            results.append(res)
            print(f"  Seed {res['seed']} | Mean PRR={res['mean_prr']:.4f} (min={res['min_prr']:.4f}) | CBR={res['mean_cbr']:.4f} | Viols={res['n_viols']}/100 (P_viol={res['p_viol']:.4f})")

    results.sort(key=lambda r: r["seed"])
    print("\n" + "=" * 80)
    print(f"{'Seed':<10} | {'Mean PRR':<12} | {'Min PRR':<12} | {'Mean CBR':<12} | {'Viols':<8} | {'P_viol':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['seed']:<10} | {r['mean_prr']:<12.4f} | {r['min_prr']:<12.4f} | {r['mean_cbr']:<12.4f} | {r['n_viols']:<8} | {r['p_viol']:<10.4f}")
    
    mean_prr = np.mean([r["mean_prr"] for r in results])
    mean_cbr = np.mean([r["mean_cbr"] for r in results])
    mean_pviol = np.mean([r["p_viol"] for r in results])
    print("-" * 80)
    print(f"{'MEAN':<10} | {mean_prr:<12.4f} | {'--':<12} | {mean_cbr:<12.4f} | {'--':<8} | {mean_pviol:<10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
