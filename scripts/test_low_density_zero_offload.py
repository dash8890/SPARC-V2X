"""Evaluate zero-offload background safety under lower density (rho = 20) with single worker."""
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


class ZeroOffloadPolicy:
    def __init__(self):
        self.beta = 0.0
        self.lam = 0.0
        self.is_pb_crl_b5_prime = True

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        return 0.0, 0.0, torch.tensor(0.0), 0.0, 0.0


def run_one_seed(seed: int, eid: int, runs_dir: Path, density: int = 20):
    bridge = Path(tempfile.gettempdir()) / f"zero_rho{density}_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    policy = ZeroOffloadPolicy()

    t0 = time.perf_counter()
    _, _, completed, _ = run_episode(
        WILAB_ROOT, bridge, eid, density, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 4, False, 175000,
        300, 12.0, "b5_prime", None,
        True, "off", "", True, 186100 + (seed - 73300), gov,
        175000, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
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
        "density": density,
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
    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "low_density_audit" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Re-run specifically seed 73302 (eid 90002) which had license collision
    seed = 73302
    eid = 90002
    res = run_one_seed(seed, eid, runs_dir, density=20)
    print(f"Seed {res['seed']} completed! PRR={res['mean_prr']:.4f} | CBR={res['mean_cbr']:.4f} | Viols={res['n_viols']}/100 (P_viol={res['p_viol']:.4f})")


if __name__ == "__main__":
    main()
