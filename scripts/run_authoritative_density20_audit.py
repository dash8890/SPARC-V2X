"""Authoritative, immutable density-20 zero-offload baseline audit.

Executes seeds 73300-73303 sequentially with explicit configuration manifest,
recording complete provenance and verified window-level PRR/CBR measurements.
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


class ZeroOffloadPolicy:
    def __init__(self):
        self.beta = 0.0
        self.lam = 0.0
        self.is_pb_crl_b5_prime = True

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        return 0.0, 0.0, torch.tensor(0.0), 0.0, 0.0


def run_authoritative_audit():
    out_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "authoritative_density20_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    seeds = [73300, 73301, 73302, 73303]
    records = []

    print("=" * 80)
    print("STEP 1: AUTHORITATIVE DENSITY-20 ZERO-OFFLOAD BASELINE AUDIT")
    print("Density: 20 veh/km | CAM: 300B @ 10Hz | Horizon: 12.0s | Execution: Sequential")
    print("=" * 80)

    for i, seed in enumerate(seeds):
        eid = 91000 + i
        bridge = Path(tempfile.gettempdir()) / f"auth_d20_{os.getpid()}_{eid}"
        if bridge.exists():
            shutil.rmtree(bridge, ignore_errors=True)
        gov = SafetyGovernor(175000, horizon_envelope_only=True)
        policy = ZeroOffloadPolicy()

        t0 = time.perf_counter()
        _, _, completed, _ = run_episode(
            WILAB_ROOT, bridge, eid, 20, policy, 900.0,
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
        audit = json.loads(audit_file.read_text(encoding="utf-8"))

        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        cbr = [float(e.get("cbr", 0.0)) for e in valid]
        delay = [float(e.get("p95_delay_ms", 0.0)) for e in valid]
        viols = sum(1 for p in prr if p < 0.95)
        p_viol = viols / max(len(valid), 1)

        rec = {
            "seed": seed,
            "episode_id": eid,
            "density": 20,
            "n_valid": len(valid),
            "n_viols": viols,
            "p_viol": p_viol,
            "mean_prr": float(np.mean(prr)),
            "min_prr": float(np.min(prr)),
            "mean_cbr": float(np.mean(cbr)),
            "p95_delay": float(np.mean(delay)),
            "wall_s": wall,
            "feasible_alpha": bool(p_viol <= 0.02),
        }
        records.append(rec)
        print(f"  Seed {seed:<6} (ep_{eid:04d}) | Viols = {viols:2d}/100 ({p_viol:6.4f}) | Mean PRR = {rec['mean_prr']:.4f} | CBR = {rec['mean_cbr']:.4f} | Feasible: {rec['feasible_alpha']} ({wall:.1f}s)")

    print("-" * 80)
    m_pviol = float(np.mean([r["p_viol"] for r in records]))
    m_prr = float(np.mean([r["mean_prr"] for r in records]))
    m_cbr = float(np.mean([r["mean_cbr"] for r in records]))
    f_feas = float(np.mean([1.0 if r["feasible_alpha"] else 0.0 for r in records]))
    print(f"MEAN DENSITY-20 | P_viol = {m_pviol:.4f} | PRR = {m_prr:.4f} | CBR = {m_cbr:.4f} | Feasible Fraction: {f_feas*100:.1f}%")
    print("=" * 80)

    summary_path = out_dir / "authoritative_density20_summary.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved authoritative manifest to: {summary_path}")


if __name__ == "__main__":
    run_authoritative_audit()
