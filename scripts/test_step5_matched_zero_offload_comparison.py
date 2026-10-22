"""Step 5: Matched Zero-Offload Feasibility Comparison on Validation Seeds.

Compares Repaired PB-CRL against the zero-offload baseline (beta = 0.0)
across validation seeds [73300, 73301, 73302, 73303] to report:
  1. Absolute safety violation rate P_viol for both conditions.
  2. Background collision floor established by zero-offload.
  3. Incremental safety degradation attributable to computation traffic:
       Delta P_viol = P_viol(PB-CRL) - P_viol(beta=0)
  4. Completed computation utility V_done achieved under the radio constraint.
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
from controllers.rl.pb_crl_policy import PBCRLPolicy, PBCRLCoreAgent, EnsembleRiskModel


class ZeroOffloadPolicy:
    def __init__(self):
        self.beta = 0.0
        self.lam = 0.0
        self.is_pb_crl_b5_prime = True

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        return 0.0, 0.0, torch.tensor(0.0), 0.0, 0.0


def run_single_condition(condition: str, seed: int, eid: int, runs_dir: Path):
    bridge = Path(tempfile.gettempdir()) / f"step5_{condition}_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    os.environ["WILAB_OUTPUT_ROOT"] = str(runs_dir)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    task_seed = 190000 + (seed - 73300)

    if condition == "zero_offload":
        policy = ZeroOffloadPolicy()
    elif condition == "pb_crl":
        policy_net = PBCRLPolicy()
        risk_model = EnsembleRiskModel()
        policy = PBCRLCoreAgent(policy_net, risk_model, K=1, total_epochs=100, delta_s=0.05, shield_enabled=False)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    t0 = time.perf_counter()
    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, 35, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 4, False, 175000,
        300, 12.0, "b5_prime", None,
        True, "off", "", True, task_seed, gov,
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
        "condition": condition,
        "seed": seed,
        "n_valid": len(valid),
        "mean_prr": float(np.mean(prr)) if prr else 1.0,
        "min_prr": float(np.min(prr)) if prr else 1.0,
        "mean_cbr": float(np.mean(cbr)) if cbr else 0.0,
        "p95_delay": float(np.mean(delay)) if delay else 0.0,
        "n_viols": viols,
        "p_viol": p_viol,
        "completed_value": float(comp_val),
        "wall_s": wall,
    }


def main():
    print("=" * 80)
    print("STEP 5: MATCHED ZERO-OFFLOAD FEASIBILITY COMPARISON")
    print("Validation Seeds: [73300, 73301, 73302, 73303] | Density: 35 | Horizon: 12.0s")
    print("=" * 80)

    base_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step5_matched_comparison"
    seeds = [73300, 73301, 73302, 73303]

    tasks = []
    for i, s in enumerate(seeds):
        # Eid numbering: 70000+ for zero_offload, 71000+ for pb_crl
        runs_zero = base_dir / "zero_offload" / "runs"
        runs_pb = base_dir / "pb_crl" / "runs"
        runs_zero.mkdir(parents=True, exist_ok=True)
        runs_pb.mkdir(parents=True, exist_ok=True)
        tasks.append(("zero_offload", s, 70000 + i, runs_zero))
        tasks.append(("pb_crl", s, 71000 + i, runs_pb))

    raw_results = {}
    # Run concurrently (up to 4 workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futs = {executor.submit(run_single_condition, *t): (t[0], t[1]) for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            cond, s = futs[fut]
            res = fut.result()
            raw_results[(cond, s)] = res
            print(f"  Finished {cond:12s} | Seed {s} | PRR={res['mean_prr']:.4f} | Viols={res['n_viols']:2d}/100 | V_done={res['completed_value']:6.2f} ({res['wall_s']:.1f}s)")

    # Produce paired comparison table
    print("\n" + "=" * 95)
    print(f"{'Seed':<6} | {'V_done':<7} | {'Zero PRR':<9} | {'PB PRR':<9} | {'Zero Viol':<10} | {'PB Viol':<10} | {'Delta Viol':<11} | {'Delta PRR':<10}")
    print("-" * 95)

    comparison_records = []
    for s in seeds:
        z_res = raw_results[("zero_offload", s)]
        pb_res = raw_results[("pb_crl", s)]
        delta_viol = pb_res["p_viol"] - z_res["p_viol"]
        delta_prr = z_res["mean_prr"] - pb_res["mean_prr"]
        record = {
            "seed": s,
            "v_done": pb_res["completed_value"],
            "zero_prr": z_res["mean_prr"],
            "pb_prr": pb_res["mean_prr"],
            "zero_viol_rate": z_res["p_viol"],
            "pb_viol_rate": pb_res["p_viol"],
            "delta_viol_rate": delta_viol,
            "delta_prr": delta_prr,
        }
        comparison_records.append(record)
        print(f"{s:<6d} | {pb_res['completed_value']:<7.2f} | {z_res['mean_prr']:<9.4f} | {pb_res['mean_prr']:<9.4f} | "
              f"{z_res['p_viol']:<10.4f} | {pb_res['p_viol']:<10.4f} | {delta_viol:<+11.4f} | {delta_prr:<10.4f}")

    mean_vdone = np.mean([r["v_done"] for r in comparison_records])
    mean_z_prr = np.mean([r["zero_prr"] for r in comparison_records])
    mean_pb_prr = np.mean([r["pb_prr"] for r in comparison_records])
    mean_z_viol = np.mean([r["zero_viol_rate"] for r in comparison_records])
    mean_pb_viol = np.mean([r["pb_viol_rate"] for r in comparison_records])
    mean_delta_viol = np.mean([r["delta_viol_rate"] for r in comparison_records])
    mean_delta_prr = np.mean([r["delta_prr"] for r in comparison_records])

    print("-" * 95)
    print(f"{'MEAN':<6} | {mean_vdone:<7.2f} | {mean_z_prr:<9.4f} | {mean_pb_prr:<9.4f} | "
          f"{mean_z_viol:<10.4f} | {mean_pb_viol:<10.4f} | {mean_delta_viol:<+11.4f} | {mean_delta_prr:<10.4f}")
    print("=" * 95)

    summary_file = base_dir / "step5_matched_comparison_summary.json"
    summary_file.write_text(json.dumps({
        "records": comparison_records,
        "mean_vdone": float(mean_vdone),
        "mean_zero_prr": float(mean_z_prr),
        "mean_pb_prr": float(mean_pb_prr),
        "mean_zero_viol": float(mean_z_viol),
        "mean_pb_viol": float(mean_pb_viol),
        "mean_delta_viol": float(mean_delta_viol),
        "mean_delta_prr": float(mean_delta_prr),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved matched comparison results to: {summary_file}")
    print("\n" + "=" * 80)
    print("STEP 5 PASS: Matched zero-offload feasibility comparison complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
