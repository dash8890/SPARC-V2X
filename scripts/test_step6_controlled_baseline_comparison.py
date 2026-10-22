"""Step 6: Controlled Comparison against Strongest Baselines (B4 and B5′).

Compares Repaired PB-CRL against the strongest baselines:
  1. B4 (Certainty-Equivalent MILP with learned risk weights)
  2. B5′ (Two-head policy baseline from preserved checkpoint)
  3. Repaired PB-CRL (Full 3-head policy architecture with DeepSets, GRU, budget block)
Across the exact same validation seeds [73300, 73301, 73302, 73303] and workloads.
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
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.pb_crl_b5_prime import B5PrimePolicy
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.hq_spo_baselines import CeRiskWeights
from controllers.rl.pb_crl_policy import PBCRLPolicy, PBCRLCoreAgent, EnsembleRiskModel

ENVELOPE = 175000
B5P_CKPT = ROOT / "checkpoints" / "b5_prime_best_preserved_43p86.pt"


def make_method_policy(method: str):
    if method == "B4_ce_milp":
        pol = ResidualTaskPolicy(logit_mode="direct", logit_scale=100.0)
        pol.ce_weights = CeRiskWeights.load()
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        return pol, "ce_milp"
    elif method == "B5_prime":
        if not B5P_CKPT.is_file():
            raise FileNotFoundError(f"Missing required checkpoint for B5': {B5P_CKPT}")
        pol = B5PrimePolicy(state_dim=16, hidden=64)
        ck = torch.load(B5P_CKPT, map_location="cpu")
        pol.load_state_dict(ck["policy"], strict=False)
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        return pol, "b5_prime"
    elif method == "PB_CRL_repaired":
        policy_net = PBCRLPolicy()
        risk_model = EnsembleRiskModel()
        pol = PBCRLCoreAgent(policy_net, risk_model, K=1, total_epochs=100, delta_s=0.05, shield_enabled=False)
        return pol, "b5_prime"
    else:
        raise ValueError(f"Unknown method: {method}")


def run_single_method(method: str, seed: int, eid: int, runs_dir: Path):
    bridge = Path(tempfile.gettempdir()) / f"step6_{method}_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)
    task_seed = 190000 + (seed - 73300)

    policy, mode = make_method_policy(method)

    t0 = time.perf_counter()
    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, 35, policy, 900.0,
        False, False, 100.0, 20.0, seed, 100,
        None, 4, False, ENVELOPE,
        300, 12.0, mode, None,
        True, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
    )
    wall = time.perf_counter() - t0

    ep_dir = runs_dir / f"episode_{eid:04d}"
    audit_file = ep_dir / "bridge_audit.json"
    if not audit_file.is_file():
        raise FileNotFoundError(f"Missing required audit file for episode {eid}: {audit_file}")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))

    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    if not valid:
        raise RuntimeError(f"No valid radio sensing windows found for episode {eid} in {audit_file}")
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    cbr = [float(e.get("cbr", 0.0)) for e in valid]
    delay = [float(e.get("p95_delay_ms", 0.0)) for e in valid]
    offered_b = sum(int(e.get("computation_bytes_offered", 0)) for e in audit)
    viols = sum(1 for p in prr if p < 0.95)
    p_viol = viols / max(len(valid), 1)

    return {
        "method": method,
        "seed": seed,
        "V_done": float(comp_val),
        "offered_bytes": offered_b,
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
    print("STEP 6: CONTROLLED COMPARISON AGAINST STRONGEST BASELINES (B4 & B5')")
    print("Methods: [B4_ce_milp, B5_prime, PB_CRL_repaired]")
    print("Validation Seeds: [73300, 73301, 73302, 73303] | Density: 35 | Horizon: 12.0s")
    print("=" * 80)

    base_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step6_baseline_comparison"
    methods = ["B4_ce_milp", "B5_prime", "PB_CRL_repaired"]
    seeds = [73300, 73301, 73302, 73303]

    tasks = []
    eid_cur = 80000
    for s in seeds:
        for m in methods:
            eid_cur += 1
            runs_m = base_dir / m / "runs"
            runs_m.mkdir(parents=True, exist_ok=True)
            tasks.append((m, s, eid_cur, runs_m))

    raw_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futs = {executor.submit(run_single_method, *t): (t[0], t[1]) for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            raw_results.append(res)
            print(f"  Finished {res['method']:<16} | Seed {res['seed']} | V_done={res['V_done']:6.2f} | PRR={res['mean_prr']:.4f} | Viols={res['n_viols']:2d}/100 ({res['wall_s']:.1f}s)")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<18} | {'Mean V_done':<12} | {'Mean Bytes':<12} | {'Mean PRR':<10} | {'Mean CBR':<10} | {'P_viol':<10}")
    print("-" * 80)

    summary_records = {}
    for m in methods:
        m_rows = [r for r in raw_results if r["method"] == m]
        mV = float(np.mean([r["V_done"] for r in m_rows]))
        mB = float(np.mean([r["offered_bytes"] for r in m_rows]))
        mPRR = float(np.mean([r["mean_prr"] for r in m_rows]))
        mCBR = float(np.mean([r["mean_cbr"] for r in m_rows]))
        mPviol = float(np.mean([r["p_viol"] for r in m_rows]))
        summary_records[m] = {
            "mean_v_done": mV,
            "mean_bytes": mB,
            "mean_prr": mPRR,
            "mean_cbr": mCBR,
            "mean_p_viol": mPviol,
            "seed_details": m_rows,
        }
        print(f"{m:<18} | {mV:<12.2f} | {mB:<12.0f} | {mPRR:<10.4f} | {mCBR:<10.4f} | {mPviol:<10.4f}")
    print("=" * 80)

    out_file = base_dir / "step6_baseline_comparison_summary.json"
    out_file.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
    print(f"\nSaved Step 6 controlled baseline comparison summary to: {out_file}")
    print("\n" + "=" * 80)
    print("STEP 6 PASS: Controlled baseline comparison complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
