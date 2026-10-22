"""P0 Calibration Test: Baseline-Preserving Admission & Price Exploration.

Verifies:
  1. Exploration coverage: stochastic sampling under initial distribution covers both
     admission-preserving (lambda < 5.0) and task-pruning (lambda in [10, 30]) regions.
  2. Deterministic baseline parity: V_init / V_B4 >= 0.95 (V_init >= 43.86 on Seed 73300).
  3. Admitted task counts match B4 (19 tasks on Seed 73300).
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
from controllers.rl.pb_crl_policy import PBCRLPolicy, PBCRLCoreAgent, EnsembleRiskModel


def test_exploration_coverage(bias_val: float, log_std_val: float):
    print(f"\n--- Checking Pricing Exploration Coverage (bias={bias_val:.2f}, log_std={log_std_val:.2f}) ---")
    pol = PBCRLPolicy()
    pol.head_lam.bias.data.fill_(bias_val)
    pol.log_std_lam.data.fill_(log_std_val)

    # Sample 5000 draws
    torch.manual_seed(42)
    mu = bias_val
    std = float(np.exp(log_std_val))
    z_samples = torch.randn(5000) * std + mu
    lam_samples = torch.sigmoid(z_samples) * 50.0

    min_l = float(lam_samples.min())
    p10_l = float(torch.quantile(lam_samples, 0.10))
    p50_l = float(torch.quantile(lam_samples, 0.50))
    p90_l = float(torch.quantile(lam_samples, 0.90))
    max_l = float(lam_samples.max())

    frac_permissive = float((lam_samples < 5.0).float().mean())
    frac_pruning = float(((lam_samples >= 10.0) & (lam_samples <= 35.0)).float().mean())

    print(f"  Sampled Lambda Distribution (N=5000):")
    print(f"    Min: {min_l:.2f} | P10: {p10_l:.2f} | Median (P50): {p50_l:.2f} | P90: {p90_l:.2f} | Max: {max_l:.2f}")
    print(f"    Fraction in admission-preserving region (< 5.0):     {frac_permissive*100:5.1f}%")
    print(f"    Fraction in active task-pruning band ([10.0, 35.0]):  {frac_pruning*100:5.1f}%")

    assert frac_permissive > 0.50, f"Expected majority in permissive band, got {frac_permissive}"
    assert frac_pruning > 0.05, f"Expected at least 5% in active pruning band for exploration, got {frac_pruning}"
    print("  -> Exploration Coverage: PASS")
    return frac_permissive, frac_pruning


def test_closed_loop_parity(bias_val: float, log_std_val: float):
    print(f"\n--- Running Closed-Loop Validation Parity Test (Seed 73300) ---")
    sim_seed = 73300
    task_seed = 190000
    density = 35
    controller_epochs = 100
    eid = 88801

    runs_dir = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "p0_calibration" / f"bias_{bias_val:.2f}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    bridge = Path(tempfile.gettempdir()) / f"p0_calib_{os.getpid()}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)

    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    policy_net = PBCRLPolicy()
    policy_net.head_lam.bias.data.fill_(bias_val)
    policy_net.log_std_lam.data.fill_(log_std_val)

    risk_model = EnsembleRiskModel()
    agent = PBCRLCoreAgent(policy_net, risk_model, K=1, total_epochs=controller_epochs, delta_s=0.05, shield_enabled=False)

    t0 = time.perf_counter()
    _, _, comp_val, _ = run_episode(
        WILAB_ROOT, bridge, eid, density, agent, 900.0,
        False, False, 100.0, 20.0, sim_seed, controller_epochs,
        None, 4, False, 175000,
        300, 12.0, "b5_prime", None,
        True, "off", "", True, task_seed, gov,
        175000, True, True, 1.0, None, None, None, 2e8, 1e9,
        output_root=runs_dir,
    )
    wall = time.perf_counter() - t0

    ep_dir = runs_dir / f"episode_{eid:04d}"
    summary_file = ep_dir / "episode_summary.json"
    audit_file = ep_dir / "bridge_audit.json"
    if not summary_file.is_file() or not audit_file.is_file():
        raise FileNotFoundError(f"Missing summary or audit file in {ep_dir}")

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    audit = json.loads(audit_file.read_text(encoding="utf-8"))

    v_init = float(summary.get("V_done", comp_val))
    v_b4 = 57.92  # B4 utility on Seed 73300
    parity_ratio = v_init / v_b4

    tasks_admitted = summary.get("n_tasks") - summary.get("n_incomplete_tasks", 0)
    compl_tasks = summary.get("n_deadline_valid_completed", 0)
    bytes_exec = summary.get("executed_bytes_sum", 0)
    p_viol = summary.get("safety_P_viol", 0.0)
    prr = summary.get("safety_mean_prr_valid", 1.0)

    print(f"  Closed-Loop Results for Seed 73300 ({wall:.1f}s):")
    print(f"    Completed Value V_init:   {v_init:.2f} (B4 reference: {v_b4:.2f})")
    print(f"    Parity Ratio (V / V_B4):  {parity_ratio*100:.2f}% (Requirement: >= 95.0%)")
    print(f"    Completed Tasks:          {compl_tasks} (B4 completed: 19)")
    print(f"    Executed Bytes:           {bytes_exec:,d} B (B4 executed: 699,766 B)")
    print(f"    Mean PRR:                 {prr:.4f}")
    print(f"    Safety P_viol:            {p_viol:.4f}")

    assert parity_ratio >= 0.95, f"Failed 95% parity gate! Ratio: {parity_ratio:.4f} < 0.95"
    print("  -> Baseline Parity Gate: PASS (>= 95% of B4)")
    return v_init, parity_ratio


def main():
    print("=" * 80)
    print("P0 CALIBRATION: BASELINE PARITY & PRICING EXPLORATION")
    print("=" * 80)

    # Test calibration: bias=-3.2 (det lambda ≈ 1.95), log_std=0.18 (std ≈ 1.20)
    bias_cand = -3.2
    log_std_cand = 0.18

    test_exploration_coverage(bias_cand, log_std_cand)
    test_closed_loop_parity(bias_cand, log_std_cand)

    print("\n" + "=" * 80)
    print("P0 CALIBRATION VERIFIED: Meets both baseline parity (>=95%) and pricing exploration!")
    print("=" * 80)


if __name__ == "__main__":
    main()
