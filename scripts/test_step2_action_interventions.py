"""Step 2: Deterministic Action-Intervention Verification Test.

Verifies that downstream execution components react specifically to each action:
  1. Lambda intervention: varying lambda changes task scoring, priorities, and prunes low-density tasks.
  2. Beta intervention: varying beta directly modulates byte capacity and admitted byte volume.
  3. Gain (g) intervention: varying g dynamically scales p_allow and predictive shield beta_exec.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

import torch
from task_layer import TaskLayer
from controllers.rl.pb_crl_packer import price_budget_pack
from controllers.rl.pb_crl_policy import EnsembleRiskModel, PredictiveRiskAccount


def test_step2():
    print("=" * 80)
    print("STEP 2: ACTION-INTERVENTION VERIFICATION TEST")
    print("=" * 80)

    # 1. Setup real candidate pool from validation seed 73300 (epoch 0)
    sim_seed = 73300
    task_seed = 190000
    density = 35
    layer = TaskLayer(density, task_seed, tasks_per_source=4, controller_epochs=100)
    layer.set_epoch(0)
    cbr = 0.085
    cands = layer.candidates(cbr, 0.0)
    print(f"Candidate pool loaded: {len(cands)} candidate edges.")
    assert len(cands) > 0, "No candidates available for test!"

    # -------------------------------------------------------------------------
    # TEST 2A: LAMBDA INTERVENTION (Fixed beta=1.0, varying lambda)
    # -------------------------------------------------------------------------
    print("\n--- 2A: Lambda Action Intervention (Price Authority & Pruning) ---")
    lam_values = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    results_lam = []
    prev_n_tasks = float("inf")

    for lam in lam_values:
        pack = price_budget_pack(cands, lam=lam, beta_exec=1.0, b_max=175000, unique_sources=True)
        n_tasks = len(pack)
        tot_bytes = sum(int(p["payload_bytes"]) for p in pack)
        task_ids = tuple(sorted(p["task_id"] for p in pack))
        results_lam.append((lam, n_tasks, tot_bytes, task_ids))
        print(f"  lambda = {lam:4.1f} -> admitted tasks = {n_tasks:2d}, bytes = {tot_bytes:6d} B")

    # Invariants for Lambda:
    # - As lambda increases, admitted tasks must be monotonically non-increasing
    # - Active pruning must occur (n_tasks at lam=0 must be strictly greater than at lam=50)
    # - Distinct task sets must be produced across the active band
    for i in range(len(results_lam) - 1):
        assert results_lam[i][1] >= results_lam[i + 1][1], (
            f"Monotonicity violation in lambda: {results_lam[i][0]} has {results_lam[i][1]} tasks, "
            f"but {results_lam[i+1][0]} has {results_lam[i+1][1]} tasks!"
        )

    assert results_lam[0][1] > results_lam[-1][1], (
        f"Lambda is completely inert! At lam=0: {results_lam[0][1]} tasks, at lam=50: {results_lam[-1][1]} tasks."
    )
    # Check that task sets actually differ
    distinct_task_sets = len(set(r[3] for r in results_lam))
    print(f"  Distinct task sets across lambda range: {distinct_task_sets}")
    assert distinct_task_sets >= 3, f"Expected at least 3 distinct task sets, got {distinct_task_sets}"
    print("  -> Lambda intervention PASS: Pruning is active, monotonic, and discriminative.")

    # -------------------------------------------------------------------------
    # TEST 2B: BETA INTERVENTION (Fixed lambda=15.0, varying beta)
    # -------------------------------------------------------------------------
    print("\n--- 2B: Beta Action Intervention (Capacity Authority & Volume) ---")
    beta_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    results_beta = []

    for beta in beta_values:
        pack = price_budget_pack(cands, lam=15.0, beta_exec=beta, b_max=175000, unique_sources=True)
        n_tasks = len(pack)
        tot_bytes = sum(int(p["payload_bytes"]) for p in pack)
        results_beta.append((beta, n_tasks, tot_bytes))
        print(f"  beta = {beta:4.2f} -> admitted tasks = {n_tasks:2d}, bytes = {tot_bytes:6d} B (cap = {int(beta * 175000):6d} B)")

    # Invariants for Beta:
    # - At beta=0.0, admitted tasks and bytes must be EXACTLY 0
    # - Admitted bytes must never exceed the budget cap
    # - Admitted bytes and tasks must be monotonically non-decreasing
    assert results_beta[0][1] == 0 and results_beta[0][2] == 0, (
        f"At beta=0.0, tasks={results_beta[0][1]}, bytes={results_beta[0][2]} (expected 0)"
    )

    for i in range(len(results_beta) - 1):
        assert results_beta[i][2] <= int(results_beta[i][0] * 175000) + 1, "Budget cap exceeded!"
        assert results_beta[i][2] <= results_beta[i + 1][2], (
            f"Monotonicity violation in beta bytes: {results_beta[i][0]} has {results_beta[i][2]} bytes, "
            f"but {results_beta[i+1][0]} has {results_beta[i+1][2]} bytes!"
        )

    assert results_beta[-1][2] > results_beta[1][2], "Beta does not increase admitted bytes across [0.1, 1.0]!"
    print("  -> Beta intervention PASS: Strict zero at beta=0, caps obeyed, monotonic volume scaling.")

    # -------------------------------------------------------------------------
    # TEST 2C: GAIN (g) INTERVENTION (Predictive Risk Account & Shield)
    # -------------------------------------------------------------------------
    print("\n--- 2C: Gain (g) Action Intervention (Risk Pacing & Shield Authority) ---")
    risk_model = EnsembleRiskModel()
    account = PredictiveRiskAccount(risk_model, K=1, total_epochs=100, delta_s=0.05)
    obs_tensor = torch.zeros(1, 16, dtype=torch.float32)

    g_values = [0.5, 0.75, 1.0, 1.5, 2.0]
    p_allows = []
    betas_exec = []

    # Propose high beta (1.0) under risk constraint
    beta_prop = 1.0
    for g in g_values:
        # Reset account state for fair comparison
        account.reset(K=1)
        account.current_epoch = 10  # 90 epochs left
        account.m_rem = 0.05
        
        beta_exec, q_plus_exec, p_allow = account.evaluate_shield(obs_tensor, beta_prop=beta_prop, g_t=g)
        p_allows.append(p_allow)
        betas_exec.append(beta_exec)
        print(f"  g = {g:4.2f} -> p_allow = {p_allow:8.6f}, beta_exec = {beta_exec:.4f}, q_plus = {q_plus_exec:.6f}")

    # Invariants for Gain g:
    # - p_allow must strictly increase with g: p_allow(g) ~ g * m_rem / t_left
    for i in range(len(p_allows) - 1):
        assert p_allows[i] < p_allows[i + 1], (
            f"Strict monotonicity failure in p_allow: g={g_values[i]} ({p_allows[i]}) vs g={g_values[i+1]} ({p_allows[i+1]})"
        )
    # Check linear proportionality: p_allow(g=2.0) / p_allow(g=1.0) should be approx 2.0
    ratio = p_allows[-1] / p_allows[2]
    print(f"  p_allow ratio (g=2.0 / g=1.0): {ratio:.4f} (expected 2.0000)")
    assert abs(ratio - 2.0) < 1e-4, f"Pacing gain scaling incorrect: ratio={ratio}"

    # Also test that higher g enables higher beta_exec when constrained
    # Construct an artificial risk model scenario where risk is sensitive to beta
    # to demonstrate beta_exec responds to g
    class SensitiveRiskModel(torch.nn.Module):
        def forward(self, obs, beta):
            # risk scales linearly: beta=0 -> 0.0, beta=1 -> 0.001
            val = beta * 0.0008
            return val, val, torch.zeros_like(val)

    account_sens = PredictiveRiskAccount(SensitiveRiskModel(), K=1, total_epochs=100, delta_s=0.05)
    account_sens.current_epoch = 10
    account_sens.m_rem = 0.05

    b_low, _, p_low = account_sens.evaluate_shield(obs_tensor, beta_prop=1.0, g_t=0.5)
    b_high, _, p_high = account_sens.evaluate_shield(obs_tensor, beta_prop=1.0, g_t=2.0)
    print(f"  Constrained shield response: g=0.5 -> beta_exec={b_low:.4f}; g=2.0 -> beta_exec={b_high:.4f}")
    assert b_low < b_high, f"Shield did not throttle beta according to pacing gain! {b_low} >= {b_high}"

    print("  -> Gain (g) intervention PASS: p_allow scales proportionally, shield responds dynamically.")

    print("\n" + "=" * 80)
    print("STEP 2 PASS: All downstream action interventions verified deterministically.")
    print("=" * 80)


if __name__ == "__main__":
    test_step2()
