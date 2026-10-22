# Performance Evaluation Execution Plan & Gap Analysis

**Document:** `sparc_performance_evaluation/reports/PERFORMANCE_EVALUATION_PLAN.md`  
**Protocol Source:** [`sparc_performance_evaluation/protocol/EC_SPO_PERFORMANCE_EVALUATION_PROTOCOL.md`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/sparc_performance_evaluation/protocol/EC_SPO_PERFORMANCE_EVALUATION_PROTOCOL.md)  
**Date:** 2026-10-20  
**Scope:** 8 Figures (32 panels), 4 Tables, 5 Headline Cells (3 in-distribution, 2 drift), 7 Headline Methods (B1–B5, B5′, PB-CRL-core) + B5′+shield + Ceiling.

---

## 1. What We Already Have

### 1.1 Shared Interface & Execution Core
* **Shared Knapsack Packer (`controllers/rl/pb_crl_packer.py`):**
  * Sort-and-take linear knapsack scoring: $\text{score}_i = v_i - \lambda_t (b_i / B_{\max})$.
  * Execution throttle: $\sum b_i \le \beta_{\text{exec}} \cdot B_{\max}$ ($175\text{ kB}$).
  * Source and task uniqueness enforced.
* **Safety Governor & Simulator Integration:**
  * Mode-2 `SafetyGovernor` and `check_deadline()` integration.
* **Trained Policies & Checkpoints:**
  * **PB-CRL Foundation Policy (B5′):** Checkpoint [`b5_prime_best_preserved_43p86.pt`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/sparc_performance_evaluation/results/raw/pb_crl_b5_prime_train/b5_prime_best_preserved_43p86.pt) achieving $V = 43.203$ (peak $43.86$), beating Baseline B5 ($39.906$).
  * This satisfies the protocol requirement (§8 Step 5 / v3.1 §5.2) establishing $\pi_{\text{ref}} = \text{B5}'$.
* **Statistical Infrastructure:**
  * Exact 1-sided 95% Clopper–Pearson binomial upper bound on $P(Z > K)$ implemented in [`controllers/rl/pb_crl_c2.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/pb_crl_c2.py).
  * Logging of window violation flags $z_w$ based on PRR and p95 delay.

---

## 2. What We Need To Do (Aligned with Protocol §8)

### Stage A: Foundation & Characterization (Gates 1–3)
1. **Regime Calibration & Cell Definition (§2.1–§2.3):**
   * Calibrate Low, Mid, and High density cells under unconstrained offloading to ensure C2 constraint binds.
   * Lock the 5 headline cells: 3 in-distribution + 2 drift (held-out high density; CAM+CPM mix with faster mobility).
2. **Coupling Characterization (Fig 1 / Gate G7):**
   * Run sweep over offered load $L = \text{bytes}(A_t)/B_{\max}$ across 3 densities to measure coupling index $\chi$ (slope of window PRR and reselection rate vs. load) and confirm coupling is real and decision-relevant.
3. **Proxy Environment & Fidelity Gate (Table 4):**
   * Establish the fast proxy (~0.2 s/ep) for sweeps/frontiers and validate rank agreement (Kendall's $\tau$) against WiLab rollouts.

### Stage B: Baseline Implementation & Per-Cell Tuning (§3)
4. **B1 (Greedy Value-Density):** Implement cap tuner over validation seeds per cell.
5. **B2 (CBR-Threshold Admission):** Grid-search over $(\text{th}_{\text{lo}}, \text{th}_{\text{hi}}, \text{th}_{\text{resel}})$ on validation seeds per cell.
6. **B3 (Myopic Dual Packing):** Grid-search over $\eta$ and $\lambda_0$ on validation seeds per cell.
7. **B4 (Certainty-Equivalent MILP):** Fit offline logistic risk model $P(\text{viol}) = \sigma(w_0 + w_1 \text{CBR} + w_2 \text{resel} + w_3 b(A))$; grid search over safety margin.
8. **B5 & B5′:** Freeze PPO-Lagrangian models under the same search budget $N_{\text{tune}}$.

### Stage C: PB-CRL-Core Shield & Training (§3, §8 Step 6)
9. **Calibrate Risk Account & Shield (§4):**
   * Fit ensemble safety critic $Q_c^+$ and tune Poisson risk account parameters $(\delta_s, c_q)$ on validation seeds.
10. **Fine-tune PB-CRL-core:**
    * Initialize from preserved checkpoint `b5_prime_best_preserved_43p86.pt` with PID-Lagrangian and paired control variates.

### Stage D: Headline Evaluation & Reporting (Figs 3, 5, 6; Tables 1–3)
11. **WiLab Test Runs:**
    * Execute 120 test episodes per method per cell across all 5 cells with Common Random Numbers (CRN).
12. **Statistical Aggregation:**
    * Clopper-Pearson C2 upper bounds vs. $\delta$; mark infeasible methods.
    * 10,000 paired bootstrap resamples for utility differences against $\pm 1\%$ equivalence margin.
    * Holm-adjusted superiority testing; determine **Branch A** vs. **Branch B**.
13. **Generate 32 Panels (8 Figures) and 4 Tables.**

---

## 3. What We Need To Update / Reconcile

1. **Logging Checklist Compliance (§12):**
   * Update the simulation runner/evaluator to log the complete required schema: per-window PRR (with distance bins), PIR distribution, reselection rate, reservation occupancy, loss breakdown (collisions vs. half-duplex), decision latencies (p50/p99), and risk-account traces.
2. **Evaluation Harness Refactor:**
   * Unify evaluation under a single orchestrator (`run_ec_spo_evaluation.py`) that strictly enforces:
     * Disjoint seed partitions.
     * Per-cell baseline tuning on validation seeds.
     * Single universal model for PB-CRL-core.
     * Hard C2 Clopper-Pearson feasibility filtering.
3. **Reporting Template Generation:**
   * Script automated table exporters for Tables 1–4 and plotting pipeline for Figures 1–8.
