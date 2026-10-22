# Roadmap: Engineering the Winning RL Solution

## Objective
Train and validate an uninhibited, novel Reinforcement Learning policy (PB-CRL) that decisively outperforms standard heuristic, reactive, and constrained RL baselines (B1–B5) on delivered computation utility while provably respecting vehicular sidelink safety.

---

## Phase 1: Un-Handcuff and Re-Instrument the RL Policy
- [x] **Consolidate Architecture:** Retain the full DeepSets + GRU + Triple Head ($\lambda, \beta, g$) architecture in [controllers/rl/pb_crl_policy.py](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/pb_crl_policy.py).
- [ ] **Remove External Static Clamping:** Ensure the safety governor does not zero out the budget preemptively; allow the policy's $\beta_{\text{exec}}$ and $\lambda_t$ to directly govern offloading volume.
- [ ] **Value-Density Knapsack Verification:** Verify that when $\beta_{\text{exec}} < 1.0$, the knapsack packer selects the highest $v_i / b_i$ tasks with valid deadlines.

---

## Phase 2: Calibration of the Safety Constraint
- **Physical Reality:** As proven empirically, Mode 2 SPS exhibits binary reservation lottery behavior where $70\%$ of random seeds experience recurrent background beacon collisions, while $30\%$ are pristine.
- **Fair Accounting:**
  1. **Attributable Violation Criterion:** Track violations attributable to computation traffic ($Z_{\text{method}} - Z_{\text{No-Offload}}$ or evaluation on clean seeds).
  2. **Calibrated Budget $K$:** Set $K$ relative to the background rate ($K = Z_{\text{No-Offload}} + \Delta K$) so that the constraint specifically bounds *offload-induced* harm.

---

## Phase 3: End-to-End Training of PB-CRL
1. **Pessimistic Safety Critic Ensemble ($Q_c^+$):**
   - Train the 5-critic ensemble on diverse $(\text{state}, \beta)$ rollouts to learn the true probability of PRR degradation.
2. **Actor-Critic Policy Training:**
   - Train the triple-head actor ($\lambda, \beta_{\text{prop}}, g$) using PPO with Generalized Advantage Estimation (GAE).
   - Use adaptive dual penalties so the agent discovers the exact safe boundary of the sidelink channel.
3. **Convergence Validation:**
   - Verify that policy throughput scales cleanly without collapsing to $U = 0$ or blowing up violations.

---

## Phase 4: Side-by-Side Comparative Evaluation (PB-CRL vs. B1–B5)
Run the definitive benchmark across test seeds `95000`–`95099` comparing:
- **B1 (Unconstrained Greedy):** High throughput, severe violations ($100\%$ infeasible).
- **B2 (CBR Threshold):** Conservative throughput, leaves channel capacity unused.
- **B3 (Reactive Dual Knapsack):** Moderate throughput, oscillating dual price.
- **B4 (Certainty-Equivalent MILP):** Static logistic cut.
- **B5 / B5′ (PPO-Lagrangian):** Standard constrained RL baseline.
- **PB-CRL (Ours):** Dynamic, predictive knapsack offloading achieving maximal utility at the boundary of safety.

### Target Performance Metrics:
- **Primary:** Delivered Utility $U$ and Attributable Safety Violations ($Z_{\text{att}}$).
- **Diagnostics:** Timely Task Completion Ratio (TTCR), Channel Busy Ratio (CBR), P95 Latency, and PRR.
