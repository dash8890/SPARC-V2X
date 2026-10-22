# SPARC-V2X Research Roadmap: The Definitive Path to Victory

## 1. Grounding: Reclaiming the Winning Foundation

Forensic analysis ([`docs/FORENSIC_ROOT_CAUSE_AND_RESTORATION.md`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/docs/FORENSIC_ROOT_CAUSE_AND_RESTORATION.md)) established that:
- **Baseline B5 decisively beat B1–B4 ($39.91$ vs. $30.78\text{--}32.77$)** via deep RL task prioritization.
- **PB-CRL Foundation (B5′) decisively beat Baseline B5 ($43.86$ vs. $39.91$)** via continuous $(\lambda_t, \beta_t)$ control paired with the helper-aware knapsack packer (`price_budget_pack`).
- Degradation to $U = 5.75$ occurred only when an external Poisson risk formula suffocated the policy by setting $p_{\text{allow}} \le 0.007$, clamping $\beta_{\text{exec}} \to 0$.

The roadmap from this point forward eliminates all exploratory side-quests and follows the proven engineering trajectory.

---

## 2. Execution Phases

### Phase 1: Un-Handcuff the Policy Action Space
- [ ] **Direct Execution in `controllers/rl/pb_crl_policy.py`:**
  - Remove the external zeroing clamp in `act_numpy`.
  - Allow the neural actor's $(\lambda_t, \beta_t)$ to directly govern the helper-aware multiple-choice knapsack packer (`price_budget_pack`).
  - Retain safety evaluation as a diagnostic and a training loss/gradient, not as a runtime execution gag.
- [ ] **Knapsack Value-Density Audit:**
  - Verify that when the policy modulates $\beta_t < 1.0$ during congestion, it prioritizes tasks with the highest utility density $(v_i - \lambda_t b_i/B_{\max})/b_i$ and viable helper queues.

### Phase 2: Environment & Physics Realignment
- [ ] **Standardize Simulation Horizon:**
  - Set `simulation_time = 12.0 s` ($100$ controller epochs) matching the proven environment where B5 ($39.91$) and B5′ ($43.86$) were established.
  - This allows 3GPP Mode 2 SPS reselection ($C_{\text{resel}} \approx 0.5\text{--}1.5\text{ s}$) to resolve initial random collisions and measure true steady-state offloading throughput.
- [ ] **Workload Standardization:**
  - `tasks_per_source = 4`, `density = 35`, `cluster_share = 0.50` (or `0.70`).

### Phase 3: End-to-End Training of PB-CRL
- [ ] **Train Unified 3-Head Actor-Critic:**
  - Architecture: DeepSets task encoder + 10-epoch radio history GRU + Budget block + Trunk MLP.
  - Heads: $\lambda_t \in [0, \lambda_{\max}]$, $\beta_t \in [0, 1]$, $g_t \in [0.5, 2.0]$.
  - Twin Critics: $V_r(s)$ (timely utility) and $V_c(s)$ (safety constraint cost).
- [ ] **Lagrangian Advantage Formulation:**
  - Update policy using PPO with dual Lagrangian advantage:
    $$A(\tau) = \frac{A^r(\tau) - \lambda_{\text{dual}} A^c(\tau)}{1 + \lambda_{\text{dual}}}$$
  - The policy learns to pull back $\beta_t$ dynamically as CBR rises, discovering the exact safe channel frontier without artificial external clamping.
- [ ] **Validation Target:**
  - Exceed B5′ ($V > 43.86$) at verified safety compliance.

### Phase 4: Definitive Head-to-Head Benchmark
- [ ] Run PB-CRL vs. B1–B5 on test seeds `73100–73107` and headline test seeds:
  - **B1 (Unconstrained Greedy):** Reference throughput, high violations.
  - **B2 (CBR-Threshold Cap):** Leaves capacity unused.
  - **B3 (Myopic Dual Knapsack):** Reactive dual lag.
  - **B4 (Certainty-Equivalent MILP):** Rigid offline risk cut.
  - **B5 (PPO-Lagrangian on Logits):** Strong baseline ($V = 39.91$).
  - **B5′ (PPO-Lagrangian on Packer):** Foundation baseline ($V = 43.86$).
  - **PB-CRL (Ours):** Full predictive budget-constrained RL.
- [ ] Generate headline comparison table: Mean Delivered Utility $V$, Task Completion Ratio (TTCR), PRR, CBR, and Safety Compliance.
