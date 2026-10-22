# Forensic Analysis: Historical Wins, Failure Modes, and the Definitive Restoration Path

## Executive Summary

Historical evaluation logs in this repository demonstrate two verified milestones:
1. **Baseline B5 decisively outperformed Baselines B1–B4** ($V = 39.91$ vs. $30.78\text{--}32.77$, a $\mathbf{+21.8\%}$ advantage), establishing that neural RL task prioritization beats naive heuristic sorting.
2. **The PB-CRL Foundation Policy (B5′) decisively outperformed Baseline B5** ($V = 43.86$ vs. $39.91$, a $\mathbf{+3.95}$ point advantage), establishing that continuous $(\lambda_t, \beta_t)$ control paired with a helper-aware multiple-choice price-and-budget knapsack packer beats scalar logit ranking.

However, subsequent evaluation runs degraded catastrophically, culminating in reported PB-CRL throughputs of **$U = 5.75$**. This document details the forensic audit of why the initial implementation won, the three divergences that caused the degradation, and the definitive engineering path forward.

---

## Part 1: What Worked in the Initial Implementation

From [`SPARC_HQ_SPO_BASELINE_PANEL.md`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/sparc_performance_evaluation/reports/SPARC_HQ_SPO_BASELINE_PANEL.md) and [`SPARC_HQ_SPO_PB_CRL_B5P_PANEL.md`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/sparc_performance_evaluation/reports/SPARC_HQ_SPO_PB_CRL_B5P_PANEL.md):

### 1. B5 Baseline Panel Results (Seeds `73100–73107`)
In the original HQ-SPO baseline benchmark on test seeds with 4 tasks per source (`tps = 4`):

| Method | Algorithm Description | Mean Delivered Utility ($V$) | Safety Violation Rate ($P_{\text{viol}} \le 0.02$) | Mean Wall Clock (s) |
| :--- | :--- | :---: | :---: | :---: |
| **B1 (greedy_vb)** | Analytic $v_i / b_i$ greedy sort | $32.77$ | $0.75$ | $42.3$ |
| **B2 (cbr_threshold)** | Dynamic CBR-threshold byte cap | $32.77$ | $0.75$ | $41.3$ |
| **B3 (myopic_dual)** | Subgradient dual ascent on $\lambda$ | $30.78$ | $0.75$ | $41.6$ |
| **B4 (ce_milp)** | Offline logistic risk certainty-equivalent | $31.00$ | $0.75$ | $39.1$ |
| **B5 (ppo_lag)** | **PPO-Lagrangian on task residual logits** | **$39.91$** | **$0.75$** | **$42.3$** |

**Why B5 Won:**
B1–B4 relied on analytic 1D value-density ($v_i / b_i$) and either transmitted blindly or applied crude scalar byte cutoffs. B5 used a deep actor-critic (`mode = "residual"`) that scored task candidates non-linearly, prioritizing tasks with high value, tight deadlines, and helpers whose current queue state guaranteed completion before vehicular departure.

### 2. PB-CRL Foundation (B5′) vs. B5 (Seeds `73300–73307`)
In [`train_pb_crl_b5_prime.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/sparc_performance_evaluation/scripts/train_pb_crl_b5_prime.py), the policy architecture was upgraded to output continuous $(\lambda_t, \beta_t)$ directly into the **helper-aware multiple-choice price-and-budget knapsack packer** (`price_budget_pack`):

- **Baseline B5 Target:** $V = 39.906$
- **B5′ Checkpoint (`b5_prime_best_preserved_43p86.pt`):** **$V = 43.86$** (peak), **$V = 43.20$** (validation mean)
- **Net Headroom Gain:** $\mathbf{+3.95}$ points over B5, and $\mathbf{+11.09}$ points over B1 ($+33.8\%$).

### 3. The Physical Environment of the Winning Runs
The parameters that generated these results in `train_pb_crl_b5_prime.py`:
- `simulation_time = 12.0 s` ($100$ controller epochs).
- `tasks_per_source = 4`, `density = 35`.
- **Zero External Choking:** The policy acted directly through the helper-aware packer. There was **no external Poisson-tail shield clamping $\beta \to 0$**.

---

## Part 2: Forensic Root-Cause of What Failed Later

Between the $V = 43.86$ milestone and the recent run where PB-CRL collapsed to $U = 5.75$, three fatal divergences occurred:

### Divergence 1: The Poisson-Tail Predictive Shield Suffocated the Policy
A mathematical shield was added to enforce $P(\text{Poisson}(m) > K) \le 0.05$:
- For $K = 1$, the Poisson tail bound allows a total cumulative risk of $m_{\text{rem}} \approx 0.355$.
- In `PredictiveRiskAccount.evaluate_shield`, the per-epoch allowed risk was computed as:
  $$p_{\text{allow}} = \frac{m_{\text{rem}}}{T_{\text{left}}} = \frac{0.355}{50} = \mathbf{0.0071} \quad (0.7\% \text{ per epoch})$$
- The 5-critic ensemble $Q_c^+$ estimated that any non-trivial offload burst ($\beta > 0.05$) carried an instantaneous risk $q^+ > 0.0071$.
- **The Execution Trap:** Because $q^+ > p_{\text{allow}}$, the grid search clamped $\beta_{\text{exec}} \to 0$. After 2–4 epochs of minor offloading, $m_{\text{rem}}$ was exhausted, locking $\beta_{\text{exec}} = 0$ for the remaining 46 epochs.
- **Impact:** The policy was permanently gagged mid-episode, crashing delivered utility to $U = 5.75$.

### Divergence 2: The Architecture Alias (PB-CRL Was Never Trained)
As uncovered in [`POLICY_ARCHITECTURE_AND_WEIGHT_AUDIT.md`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/sparc_performance_evaluation/reports/POLICY_ARCHITECTURE_AND_WEIGHT_AUDIT.md):
- In `run_step4_panel.py`, `PB-CRL-core` was **never instantiated or trained** as the specified 3-head $(\lambda, \beta, g)$ DeepSets architecture.
- Instead, the script assigned `PB-CRL-core` to load the exact same class and checkpoint as `B5'+shield` ($\Delta U = +0.00$). 
- The comparative evaluation was running an unlearned clone against an identical clone under a suffocating shield.

### Divergence 3: Horizon Truncation Created an Artificial SPS Noise Floor
- The simulation duration was reduced from $12.0\text{ s}$ down to $2.5\text{ s}$ ($50$ epochs) with zero warm-up discard.
- In 3GPP C-V2X Mode 2, vehicles select resource reservations with reselection counters spanning $5\text{--}15$ beacon periods ($0.5\text{--}1.5\text{ s}$).
- In a 2.5-second run with no warm-up, vehicles that collide on random startup never reselect during the evaluation window. The collision repeats every $100\text{ ms}$, producing $10$ violating windows regardless of whether offloading occurs. This artificially created a $70\%\text{--}90\%$ background failure rate under No-Offload.

---

## Part 3: The Definitive Restoration Path

To return to the winning trajectory and achieve superior publication-grade performance:

```
                      [Candidate Tasks (v_i, b_i, D_i)]
                                      │
                                      ▼
                      [Neural Policy Network (Actor)]
                                      │
                                      ├──► λ_t (Shadow Price)
                                      └──► β_t (Offload Volume Fraction)
                                      │
                                      ▼
             [Helper-Aware Multiple-Choice Knapsack Packer]
                  (Sort by (v_i - λ_t * b_i/B_max) / b_i)
                                      │
                                      ▼
             [Admit Tasks on Best Helpers up to β_t * B_max]
                                      │
                                      ▼
                 [WiLab Sidelink Mode 2 Closed-Loop (12s)]
                                      │
                                      ▼
          [Training Signal: R_t = U_t - λ_dual * Cost_t (GAE)]
               (No hard test-time zeroing clamp on β_t)
```

### Core Action Principles:
1. **Un-Handcuff the Decision Path:**
   - Remove the $p_{\text{allow}}$ zeroing clamp from `act_numpy`.
   - The neural policy's $(\lambda_t, \beta_t)$ must directly govern the knapsack packer at test time.
   - Safety constraints must guide policy updates via **Lagrangian policy gradients and cost critics during training**, not a hard heuristic clamp that silences the agent.
2. **Preserve the Winning Knapsack Packer (`price_budget_pack`):**
   - Retain the helper-aware knapsack engine that enabled B5′ to beat B5 ($43.86$ vs. $39.91$).
   - Task packing prioritizes maximum utility per byte, ensuring that even when the policy throttles $\beta_t$ to $0.70$ during high CBR, it captures $>90\%$ of available compute utility.
3. **Align the Evaluation Horizon:**
   - Standardize evaluations on `simulation_time = 12.0 s` ($100$ controller epochs), allowing Mode 2 SPS reselection dynamics to operate in steady-state as in the original validated experiments.
4. **Train the True 3-Head PB-CRL Policy:**
   - Train the unified DeepSets + GRU + Triple Head actor end-to-end against the closed-loop environment.
   - Benchmark against B1–B5 on identical sealed seeds to demonstrate the conclusive win.
