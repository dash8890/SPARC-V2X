# System Architecture: HQ-SPO, PB-CRL, and Baselines B1–B5

## 1. Problem Formulation: Vehicular Edge Computing under Sidelink Safety Constraints

In connected and autonomous vehicle (CAV) networks, vehicles generate compute-intensive perception and planning tasks that exceed onboard embedded compute limits. To meet real-time task deadlines, vehicles offload execution over 5G NR / C-V2X Sidelink (Mode 2) to nearby roadside units (RSUs) or edge-enabled helper vehicles.

However, the sidelink spectrum is shared with periodic safety-critical basic safety messages (BSMs) or cooperative awareness messages (CAMs). 
- **Channel Access (Sidelink Mode 2):** Distributed sensing-based semi-persistent scheduling (SPS) without central coordination.
- **Safety Standard:** Packet Reception Ratio (PRR) for safety beacons must remain $\ge 0.95$ within every evaluation window.
- **Offload Interference:** Heavy computation packet bursts raise the Channel Busy Ratio (CBR) and increase resource collisions, directly threatening beacon reception.

The objective is to maximize timely delivered computation utility $U = \sum_i v_i \cdot \mathbf{1}\{\text{finish}_i \le D_i\}$ while guaranteeing that safety violations do not exceed a tight operational budget $K$:
$$\max_{\pi} \mathbb{E}\left[ \sum_{t=0}^T U_t \right] \quad \text{s.t.} \quad \mathbb{P}\left( \sum_{t=0}^T \mathbf{1}\{\text{PRR}_t < 0.95\} > K \right) \le \delta$$

---

## 2. Baselines B1–B5 (HQ-SPO Suite)

All baselines and the novel RL policy share the same underlying environment, vehicle traces, task generation, and helper queue physics.

### B1: Unconstrained Greedy Value-Density (`greedy_value_density`)
- **Mechanism:** At each epoch $t$, task candidates are sorted descending by value density $v_i / b_i$.
- **Admission:** Packs tasks greedily up to the absolute physical radio envelope $B_{\max} = 175\text{ kB}$ under helper queue and source constraints.
- **Role:** Represents the unconstrained upper bound on computation throughput, ignoring sidelink safety. In coupled regimes, B1 causes severe packet drops and fails safety.

### B2: Channel-Aware CBR-Threshold Adaptive Cap (`cbr_effective_bytes`)
- **Mechanism:** Dynamically adjusts the allowable offload byte budget $B_{\text{eff}} \in [0, B_{\max}]$ based on measured Channel Busy Ratio (CBR) and reselection rate:
  $$B_{\text{eff}} = B_{\max} \cdot \left[ \frac{\text{CBR}_{\text{hi}} - \text{CBR}}{\text{CBR}_{\text{hi}} - \text{CBR}_{\text{lo}}} \right]_0^1 \cdot \mathbf{1}\{\text{resel} \le \text{th}_{\text{resel}}\}$$
- **Role:** Standard heuristic reactive rate-limiting. Leaves substantial capacity unused when CBR is moderately elevated even if PRR remains safe.

### B3: Reactive Myopic Dual Knapsack (`MyopicDualBaseline`)
- **Mechanism:** Solves a multiple-choice dual knapsack where task score is penalised by an adaptive shadow price $\lambda$:
  $$u_i = v_i - \lambda_t \left( \frac{b_i}{B_{\max}} \right)$$
- **Dual Ascent:** $\lambda_{t+1} = \max\left(0, \lambda_t + \eta (\mathbf{1}\{\text{PRR}_t < 0.95\} - \alpha)\right)$.
- **Role:** Standard subgradient dual ascent. Suffers from lag—only raises $\lambda$ *after* collisions have occurred.

### B4: Certainty-Equivalent Logistic Risk MILP (`ce_byte_cap`)
- **Mechanism:** Fits an offline logistic regression model to predict violation probability from state features:
  $$P(\text{viol}) = \sigma(w_0 + w_1 \cdot \text{CBR} + w_2 \cdot \text{resel} + w_3 \cdot (B / B_{\max}))$$
  Inverts the model to determine the maximum byte budget $B_{\text{eff}}$ such that $P(\text{viol}) \le \alpha$.
- **Role:** Model-based heuristic safety cap. Vulnerable to distribution shift between offline fit and online traffic bursts.

### B5 / B5′: PPO-Lagrangian on Shared Packer (`B5PrimePolicy`)
- **Mechanism:** Deep actor-critic with Lagrangian relaxation. The actor outputs dual price $\lambda_t$ and offload fraction $\beta_t$.
- **Advantage Formulation:** Evaluates dual advantages $A(\tau) = \frac{A^r(\tau) - \lambda A^c(\tau)}{1 + \lambda}$.
- **Role:** Standard constrained reinforcement learning baseline. Enforces long-term average constraint satisfaction, but cannot prevent transient burst exceedances.

---

## 3. PB-CRL v3.1: Novel Predictive Budget-Constrained RL Architecture

PB-CRL addresses the fundamental flaw of prior baselines: **safety cannot be managed purely in expectation or purely reactively; it requires predictive, short-horizon risk accounting coupled with dynamic task knapsack optimization.**

```
[Candidate Tasks] ──► [DeepSets Task Encoder] ──┐
                                                │
[Helper States]   ──► [Helper MLP Encoder]   ───┼──► [Shared Trunk MLP] ──► Action Heads:
                                                │                           ├─ λ_t (Shadow Price)
[Radio History]   ──► [10-Epoch GRU]          ──┤                           ├─ β_prop (Volume Fraction)
                                                │                           └─ g_t (Pacing Gain)
[Budget Block]    ──► [Risk State MLP]       ───┘                           
                                                                                  │
                                                                                  ▼
                                                                     [5-Critic Ensemble Q_c^+]
                                                                                  │
                                                                                  ▼
                                                                     [Predictive Risk Shield]
                                                                                  │
                                                                                  ▼ (β_exec, λ_t)
                                                                     [Helper-Aware Knapsack Packer]
                                                                                  │
                                                                                  ▼
                                                                       Admitted Offload Pack
```

### Component Details:
1. **Perception & State Representation:**
   - **DeepSets Task Encoder:** Permutation-invariant set aggregation over candidate tasks $(v_i, b_i, D_i)$.
   - **Radio History GRU:** 64-unit recurrent cell over the past 10 epochs ($200\text{ ms}$) tracking $[\text{CBR}, \text{resel}, B_t/B_{\max}, z_t]$ to capture sidelink channel congestion trajectory.
   - **Budget State Block:** Encodes remaining violation allowance $\kappa_t / K$, horizon remaining $T_{\text{left}} / T$, and remaining Poisson risk budget $m_{\text{rem}} / \nu(K)$.

2. **Policy Network Heads:**
   - $\lambda_t \in [0, \lambda_{\max}]$: Shadow price regulating knapsack efficiency.
   - $\beta_{\text{prop}} \in [0, 1]$: Proposed offload volume fraction.
   - $g_t \in [0.5, 2.0]$: Risk budget pacing gain adapting to channel stability.

3. **Pessimistic Safety Critic Ensemble $Q_c^+$:**
   - Ensemble of 5 independent neural networks predicting $P(\text{violation} \mid o_t, \beta)$.
   - Epistemic uncertainty aggregation: $Q_c^+(o_t, \beta) = \mu_Q + c_q \cdot \sigma_Q$.

4. **Predictive Risk Account & Shield:**
   - Computes allowed instantaneous risk:
     $$p_{\text{allow}} = \min\left( \frac{g_t \cdot m_{\text{rem}}}{T_{\text{left}}}, m_{\text{rem}} \right)$$
   - Filters proposed action: $\beta_{\text{exec}} = \max \{ \beta \le \beta_{\text{prop}} \mid Q_c^+(o_t, \beta) \le p_{\text{allow}} \}$.
   - Updates remaining Poisson risk budget: $m_{\text{rem}} \leftarrow \max(0, m_{\text{rem}} - Q_c^+(o_t, \beta_{\text{exec}}))$.
   - If an actual violation occurs ($z_t = 1$), resets $m_{\text{rem}} \leftarrow \nu(\kappa - 1)$ where $\nu(\kappa)$ is derived from Poisson tail bounds $\mathbb{P}(\text{Poisson}(m) > \kappa) \le \delta$.

5. **Helper-Aware Knapsack Packer (`price_budget_pack`):**
   - Implements multiple-choice knapsack optimization over available helper vehicles, ensuring tasks are only scheduled on helpers that can meet deadlines before vehicular departure.
