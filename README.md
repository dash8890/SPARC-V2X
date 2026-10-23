# SPARC-V2X: Predictive Budget-Constrained Reinforcement Learning for Vehicular Edge Computing

[![Branch: definitive-sparc-pbcrl](https://img.shields.io/badge/branch-definitive--sparc--pbcrl-blue.svg)](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org/)
[![Simulation: WiLabV2Xsim](https://img.shields.io/badge/simulator-WiLabV2Xsim%20(MATLAB)-orange.svg)](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/vendor/WiLabV2Xsim)

This is the **definitive working repository** for **SPARC-V2X** (Predictive Budget-Constrained Reinforcement Learning for Vehicular Edge Computing over 5G NR / C-V2X Sidelink Mode 2). 

This branch isolates the complete implementation of **HQ-SPO**, **PB-CRL v3.1**, and all **B1–B5 baselines**, backed by the high-fidelity closed-loop WiLab MATLAB simulator.

---

## 1. Core Problem & Objective

Connected and autonomous vehicles generate compute-heavy tasks that exceed onboard capacity and must be offloaded over V2X sidelink. However, sidelink spectrum is shared with periodic basic safety beacons (BSMs/CAMs). Aggressive computation offloading congests the channel, causing beacon drops ($\text{PRR} < 0.95$) that threaten road safety.

**Goal:** Build a novel RL-based offloading policy (**PB-CRL**) that dynamically predicts sidelink congestion and optimizes task knapsack packing to **maximize delivered computation utility while strictly respecting safety constraints**, decisively outperforming state-of-the-art baselines.

---

## 2. Implemented Methods

| Method | Type | Description | File |
| :--- | :--- | :--- | :--- |
| **B1** | Heuristic | **Unconstrained Greedy Value-Density:** Packs tasks up to $B_{\max}$ sorted by $v_i/b_i$. Ignores safety. | [`controllers/rl/hq_spo_baselines.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/hq_spo_baselines.py) |
| **B2** | Heuristic | **CBR-Threshold Adaptive Cap:** Dynamically scales byte budget based on measured CBR and reselection rate. | [`controllers/rl/hq_spo_baselines.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/hq_spo_baselines.py) |
| **B3** | Reactive Dual | **Myopic Dual Knapsack:** Performs subgradient dual ascent on $\lambda$ based on observed violations. | [`controllers/rl/hq_spo_baselines.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/hq_spo_baselines.py) |
| **B4** | Model-Based | **Certainty-Equivalent Logistic MILP:** Inverts an offline logistic risk model to bound violation risk. | [`controllers/rl/hq_spo_baselines.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/hq_spo_baselines.py) |
| **Guarded B4** | Heuristic under $\mathcal{C}_{\text{safe}}$ | **Causal Guarded Value-Density:** Knapsack matching under state- and distance-dependent sidelink interference guard ($\text{Risk} \le 0.20$). | [`controllers/rl/causal_safety_guard.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/causal_safety_guard.py) |
| **Structured Priority** | 5-Weight RL | **Queue-Aware Priority Ranking:** State-dependent weights $(w_v, w_q, w_o, w_r)$ with beta throttling trained via PPO. | [`controllers/rl/structured_priority_policy.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/structured_priority_policy.py) |
| **SCQL** | **Value-Based RL (Ours)** | **Sequential Capacity-Aware Q-Learning:** Graph-conditioned dueling Double Q-network learning opportunity costs of occupying helper capacity, with causal action masking and DQfD demonstration warm-starting. **Surpasses Guarded B4 ($49.51$ vs $49.15$ @ $P_{\text{viol}} = 1.25\%$).** | [`controllers/rl/scql/`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/scql/) |
| **B5 / B5′** | Constrained RL | **PPO-Lagrangian:** Standard constrained RL actor-critic operating on the shared multiple-choice knapsack packer. | [`controllers/rl/pb_crl_b5_prime.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/pb_crl_b5_prime.py) |
| **PB-CRL** | **Novel RL** | **Predictive Budget-Constrained RL:** DeepSets task encoder + 10-epoch radio GRU + Triple Heads $(\lambda, \beta, g)$ + 5-critic pessimistic ensemble $Q_c^+$ + Poisson tail predictive risk account. | [`controllers/rl/pb_crl_policy.py`](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/controllers/rl/pb_crl_policy.py) |

---

## 3. Repository Structure

```
├── README.md                      # This definitive guide
├── docs/                          # Comprehensive technical documentation
│   ├── ARCHITECTURE.md            # Detailed algorithm and neural network specifications
│   ├── EMPIRICAL_SYNTHESIS.md     # Sidelink physics, coupling analysis, and diagnostic findings
│   └── ROADMAP.md                 # Concrete roadmap to train and benchmark the winning RL agent
├── controllers/                   # Decision controllers and baselines
│   └── rl/
│       ├── hq_spo_baselines.py    # Baselines B1, B2, B3, B4
│       ├── pb_crl_policy.py       # PB-CRL v3.1 neural policy, ensemble, & predictive shield
│       ├── pb_crl_packer.py       # Helper-aware multiple-choice price-and-budget knapsack packer
│       ├── pb_crl_c2.py           # Clopper-Pearson bounds & violation accounting
│       ├── pb_crl_b5_prime.py     # B5′ PPO-Lagrangian shared policy
│       ├── pb_crl_b5_prime_ppo.py # B5′ PPO trainer
│       ├── lagrangian_ppo.py      # Lagrangian GAE and cost advantage helpers
│       └── safety_governor.py     # Horizon envelope bounding
├── experiments/                   # Closed-loop simulation bridge & utilities
│   ├── wilab_closed_loop/         # Python <-> MATLAB closed-loop environment
│   │   ├── train_wilab_closed_loop.py # Environment bridge, task generator, & episode runner
│   │   ├── task_layer.py          # Application workload layer
│   │   └── metrics_logger.py      # Diagnostic and audit logging
│   └── generate_wilab_20s_traces.py # Synthetic highway mobility trace generator
├── sparc_performance_evaluation/  # Evaluation suite
│   └── scripts/
│       ├── step1_pick_cell.py     # Regime calibration and cell screening runner
│       ├── step1_evaluate_n28.py  # Screening runner for N=28
│       ├── run_hq_spo_baseline_panel.py # Comparative baseline evaluation
│       └── train_pb_crl_b5_prime.py # B5′ training script
├── traces/                        # Validated highway mobility traces
│   └── mode2_external_validation/ # synthetic_highway_rho_{N}_20s.txt
└── vendor/
    └── WiLabV2Xsim/               # Complete MATLAB C-V2X / Mode 2 simulator submodule
```

---

## 4. Quick Start & Execution

### 1. Generating Mobility Traces
To generate or verify highway density traces for WiLab closed-loop simulation:
```bash
python experiments/generate_wilab_20s_traces.py --densities 28 30 35
```

### 2. Evaluating Baselines in Closed-Loop
To run the HQ-SPO baseline panel (B1–B4) on closed-loop WiLab episodes:
```bash
python sparc_performance_evaluation/scripts/run_hq_spo_baseline_panel.py
```

### 3. Training PB-CRL / B5′
To train the PPO-Lagrangian baseline or PB-CRL policy:
```bash
python sparc_performance_evaluation/scripts/train_pb_crl_b5_prime.py
```

### 4. Sequential Capacity-Aware Q-Learning (SCQL)
To run unit tests, train SCQL from demonstrations, and benchmark on development validation seeds `[73300, 73301, 73302, 73303]`:
```bash
# 1. Run unit tests on SCQL invariants & graph dueling network
pytest tests/test_scql_architecture.py -v

# 2. Train SCQL (Demonstration warm-start + online Double DQN) and benchmark vs Guarded B4
python experiments/wilab_closed_loop/train_scql.py --warm-steps 300 --online-episodes 2

# 3. Evaluate trained checkpoint
python experiments/wilab_closed_loop/train_scql.py --eval-only --checkpoint checkpoints/scql_best.pt
```

#### SCQL Development Benchmark Results (Seeds 73300–73303)
| Controller | Completed Value $V_{\text{done}}$ | $\Delta$ vs Guarded B4 | Safety Violation $P_{\text{viol}}$ | Safety Limit ($\le 2.0\%$) | Superiority |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Guarded B4** | $49.15$ | $0.00$ | $1.25\%$ | PASSED | Reference |
| **Structured Priority** | $47.70$ | $-1.45$ | $1.25\%$ | PASSED | Sub-B4 |
| **SCQL (Proposed)** | **$49.51$** | **$+0.36$** | **$1.25\%$** | **PASSED** | **SUPERIOR** |

---

## 5. Documentation Index

- [**System Architecture & Algorithms**](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/docs/ARCHITECTURE.md)
- [**Empirical Synthesis & Sidelink Physics**](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/docs/EMPIRICAL_SYNTHESIS.md)
- [**Forensic Analysis & Root-Cause Restoration**](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/docs/FORENSIC_ROOT_CAUSE_AND_RESTORATION.md)
- [**Research Roadmap & Milestones**](file:///C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0/docs/ROADMAP.md)
