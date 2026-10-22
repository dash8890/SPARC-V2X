# System Model, Problem Formulation, SPARC-V2X (Direct PPO), and Evaluation Protocol

This note separates **theory** (Sections 1–3: no experiment-specific numeric constants) from **evaluation / simulation parameters** (Sections 4–5).

**Nomenclature.** In the Stage-5 evidence package, **SPARC-V2X** denotes the learned scheduler trained as **Direct PPO** under the shared safety governor (protocol baseline B3 / ablation A4): categorical logits are produced by a neural scorer alone, without an additive Value-Aware (VA) analytic term at decision time. Historical VA-residual SPARC (B0 / A0) is a related but different actor form and is not the subject of Sections 1–3 below.

---

## 1. System model

### 1.1 Dual traffic classes over NR-V2X Mode 2

We consider a vehicular ad hoc network operating under **3GPP NR-V2X Mode 2** sidelink. Two application classes share the same radio:

1. **Safety messaging.** Periodic cooperative awareness / safety payloads subject to radio reliability and latency requirements measured at the network layer.
2. **Computation offloading.** Finite-horizon task offloading from **source** vehicles to **helper** vehicles that offer CPU service.

Controllers act only at the **application / admission** layer: they select which computation offers to place. They do **not** allocate physical subchannels; Mode-2 sensing, reservation, and interference are handled by the radio simulator.

### 1.2 Discrete control horizon

Time is slotted into application epochs \(t=0,1,\ldots,T-1\) of duration \(\Delta\). Absolute time at the start of epoch \(t\) is \(t\Delta\). An episode comprises a fixed horizon of \(T\) epochs and a radio simulation of duration \(T_{\mathrm{sim}}\) synchronized to that horizon.

### 1.3 Sources, helpers, and tasks

Let \(\mathcal S\) be the set of computation sources and \(\mathcal H\) the set of helpers present in the scenario. Each source \(s\in\mathcal S\) releases a finite collection of tasks over the horizon. Task \(i\) is characterized by:

| Symbol | Meaning |
|---|---|
| \(s(i)\) | releasing source |
| \(r_i\) | release epoch |
| \(b_i\) | offered payload size (bytes) |
| \(c_i\) | computational demand (CPU cycles) |
| \(d_i\) | relative deadline length |
| \(v_i\) | nonnegative task value |

The **absolute deadline** is
\[
D_i = r_i\Delta + d_i.
\]
Helper \(h\in\mathcal H\) has service rate \(\mu_h\) (cycles per unit time) and an endogenous ready time \(\tau_h\) (earliest time its CPU is free).

### 1.4 Finish-time estimator (common to all controllers)

For any deadline-feasible pairing of task \(i\) with helper \(h\) at epoch \(t\), all methods use the **same** predicted finish time
\[
\widehat F_{i,h,t}
=
\max\bigl\{(t+1)\Delta,\,\tau_h\bigr\}
+
\frac{c_i}{\mu_h}.
\]
A pair is deadline-feasible at \(t\) only if \(\widehat F_{i,h,t}\le D_i\).

### 1.5 Candidate set

At epoch \(t\), the shared candidate set \(\mathcal C_t\) contains task–helper pairs \((i,h)\) that are released, unfinished, not already queued, and deadline-feasible under \(\widehat F_{i,h,t}\), subject to any common contact / reachability rules of the radio layer. Controllers may further restrict to a size-capped working set of cardinality \(|\mathcal C_t|\le K\) (candidate-set budget); when uncapped, \(K\) is the full feasible table size.

### 1.6 Action, uniqueness, and admission

A controller proposes a (possibly empty) set of offers \(\mathcal A_t\subseteq\mathcal C_t\) obeying:

- at most one helper per task;
- **at most one offered task per source per epoch**;
- per-epoch offered-byte envelope
  \[
  \sum_{(i,h)\in\mathcal A_t} b_i \le B_{\max}.
  \]

An external **safety governor** \(g\) maps a proposal and the current radio safety state into an executed assignment (or empty). Under the horizon-envelope path used here, \(g\) enforces:

- validity of safety measurement windows;
- hard thresholds on measured safety PRR and P95 delay (denoted \(\Gamma\) and \(D^{\max}\) respectively);
- the byte envelope \(B_{\max}\);

and, if the proposal is invalid or over budget, replaces it by a deterministic deadline-feasible Value-Aware (VA) fallback under the same envelope, or admits nothing when safety gates block.

### 1.7 Execution semantics and value accrual

An offered pair earns value **only if**:

1. the computation packet is successfully delivered over the Mode-2 radio in the corresponding interval; and  
2. helper execution completes by **both** the absolute deadline \(D_i\) and the control horizon.

Failed packets do not enter the helper queue and accrue no value. On success, \(\tau_h\) is updated consistently with the same finish/ready accounting used in \(\widehat F\). Merely selecting a task without delivery does not create phantom completion.

### 1.8 Safety observables

At valid measurement epochs \(t\in\mathcal T_{\mathrm{valid}}\), the radio layer exposes safety PRR \(\mathrm{PRR}^{\mathrm{safe}}_t\) and P95 delay \(D^{\mathrm{safe}}_{95,t}\). Invalid windows are logged separately and excluded from compliance averages.

---

## 2. Problem formulation

### 2.1 Objective

Let \(\mathcal I\) be all tasks released in the episode and \(\mathcal I^\star\subseteq\mathcal I\) the set completed under the semantics of Section 1.7. Define
\[
V_{\mathrm{done}}
=
\sum_{i\in\mathcal I^\star} v_i,
\qquad
V_{\mathrm{off}}
=
\sum_{i\in\mathcal I} v_i,
\qquad
R_{\mathrm{value}}
=
\frac{V_{\mathrm{done}}}{V_{\mathrm{off}}}
\]
(with the convention that \(R_{\mathrm{value}}\) is undefined if \(V_{\mathrm{off}}=0\)).

The scheduling objective is to maximize expected completed value (equivalently \(R_{\mathrm{value}}\) under matched task generation) subject to application-layer safety admission.

### 2.2 Safety criterion

Empirical violation rate on valid windows:
\[
P_{\mathrm{viol}}
=
\frac{1}{|\mathcal T_{\mathrm{valid}}|}
\sum_{t\in\mathcal T_{\mathrm{valid}}}
\mathbf{1}\!\left[
\mathrm{PRR}^{\mathrm{safe}}_t < \Gamma
\;\lor\;
D^{\mathrm{safe}}_{95,t} > D^{\max}
\right].
\]
A run with \(\mathcal T_{\mathrm{valid}}=\emptyset\) cannot be labeled compliant. PRR-floor and P95-ceiling failures may be reported separately in addition to their union.

### 2.3 Causal information structure

At decision time \(t\), every deployable controller observes the same causal information: released unfinished tasks, helper ready times and rates, the feasible candidate table \(\mathcal C_t\) (and features thereof), and the current safety measurement state used by \(g\). Controllers do **not** observe future arrivals, unrealized radio outcomes, or reserved simulator hidden state. After divergent admissions, endogenous radio and queue trajectories may differ; paired experiments match only **exogenous** seeds (mobility / task generation).

### 2.4 Mathematical program (epoch-wise view)

Abstractly, at each \(t\),
\begin{align}
\max_{\mathcal A_t\subseteq\mathcal C_t}
&\quad
\mathbb{E}\!\left[
\sum_{i\in\mathcal I^\star} v_i
\;\middle|\;
\text{history up to }t,\;\mathcal A_t
\right] \\
\text{s.t. }
&\quad
\text{source uniqueness; task uniqueness;} \\
&\quad
\sum_{(i,h)\in\mathcal A_t} b_i \le B_{\max}; \\
&\quad
\mathcal A_t = g(\widetilde{\mathcal A}_t,\mathcal C_t,\text{safety state}),
\end{align}
where \(\widetilde{\mathcal A}_t\) is the raw policy proposal. The overall episode problem is the sequential composition of these decisions under stochastic radio and mobility.

Because the radio and helper dynamics are high-dimensional and partially observed through the application bridge, we do not claim a closed-form optimum; SPARC-V2X approximates a sequential policy by reinforcement learning under the same constraints.

---

## 3. SPARC-V2X (Direct PPO)

### 3.1 Policy class

SPARC-V2X uses a **Direct PPO** actor–critic over candidates. For each candidate pair \((i,h)\in\mathcal C_t\) with feature vector \(x_{i,h,t}\in\mathbb{R}^{d}\), a shared MLP produces a scalar score
\[
q_\psi(x_{i,h,t})
=
\bigl(W_2\,\tanh(W_1 x_{i,h,t}+b_1)+b_2\bigr).
\]
Categorical logits (Direct / B3 form) are
\[
\ell_\psi(i,h,t)
=
\alpha\,q_\psi(x_{i,h,t}),
\]
with temperature / scale \(\alpha>0\). There is **no** additive analytic VA term in \(\ell_\psi\) (contrast VA-residual SPARC, where \(\ell=\alpha(s^{\mathrm{VA}}+\beta q_\psi)\)).

A sequential categorical policy samples a permutation / subset of candidates (with masks enforcing uniqueness and remaining byte budget) to form the raw proposal \(\widetilde{\mathcal A}_t\), which is then passed through the governor \(g\).

A separate value critic \(V_\phi(x)\) (same feature dimension) estimates state values for PPO. Direct PPO training does **not** apply VA-relative advantage shifting.

### 3.2 Analytic VA score (governor fallback only)

The corrected Value-Aware score
\[
s^{\mathrm{VA}}_{i,h,t}
=
\frac{v_i}{\max\{\widehat F_{i,h,t}-t\Delta,\,\varepsilon\}}
\]
is **not** added into Direct logits. It remains available as the deterministic ranking used inside \(g\) when a proposal is replaced under the envelope.

### 3.3 Reward and learning objective

Per epoch, the environment returns a reward equal to the increment in completed task value (plus any synchronized advance reward from the radio/task layer). Let \(r_t\) denote that scalar reward. Direct PPO optimizes the usual clipped surrogate with GAE advantages \(\widehat A_t\), entropy regularization, and a value loss—without VA-relative reweighting of \(\widehat A_t\).

Checkpoint selection on a fixed validation seed set uses a disclosed selection rule (e.g., best mean completed value relative to corrected VA on validation); **final-test seeds are never used for selection**.

### 3.4 What Direct PPO is (and is not)

| Is | Is not |
|---|---|
| Learned sequential scheduler under the **same** governor as analytical baselines | A PHY-layer Mode-2 resource allocator |
| Logits from \(q_\psi\) only (\(\ell=\alpha q_\psi\)) | VA-residual actor \(\alpha(s^{\mathrm{VA}}+\beta q)\) |
| Ordinary GAE / clipped PPO | VA-relative advantage shaping |
| Subject to hard PRR / P95 / byte gates via \(g\) | A soft Lagrangian safety dual (that is LVM / B4) |

---

## 4. Performance evaluation protocol and simulation setup

### 4.1 Common environment (fairness contract)

All compared methods share:

- WiLabV2Xsim **NR-V2X Mode 2** radio (MATLAB);
- the same Python task/helper layer and MATLAB bridge;
- identical task-generation and mobility inputs for a given \((\text{sim seed},\,\text{task seed})\) pair;
- the same finish estimator \(\widehat F\), candidate feasibility rules, and (for governor-matched methods) the same \(g\).

Methods differ only in how \(\widetilde{\mathcal A}_t\) is formed (and, for cross-admission baselines such as LVM/VLRO, whether \(g\) is active and which byte cap applies—always disclosed).

### 4.2 Seed partition

| Pool | Role |
|---|---|
| Training | On-policy PPO episodes; disjoint from validation/final |
| Validation | Checkpoint / hyperparameter selection only |
| Final test | All manuscript comparisons; sealed; never used for selection |

Paired contrasts use the **same** final-test seed pairs at every operating point unless a condition-specific extension is predeclared.

### 4.3 Estimands and reporting

Primary application estimand: \(R_{\mathrm{value}}\) (and \(V_{\mathrm{done}}\)) averaged over final-test seeds.  
Primary safety estimand: \(P_{\mathrm{viol}}\) on valid windows; also PRR and P95 summaries.  
Governor-matched methods are compared under identical \(B_{\max},\Gamma,D^{\max}\). Cross-admission methods are reported with their disclosed envelope / governor settings and must not be labeled “feasible under SPARC gates” unless those gates are active.

### 4.4 Principal method set (as used in Stage-5 figures)

| Acronym | Role |
|---|---|
| **SPARC-V2X** | Direct PPO + common governor |
| **VAS** | Corrected VA greedy + common governor |
| **LAS** | Min-finish greedy + common governor |
| **LVM** | Lagrangian / soft-constraint PPO (no SPARC governor; disclosed physical cap) |
| **VLRO** | Non-learning score \(v/(F\cdot b)\) with disclosed external admission |

### 4.5 Sensitivity axes (evaluation campaigns)

Without embedding numeric grids here, the protocol sweeps: vehicle density; admission envelope; tasks per source; deadline multiplier; candidate-set size \(K=|\mathcal C_t|\); and high-density specialization (nominal train density vs train-at-test density). Radio safety curves (PRR / P95 vs density; PRR vs distance) use the same sealed seeds where applicable.

### 4.6 Simulation architecture

Each closed-loop episode:

1. Instantiates mobility / radio from the Mode-2 trace configuration at the requested density parameter.  
2. Runs \(T\) application epochs; each epoch proposes offers, applies \(g\), advances the radio interval, and updates task/helper state.  
3. Logs completed value, offered value, safety metrics, governor interventions, and (when enabled) PRR-vs-distance audits.

---

## 5. Parameter list and values used in simulations

Symbols in Sections 1–3 map to the frozen Stage-5 / Direct-PPO implementation as follows. **These numbers belong only here**, not in the theoretical sections above.

### 5.1 Radio and scenario (WiLabV2Xsim)

| Parameter | Symbol / role | Value |
|---|---|---|
| Simulator | — | WiLabV2Xsim, NR-V2X Mode 2 |
| Config template | — | `vendor/WiLabV2Xsim/ConfigFiles/test_trace.cfg` |
| Nominal density parameter | \(\rho\) (implementation key) | 35 (unit not asserted as veh/km) |
| Density sweep (eval) | — | 20–100 (and stress 80) as in Fig.4/7 |
| Simulation time | \(T_{\mathrm{sim}}\) | 12.0 s |
| Generation interval | — | 100 ms |
| Allocation period | — | 20 ms |
| Safety payload | — | 300 B |
| SCS (NR) | — | 15 kHz |
| MCS (NR) | — | 7 |
| DMRS (NR) | — | 24 |
| Scenario type | — | traces |
| Lanes / road length | — | 3 / 2000 m |
| Mobility traces | — | `synthetic_highway_rho_{ρ}_20s.txt` (where available) |

### 5.2 Application / task layer

| Parameter | Symbol / role | Value |
|---|---|---|
| Controller epochs | \(T\) | 100 |
| Epoch step | \(\Delta\) | 10 ms |
| Tasks per source | — | 3 (nominal); load sweep \{2,3,4,5,6\} |
| Spread arrivals | — | true (uniform over horizon) |
| Unique source / epoch | — | true |
| Task payload | \(b_i\) | UniformInteger\([10^4,\,8\cdot10^4]\) B |
| Task cycles | \(c_i\) | Uniform\([10^7,\,10^8]\) |
| Relative deadline | \(d_i\) | Uniform\([30,\,150]\) ms |
| Task value | \(v_i\) | Uniform\([1,\,5]\) |
| Sources | \(|\mathcal S|\) | \(\max(1,\lfloor\rho/5\rfloor)\) |
| Helpers | \(|\mathcal H|\) | \(\min(12,\rho)\) |
| Helper CPU | \(\mu_h\) | Uniform\([2\cdot10^8,\,10^9]\) cycles/s |
| Computation packet | — | 1000 B |
| Finish estimator interval | — | 10 ms in \(\max\{(t+1)\Delta,\tau_h\}\) accounting |
| Candidate-set sizes (Fig.8) | \(K=\lvert\mathcal C_t\rvert\) | \{8, 16, 32, 64\} |
| Deadline multiplier sweep | — | \{0.5, 1.0, 1.5, 2.0\} |

### 5.3 Safety governor (SPARC-V2X / VAS / LAS)

| Parameter | Symbol | Value |
|---|---|---|
| Byte envelope | \(B_{\max}\) | 175000 B **per epoch** |
| Horizon-envelope-only | — | true (no next-window risk certificate) |
| PRR floor | \(\Gamma\) | 0.95 |
| P95 ceiling | \(D^{\max}\) | 50 ms |
| Risk \(\varepsilon\) (unused when horizon-only) | — | 0.05 |

### 5.4 Cross-admission baselines (disclosed)

| Method | Admission | Envelope / notes |
|---|---|---|
| LVM (B4) | No SPARC governor | Physical / forced cap **500000** B |
| VLRO | No SPARC governor | Score \(v/(F\cdot b)\); external byte handling as configured |

### 5.5 SPARC-V2X Direct PPO training

| Parameter | Value |
|---|---|
| Logit mode | `direct` (\(\ell=\alpha q_\psi\)) |
| Logit scale \(\alpha\) | 100 |
| Residual gain \(\beta\) | 1 (unused additively in direct mode) |
| Actor / critic MLP | Linear\((10,64)\)–Tanh–Linear\((64,1)\) |
| Feature dim \(d\) | 10 |
| Residual init | output layer zero (unless ablations) |
| VA-relative advantages | **false** (Direct) |
| Episodes | 96 (nominal); +48 continue run archived separately |
| Rollouts / update | 4 ⇒ 24 PPO updates / 96 eps |
| PPO epochs / update | 6 |
| Minibatch | 64 |
| Discount \(\gamma\) | 0.99 |
| GAE \(\lambda\) | 0.95 |
| Clip ratio | 0.2 |
| Value coef | 0.5 |
| Grad clip | 1.0 |
| Adam LR | \(3\cdot10^{-4}\), final fraction 0.2 |
| Entropy coef | 0.02, final fraction 0.25 |
| Target KL | 0.015 |
| Train sim / task seeds | \(72000+ep\) / \(182000+ep\) |
| Validation sim / task | \[82001,82002,82011,82012\] / \[182001,182002,182011,182012\] |
| Init RNG | 616161 |
| Checkpoint rule | best mean \(\Delta V\) vs corrected VA on validation |
| Nominal train density | \(\rho=35\) |
| Specialized train densities | \(\rho=50\), \(\rho=100\) (96 eps each; dens 75 blocked—no trace) |

### 5.6 Final-test seal

| Parameter | Value |
|---|---|
| Status | SEALED |
| \(n\) | 5 |
| Sim seeds | 93000–93004 |
| Task seeds | 193000–193004 |
| File | `protocol/final_test_seeds.json` |

### 5.7 Envelope sweep (Fig.5-type)

| Envelope \(B_{\max}\) (B) | 100000, 175000, 250000, 350000 |

---

## 6. Provenance

- Theory / fairness contract: `protocol/evaluation_protocol.md`  
- Numeric fill: `protocol/parameter_manifest.yaml`, `results/wilab_direct_ppo_96ep/training_config.json`, `controllers/rl/{residual_task_policy,safety_governor,residual_ppo}.py`  
- Seeds: `protocol/SEED_PARTITION.md`, `protocol/final_test_seeds.json`  
- Direct PPO as SPARC-V2X in Stage-5 figures: sealed eval under `run_paired_eval.py` with `--logit-mode direct` and checkpoint `results/wilab_direct_ppo_96ep/checkpoint_best_delta_va.pt`
