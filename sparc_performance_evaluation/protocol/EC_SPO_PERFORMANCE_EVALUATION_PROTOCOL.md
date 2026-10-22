# EC-SPO Performance Evaluation Protocol

**Scope:** 8 figures (4 subfigures each, 32 panels) and 4 tables, excluding the system-model and method figures and the symbols/parameters table.  
**Math:** unchanged. Objective \(\mathbb E[U(\tau)]\) with per-task VoI \(v_i\); protection constraint C2, \(P(\sum z_w > K) \le \delta\); methods B1–B5, B5′, PB-CRL-core, and the v3.1 ablations.  
**Status:** locked protocol. No result here has been measured. Numbers marked "proposal" are choices to fix before running.

---

## 0. Rules that apply to every experiment

1. **Feasibility first.** In each scenario cell, a method is *feasible* if the exact one-sided 95% Clopper-Pearson upper bound on its C2 exceedance probability is \(\le \delta\). Utility is ranked only among feasible methods. An infeasible method is shown, marked infeasible, and counts as a loss in that cell. Utility is never averaged across cells where the feasible sets differ.
2. **Same information, latency budget, and tuning effort** for every method (as in the baseline spec). Baselines are tuned **per cell** on that cell's validation seeds. Learned policies are trained once across the randomized scenario distribution and are never tuned per cell. This favors the baselines and is stated in the paper.
3. **Drift cells are never seen** in training, tuning, or calibration by any method.
4. **Common random numbers.** All methods in a comparison use the same exogenous seeds (arrivals, background traffic, mobility, fading). Comparisons are paired.
5. **Train / validation / test seed sets are disjoint.** The test set is touched once per reported number.
6. **Two fidelity tiers.** WiLab results (full fidelity) support every headline claim. Proxy-environment results support sweeps, frontiers, ablations, and training only, and each reported sweep is spot-checked in WiLab (Table 4). If proxy fidelity fails the Table 4 gate, the affected sweeps are rerun in WiLab at reduced size.
7. **Pre-registered decisions** (§9) are written down before test runs.
8. **Failures are reported.** Cells or metrics where PB-CRL-core loses or is infeasible stay in the figures and tables.

---

## 1. Questions, claims, and evidence

| RQ | Question | Claim it supports | Evidence |
|---|---|---|---|
| RQ1 | Does offloaded load measurably degrade Mode 2 reliability, and through which mechanism? | The coupling is real and decision-relevant (gate G7) | Fig 1 |
| RQ2 | What is the exchange rate between timely service and protection? | The frontier and protected capacity are the object of study | Fig 2, Fig 4 |
| RQ3 | Does the controller beat strong baselines at the specified protection level? | Method claim (Branch A) or formulation-plus-shield claim (Branch B) | Fig 3, Table 1, Table 2 |
| RQ4 | Is safety traffic actually protected, in standard KPIs? | Protection claim | Fig 5 |
| RQ5 | What happens inside the network, and how does the controller spend protection budget? | Network-aspects contribution (TCCN) | Fig 6 |
| RQ6 | Do the conclusions survive drift, calibration error, and KPI choices? | Robustness | Fig 7 |
| RQ7 | Where does the gain come from, and is it deployable and affordable? | Attribution and practicality | Fig 8, Table 2, Table 3 |

---

## 2. Scenario design

### 2.1 Factors

| Factor | Levels (proposal, fix from stack) | Role |
|---|---|---|
| Density, \(N_{\mathrm{UE}} / N_r\) | Low, Mid, High, plus held-out Higher (drift) | Saturation regime |
| Offered task rate (per vehicle) | Low, Mid, High | Demand |
| Background traffic mix | Periodic CAM only; CAM + variable-size CPM (drift) | Reselection churn |
| Latency tier \(D_c\) | Tiers taken from TS 22.186 emulated rows | Deadlines |
| Pack size \(b_i\) distribution | Narrow, wide | Load shape |
| Cluster share \(n_c / N\) | Small, large | Coupling strength |
| Helper service rate | Low, Mid, High | Compute-side coupling |
| Mobility / speed | Baseline; faster (drift) | Mobility shift |
| Protection budget \(K\) and \(\delta\) | Main setting; tighter and looser for frontier sweeps | Constraint level |

### 2.2 Regime calibration rule (so the constraint actually binds)
Define regimes by the behavior of the **unconstrained** offloading policy (offer everything feasible):
- **Low:** its C2 exceedance probability is already \(\le \delta\). Every method should tie here. This is a sanity check.
- **Mid:** its exceedance probability is well above \(\delta\), but a feasible policy keeps a large share of service.
- **High:** a feasible policy must sacrifice a large share of service.

If a cell never leaves Low, it carries no information and should be dropped from the headline set.

### 2.3 Cells
- **Headline cells (WiLab):** 3 in-distribution cells (Low/Mid/High density at Mid demand level) and 2 drift cells (held-out higher density; CAM+CPM mix with faster mobility). Total 5.
- **Sweep cells (proxy, with WiLab spot checks):** full factorial subset needed for Figs 2, 4, 7.

### 2.4 Seed sets
Disjoint pools for training, validation (tuning and calibration), and test. Test seeds are fixed in advance and shared across methods.

---

## 3. Methods and tuning

| Label | Method | Tuning rule |
|---|---|---|
| B1 | Greedy value-density packing with a tuned byte cap | Grid over the cap |
| B2 | CBR-threshold admission (CBR and reselection-rate gates) | Grid over three thresholds |
| B3 | Myopic dual packing | Grid over \(\eta\) and \(\lambda_0\) |
| B4 | One-step certainty-equivalent MILP with safety cut | Fit logistic risk model on logs; grid over the margin |
| B5 | PPO-Lagrangian, original (residual / pack-logit + governor) | Fixed search budget \(N_{\mathrm{tune}}\) |
| B5′ | PPO-Lagrangian on the shared packer | Same budget \(N_{\mathrm{tune}}\) |
| B5′+shield | B5′ with the same risk account and shield as PB-CRL | Same budget; shield calibrated as in v3.1 |
| PB-CRL-core | The proposed method (three scalar heads, risk account, shield, PID, paired rollouts, warm start) | v3.1 §5 |
| Ceiling (proxy only) | Clairvoyant planner with future knowledge, drawn as a dashed reference | Not a competitor |

The v3.1 ablations (§9 there) and the \(+\delta\) extension are used only in Fig 8.

**Operating-point rule.** Each baseline is tuned to the highest validation VoI-weighted throughput among configurations whose validation CP upper bound on C2 exceedance is \(\le \delta\). If none exist, the method is infeasible in that cell.

---

## 4. Metrics and estimators

**Primary**
- **VoI-weighted timely throughput** = \(U(\tau)\) per unit time, per vehicle.
- **C2 exceedance probability** = fraction of test episodes with \(\sum_w z_w > K\), with exact CP upper bound. Feasibility flag per cell.

**Secondary service**
- **TTCR** = fraction of generated tasks completed by deadline (the \(v_i = 1\) special case, computed post hoc).
- **Result age** (post-hoc diagnostic only).

**Protection (TR 37.885 definitions)**
- PRR type 1 in distance bins, and average PRR at the reference distance.
- PIR distribution (CDF) at the reference distance.
- **C1 fraction**: mean violating-window fraction, reported alongside C2.

**Network**
- CBR time series; **reselection / reassignment rate**; reservation occupancy.
- Loss breakdown: collisions vs half-duplex vs other.
- **Coupling index \(\chi\)** = slope of window PRR (and of reselection rate) against own offered load \(L = \mathrm{bytes}(A_t) / B_{\max}\) at fixed background (CRN seeds), with bootstrap CI.
- **Churn effect** = change in PRR and reselection rate between equal-volume stable and varying pack size or timing.

**Capacity**
- **Protected timely offloading capacity** \(\lambda^\star(\delta, \tau)\) = largest offered task rate with \(\mathrm{TTCR} \ge \tau\) and C2 feasible. Estimated by bisection on arrival rate with CRN seeds and \(n\) episodes per probe, with bootstrap CI over seed sets. \(\tau\) is fixed in advance (proposal: report \(\tau \in \{0.8, 0.95\}\)).

**Cost**
- Decision latency (median and 99th percentile), model size, training budget in WiLab and proxy episodes.

---

## 5. Statistical protocol

**Episodes per cell for the feasibility test.** Exceedance is a per-episode event, so \(n\) counts episodes. Normal-approximation guide for the sample size needed to show \(\mathrm{UB} \le \delta\) when true rate is half of \(\delta\) (confirm with exact CP):

| \(\delta\) | True rate | Approximate \(n\) |
|---|---|---|
| 0.10 | 0.05 | 60–70 |
| 0.05 | 0.025 | 110–125 |
| 0.02 | 0.01 | 280–300 |

Proposal: \(n = 120\) test episodes per method per cell if \(\delta \approx 0.05\). Scale with chosen \(\delta\).

**Paired utility comparison.** Per-seed differences in VoI-weighted throughput, 10,000 bootstrap resamples, 95% CI. Superiority claim: CI lower bound above the equivalence margin. Equivalence: 95% CI inside the margin (pre-registered, \(\pm 1\%\) of reference utility at matched feasibility).

**Multiple comparisons.** Holm correction across cells for superiority claims. Equivalence is reported per cell, uncorrected, as intervals.

**Learned methods.** At least 5 training seeds. Report across-seed spread, and report the checkpoint chosen on validation, never on test.

**Aggregation.** Report per-cell results plus win/tie/loss counts and the feasibility rate across cells. Aggregate utility only over cells in which all compared methods are feasible.

---

## 6. Figures (8 figures, 4 subfigures each)

* **Fig 1. Coupling and mechanism (evidence for RQ1):** (a) Window PRR vs load \(L\), slope \(\chi\). (b) CBR and reselection rate vs \(L\). (c) Churn test (stable vs varying pack). (d) Window violation probability vs \(L\).
* **Fig 2. Service-protection frontier (RQ2, RQ3):** (a) Throughput vs C2 exceedance at Mid density. (b) High density. (c) TTCR version. (d) Zoom on \(\delta\) boundary.
* **Fig 3. Headline comparison at specified protection level (RQ3):** (a) VoI throughput per method/cell with paired 95% CIs. (b) Paired gain over best feasible baseline, \(\pm 1\%\) margin shaded. (c) C2 exceedance CP UB against \(\delta\). (d) TTCR per method/cell.
* **Fig 4. Protected timely offloading capacity (RQ2):** (a) \(\lambda^\star\) vs density. (b) \(\lambda^\star\) vs pool size. (c) Timely service vs offered task rate. (d) Service vs protection budget \(K\).
* **Fig 5. Safety traffic under offloading, standard KPIs (RQ4):** (a) PRR vs distance vs requirement line. (b) PIR CDF. (c) Average PRR at reference distance. (d) Window PRR distribution.
* **Fig 6. Network internals (RQ5):** (a) CBR over time. (b) Reselection rate over time / CDF. (c) Loss breakdown (collision, half-duplex, other). (d) Representative episode traces: \(\beta_t\), \(\kappa_t\), risk-account level vs baselines.
* **Fig 7. Robustness (RQ6):** (a) Held-out density. (b) Held-out traffic mix and faster mobility. (c) Drift guard calibration ratio \(r\). (d) Reliability diagram of \(Q_c\).
* **Fig 8. Attribution and learning behavior (RQ7):** (a) Paired gain for ablations 1–7. (b) Attribution ratio \(\rho_\delta\). (c) Paired gain & gradient-variance with/without CV. (d) Invariance check under \(v_i\) rescaling.

---

## 7. Tables (4)

* **Table 1. Main results at the specified protection level:** Rows: B1, B2, B3, B4, B5, B5′, B5′+shield, PB-CRL-core. Columns: Throughput, TTCR, C2 exceedance (point + CP UB), C1 fraction, feasibility rate.
* **Table 2. Paired comparisons and pre-registered fork:** PB-CRL-core vs B5′, vs B5′+shield, vs best baseline. Gain, Holm superiority, equivalence, \(\rho_\delta\), branch verdict (A or B).
* **Table 3. Cost and deployment:** Latency (p50/p99 vs 10 ms), model size, training budget (WiLab + proxy), wall-clock time, tuning budget used.
* **Table 4. Proxy fidelity and transfer:** Spot-checked sweep points: calibration error, utility error, Kendall's \(\tau\), gate pass/fail.

---

## 8. Execution order and gates

1. Proxy fidelity gate (Table 4).
2. Coupling characterization (Fig 1, Gate G7).
3. Regime calibration (§2.2).
4. Baseline tuning per cell on validation seeds.
5. B5 / B5′ training (\(\pi_{\mathrm{ref}}\) selection).
6. PB-CRL-core training.
7. Headline evaluation (Fig 3, Table 1, Table 2, Fig 5) on WiLab test seeds.
8. Sweeps and capacity (Figs 2, 4) in proxy with WiLab spot checks.
9. Drift cells and calibration (Fig 7).
10. Ablations and attribution (Fig 8).
11. Network internals (Fig 6).

---

## 9. Pre-registered decisions

- Equivalence margin: \(\pm 1\%\) of reference utility at matched feasibility, 95% paired CI.
- Fork: **Branch A** if PB-CRL-core beats B5′+shield beyond margin. **Branch B** if B5′+shield is within margin.
- \(\delta_s\) and \(c_q\) chosen on validation scenarios only.
- Attribution ratio \(\rho_\delta > 0.5 \implies\) per-task scoring drives gain.

---

## 10. Compute budget (plan)

| Item | Episodes | Time (single core) |
|---|---|---|
| Coupling characterization | 720 | 8.0 h |
| Headline test (7 methods \(\times\) 5 cells \(\times\) 120) | 4,200 | 46.7 h |
| WiLab spot checks of proxy sweeps | 600 | 6.7 h |
| Training-side WiLab data, refits, selection | 710 | 7.9 h |
| **Total** | **6,230** | **~69 h** (~9 h on 8 cores) |

---

## 11. Threats to validity
Mitigations for proxy gap, shield miscalibration, baseline under-tuning, scenario cherry-picking, KPI dependence, seed sensitivity, simulator-specific parameters.

---

## 12. Logging checklist
Per-window \(z_w\), PRR/PIR, CBR, reselection rate, reservation-table summary, task timestamps, offered bytes, loss-cause tags, decision latency, seeds, policy actions, shield flags, risk-account state.
