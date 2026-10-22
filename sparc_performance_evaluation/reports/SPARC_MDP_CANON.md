# SPARC-V2X — Program Canon (all load-bearing decisions)

**Purpose:** Nothing important may exist only in chat. This file indexes and states every load-bearing decision of the MDP correction program.  
**Read order:** `SPARC_MDP_HANDOFF.json` → **this file** → linked deep specs.  
**Authorization to run code/MATLAB is still only in HANDOFF** (`authorized_now`).

---

## 0. Closed and frozen

| Item | Decision | Where |
|------|----------|--------|
| RAO Direct↔POPL causal selector | **CLOSED** after G4 FAIL | `RAO_PHASE1_DUAL_POLICY_SELECTOR_CLOSED.flag`, `RAO_PHASE1_G1_G7_REPORT.md` |
| G4 result | Δvs Direct = 0; Δvs POPL = −1.154; selector ≡ always-Direct on held-out | same |
| CA-PPO / W-advantage path | **CLOSED** after Phase 2d FAIL | `SPARC_MDP_PHASE2D_REPORT.md`, HANDOFF |
| Frozen-generator regime routers (pressure-gate v1/v2, Direct↔POPL↔LVM) | **CLOSED** — same failure class as RAO; v1 FAIL (1/7 ≥ best); v2 aborted | `PRESSURE_GATE_EVAL_REPORT.md`, `FROZEN_GENERATOR_ROUTING_CLOSED.md` |
| POPL pack-defect fix (omit + helper bind) | **FAIL** (2026-10-19) — decode-only A+B hurt all 4 nominal seeds (−8.5 vs best); path exhausted on this checkpoint | `POPL_PACK_DEFECT_FIX_REPORT.md` |
| Research diagnosis + objective lock | **LOCKED** — algorithm-first was the failure; success=beat LVM on \(J\); next=formulation then consequentiality-vs-LVM gate; no new RL until both | `SPARC_RESEARCH_DIAGNOSIS_AND_OBJECTIVE.md` |
| LVM-RPI-Distill | **CLOSED (FAIL)** — residual-on-LVM dropped; headroom premise fails under locked env; C0/distill not to run | `SPARC_LVM_RPI_PATH_CLOSED.md`, `SPARC_FORMULATION_LOCK.md` (historical) |
| Next path | **Fork choice** A limits / B packing-pressure / C LVM misspec repair — no method until fork kill gate | `SPARC_NEXT_ROADMAP.md` |
| Problem pivot | tps4 packing pressure **kept**; frozen POPL/Direct mean probe FAIL only | `SPARC_PROBLEM_PIVOT_PACKING_PRESSURE.md`, `SPARC_PACKING_PRESSURE_LVM_GATE_REPORT.md` |
| BUILD TO WIN | Superseded as method path; lesson retained | `SPARC_BUILD_TO_WIN.md` |
| EC-SPO novel problem | **CLOSED** — G9 FAIL×2; radio-endogenous claim unsupported | `SPARC_NOVEL_RL_NECESSARY_PROBLEM.md` |
| HQ-SPO (helper-queue endogenous) | **ACTIVE** problem | `SPARC_HQ_SPO_PROBLEM.md` |
| HQ-SPO baselines to beat | **LOCKED** — (1) greedy \(v/b\) (2) CBR-threshold admission (3) myopic dual/LVM-class (4) one-step CE MILP (5) PPO-Lagrangian; eval protocol in suite doc. Success = flat win vs best feasible of these five — **not** “beat LVM/Direct only” | `SPARC_HQ_SPO_BASELINE_SUITE.md` |
| HQ-SPO method learning | **LOCKED** — convergence + **substantial learning over time** are non-negotiable; a short smoke / fluke beat without a learning curve does **not** count as a method win | user 2026-10-20 |
| HQ-SPO method algorithm | **SPEC = PB-CRL v3.1** (supersedes v3/v2/v1). Core: \((\lambda,\beta_{\mathrm{prop}},g)\) + shared price-and-budget packer; B5′ fair comparator; conditional warm-start; C2 risk account + drift guard; matched-C2 headroom; two-tier proxy+WiLab; pre-registered B5′+shield fork. **Not run.** **Not** pack-logit PPO. | `SPARC_HQ_SPO_PB_CRL.md`, user 2026-10-20 |
| B5 pack-logit “method” retrain | **FORBIDDEN / CLOSED mistake** — `train_hq_spo_method_converge.py` is B5, not the method | user 2026-10-20 |
| PB-CRL implementation | **Not authorized** until user phrase; next when authorized = Tier-2 data + B5′ rebuild / headroom — not bulk train | HANDOFF |
| Weak pack-search-as-paper | **FORBIDDEN** as deliverable | user reject |
| Final-test seeds `9300x` | **Sealed** | `rao_phase1_seeds.json` |
| Locked val `82001/02/11/12` | Forbidden for fitting / threshold / model selection in this program | same |
| Baselines Direct, LVM, VAS, LAS, VLRO | Frozen comparators | Stage-0 / POPL reports |
| Governor + 175 kB envelope | Unchanged unless a later phase explicitly re-locks | runbook |

**Do not:** reopen RAO selector tuning; retune frozen-generator routers; start CA-PPO training before Phase 1 PASS; open final-test; claim 15% from unrun methods; run another “one more gate/eval” loop on locked val.

---

## 1. Why we are here (evidence, not vibes)

### 1.1 Three possibilities (research fork)

| # | Hypothesis | Status after RAO + POPL |
|---|------------|-------------------------|
| (1) | Headroom exists; learning/credit fails | Real for residual ~2% oracle gap + selector miss (e.g. pair 47); **not** an explanation for missing 15% |
| (2) | Little headroom under locked env / additive `V_done` | **Primary:** Direct∪POPL oracle ≈ +1.9% vs Direct (n=48); 69% disagreeing packs have ΔG=0; val POPL +0.39 vs LVM |
| (3) | Protocol obscures effects | Secondary: force_match fails with G=0/P_viol=1; sparse coverage — compresses signal, does not invent 15% |

### 1.2 Objective / MDP under-attended

Failed branches (JIPI/RHO/POPL/RAO) mostly changed **generators**. The **utility and MDP** stayed thin: additive `V_done`, delayed credit, governor outside `logπ`, Direct `γ=0.99` vs eval `γ=1`.

### 1.3 Key numbers to keep

| Fact | Value |
|------|--------|
| LVM nominal mean (val context) | ≈ 46.82 |
| 15% aspirational target | ≈ 53.84 (+~7) |
| POPL vs Direct (locked val) | +1.15 (inconclusive CI) |
| POPL vs LVM (locked val) | +0.39 |
| RAO oracle vs Direct (48 train-dev pairs) | ≈ +1.9% |
| RAO frac disagree with ΔG=0 | ≈ 69% |

---

## 2. MDP reformulation (canonical math)

**Full math:** `SPARC_MDP_REFORMULATION_MATH.md` (do not paraphrase away \(S_t^+\)).

**One line:** choose pack \(A_t\) → governor → \(A_t^{\mathrm{exec}}\) → deterministic commitments \(S_t^+=\Phi(S_t,A_t^{\mathrm{exec}})\) → stochastic radio/compute → optimize \(\mathbb{E}[G_0]\) with \(\gamma=1\), \(G_t=\sum_{k\ge t}r_k\).

**Learning intent (Phase 2+ only):** auxiliary \(W(Z,A^{\mathrm{exec}})\approx\mathbb{E}[G_t\mid Z,A^{\mathrm{exec}}]\) on realized returns; actor `logπ` on **proposed** pack.

---

## 3. Utility thesis (Phase-0 text — already drafted here)

When user says `AUTHORIZE Phase 0`, **copy §§3.1–3.4 below** into `SPARC_MDP_PHASE0_UTILITY_LOCK.md` (may tighten wording; must not contradict Canon or reformulation math).

### 3.1 Utility
- Primary scalar: undiscounted deadline-valid completed value \(V_{\mathrm{done}}\) / \(G_0\) with \(\gamma=1\).
- Safety: **constraint-only** (SafetyGovernor + 175 kB + feasible set). Not a soft penalty in \(r_t\) under this lock.
- Forbidden as training reward: RHO-style analytic \(E[V]\), slack-only scores, invented \(\hat p\), RAO selector retuning objectives.

### 3.2 Decision problem
As in reformulation math: pack action, governor in transition, proposed vs executed split, post-decision commitments \(S_t^+\).

### 3.3 Success criteria by phase
- **Phase 1:** commitment features lift + material pack-choice swing (runbook thresholds) — **not** +15%.
- **Phase 2:** non-regression then staged hygiene gates (2a–2d).
- **Phase 3:** matched \(V_{\mathrm{done}}\) vs frozen Direct and LVM under authorized budget.
- **Aspirational 15%:** recorded, not a gate for Phase 1–2a.

### 3.4 Closed paths
RAO dual-policy selector CLOSED; cite flag.

---

## 4. Four-phase ladder (what / how / stop)

Deep procedure: `SPARC_MDP_EXECUTION_RUNBOOK.md`. Overview: `SPARC_MDP_CORRECTION_PROGRAM.md`. Live gate: `SPARC_MDP_HANDOFF.json` + `SPARC_MDP_PROGRAM_STATUS.md`.

| Phase | What | How (concrete) | Stop |
|-------|------|----------------|------|
| **0** | Utility lock doc | Copy Canon §3 → `SPARC_MDP_PHASE0_UTILITY_LOCK.md`; cite math file | No MATLAB/PPO; then wait for Phase 1 auth |
| **1** | Headroom + commitment diagnostic | Script `run_mdp_phase1_headroom.py` on `rao_phase1/pairs.json`; features + PASS numbers in runbook | FAIL → `SPARC_MDP_CA_PPO_PATH_CLOSED.flag`; no Phase 2 |
| **2a** | γ=1 + drain | Touch `residual_ppo.py` / train entry per runbook | Non-regression or stop |
| **2b** | Full-pack `logπ`; proposed vs exec | Likelihood hygiene + unit test | Then wait 2c auth |
| **2c** | Auxiliary \(W\) only | MSE on realized \(G_t\); actor not driven by \(W\) | FAIL → no 2d |
| **2d** | \(W\) in advantage | Only if 2c PASS | Then wait Phase 3 auth |
| **3** | Bounded CA-PPO | Requires explicit budget in user auth message | Report vs Direct/LVM |

**Rule:** Finishing phase \(k\) never starts \(k+1\). Only HANDOFF `authorized_now` + user phrase.

---

## 5. Phase-1 executable contract (so it cannot evaporate after Phase 0)

| Field | Value |
|-------|--------|
| Inputs | `results/raw/rao_phase1/pairs.json`, audits under `rao_phase1/runs/` |
| Script | `sparc_performance_evaluation/scripts/run_mdp_phase1_headroom.py` |
| Command | `python -u sparc_performance_evaluation/scripts/run_mdp_phase1_headroom.py --i-authorize-mdp-phase1` |
| Fit / hold | pair_idx 0–31 / 32–47 |
| Commitment features | Named list in runbook §Phase1 (`exec_bytes`, `helper_bytes_hhi`, …) |
| Swing PASS | `frac(|ΔG|>0.1)≥0.25` OR `mean(oracle−G_Direct)≥1.0` OR `mean(|ΔG|\|ΔG≠0)≥2.0` |
| Lift PASS | held-out bacc(`F_both`) ≥ bacc(`F_base`)+0.02 OR Spearman rule ≥+0.05 |
| Overall PASS | Swing PASS **and** Lift PASS |
| Report | `SPARC_MDP_PHASE1_HEADROOM_REPORT.md` |
| Raw | `results/raw/mdp_phase1/` |

After Phase 0 DONE, HANDOFF must set `next_user_phrase_to_start` = `AUTHORIZE Phase 1` and point `next_agent_action_when_authorized` at this contract.

---

## 6. CA-PPO relationship

Commitment-aware / post-decision critic is the **intended Phase 2–3 mechanism**.  
It is **not** authorized by this Canon. Phase 1 FAIL closes that training path without another corrective-tuning cycle on dual-policy selection.

---

## 7. File map (complete)

| File | Role |
|------|------|
| `SPARC_MDP_HANDOFF.json` | Authorized **now**; next actions |
| `SPARC_MDP_CANON.md` | **This file** — all load-bearing decisions |
| `SPARC_MDP_REFORMULATION_MATH.md` | MDP math |
| `SPARC_MDP_EXECUTION_RUNBOOK.md` | Commands, features, thresholds |
| `SPARC_MDP_CORRECTION_PROGRAM.md` | Ladder overview |
| `SPARC_MDP_PROGRAM_STATUS.md` | Human status table |
| `SPARC_MDP_PHASE0_UTILITY_LOCK.md` | Created only after Phase 0 auth (from §3) |
| `SPARC_MDP_PHASE1_HEADROOM_REPORT.md` | Created only after Phase 1 runs |
| `SPARC_LVM_RPI_PATH_CLOSED.md` | RPI residual-on-LVM FAIL close |
| `SPARC_NEXT_ROADMAP.md` | Post-RPI forks A/B/C |
| `RAO_PHASE1_*` | Prior closed branch evidence |
| `SPARC_HQ_SPO_PB_CRL.md` | **HQ-SPO method SPEC** (PB-CRL v3.1) |
| `SPARC_HQ_SPO_BASELINE_SUITE.md` | Five baselines B1–B5 |

---

## 8. Agent non-negotiables

1. Do not answer SPARC scheduling questions from chat memory alone — open Canon + HANDOFF.  
2. Do not lock only the last thing the user mentioned — update Canon when any load-bearing decision changes.  
3. Do not start work when `authorized_now` is false except editing these governance docs if the user asks to fix process.  
4. After any phase: update HANDOFF + STATUS; set `authorized_now=false`; STOP.
