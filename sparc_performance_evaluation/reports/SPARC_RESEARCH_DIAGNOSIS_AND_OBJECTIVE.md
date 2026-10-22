# SPARC-V2X — Research Diagnosis, Objective Lock, and Next Gate

**Date:** 2026-10-19  
**Status:** Load-bearing. Chat is not the plan; this file + Canon index the decision.  
**Related:** success/fail map from `POPL_P5_ROBUSTNESS_REPORT.md`, `POPL_P5_EXECUTION_AUDIT.md`, `rao_phase1` pairs; closed paths in Canon §0; pack-defect FAIL in `POPL_PACK_DEFECT_FIX_REPORT.md`.

---

## 0. One-line diagnosis

We designed RL algorithms before establishing that the formulated problem contained **sufficiently consequential, controllable decisions** for learning to beat a strong deterministic scheduler (LVM). Most later failures are consequences of that order error—not of missing PPO tricks.

---

## 1. Success vs failure map (already established)

### Success regimes / seeds (POPL vs matched baselines)

| Condition | Pattern | Mechanism (audits) |
|---|---|---|
| `workload_tps4` | POPL beats Direct (+3.75) and LVM (+1.33); all 4 seeds | Pack admission + co-scheduling companions |
| `density_35_nominal` | Mild mean gain; **82002 / 82012** win; **82001** loses | Thin-tailed: few task events |
| `deadline_1.5` | Mixed / mild | Slack allows pack interactions |

### Failure regimes / seeds

| Condition | Pattern | Mechanism (audits) |
|---|---|---|
| `deadline_0.5` | Trails LVM hard (−3.01) | Tight slack; LVM value-greedy wins |
| `workload_tps2` | Trails Direct/LVM | Light load; early mistakes dominate |
| `helper_homog` / `helper_heterog` | Trails best baseline | Helper signal absent or mis-used |
| Seed **82001** | Systematic loser on several axes | Epoch-0 omit of valuable task vs Direct |

### Structural contrast

| | Success | Failure |
|---|---|---|
| Workload / slack | High packing pressure | Tight deadline / light load |
| Pack vs Direct | Different admission or companions | Omit valuable task, or same tasks / worse helpers |
| Outcome density | Some \(\Delta V\neq0\) | Many pack disagreements with \(\Delta G=0\) (~69% in Direct/POPL pairs) |
| Oracle headroom (Direct∪POPL) | — | ≈ +1.9% vs Direct — not panel dominance |

### Failed “fixes” that rearrange frozen generators

| Attempt | Result |
|---|---|
| RAO dual-policy selector | G4 FAIL — CLOSED |
| MDP W-advantage (Phase 2d) | FAIL — CA-PPO CLOSED |
| Pressure-gate Direct↔POPL(/LVM) | FAIL / aborted — CLOSED |
| POPL decode A+B (no early TERMINATE + min-finish rebind) | **FAIL** — worse on all 4 nominal seeds (−8.5 vs best) |

**Implication:** Audit events (omit, helper flips) are real symptoms. They are **not** proven to be fixable by decode heuristics or routers on this frozen checkpoint. Forcing fuller packs + greedier helpers destroyed whatever the frozen policy did right under pressure.

---

## 2. Alignment with the research diagnosis

| Diagnosis claim | Map / history support |
|---|---|
| Many disagreements do not change realized value | ~69% disagreeing Direct/POPL packs \(\Delta G=0\) |
| Wins are thin and interaction-driven | `tps4`, companion-pack / admission events |
| Local scoring ≠ system value | RHO surrogates; decode “fix” hurt realized \(V\) |
| Switching among frozen packs is not the research | RAO / gates / pack-defect all failed |
| Safety-aware novelty weak if governor rarely intervenes | Nominal ~0% intervention (reported) — keep governor; don’t sell unused constraint as contribution |

Central failure of the program: **algorithm-first**. We pursued “can RL beat baselines?” without first settling which consequential scheduling weakness RL must overcome versus LVM.

---

## 3. Corrected objectives (locked)

### A. Scientific objective

Develop a safety-constrained, computation-aware scheduling framework for NR-V2X Mode 2 that maximizes the **realized** value of deadline-constrained vehicular computing tasks, accounting for competition over radio resources and helper execution capacity.

Novelty must come from the **joint communication–computation scheduling problem** and how the method handles it—not from attaching PPO to a simulator.

### B. Algorithmic objective

Learn joint task-selection and helper-assignment decisions that beat a **strong deterministic scheduler (LVM)** by accounting for **future contention and opportunity cost of current commitments**—not by reproducing LVM, patching Direct, or optimizing a proxy pack score.

### C. Experimental success criterion (frozen)

\[
\begin{aligned}
&J_{\mathrm{proposed}} > J_{\mathrm{LVM}} &&\text{on fair, matched comparison},\\
&P_{\mathrm{viol}} \le 0.02,\\
&T_{\mathrm{decision}} \le 10\,\mathrm{ms}.
\end{aligned}
\]

- Primary scalar remains undiscounted deadline-valid completed value \(V_{\mathrm{done}}\) / \(J(\pi)\).
- ~15% remains an **aspirational** target, not a design gate.
- Publishable bar: paired superiority + useful robustness—not only a favorable nominal mean.
- **Hard stop:** if a fair implementation still cannot establish meaningful superiority over LVM, conclude RL benefit unsupported under this workload. Do **not** weaken LVM, change the metric, or retarget an easier baseline.

### System objective (math)

\[
J(\pi)=\mathbb{E}_\pi\left[
\sum_{i\in\mathcal{I}} v_i\,
\mathbf{1}\bigl\{T_i^{\mathrm{rx}}<\infty,\; T_i^{\mathrm{done}}\le d_i\bigr\}
\right].
\]

Assignment → radio reception → helper queue → execution → completion/expiry. An individually high-value task can have **negative incremental system benefit**. Intermediate utilities (slack, \(\widehat p\), pack scores) may inform decisions but must not replace \(J\).

---

## 4. What the solution direction must look like (design, not yet an algorithm)

Four properties (design requirements; not a validated controller):

1. **Control the decision that matters** — feasible pack of task–helper assignments at the frozen simulator interface; no invented PHY authority.
2. **Account for cost on other tasks** — displacement and future opportunity cost, not isolated pair ranking.
3. **Preserve true objective through learning** — optimize expected realized \(V_{\mathrm{done}}\) over the finite horizon (outstanding work after the assignment epoch included).
4. **Clear RL vs deterministic split** — deterministic code enforces known feasibility / governor / envelope; RL (if used) owns uncertain long-horizon commitment choice that LVM does not already capture.

---

## 5. What to do now (authorized sequence)

**Do not** start another PPO variant, surrogate, multi-agent method, counterfactual learner, router, decode tweak, or indefinite diagnostic sweep.

### Deliverable 1 — Formulation lock (document)

One internally consistent problem-and-method formulation that settles together:

| Decision | Required resolution |
|---|---|
| Research question | Joint safety-constrained radio/computation scheduling for realized deadline-valid value |
| Controlled variables | Only assignment / computation-demand decisions at the frozen interface |
| Dynamics | Assignment → reception → helper queue → execution → completion/expiry |
| Constraints | Existing envelope, feasibility, measured safety, runtime ≤10 ms |
| Objective | Undiscounted realized weighted deadline-valid completion |
| MDP/POMDP | Observable state, complete pack action, physical horizon, terminal accounting |
| Learning contribution | System-wide opportunity-cost decisions, not isolated candidate scoring |
| Deterministic contribution | Feasibility and operations that do not require learning |
| Baselines | LVM + frozen competitors; identical radio + governor |
| Success | Paired improvement on \(J\) without safety/runtime violations |

### Deliverable 2 — Consequentiality gate vs LVM (before any new training)

On a **fresh train/dev split** (locked val `82001/02/11/12` forbidden for fitting; final-test sealed):

- Measure whether controllable pack decisions exist with material \(\Delta V_{\mathrm{done}}\) vs **LVM** that are frequent enough and predictable from observables.
- Use the success/fail map as **hypotheses** (more swing under high packing pressure; less or opposite under tight/light/helper)—not as tuning targets on val seeds.

| Gate result | Action |
|---|---|
| Swing vs LVM tiny or rare | **Stop the RL superiority claim** for this workload (negative/limits result or reformulate the *problem*) |
| Swing real and identifiable | Only then authorize a controller that follows the formulation |

No implementation of a new learned controller is authorized until Deliverable 1 is accepted and Deliverable 2’s protocol is written and approved.

---

## 6. Explicit forbids

- Replace SPARC with COMA / DAG / transformer because a paper used them when our apps lack matching structure.
- Change radio simulator or task distributions to manufacture RL gains.
- Rename predicted pack quality as completed value.
- Redefine success as beating Direct or another easier baseline.
- Optimize within Direct∪POPL as if that set defined the problem (oracle headroom already small).
- Another frozen-generator router, W-retune, or decode heuristic loop on POPL-P5.
- Conceal a failed LVM comparison by changing metrics or baselines.

---

## 7. Unresolved scientific boundary

The literature cannot answer: does **this** SPARC workload provide enough controllable radio–computation tradeoffs for a learned policy to outperform LVM meaningfully?

Evidence shows **some** opportunities (companion packs; admitting excluded tasks). They have not yielded a consistently superior policy. That is a limit of the evidence—not a reason for another indefinite audit cycle.

---

## 8. Status

| Item | State |
|---|---|
| Diagnosis + objective lock | **Documented (this file)** |
| Formulation lock (Deliverable 1) | Not started — requires explicit user authorize to draft |
| Consequentiality-vs-LVM gate protocol (Deliverable 2) | Not started — after formulation |
| New training / new controller | **Blocked** until 1+2 pass |
| Agent default | Stop; do not invent the next algorithm |

**Canon index:** see `SPARC_MDP_CANON.md` §0.  
**Handoff:** `SPARC_MDP_HANDOFF.json` — `authorized_now=false` until user authorizes Deliverable 1.
