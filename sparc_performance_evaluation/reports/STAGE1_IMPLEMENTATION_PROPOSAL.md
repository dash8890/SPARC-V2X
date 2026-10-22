# Stage 1 — Implementation Proposal

**Status: CONDITIONALLY APPROVED FOR EXECUTION (scoped).**  
Authorized by Stage 0 PASS (`STAGE0_DIAGNOSTIC_REPORT.md`) and user conditional approval.

**Execution order (locked):**
1. Implement context-conditioned scoring + sequential recompute (in progress / landed under this scope).
2. Deterministic correctness, feasibility, action-likelihood, and D7 responsiveness gates (`STAGE1_GATE_REPORT.md`).
3. Full PPO training **only if** step-2 gates pass (D7 authorizes training; it does **not** establish performance success).
4. Matched five-method validation on seeds `82001/82002/82011/82012`.
5. Score results with the revised dual decision (§8): engineering target \(G\ge15\%\) / \(V\approx53.84\) **and** separate scientific assessment. Sub-15% is **not** automatic failure.

**Still not authorized:** full training before D7 gates pass; sealed final-test access; manuscript revision; governor / admission / STOP / VAS-residual changes.

---

## 1. What Stage 0 establishes

Stage 0 identifies a **scheduling** problem, not an admission problem.

| Finding | Measured (dens=35, val seeds, common governor, \(B_{\max}=175\) kB) |
|---|---|
| Feasible candidate competition | \(P(\lvert\mathcal C_t\rvert\ge 2)=90.3\%\), mean \(\lvert\mathcal C_t\rvert=24.4\) |
| Materially different schedules | \(90.3\%\) of audited decision epochs |
| SPARC vs deterministic schedulers | Disagreement \(\approx 84\%\)–\(87\%\) on identical candidate sets |
| Governor interventions | **\(0\%\)** for every method; fallback value share \(0\%\) |
| Helper competition | Non-idle helper-queue epochs \(25.8\%\) |
| Deadline outcome uncertainty | Completion \(57.1\%\), miss \(14.3\%\) (84 tasks) |

**Implication:** increasing admission freedom or modifying the governor is **not justified**.

### The Stage 1 target failure (D7)

Current Direct-PPO differs from static heuristics, but channel ablations show weak conditioning on the state that is supposed to justify RL:

\[
\text{CBR ablation}=0\%,\qquad
\text{deadline ablation}=3.23\%,\qquad
\text{helper-queue ablation}=0\%.
\]

Stage 1 must produce a **state-responsive** Direct-PPO scheduler under the **same** governor and envelope.

---

## 2. Stage 1 objective

> Can SPARC exploit deadline, helper, and radio state information to outperform the strongest matched scheduler under exactly the same governor and computation envelope?

### Primary gain metric (unchanged definition)

\[
G
=
\frac{V_{\mathrm{SPARC}}-V_{\mathrm{best\ matched}}}
{V_{\mathrm{best\ matched}}}
\times 100\%.
\]

**Engineering target (ambitious, not an automatic research FAIL threshold):** \(G\ge 15\%\).  
**Mandatory safety:** \(P_{\mathrm{viol}}\le 0.02\) (with uncertainty).  
See §8 for interpretive bands and Decision A vs Decision B.

### Strongest matched comparator (Stage 0 val replay)

| Method | \(V_{\mathrm{done}}\) |
|---|---:|
| VAS | 38.0362 |
| LAS | 36.2995 |
| VLRO | 39.9785 |
| SPARC (current Direct) | 46.0571 |
| **LVM (best matched)** | **46.8173** |

Under Stage-0 validation evidence, the engineering benchmark corresponding to \(G=15\%\) vs LVM is

\[
V_{\mathrm{eng\ target}}
=
1.15\times 46.8173
\approx
\mathbf{53.84}.
\]

Beating VAS / LAS / VLRO alone is **not** sufficient for either Decision A or Decision B. The comparator is **matched LVM** unless another matched method is stronger on the locked Stage-1 val panel.

### Clarification on VLRO / LVM (matched admission)

- VLRO’s earlier apparent advantage was **admission-confounded**. Under the common governor, current SPARC already exceeds VLRO by \(\approx 15.2\%\) on this val panel.
- Matched LVM remains slightly stronger (\(+1.65\%\) vs current SPARC). Closing and then clearing the \(15\%\) gap vs LVM is the Stage 1 objective.

---

## 3. Frozen components

Stage 1 must **not** change:

| Component | Freeze |
|---|---|
| Mode-2 WiLab simulator | unchanged |
| Task generation / helper model / finish-time estimator | unchanged |
| Feasible candidate definition | unchanged |
| Source / task uniqueness | unchanged |
| Envelope | \(B_{\max}=175\) kB |
| Safety | common SafetyGovernor (PRR/P95 + envelope) |
| Reward | new deadline-valid completed value (existing definition) |
| Learning objective | PPO (same clipped objective family as current Direct) |
| Seeds | training pool; validation `82001,82002,82011,82012`; **sealed final-test closed** |
| Matched baselines | VAS, LAS, VLRO, LVM implementations and matched eval protocol |

**Explicit non-goals**

- Do **not** weaken / retune the governor to make SPARC look better (D3 gives no justification).
- Do **not** reintroduce residual VAS anchoring (`va_residual` / analytic additive branch).
- Do **not** add STOP (D5 inconclusive; requires a separate controlled replay first).
- Do **not** redesign the reward.
- Do **not** open sealed final-test seeds if Stage 1 fails.

---

## 4. Proposed architectural change

Retain **Direct PPO**. Change the **state representation / contextual scoring**, not the algorithm class.

### 4.1 Inputs

For each candidate \((i,h)\) retain local features \(\mathbf{x}_{i,h,t}\) (existing 10-d candidate encoding).

Construct an explicit decision-context vector

\[
\mathbf{z}_t
=
[
\mathrm{CBR}_t,\,
\lvert\mathcal C_t\rvert,\,
B_t^{\mathrm{rem}},\,
\tau_t,\,
\mathbf{s}^{\mathrm{deadline}}_t,\,
\mathbf{s}^{\mathrm{helper}}_t
],
\]

where:

- \(B_t^{\mathrm{rem}}\): remaining Mode-2 byte budget inside the epoch’s sequential packing;
- \(\mathbf{s}^{\mathrm{deadline}}_t\): compact stats of candidate slack (e.g. mean / min / p10 of \(D_i-\widehat F_{i,h,t}\) over \(\mathcal C_t\));
- \(\mathbf{s}^{\mathrm{helper}}_t\): compact stats of helper loading / readiness (e.g. mean / max projected queue or \(\max(0,\tau_h-t\Delta)\));
- \(\lvert\mathcal C_t\rvert\): competition level;
- \(\mathrm{CBR}_t\): radio occupancy;
- \(\tau_t\): epoch / horizon progress (already partially present as a candidate feature).

Per-candidate relative quantities \(\mathbf{r}_{i,h,t}\):

\[
\mathrm{slack}_{i,h,t}=D_i-\widehat F_{i,h,t},
\qquad
\mathrm{load}_{h,t}=\max(0,\tau_h-t\Delta),
\qquad
\mathrm{budgetFrac}_{i,t}=\frac{b_i}{B_t^{\mathrm{rem}}}.
\]

Optional but recommended relatives (still logging-compatible, no simulator change):

- rank / z-score of slack and load within the current feasible set;
- \(\lvert\mathcal C_t\rvert^{-1}\) or log-competition;
- indicator that helper \(h\) is already selected earlier in the same epoch’s sequential pack (always false at first pick; updates after picks if multi-admit).

### 4.2 Scorer (Direct)

\[
q_\psi(i,h,t)
=
f_\psi\!\left(\mathbf{x}_{i,h,t},\,\mathbf{z}_t,\,\mathbf{r}_{i,h,t}\right),
\qquad
\ell_\psi(i,h,t)=\alpha\, q_\psi(i,h,t).
\]

No additive VA term. Initialization: ordinary Direct init (not VA zero-residual).

**Suggested network (proposal default; may be tuned only on train/val):**

- Concatenate \([\mathbf{x}\,\|\,\mathbf{z}\,\|\,\mathbf{r}]\) per candidate.
- Shared MLP → scalar score (capacity comparable to current Direct head; disclose param count).
- Critic may consume a pooled context embedding of \(\mathbf{z}_t\) (same freeze on reward/PPO).

Exact tensor layout and normalization constants are implementation details to be locked in a short `STAGE1_FEATURE_SPEC.md` **at execution approval**, using Stage-0 `candidate_epochs.json` as the schema source.

### 4.3 Recompute context during sequential selection (mandatory)

Current `sample_action` builds logits **once**, then sequentially masks used tasks/sources/bytes. Stage 1 must instead, before each sequential pick \(m=1,2,\ldots\):

\[
\mathcal C_{t,m}
\;\rightarrow\;
\mathbf{z}_{t,m},\,\mathbf{r}_{i,h,t,m}
\;\rightarrow\;
q_\psi(i,h,t,m)
\;\rightarrow\;
\text{Categorical over remaining feasible choices}.
\]

After each selection:

- remove task / source;
- decrease \(B^{\mathrm{rem}}\);
- update helper-load projections used in \(\mathbf{r}\) / \(\mathbf{s}^{\mathrm{helper}}\) for remaining candidates;
- refresh competition stats \(\lvert\mathcal C_{t,m}\rvert\) and slack aggregates.

This is the primary mechanism expected to exploit D1 / D2b / D6. Log-probabilities for PPO must include the full sequential conditional chain (already required for correct on-policy likelihood).

**Non-change:** governor still wraps the **final** proposed assignment set; no change to admission gates.

---

## 5. What must not be added in Stage 1

| Item | Reason |
|---|---|
| Explicit STOP action | D5 INCONCLUSIVE; needs controlled paired replay first |
| New / loosened safety mechanism | D3: governor already non-binding at audited OP; wrong problem |
| Residual / VAS analytical branch | D7 not solved by re-anchoring; Direct remains the algorithm |
| Reward redesign | Objective already matches completed deadline-valid value |
| Final-test evaluation | Sealed until architecture + Stage 1 PASS |

---

## 6. Cheap pre-training qualification (gate before full PPO)

**Purpose:** verify the architecture *can* represent state-dependent scheduling before a full training budget.

**Data:** existing Stage-0 `candidate_epochs.json` / decision traces (validation + optional train-episode dumps). No sealed seeds.

**Procedure (proposal):**

1. Build feature tensors with the new \(\mathbf{z},\mathbf{r}\) layout and sequential-recompute hooks (offline shadow or short smoke episodes).
2. Initialize the new scorer; optionally run a **small** supervised / behavioral-cloning or few-update PPO smoke **only** as an engineering check (not a claim run).
3. Repeat D7-style ablations on held-out Stage-0 candidate states:
   - zero / shuffle CBR in \(\mathbf{z}\);
   - ablate deadline slack channels in \(\mathbf{r}/\mathbf{s}^{\mathrm{deadline}}\);
   - ablate helper-load channels in \(\mathbf{r}/\mathbf{s}^{\mathrm{helper}}\).

### Qualification minimum

\[
\text{decision-change fraction}\ge 10\%
\]

for **at least deadline and helper-state** channels, plus demonstrable sensitivity to radio context on epochs where CBR actually varies.

This is a **capability** check, not a requirement to thrash decisions arbitrarily. If the scorer remains effectively invariant, **do not** launch full PPO training.

**Qualification outputs:** `STAGE1_RESPONSIVENESS.json` with per-channel change fractions and example flipped assignments.

---

## 7. Validation experiment (only if §6 passes)

### Train

- Policy: context-conditioned Direct PPO (§4), sequential context recompute (§4.3).
- Data: **existing training pool only** (sim `72000+`, task `182000+` partition).
- Budget: disclose episode count / early-stop rule up front; prefer matching historical Direct budget unless a smaller smoke is predeclared.
- Checkpoint selection: **validation only** (four seeds below).

### Evaluate (matched)

Seeds: `82001, 82002, 82011, 82012`  
Density: \(35\)  
Governor: ON  
Envelope: \(175\) kB  
Comparators: VAS, LAS, VLRO, LVM (same matched protocol as Stage 0)

### Report at minimum

\[
V_{\mathrm{done}},\quad R_{\mathrm{value}},\quad P_{\mathrm{viol}},\quad \text{offered bytes}
\]

plus:

- paired \(\Delta V, \Delta R\) vs **LVM** (primary);
- D7-style ablation summary on the trained checkpoint (state-use evidence);
- governor intervene rate / fallback value share (confirm admission parity);
- mean offered bytes vs LVM (flag unequal admission).

Artifacts (proposed paths, written only after execution approval):

- `results/raw/stage1_val_*`
- `reports/STAGE1_VALIDATION_REPORT.md`

---

## 8. Evaluation framework (revised)

Distinguish an **engineering target** (guides development) from a **scientific assessment** (judges contribution). The \(15\%\) / \(V\approx53.84\) benchmark is retained as the ambitious target; it is **not** an automatic research FAIL if missed.

### 8.1 Gain classification (interpretive guidelines, not new hard gates)

| Result | Classification | Interpretation |
|---|---|---|
| \(G\ge 15\%\) | Target achieved | Strong performance result, subject to safety and statistical verification |
| \(5\%\le G<15\%\) | Promising | Potentially meaningful; assess consistency, mechanism, robustness, complexity |
| \(0\%<G<5\%\) | Marginal | Positive but needs particularly strong supporting evidence |
| \(G\le 0\%\) | No demonstrated value advantage | Do not claim scheduling superiority; ask whether another substantive benefit exists |

If a matched comparator exceeds LVM on the Stage-1 val replay, that method becomes \(V_{\mathrm{best\ matched}}\) and \(G\) / the \(15\%\) engineering benchmark are recomputed (do not silently keep 53.84).

### 8.2 Mandatory requirements (not relaxed)

| Requirement | Rule |
|---|---|
| Safety | \(P_{\mathrm{viol}}\le 2\%\), with uncertainty assessment |
| Fairness | All five methods under identical matched-admission conditions |
| Statistical credibility | Paired validation results and uncertainty — not favorable averages alone |
| Mechanism | Evidence that context-conditioned scheduling meaningfully improves decisions |
| Reproducibility | Locked seeds; unchanged baselines; **no final-test access** during development |
| D7 pre-train gate | Responsiveness prerequisite before expensive training; **authorizes training only**, not performance success |

### 8.3 Dual decision logic

**Decision A — Engineering target met?**  
Strictly numerical: \(G\ge 15\%\) over the strongest matched competitor (benchmark \(\approx53.84\) vs current LVM), with mandatory safety held.

**Decision B — Scientifically meaningful improvement worth developing further?**  
Considers effect size band (§8.1), paired uncertainty, safety, decision quality / mechanism, robustness, and computational overhead.

Examples:

- \(G\in[8,10]\%\) consistent over LVM with strong mechanism evidence and acceptable overhead → may be **Decision B = yes** even though Decision A = no.
- \(G=16\%\) driven by one unusual seed without paired support → Decision A may look positive but Decision B should **not** automatically succeed.

### 8.4 Hard stops (still automatic negative)

Regardless of \(G\), Stage 1 does **not** advance if any of:

- \(P_{\mathrm{viol}}>2\%\);
- apparent gain from unequal admission;
- post-train state-feature responsiveness still negligible (D7 unresolved in the trained policy);
- improvement claimed only vs VAS/LAS while LVM remains the strongest matched comparator and is not addressed.

### 8.5 After the validation report

- **Decision A yes + Decision B yes:** may request a separate Stage 2 (sealed final-test / manuscript wording). Neither decision unseals final-test by itself.
- **Decision A no + Decision B yes (e.g. promising \(5\)–\(15\%\)):** evidence-based retain / revise choice — **not** automatic reject; **not** automatic final-test.
- **Decision B no** (including \(G\le0\) without other substantive benefit, or failed mandatories): do not open sealed final-test; do not revise manuscript architecture claim; short postmortem — no silent governor retuning.

Full write-up of this policy: `STAGE1_EVALUATION_FRAMEWORK.md`.

---

## 9. Engineering inventory (for execution approval)

| Work item | Scope |
|---|---|
| Feature builder | \(\mathbf{z}_t\), \(\mathbf{r}_{i,h,t}\) from existing candidate + governor budget state |
| Policy module | New Direct scorer \(f_\psi\); `logit_mode=direct` retained |
| `sample_action` | Sequential context recompute + correct sequential log-prob |
| Logging | Extend candidate dump with context/relatives used for D7 audits |
| Responsiveness harness | Offline ablation on Stage-0 states (§6) |
| Train entrypoint | Mirror existing Direct PPO trainer; train-pool seeds only |
| Val campaign | Five-method matched replay on validation seeds (§7) |
| Report | Decision A (engineering \(G\ge15\%\)) + Decision B (scientific assessment) per §8 |

**Out of scope until later approval:** STOP, governor changes, residual VA, reward changes, final-test, manuscript rewrite.

---

## 10. Authorization boundary

| Allowed now | Not allowed now |
|---|---|
| This proposal document | Implementing/training as the claimed SPARC path |
| Planning feature dims / file layout | Sealed final-test runs |
| | Manuscript / baseline nomenclature changes |
| | Governor or admission retuning |

**Stage 0 PASS → Stage 1 proposal (this document).**  
**Stage 1 execution** is conditionally approved within this scope (implementation + D7 gates; training only after D7).  
**Sealed final-test** requires a separate request after a Stage-1 validation report with Decision A/B stated explicitly.

---

## 11. One-sentence scientific claim Stage 1 is allowed to pursue

Turn the Stage-0 scheduling opportunity into a **state-conditioned Direct-PPO** scheduler that, under matched admission, improves completed value over the strongest matched competitor (currently LVM) with \(P_{\mathrm{viol}}\le 2\%\), aiming for the ambitious engineering target \(G\ge15\%\) (\(V\approx53.84\)) while allowing an evidence-based scientific assessment if the realized gain is smaller but well-supported.
