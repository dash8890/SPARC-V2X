# POPL-SPARC — Paired-Rollout Feasibility Specification

**Full name:** Paired-Outcome Policy Learning for SPARC  
**Status:** Design hypothesis / feasibility specification only  
**Training / implementation / replay / manuscript / final-test:** **not authorized**

---

## 0. Positioning

| Item | Decision |
|---|---|
| Stage 1 | **Closed** — not reopened |
| JIPI | **Closed** — not reopened |
| RHO surrogate stack | **Closed** — analytical \(J/\widehat p\) path not continued |
| Learning signal | **Realized** episode (or return-to-go) completed value from paired closed-loop rollouts |
| Not used as reward | Slack, pack \(E[V]\), coverage penalties, LVM agreement |
| Reference action | Frozen Direct PPO pack at paired decision epochs |
| Safety / radio / envelope | Unchanged governor, Mode 2, \(B_{\max}=175\) kB |
| Benchmark | Matched LVM validation mean **46.82**; 15% target ≈ **53.84** |

**Hypothesis:** A policy that constructs feasible packs and is trained on **paired realized \(\Delta G\)** against Direct can improve deadline-valid completed value without a completion-probability surrogate.

This document does **not** claim POPL will beat LVM. It specifies the paired-rollout mechanics, PPO rule, and **cheap gates** that must pass before any expensive training run.

---

## 1. Mechanism (informal)

```text
Same pre-decision state s_t
        |
        +-- Choice A: frozen Direct pack A_t^ref
        |
        +-- Choice B: alternative feasible pack A_t^alt  (learner / explorer)
        |
        v
Matched closed-loop branches (identical exogenous RNG through t; diverge after)
        |
        v
ΔG_t = G_t(B) - G_t(A)
        |
        v
PPO-compatible update favoring the higher realized return-to-go
```

---

## 2. Mathematical formulation

### 2.1 Realized return-to-go

\[
G_t(A_t)=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,
\quad
r_k=\sum_{i\in\mathcal I_k^{\mathrm{new}}}v_i.
\]

Primary **evaluation** uses \(\gamma=1\) (undiscounted episode completed value from the intervention onward, or full episode \(G_0\) when comparing whole trajectories). Any \(\gamma<1\) used only inside the learning estimator must be predeclared and justified; it is not the project metric.

### 2.2 Paired advantage

For the same pre-decision state \(s_t\):

\[
\Delta G_t
=
G_t(A_t^{\mathrm{alt}})
-
G_t(A_t^{\mathrm{ref}}).
\]

Reference \(A_t^{\mathrm{ref}}\) = frozen Direct. Alternative \(A_t^{\mathrm{alt}}\) = feasible pack from the learning policy (or from a declared exploration proposal distribution during data collection).

### 2.3 Learning objective

\[
\max_\theta\;\mathbb E_{\pi_\theta}[G_0].
\]

Paired outcomes supply a **relative** training signal; they do **not** replace \(r_k\) with an analytical score.

---

## 3. Actor architecture (design)

Sequential pack construction (not Stage-1 rescoring, not JIPI edit-over-Direct, not RHO beam+\(\widehat p\)):

1. Observe causal state (tasks, deadlines, helpers, radio).  
2. Select a feasible task–helper assignment, or **TERMINATE**.  
3. Update partial pack / remaining bytes / uniqueness constraints.  
4. Repeat until TERMINATE or no feasible addition.  
5. Submit pack to the **unchanged** safety governor.

**TERMINATE** is a pack-completion action, not a separate admission STOP. Under-assignment risk (RHO lesson) requires:

- Explicit termination penalty / advantage shaping only via **realized** \(G\) (no analytical coverage bonus).  
- Exploration that proposes substitutions and alternate combinations, not near-clones of Direct.  
- Optional training-time minimum exploration rate on non-Direct packs (predeclared; not fit to val).

Inference never receives future outcomes.

---

## 4. Paired closed-loop protocol (correctness requirements)

### 4.1 Counterfactual correctness

For a paired sample at epoch \(t^\*\):

| Requirement | Spec |
|---|---|
| Shared prefix | Identical sim seed, task seed, density, governor, envelope; identical bridge/radio trajectory for epochs \(< t^\*\) |
| Common state | Byte-identical `state_*` (and task-layer release set) at \(t^\*\) before the branch |
| Matched exogenous randomness | Same MATLAB/RNG stream through \(t^\*\); branch only by the forced pack at \(t^\*\) (and thereafter each branch’s own controller actions) |
| Feasibility | \(A^{\mathrm{ref}},A^{\mathrm{alt}}\in\mathcal F_{t^\*}\) under existing candidate/source/byte rules |
| Logging | Store packs, \(\Delta G\), safety windows, whether force applied |

**Forbidden:** fitting proposals to locked validation labels; using final-test seeds; unequal governor/envelope.

### 4.2 Continuation policy after the branch

**Default (recommended for credit assignment):**  
- Epoch \(t^\*\): force \(A^{\mathrm{ref}}\) or \(A^{\mathrm{alt}}\).  
- Epochs \(> t^\*\): **same continuation policy on both branches** (frozen Direct), so \(\Delta G\) attributes the fork to the paired decision plus its downstream effects under a common follower.

**Alternative (full on-policy):** continuation by \(\pi_\theta\) on the alt branch and Direct on the ref branch — higher variance; only if separately justified.

Primary POPL training design uses the **common Direct continuation** unless a later authorization changes it.

### 4.3 Regression tests (known cases — not generalization evidence)

| Case | Required discrimination |
|---|---|
| 82002 / epoch 33 / task 19 helper 2→0 | \(\Delta G>0\) for documented beneficial rebind vs Direct (matched protocol) |
| 82011 / helper-9 harmful rebind | \(\Delta G<0\) (or non-positive) for the documented harmful alternative |

These gates check **paired machinery**, not that POPL generalizes.

---

## 5. PPO learning rule using paired outcomes

### 5.1 Data unit

One **paired sample** \(i\):

\[
\bigl(s_{t},\,A^{\mathrm{ref}},\,A^{\mathrm{alt}},\,
\log\pi_\theta(A^{\mathrm{alt}}\mid s_t),\,
G^{\mathrm{ref}},\,G^{\mathrm{alt}},\,\Delta G\bigr).
\]

If the actor is sequential, \(\log\pi(A^{\mathrm{alt}}\mid s)\) is the sum of step log-probs including TERMINATE.

### 5.2 Preference / advantage construction (**superseded sketch**)

> **Correction (see `POPL_LEARNING_RULE_CORRECTNESS_REVIEW.md`):**  
> Simply setting \(\hat A=\mathrm{normalize}(\Delta G)\) inside PPO is **not** valid unless \(A^{\mathrm{alt}}\sim\pi_{\theta_{\mathrm{old}}}\) and \(\log\pi(A)\) is the full sequential likelihood (incl. TERMINATE). Direct’s return is then an action-independent baseline, so \(\Delta G\) is a legitimate score-function multiplier. Off-policy / mixed proposals require IS or a preference loss—not unweighted PPO-clip.

**Locked on-policy rule (only form cleared for a future P5):**

1. Sample \(A^{\mathrm{alt}}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)\); \(A^{\mathrm{ref}}=\) frozen Direct.  
2. \(\hat A^{\mathrm{alt}}=\mathrm{normalize}(\Delta G)\) (optional train-only batch baseline on \(\Delta G\)).  
3. PPO-clip with
   \(\rho=\pi_\theta(A^{\mathrm{alt}}\mid s)/\pi_{\theta_{\mathrm{old}}}(A^{\mathrm{alt}}\mid s)\)
   using
   \(\log\pi(A)=\sum_j\log\pi(a_j\mid s,a_{<j})+\log\pi(\mathrm{TERMINATE}\mid\ldots)\).

**Reference pack** is not a \(\pi_\theta\) action. Optional KL toward Direct (predeclared) may limit Stage-1-style drift.

### 5.3 Surrogate loss (on-policy form only)

\[
L^{\mathrm{CLIP}}
=
-\,\mathbb E_i\Bigl[
\min\bigl(
\rho_i\hat A^{\mathrm{alt}}_i,\;
\mathrm{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A^{\mathrm{alt}}_i
\bigr)
\Bigr]
+\,c_v\,L^{\mathrm{value}}
-\,c_e\,\mathcal H[\pi_\theta],
\]

with \(\rho_i\) and \(\hat A_i\) as in §5.2 (locked). Critic targets from realized returns only.

### 5.4 What is not allowed in the learning rule

- Rewarding \(\widehat p\), slack, \(J\), or coverage terms.  
- Labeling pairs using locked validation / final-test outcomes.  
- Unequal information to POPL vs matched baselines at eval.

---

## 6. Training-sample cost model (bounded budget)

### 6.1 Empirical wall-time anchor

From prior WiLab closed-loop work in this repo:

| Quantity | Estimate |
|---|---|
| Wall time per full episode | ≈ **25–55 s** (typical ~30–50 s under load) |
| Direct 96-ep train | ~24 PPO updates × 4 rollouts (single branch) |

Use **\(T_{\mathrm{ep}}=40\,\mathrm{s}\)** as a conservative planning constant unless a later timed pilot revises it.

### 6.2 Cost per paired training sample

A paired sample at a single fork epoch requires **two** episode completions (shared prefix optimizable later; worst case = 2 full episodes):

\[
C_{\mathrm{pair}}
\approx
2\,T_{\mathrm{ep}}
\approx
80\,\mathrm{s}
\quad\text{(conservative; no prefix sharing)}.
\]

With prefix sharing (one sim to \(t^\*\), then two suffixes), expected savings depend on \(t^\*/T\); **do not assume** savings until measured in the feasibility pilot.

### 6.3 Example bounded training envelopes (planning only)

| Envelope | Paired samples | Wall time @ 80 s/pair | Notes |
|---|---:|---:|---|
| Smoke / machinery | 8 | ≈ 0.2 h | Regression forks only |
| Learning pilot (train seeds) | 64 | ≈ 1.4 h | Gate: distinguish \(+/-\,\Delta G\) |
| Full POPL (if ever authorized) | 384 (= 96×4) | ≈ 8.5 h | Same order as ~2× Direct 96-ep |
| Aggressive | 960 | ≈ 21 h | Requires separate budget approval |

**Recommendation:** commit only after gates in §7; start with **≤64** train-only paired samples for the learning-feasibility pilot, not a 96-ep double budget by default.

### 6.4 Fork schedule (to bound cost)

Predeclare one of:

- **Sparse forks:** at most \(K\) decision epochs per episode (e.g. \(K=1\) random feasible epoch, or first epoch with \(|\mathcal C_t|\ge n_{\min}\)).  
- **Dense forks:** every epoch — generally **too expensive**; not default.

Default design: **\(K=1\)** fork per paired episode.

---

## 7. Prerequisite gates (cheap evidence before training authorization)

| ID | Gate | PASS requirement |
|---|---|---|
| P1 | Counterfactual correctness | Shared prefix + matched RNG; force applied; documented tolerances |
| P2 | Feasibility | All alt packs in \(\mathcal F_t\) |
| P3 | Outcome discrimination | 82002 beneficial swap \(\Delta G>0\); 82011 harmful helper-9 \(\Delta G\le 0\) under paired protocol |
| P4 | Computational feasibility | Measured \(C_{\mathrm{pair}}\) and projected pilot/full budget written down; within pre-agreed wall-time cap |
| P5 | Learning feasibility | Train-seed-only pilot (e.g. ≤64 pairs): policy’s preferred alt correlates with sign(\(\Delta G\)) above chance; no val fitting |
| P6 | Fairness | No privileged obs; baselines unchanged; final-test sealed |

**All six required** before authorizing a large POPL PPO run. Known cases (P3) are **machinery tests**, not generalization claims.

---

## 8. Fair evaluation (when later authorized)

Matched common-governor panel: Direct, VAS, LAS, VLRO, LVM, POPL.  
Primary: \(V_{\mathrm{done}}\), \(R_{\mathrm{value}}\), \(P_{\mathrm{viol}}\).  
Ablation: POPL without paired signal (standard on-policy PPO only) if compute allows — separate approval.

---

## 9. Authorization boundary

| Action | Status |
|---|---|
| This feasibility + PPO-rule specification | **Authorized / done** |
| Implement paired harness / run P1–P5 | **Not authorized** (needs separate approval) |
| POPL training (pilot or full) | **Not authorized** |
| Validation / final-test / manuscript | **Not authorized** |
| Reopen Stage 1 / JIPI / RHO surrogate | **Forbidden** |

---

## 10. Immediate engineering deliverable (this document)

Delivered here:

1. **Bounded paired-rollout protocol** (shared state, matched RNG, common Direct continuation, \(K=1\) forks).  
2. **Exact cost model** (\(C_{\mathrm{pair}}\approx 2T_{\mathrm{ep}}\); planning tables).  
3. **PPO-compatible paired learning rule** (advantage from realized \(\Delta G\); clip/entropy/value; no analytical reward).  
4. **Gate checklist P1–P6** before any expensive RL run.

**Next approval (human):** authorize implementation of the paired harness and gates P1–P4 only — still no POPL weight training until P5 is separately approved.

---

## Related

- `RHO_SURROGATE_BRANCH_CLOSURE.md` — surrogate path closed  
- `STAGE2_JIPI_CLOSURE.md` / `STAGE1_CLOSURE.md` — prior RL attempts closed  
- Locked LVM mean 46.82; Direct remains SPARC reference until a future matched validation promotes otherwise  
