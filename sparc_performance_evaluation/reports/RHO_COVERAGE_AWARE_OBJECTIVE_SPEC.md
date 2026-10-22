# RHO-SPARC — Coverage-Aware Marginal-Value Objective

**Status:** Specification only  
**Implementation / replay / PPO training:** **not authorized**  
**Supersedes for planning purposes:** the interim pack objective in §3 of `RHO_SPARC_ALGORITHM_SPECIFICATION.md` (rejected by the surrogate audit)

---

## 0. Positioning

| Item | Decision |
|---|---|
| What failed | Interim \(\sum v_i\widehat p_i - \lambda_B B - \sum\lambda_h L_h\) favored **sparse, high-\(\widehat p\)** packs (mean n=2.0 vs Direct 3.5 / LVM 4.25) |
| What stays | Feasibility \(\mathcal F_t\), governor, \(B_{\max}=175\) kB, Mode 2, beam-search *structure*, latency budget |
| What changes | **Planning objective only** — coverage-aware marginal value of the full pack |
| Not authorized | PPO training; architecture expansion; additional closed-loop replay; manuscript; final-test |
| References | Direct = validated SPARC; Stage 1 / JIPI closed |

**Design principle:** Every included assignment must earn its place by a **positive estimated marginal contribution** to deadline-valid completed value, after opportunity costs from excluded eligible work and helper-load externalities. Do **not** reward cardinality for its own sake.

---

## 1. Failure of the interim objective (retained finding)

| Mechanism | Effect |
|---|---|
| Score only sums over **included** tasks | Omitting tasks removes their \(v_i\widehat p_i\) term **and** often raises mean \(\widehat p\) on the remainder |
| Byte / load penalties too weak vs coverage loss | Sparse packs still win analytically |
| Evaluation horizon mismatch | Per-epoch \(E\) (~4–8) ranked against episode \(V\) (~35–59) without paired Δ alignment (0/4 sign agreement) |

The search was fast enough; it optimized the wrong scalar.

---

## 2. Revised planning objective

### 2.1 Eligible residual set

At epoch \(t\), let \(\mathcal C_t\) be the common feasible candidate set.  
Let \(A_t\subseteq\mathcal C_t\) be a feasible pack (\(A_t\in\mathcal F_t\)).

Define the **eligible task set**
\[
\mathcal I_t = \{i:\exists h\ (i,h)\in\mathcal C_t\}
\]
and the **uncovered eligible tasks**
\[
U(A_t)=\mathcal I_t\setminus\{i:\exists h\ (i,h)\in A_t\}.
\]

### 2.2 Pack expected completion value (included)

Retain pack-dependent completion estimates (still a prerequisite; must eventually be causally validated):
\[
V_{\mathrm{in}}(A_t\mid s_t)
=
\sum_{(i,h)\in A_t}
v_i\,\widehat p_{i,h}(A_t\mid s_t).
\]

### 2.3 Opportunity cost of exclusion (coverage term)

For each uncovered eligible task \(i\in U(A_t)\), estimate the **best alternative inclusion** under the current pack’s helper state after \(A_t\):
\[
\widehat p^{\mathrm{alt}}_{i}(A_t\mid s_t)
=
\max_{h:\ (i,h)\in\mathcal C_t}
\widehat p_{i,h}\!\left(A_t\cup\{(i,h)\}\mid s_t\right)
\quad
\text{subject to }
A_t\cup\{(i,h)\}\in\mathcal F_t,
\]
or \(0\) if no feasible one-step extension exists.

**Exclusion opportunity cost:**
\[
C_{\mathrm{excl}}(A_t\mid s_t)
=
\sum_{i\in U(A_t)}
v_i\,\widehat p^{\mathrm{alt}}_{i}(A_t\mid s_t).
\]

Interpretation: value the planner forgoes by not covering \(i\) now, evaluated at the margin of adding \(i\) to the **current** pack (not as an independent singleton that ignores congestion).

### 2.4 Helper externality (unchanged role, explicit)

\[
C_{\mathrm{load}}(A_t\mid s_t)
=
\sum_h \lambda_h(s_t)\,L_h(A_t)
+
C_{\mathrm{pending}}(A_t\mid s_t),
\]
where \(L_h\) uses existing queue semantics and \(C_{\mathrm{pending}}\) is the projected value of **already queued / pending** tasks rendered deadline-infeasible by \(A_t\) (same spirit as JIPI queue-risk accounting; no future labels).

Byte price remains an opportunity cost on radio capacity:
\[
C_{B}(A_t\mid s_t)=\lambda_B(s_t)\sum_{(i,h)\in A_t}b_i.
\]

### 2.5 Net pack value (what search maximizes)

\[
\boxed{
\begin{aligned}
J(A_t\mid s_t)
&=
V_{\mathrm{in}}(A_t\mid s_t)
-
C_{\mathrm{excl}}(A_t\mid s_t)
-
C_{B}(A_t\mid s_t)
-
C_{\mathrm{load}}(A_t\mid s_t).
\end{aligned}
}
\]

\[
A_t^\*=\arg\max_{A_t\in\mathcal F_t} J(A_t\mid s_t).
\]

**Contrast with interim objective:** the interim score omitted \(C_{\mathrm{excl}}\). That omission systematically preferred under-coverage.

### 2.6 Marginal contribution rule (anti–“just add more”)

Beam / greedy expansion may add \((i,h)\) to \(A\) only if the **marginal net value** is strictly positive:
\[
\Delta J\bigl((i,h)\mid A,s_t\bigr)
=
J(A\cup\{(i,h)\}\mid s_t)-J(A\mid s_t)
> 0
\]
(and \(A\cup\{(i,h)\}\in\mathcal F_t\)).

Equivalently, an assignment is admitted only when its contribution to \(V_{\mathrm{in}}\) exceeds the induced increase in exclusion-adjusted opportunity costs (lost alternatives for other tasks, bytes, load, pending risk).

Empty pack remains feasible; KEEP-like behavior is allowed when no positive-ΔJ extension exists.

### 2.7 What this does *not* claim

- \(J\) is **not** episode \(V_{\mathrm{done}}\) and is not a training reward.  
- \(\widehat p\) remains an estimate; this rewrite does not by itself prove causal validity.  
- \(\lambda_B,\lambda_h\) stay fixed design constants (or later PPO profile controls) — **not** fit to locked validation outcomes.

---

## 3. Consistent evaluation horizon (mandatory for any future test)

| Comparison | Allowed? |
|---|---|
| Per-epoch \(J\) or \(V_{\mathrm{in}}\) vs full-episode \(V_{\mathrm{done}}\) by magnitude | **No** |
| Rank packs by \(J\) vs rank by episode \(V\) after **identical** single-epoch force intervention | Yes (paired Δ) |
| Full-policy RHO vs full-policy LVM / Direct under matched governor | Yes (when replay later authorized) |
| Force LVM@epoch-0 + Direct rest vs locked full LVM mean | **No** (already shown to confound) |

**Primary offline alignment metric (when implemented later):**
\[
\mathrm{sign}\!\left(J(A^{\mathrm{RHO}})-J(A^{\mathrm{Direct}})\right)
\;=\;
\mathrm{sign}\!\left(V^{\mathrm{force}}(A^{\mathrm{RHO}})-V^{\mathrm{force}}(A^{\mathrm{Direct}})\right)
\]
on held-out **development** states, using a predeclared intervention protocol — **not** authorized in this document.

---

## 4. Offline validation specification (no implementation / no replay yet)

### 4.1 Purpose

Before any new closed-loop campaign, specify how the revised \(J\) will be shown to improve **alignment** with realized outcomes under a **consistent horizon**. Analytical score inflation alone is insufficient (lesson from interim \(E\)).

### 4.2 Phase A — Trace-only dry run (first gate; still not authorized to code here)

**Inputs:** Existing Stage-0 Direct / LVM audits + candidate tables; previously recorded force-pack packs from `rho_causal_pack_ranking_validation` (already executed).

**Compute (when later authorized to implement):** For each stored state and packs \(\{A^{\mathrm{Direct}}, A^{\mathrm{LVM}}, A^{\mathrm{RHO\_old}}, A^{\mathrm{hyp}}\}\):

1. \(J\) under the revised objective (and interim \(E\) for contrast).  
2. Coverage diagnostics: \(n\), bytes, \(|U|\), \(C_{\mathrm{excl}}\), marginal ΔJ of omitted Direct tasks.  
3. **Retrospective alignment:** using **already collected** realized \(V^{\mathrm{force}}\) for Direct / LVM / old-RHO packs only — does ranking by \(J\) recover realized order better than interim \(E\)?  
   - Metrics: top-1 agreement; sign(\(\Delta J\)) vs sign(\(\Delta V\)) for RHO_old−Direct and LVM−Direct.  
4. **Hypothetical pack check (analysis only):** construct \(A^{\mathrm{hyp}}\) by greedily adding positive-ΔJ extensions from empty under \(J\) on the same candidate table; report whether \(C_{\mathrm{excl}}\) falls and whether \(A^{\mathrm{hyp}}\) recovers Direct’s omitted tasks — **without** claiming realized \(V\) until a new replay is authorized.

**PASS (Phase A):** On the four existing force-pack seeds, revised \(J\) achieves sign agreement with stored \(\Delta V\) **strictly better than** interim \(E\) (which was 0/4), **or** clearly diagnoses remaining \(\widehat p\) failure if \(J\) still prefers sparse packs that lost in replay.

**FAIL (Phase A):** \(J\) still ranks old RHO packs above Direct while realized \(\Delta V\le 0\), without a documented \(\widehat p\)/pending-cost bug.

### 4.3 Phase B — New matched force-pack replay (explicitly **not** authorized now)

Only after Phase A PASS and a separate authorization:

- Same seeds/epochs predeclared.  
- Force \(A^{\mathrm{Direct}}\), \(A^{\mathrm{LVM}}\), \(A^{\mathrm{hyp}}\) (revised objective pack).  
- Rest-of-episode policy: **Direct** (identical continuation).  
- Compare \(\Delta J\) vs \(\Delta V\) under that shared horizon.  
- Separately, if ever authorized: full-episode revised-RHO vs full LVM (do not mix protocols).

### 4.4 Explicit non-goals of the offline spec

- No PPO.  
- No fitting \(\lambda\) or \(\widehat p\) temperature to locked validation labels.  
- No manuscript claims.  
- No final-test.

---

## 5. Relationship to RHO PPO (future only)

If Phase A/B eventually PASS, PPO would still select **planning profiles** \((\lambda_B,\lambda_h,\text{risk})\) over this \(J\), not resurrect the interim score. The non-RL mid-profile planner remains the required ablation.

Until then: **do not train.**

---

## 6. Authorization boundary

| Action | Status |
|---|---|
| This revised objective + offline validation **specification** | **Authorized / done** |
| Implement \(J\) / Phase A scripts | **Not authorized** (next approval) |
| Additional closed-loop replay | **Not authorized** |
| PPO training / architecture expansion | **Not authorized** |
| Manuscript / final-test | **Not authorized** |

---

## 7. Summary

The corrected scalar is net pack value
\[
J = V_{\mathrm{in}} - C_{\mathrm{excl}} - C_B - C_{\mathrm{load}},
\]
with **positive marginal \(\Delta J\)** required for each addition. Validation must use **paired intervention deltas** under one horizon—not higher analytical scores alone.

Immediate follow-on when approved: implement Phase A on existing traces. Until then, no code path changes and no replay.
