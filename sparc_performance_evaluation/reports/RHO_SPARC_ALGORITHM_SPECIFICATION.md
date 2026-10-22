# RHO-SPARC — Algorithm Specification

**Full name:** Risk-aware Horizon Optimization with Reinforcement Learning for SPARC  
**Status:** Design recommendation / specification only  
**Training / matched validation / final-test / manuscript:** **not authorized**

---

## 0. Positioning

| Item | Decision |
|---|---|
| Relationship to Stage 1 | **Closed.** Do not reopen conditional Direct rescoring |
| Relationship to JIPI | **Closed.** Do not reopen pack-edit RL over frozen Direct |
| Fundamental change | Replace candidate-ranking / schedule-editing with **RL-guided joint task–helper pack optimization** |
| What RL learns | A discrete **planning profile** (deadline-risk, helper congestion, byte opportunity prices) |
| What the optimizer does | Constructs a complete feasible assignment pack \(A_t\) under the selected profile |
| What stays frozen | Safety governor, Mode 2 radio, candidate feasibility rules, \(B_{\max}=175\) kB, reward definition |
| Primary comparator | Strongest matched method (currently LVM at 46.82) |
| Final-test access | Prohibited |

### Diagnostic lineage motivating this formulation

| Finding | Implication for RHO |
|---|---|
| LVM 46.82 > Direct 46.06 | Ranking-style RL has not established value superiority |
| \(P_{\mathrm{viol}}=0\); PRR/P95 comfortable | Do not treat safety relaxation as the performance lever |
| JIPI KEEP ≈ 99.8%; ΔV vs Direct = 0 | Editing Direct’s pack is not an effective new policy |
| Universal helper rebind harms other completions | Optimize **packs**, not isolated task–helper scores |
| 82002 / 82011 / 82012 | Same allocation problem: pack-dependent completion + helper load |

**Scientific hypothesis:** Learning adaptive opportunity prices that guide a joint pack search can raise deadline-valid completed value by accounting for assignment interactions that independent ranking and post-hoc editing miss.

This document is **not** evidence that RHO will beat LVM.

---

## 1. Design decisions

| Component | Specification |
|---|---|
| Online planner | Bounded beam search over feasible packs |
| RL algorithm | PPO over a **fixed** discrete profile set (e.g. \(3^3=27\) actions) |
| Profile dimensions | Deadline-risk sensitivity; helper congestion price; byte opportunity price |
| Profile levels | Fixed before training; not tuned to validation seeds |
| Completion model | Causal \(\widehat p_{i,h}(A_t\mid s_t)\); train-calibrated only |
| Safety governor | Existing governor, unchanged (final admission authority) |
| Computation byte envelope | 175,000 bytes |
| Reward | Realized deadline-valid completed task value (unchanged) |
| Required ablation | Joint optimizer **without** RL (fixed/fair profile) |
| Exact MIP | Offline reference only; not required online |

---

## 2. Core mechanism

```text
Observed tasks, deadlines, helpers, radio state
          |
          v
   PPO policy (planning profile)
          |
          v
   Joint pack optimizer (beam search)
          |
          v
   Value-maximizing feasible pack A_t*
          |
          v
   Unchanged safety governor → Mode 2 → helpers
```

1. Build the common feasible candidate set \(\mathcal C_t\).
2. PPO selects a planning profile \(\pi_t\) from a fixed discrete set.
3. Beam search expands feasible packs in \(\mathcal F_t\), recomputing pack-dependent completion estimates and helper loads after each addition.
4. Score packs with the profile-weighted objective (Section 3).
5. Return the best feasible pack before the decision deadline.
6. Governor admits; Mode 2 executes; reward = realized completed value.

Unlike Stage 1, interactions are evaluated by search, not only by sequential neural scoring.  
Unlike JIPI, the planner **constructs** \(A_t\) rather than editing Direct’s \(A_t^0\).

---

## 3. Planning objective

\[
\boxed{
\begin{aligned}
A_t^*=\arg\max_{A_t\in\mathcal F_t}
\Bigg[
&\sum_{(i,h)\in A_t} v_i\,\widehat p_{i,h}(A_t\mid s_t)
-\lambda_B(s_t)\sum_{(i,h)\in A_t}b_i
-\sum_h \lambda_h(s_t)\,L_h(A_t)
\Bigg].
\end{aligned}
}
\]

| Symbol | Meaning |
|---|---|
| \(\mathcal F_t\) | Feasible packs under existing source/task/helper/candidate/deadline/\(B_{\max}\) rules |
| \(v_i,b_i\) | Task value and payload bytes |
| \(\widehat p_{i,h}(A_t\mid s_t)\) | Pack-dependent Pr(deadline-valid completion) |
| \(L_h(A_t)\) | Incremental helper workload (existing queue semantics) |
| \(\lambda_B,\lambda_h\) | Profile-driven opportunity prices (not new physical constraints) |

**Reward vs objective:** The expression guides online planning only. PPO is trained on realized deadline-valid completed value. Prices are opportunity-cost approximations — **not** a guarantee of global optimality.

### How known failure modes map

| Case | How the objective addresses it |
|---|---|
| 82002 / task 19 helper swap | Compare \(\widehat p\) under helper 0 vs current helper in pack context |
| 82011 / helper 9 | Pack-dependent \(\widehat p\) + \(L_h\) penalize collateral pending-work risk |
| 82012 / task1 vs task13 | Compare expected completed value of substitution including displacement/queue |

---

## 4. What RL learns

Fixed discrete action set (example): three levels \(\{\mathrm{low},\mathrm{mid},\mathrm{high}\}\) for each of

1. Deadline-risk sensitivity  
2. Helper congestion price  
3. Capacity (byte) opportunity price  

→ 27 profiles, levels frozen before any training.

PPO observes causally available state and selects one profile; the joint optimizer builds the pack.

---

## 5. Bounded joint search (online)

**Must be fixed before training** (beam width, expand limit, wall-clock budget ≤ 10 ms decision path).

1. Generate existing feasible candidates.  
2. Start from empty pack.  
3. Expand with feasible assignments (unique source/task rules).  
4. After each expansion, recompute projected helper workloads and affected \(\widehat p\).  
5. Score with the selected profile.  
6. Keep top-\(W\) partial packs (beam).  
7. Return best complete feasible pack before the deadline.

Exact MIP allowed offline for reference only.

---

## 6. Causal completion-probability prerequisite

\[
\widehat p_{i,h}(A_t\mid s_t)
=
\Pr\!\left(T_i^{\mathrm{complete}}\le D_i \mid s_t,A_t\right).
\]

**Requirements**

- Inputs: helper ready times, queued computation, remaining deadlines, radio-delivery conditions available at \(t\).  
- **Train-calibrated only** — never fit to the four locked validation outcomes or sealed final-test.  
- Binary finish-feasibility alone is known insufficient; need ranking of **counterfactual packs**, not only observed-task accuracy.  
- Gate: if a reliable, causally valid \(\widehat p\) cannot be built, **do not** proceed to RHO PPO training.

---

## 7. Fair evaluation (when later authorized)

| Method | Purpose |
|---|---|
| Direct PPO | Original SPARC reference |
| VAS / LAS / VLRO | Heuristic baselines |
| LVM | Strongest current matched RL competitor |
| Joint optimizer without RL | Isolates combinatorial search |
| RHO-SPARC | Tests adaptive RL-guided planning |

Matched protocol: identical candidates, feasibility, 175 kB, common governor.  
Non-RL joint optimizer must use a fair fixed or train-tuned profile under a **comparable** tuning budget.

Engineering target remains \(G\ge 15\%\) vs strongest matched (≈ 53.84 vs current LVM); smaller reproducible gains judged on merit. Safety: \(P_{\mathrm{viol}}\le 2\%\).

---

## 8. Authorization boundary

| Action | Status |
|---|---|
| This specification | **Authorized / done** |
| Stage 1 / JIPI reopen | **Forbidden** |
| Causal \(\widehat p\) + joint optimizer modules | Spec/implement for audit only |
| Joint-allocation opportunity audit (offline / bounded replay) | **Next** — separate closed-loop replay needs its own approval |
| RHO PPO training (96 ep or otherwise) | **Not authorized** |
| Locked validation / final-test / manuscript | **Not authorized** |

---

## 9. Immediate next action

Specify and test causal completion-probability and joint-pack optimization on existing **development** artifacts:

→ **Joint-allocation opportunity audit** (`RHO_JOINT_ALLOCATION_OPPORTUNITY_AUDIT.md`).

**PASS:** Positive net completed-value opportunities from joint allocation; no unacceptable safety/timing regression.  
**FAIL:** Rearrangement without realized-value gain, or needs future information.  
**INCONCLUSIVE:** Analytical opportunity without closed-loop / causal-\(\widehat p\) confirmation.

No RHO training until that gate is reviewed.
