# RAO-SPARC — Stage 3 Algorithm Specification

**Status:** Design specification only. **No performance advantage claimed. No RAO PPO training authorized.**  
**Parent evidence:** POPL-P5 frozen (`POPL_P5_RESEARCH_DECISION.md`). Direct PPO remains SPARC reference.  
**Immediate authorized next step:** Phase-1 prerequisite gate design (`RAO_PREREQUISITE_GATE.md`) — collection of paired data requires a **separate** execution authorization.

---

## 1. Objective

\[
\max_\pi \mathbb E[V_{\mathrm{done}}(\pi)], \qquad P_{\mathrm{viol}}\le 0.02,
\]

under the existing 175 kB envelope, common governor, NR-V2X Mode 2 simulator, and 10 ms decision budget.

---

## 2. Method summary

**RAO-SPARC** (Regime-Adaptive Outcome Optimization):

1. Frozen Direct PPO reference pack \(A_t^D=\pi_D(s_t)\).
2. Adaptive outcome-trained pack actor \(A_t^N\sim\pi_\theta(\cdot\mid s_t)\) (sequential + TERMINATE).
3. Outcome-trained selector \(g_\phi\) choosing reference vs alternative from **causal** features only.

Learning targets are realized completed-value outcomes (and matched \(\Delta G\)), never analytical finish-time surrogates.

---

## 3. Staged development (no automatic progression)

| Phase | Scope | Authorization |
|---|---|---|
| **1** | Frozen Direct + frozen POPL as dual generators; train/eval a selector on training-development pairs; G1–G7 | Spec now; **paired collection / fitting needs separate approval** |
| **2** | Train new adaptive actor with outcome PPO | Only if Phase 1 PASS |
| **3** | Freeze actor+selector; full-controller validation | Only if Phase 2 PASS |

---

## 4. Design responses to prior failures

| Failure | Response (unverified until gated) |
|---|---|
| Stage 1 over-editing | Explicit keep-or-depart via selector |
| JIPI KEEP≈99.8% | Train selector on both +/− realized \(\Delta G\) |
| RHO surrogate misrank | Labels from realized \(V_{\mathrm{done}}\) |
| POPL tight-deadline / light-load harm | Reference pack always available; regime features include deadline/workload |
| Helper collateral | Pack-diff features include helper workload displacement |

---

## 5. Comparison set (when full RAO is later evaluated)

VAS, LAS, VLRO, LVM, Direct PPO, POPL-P5, Direct/POPL selector, RAO full, RAO-without-selector.

Primary ambitious target: ≥15% vs strongest matched competitor; \(P_{\mathrm{viol}}\le 0.02\) with uncertainty; ≤10 ms. \(P_{\mathrm{viol}}\) is a **fraction** (0.25 = 25%).

---

## 6. STOP conditions

See `RAO_PREREQUISITE_GATE.md` §STOP. Failures of G2/G3/G4 reject adaptive selection for this branch without authorizing val-seed rescue or larger networks as a substitute for evidence.

---

## 7. Module layout (target)

```
controllers/rl/rao/
    regime_features.py
    adaptive_pack_policy.py
    outcome_selector.py
    rao_controller.py
    paired_outcome_trainer.py
    policy_likelihood.py
    feasibility_checks.py
    inference_runtime.py
```

Filenames specify responsibilities. Implementation is **not** authorized by this document alone.
