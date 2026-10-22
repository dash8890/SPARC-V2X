# Stage 1 — Closure

**Verdict: Stage 1 did not improve scheduling performance. The conditional-rescoring architecture should not be promoted.**

| Metric | Result |
|---|---|
| Prior Direct SPARC | 46.06 |
| Stage-1 SPARC | 41.05 |
| Strongest matched competitor (LVM) | 46.82 |
| Stage-1 vs prior Direct | −10.9% |
| Stage-1 vs LVM | −12.3% |
| Safety violations | 0% |
| Training | 96/96 episodes |
| Selected checkpoint | Update 8 |
| Final-test status | **Sealed** |

## What the result establishes

1. **D7 responsiveness was not the underlying solution.** The new actor became responsive to deadline/helper information, but that did not translate into better scheduling decisions.
2. **Additional architectural expressiveness was insufficient** under the tested configuration.
3. **Safety held** (\(P_{\mathrm{viol}}=0\)); it does not compensate for the primary-objective regression.
4. **Prior Direct PPO remains the stronger SPARC implementation** and the reference for subsequent development.

## What this result does not establish

It does **not** prove conditional scheduling is inherently inferior. Optimization difficulty, representation choices, training instability, or limited value of context-dependent decisions remain hypotheses — not grounds to disregard the measured regression.

The early best checkpoint (update 8) motivates examining training dynamics; it does **not** justify automatically extending training or launching a hyperparameter search.

## Decision

- Close Stage 1 as a **failed validation** of this particular architectural change.
- Do **not** open sealed final-test.
- Do **not** promote the Stage-1 checkpoint.
- Do **not** revise the manuscript to describe the conditional scheduler as an improvement.

## Diagnostics completed (training-free)

1. `STAGE1_VS_DIRECT_DIAGNOSTIC.md` — ranking churn + early-peak training; governor not implicated.
2. `STAGE1_VALUE_DEFICIT_DECOMPOSITION.md` — **full ΔV accounted for**:

| Source of −5.01 mean ΔV | Value/seed | Share |
|---|---:|---:|
| Tasks Direct completes that Stage-1 **never assigns** | 2.64 | 52.7% |
| Tasks Stage-1 assigns but **deadline-misses** (Direct completed) | 2.37 | 47.3% |
| Stage-1-only completions (gains) | 0.00 | — |
| Radio incomplete / failed delivery (assigned) | 0.00 | — |

Lower offered bytes = **fewer assignments** (not smaller mean payload). No Stage-1-only wins on this panel.

**Status locked** (`STAGE1_FINAL_DEFICIT_ASSESSMENT.md`): Stage 1 closed; prior Direct retained; final-test sealed.

**Not authorized:** further Stage-1 training, STOP, architectural revision on this evidence, manuscript promotion, or any new implementation/experiment from this assessment alone.
