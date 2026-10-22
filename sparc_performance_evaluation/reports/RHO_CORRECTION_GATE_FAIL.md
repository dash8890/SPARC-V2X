# RHO-SPARC — Correction Gate Verdict: FAIL

**PPO training:** NO-GO  
**Surrogate \(\widehat p\) / pack objective:** rejected for training  
**RHO formulation:** retained as unvalidated research candidate (search latency only)  
**Stage 1 / JIPI:** remain closed  
**SPARC reference:** Direct  
**Final-test:** sealed  

| Gate | Result | Decision |
|---|---|---|
| Search latency | 2.46 ms maximum | PASS |
| Predicted vs realized ranking | Top-1 0%; Spearman 0.25 | FAIL |
| Improvement over Direct | 0/4 seeds | FAIL |
| Improvement over LVM | 0/4 seeds | FAIL |
| Safety | 0% violations | PASS |

Bottleneck: **predictive objective**, not search speed.

Follow-on: `RHO_SURROGATE_OBJECTIVE_AUDIT.md` (trace-only; no new replay/training).

**Revised objective (spec only):** `RHO_COVERAGE_AWARE_OBJECTIVE_SPEC.md` — coverage-aware \(J=V_{\mathrm{in}}-C_{\mathrm{excl}}-C_B-C_{\mathrm{load}}\) with positive marginal \(\Delta J\); Phase A offline validation specified; implementation/replay/PPO still not authorized.
