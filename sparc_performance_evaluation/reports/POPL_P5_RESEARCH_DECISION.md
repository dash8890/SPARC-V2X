# POPL-SPARC — Research Decision (P5 Freeze)

**Date:** 2026-10-18  
**Verdict:** POPL passes operational feasibility but does **not** demonstrate robust scheduling superiority.  
**Action:** Freeze POPL-P5 as a completed research candidate. Retain Direct PPO as the SPARC reference.

---

## Decision

| Item | Status |
|---|---|
| Optimized POPL inference (≤10 ms) | Accepted (PASS) |
| Nominal val mean V vs Direct / LVM | Positive, statistically inconclusive |
| Nine-condition robustness | `MIXED_INCONCLUSIVE` — no dependable advantage |
| Additional PPO / HPO on this checkpoint | **Not authorized** |
| Manuscript superiority claim | **Not justified** |
| Sealed final-test access | **Remains closed** |
| SPARC reference method | **Direct PPO** (unchanged) |
| POPL role | Experimental alternative (frozen) |

---

## Evidence summary

| Metric | Value |
|---|---|
| Conditions outperforming strongest baseline | **2 / 9** |
| Mean ΔV vs strongest baseline | **−0.91** |
| Nominal gain vs LVM (d35) | **+0.39** |
| Conditions with POPL > Direct | **4 / 9** |
| Conditions with POPL > LVM | **3 / 9** |

Pattern (locked val seeds only):

- **Strongest:** `workload_tps4` — POPL − Direct = +3.75; POPL − LVM = +1.33; all four seed deltas vs Direct positive.
- **Weakest:** `deadline_0.5` — POPL trails LVM by 3.01 and Direct by 1.15.
- Other weaknesses: light workload, homogeneous helpers, heterogeneous helpers.
- Density: better than LVM at 35, worse at 50.

The hypothesis that paired-outcome learning *generally* improves task–helper scheduling is **not** supported by this validation panel.

---

## Safety unit resolution

`P_viol` is a **fraction** of valid safety windows with a PRR/P95 violation:

\[
P_{\mathrm{viol}} = \frac{n_{\mathrm{violating\ valid\ windows}}}{n_{\mathrm{valid\ windows}}}
\in [0,1].
\]

Implementation: `safety_from_audit` in `run_paired_eval.py` (`viol / n_valid`).

| Report value | Meaning |
|---|---|
| `0.25` | **25%** of valid windows violate (not 0.25%) |
| Locked limit | \(P_{\mathrm{viol}} \le 0.02\) (**2%**) |

At `density_50`, mean \(P_{\mathrm{viol}}=0.25\) for **POPL, Direct, LVM, VAS, LAS, and VLRO alike** (seed 82012 has \(P_{\mathrm{viol}}=1.0\) under all methods). This is shared radio stress under the common governor, not a POPL-only regression. It still forbids generalizing “safety-compliant” claims from nominal d35 to all operating conditions.

---

## Why freeze (no further P5 training)

1. No dependable advantage over the strongest matched baseline across conditions.
2. Further tuning on the same four repeatedly examined validation seeds would increase overfitting risk.
3. Prior Stage-1 / JIPI / RHO cycles of localized heuristics and surrogates did not yield reliable superiority; another corrective loop on this panel is not authorized.

---

## What is retained

| Artifact | Path / role |
|---|---|
| Frozen POPL weights | `sparc_performance_evaluation/results/raw/popl_p5_pilot/popl_p5_checkpoint.pt` |
| Optimized inference path | Batched scoring + lean action I/O (behavior-preserving; latency PASS) |
| P5 pilot report | `POPL_P5_PILOT_REPORT.md` |
| Val / inference-opt / robustness | `POPL_P5_VAL_REPORT.md`, `POPL_P5_INFERENCE_OPT_REPORT.md`, `POPL_P5_ROBUSTNESS_REPORT.md` |
| SPARC reference | Direct PPO (`results/wilab_direct_ppo_96ep/…`) |

Stage 1, JIPI, and RHO remain closed.

---

## Path for any future RL scheduler work

If the project resumes toward a stronger learned scheduler, it must use a **fresh training/development split** and a **predeclared protocol**, not repeated optimization against seeds `82001/82002/82011/82012`. The sealed final-test set remains untouched until an explicit, separate release decision.

---

## Status line

**POPL is computationally feasible; its performance advantage is condition-dependent and insufficiently established. Direct PPO remains the SPARC reference. No manuscript superiority claim or final-test release is justified by this evidence.**
