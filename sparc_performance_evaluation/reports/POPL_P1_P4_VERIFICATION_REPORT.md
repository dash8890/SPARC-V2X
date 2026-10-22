# POPL-SPARC — P1–P4 Verification Report

**Overall:** `PASS`
**P5 / PPO training:** not authorized

## Gate summary

| Gate | Status | Key evidence |
|---|---|---|
| P1 Counterfactual | `PASS` | identity ‖ΔV‖=0.00e+00; prefix OK=True |
| P2 Feasibility | `PASS` | all packs in F_t under 175 kB |
| P3 Discrimination | `PASS` | 82002 ΔV=+2.4539; 82011 ΔV=-2.6469 |
| P4 Measured cost | `PASS` | mean pair=61.3s (assumption 80s; ratio=0.77) |

## Paired samples

| Name | ΔV_done | ΔG_from_fork | pair wall (s) | force OK | feas |
|---|---:|---:|---:|:---:|:---:|
| identity_82002_ep33 | +0.0000 | +0.0000 | 61.8 | Y | Y |
| beneficial_82002_t19_h2to0 | +2.4539 | +2.4539 | 58.5 | Y | Y |
| harmful_82011_t15_h4to9 | -2.6469 | -2.6469 | 63.8 | Y | Y |

## P4 runtime detail

- Mean episode wall: **30.63 s**
- Mean pair wall (2 episodes): **61.35 s**
- Max pair wall: **63.78 s**
- Planning assumption was 80 s/pair; measured/assumption = **0.77**

## Decision

P1–P4 **PASS**. A train-only **P5 learning pilot** may be considered under a **separate** authorization. PPO weight training remains unauthorized until P5 and the learning-rule correctness review both clear.

See also: `POPL_LEARNING_RULE_CORRECTNESS_REVIEW.md` (R1–R4 CONDITIONAL PASS; P5 still not authorized).

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\popl_p1_p4_verification`
