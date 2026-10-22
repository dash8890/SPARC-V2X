# RHO-SPARC — Audit Correction Gate Report

**Correction-gate verdict:** FAIL  
**PPO training:** NO-GO (surrogate rejected)

See also: `RHO_CORRECTION_GATE_FAIL.md`, `RHO_SURROGATE_OBJECTIVE_AUDIT.md`.

## 1. Search latency (requirement 1)

| Metric | Value |
|---|---:|
| max search ms | 2.4587996304035187 |
| high-load max ms (|C|≥40) | 2.4587996304035187 |
| frac ≤10 ms | 1.0 |
| incremental≡full | 53/53 |

**Latency gate:** `PASS`

## 2. Causal ranking validation (requirement 2)

**Gate:** `FAIL_OR_INCONCLUSIVE_RANKING`

| Pack | mean predicted E[V] | mean realized V_done |
|---|---:|---:|
| Direct | 4.6792 | 46.0571 |
| LVM | 7.5885 | 45.6621 |
| RHO | 7.8867 | 44.1138 |

- Top-1 pred/real agreement: 0.00
- Mean Spearman (pred E vs realized V): 0.25
- Frac RHO beats LVM (predicted / realized): 0.50 / 0.00
- Frac RHO beats Direct (realized): 0.00
- max P_viol: 0.0000

## Per-seed ranks

| Seed | pred rank | realized rank | V(Direct/LVM/RHO) |
|---:|---|---|---|
| 82001 | RHO>LVM>Direct | Direct>LVM>RHO | 39.84/38.26/38.26 |
| 82002 | LVM>RHO>Direct | Direct>LVM>RHO | 49.95/49.95/46.41 |
| 82011 | LVM>RHO>Direct | Direct>LVM>RHO | 58.69/58.69/56.05 |
| 82012 | RHO>Direct>LVM | Direct>LVM>RHO | 35.74/35.74/35.74 |

## Decision

Proceed toward RHO PPO training only if latency PASS **and** ranking shows a credible positive predicted↔realized relationship **and** realized value opportunities vs LVM.

Current combined status: latency `PASS`; ranking `FAIL_OR_INCONCLUSIVE_RANKING`.

No architecture expansion, manuscript promotion, or final-test access.
