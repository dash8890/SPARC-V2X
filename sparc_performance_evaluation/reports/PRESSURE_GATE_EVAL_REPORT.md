# Pressure-Gate Dual — Closed-Loop Eval (v1)

**Mechanism:** fit-locked deterministic packing-pressure gate (not logistic RAO selector).
**Rule:** POPL iff `pending_value≥6.0` and `min_slack≥0.01` and `std_helper_cpu≥0.05`; else Direct.
**Overall:** `FAIL`

**Diagnosis:** Direct↔POPL only cannot cover LVM-dominated failure regimes (`deadline_0.5`, `tps2`, `helper_heterog`). Gate also blocked POPL on some `tps4` seeds. Successor: v2 with LVM low-pressure fallback → `PRESSURE_GATE_EVAL_V2_REPORT.md`.

## Per-condition mean V_done

| Condition | PG | Direct | LVM | POPL | ΔvsD | ΔvsL | ΔvsP | best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `density_35_nominal` | 46.99 | 46.06 | 46.82 | 47.21 | +0.93 | +0.17 | -0.22 | POPL |
| `deadline_0.5` | 19.83 | 19.83 | 21.69 | 18.68 | +0.00 | -1.86 | +1.15 | LVM |
| `deadline_1.5` | 53.76 | 53.78 | 54.86 | 54.86 | -0.02 | -1.09 | -1.09 | LVM |
| `workload_tps2` | 27.11 | 29.53 | 30.38 | 27.82 | -2.42 | -3.27 | -0.70 | LVM |
| `workload_tps4` | 55.54 | 54.70 | 57.13 | 58.45 | +0.83 | -1.59 | -2.92 | POPL |
| `helper_homog` | 24.57 | 24.57 | 22.84 | 22.84 | +0.00 | +1.74 | +1.74 | PG(≥Direct) |
| `helper_heterog` | 54.80 | 55.47 | 55.61 | 54.36 | -0.68 | -0.81 | +0.43 | LVM |

## Aggregate

- Conditions PG > Direct: **2** / 7
- Conditions PG > LVM: **2** / 7
- Conditions PG > POPL: **3** / 7
- Conditions PG ≥ best(Direct,LVM,POPL): **1** / 7
- Mean ΔV vs Direct: **-0.192**
- Mean ΔV vs LVM: **-0.960**
- Mean ΔV vs POPL: **-0.230**

Artifacts: `sparc_performance_evaluation/results/raw/pressure_gate_eval`
