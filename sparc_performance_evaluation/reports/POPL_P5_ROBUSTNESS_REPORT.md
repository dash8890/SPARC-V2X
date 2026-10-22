# POPL-SPARC — Bounded Robustness Evaluation (Training-Free)

**Overall:** `MIXED_INCONCLUSIVE`
**Policy:** frozen optimized POPL checkpoint (no PPO updates).
**Seeds:** locked validation `82001/02/11/12` only (final-test sealed).

## Condition grid

| Condition | Axis | density | deadline× | tasks/src | helper CPU |
|---|---|---:|---:|---:|---|
| `density_20` | density | 20 | 1.0 | 3 | [2e+08, 1e+09] |
| `density_35_nominal` | density | 35 | 1.0 | 3 | [2e+08, 1e+09] |
| `density_50` | density | 50 | 1.0 | 3 | [2e+08, 1e+09] |
| `deadline_0.5` | deadline | 35 | 0.5 | 3 | [2e+08, 1e+09] |
| `deadline_1.5` | deadline | 35 | 1.5 | 3 | [2e+08, 1e+09] |
| `workload_tps2` | workload | 35 | 1.0 | 2 | [2e+08, 1e+09] |
| `workload_tps4` | workload | 35 | 1.0 | 4 | [2e+08, 1e+09] |
| `helper_homog` | helper | 35 | 1.0 | 3 | [5e+08, 5e+08] |
| `helper_heterog` | helper | 35 | 1.0 | 3 | [1e+08, 2e+09] |

## Mean completed value by condition

| Condition | POPL | Direct | LVM | VAS | LAS | VLRO | Best base | Δ vs Direct | Δ vs LVM | Δ vs best |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `density_20` | 27.60 | 27.60 | 27.20 | 25.93 | 23.95 | 25.93 | Direct | +0.00 | +0.39 | +0.00 |
| `density_35_nominal` | 47.21 | 46.06 | 46.82 | 38.04 | 36.30 | 39.98 | LVM | +1.15 | +0.39 | +0.39 |
| `density_50` | 46.67 | 44.77 | 47.98 | 36.71 | 33.38 | 36.08 | LVM | +1.89 | -1.32 | -1.32 |
| `deadline_0.5` | 18.68 | 19.83 | 21.69 | 17.23 | 16.43 | 16.02 | LVM | -1.15 | -3.01 | -3.01 |
| `deadline_1.5` | 54.86 | 53.78 | 54.86 | 52.01 | 52.31 | 52.13 | LVM | +1.08 | +0.00 | +0.00 |
| `workload_tps2` | 27.82 | 29.53 | 30.38 | 24.29 | 23.94 | 25.33 | LVM | -1.72 | -2.57 | -2.57 |
| `workload_tps4` | 58.45 | 54.70 | 57.13 | 49.57 | 47.42 | 49.94 | LVM | +3.75 | +1.33 | +1.33 |
| `helper_homog` | 22.84 | 24.57 | 22.84 | 23.08 | 23.63 | 24.60 | VLRO | -1.74 | +0.00 | -1.76 |
| `helper_heterog` | 54.36 | 55.47 | 55.61 | 54.00 | 52.31 | 52.90 | LVM | -1.11 | -1.25 | -1.25 |

## Aggregate robustness summary

- Conditions evaluated: **9**
- Conditions where POPL mean V > Direct: **4** / 9
- Conditions where POPL mean V > LVM: **3** / 9
- Conditions where POPL mean V > best matched baseline: **2** / 9
- Mean ΔV vs Direct across conditions: **+0.240**
- Mean ΔV vs LVM across conditions: **-0.670**
- Mean ΔV vs best baseline across conditions: **-0.909**
- Safety: max mean P_viol (POPL) = **0.25** = **25%** of valid windows (fraction in \([0,1]\), not 0.25%)

### Safety note

`P_viol = n_violating_valid_windows / n_valid_windows`. The locked limit is \(P_{\mathrm{viol}}\le 0.02\) (2%). At `density_50`, mean P_viol≈0.25 for **all six methods** (seed 82012 has P_viol=1.0 under every method) — shared radio stress, not POPL-only. Do not generalize nominal d35 safety compliance to all conditions.

## Per-seed detail (POPL − Direct)

| Condition | 82001 | 82002 | 82011 | 82012 |
|---|---:|---:|---:|---:|
| `density_20` | +0.00 | +0.00 | +0.00 | +0.00 |
| `density_35_nominal` | -1.58 | +2.45 | +0.00 | +3.74 |
| `density_50` | +0.90 | -3.08 | +9.75 | +0.00 |
| `deadline_0.5` | -2.06 | -1.23 | -1.29 | +0.00 |
| `deadline_1.5` | +0.00 | -1.54 | +4.37 | +1.47 |
| `workload_tps2` | -1.58 | -1.56 | +0.00 | -3.74 |
| `workload_tps4` | +2.44 | +7.64 | +3.44 | +1.48 |
| `helper_homog` | -3.77 | +0.00 | -0.97 | -2.21 |
| `helper_heterog` | -3.20 | +0.00 | +0.00 | -1.24 |

## Verdict

- Outcome code: `MIXED_INCONCLUSIVE`
- No PPO updates, parameter tuning from these results, manuscript superiority claims, or final-test access.
- Nominal density_35 reuses Stage-0 baselines + optimized POPL val traces.
- **Research decision:** POPL-P5 is frozen as a completed experimental candidate; Direct PPO remains the SPARC reference. See `POPL_P5_RESEARCH_DECISION.md`.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\popl_p5_robustness`
