# POPL-SPARC — Frozen-Checkpoint Validation Report

**Outcome:** `POSITIVE_BUT_INCONCLUSIVE_VS_DIRECT`
**Protocol:** training-free full-episode POPL on locked validation seeds `82001/02/11/12` under common governor + 175 kB envelope.
**Checkpoint:** `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\popl_p5_pilot\popl_p5_checkpoint.pt` (frozen; no PPO updates).

## POPL aggregate

- V_done mean: **47.2101** (95% CI [31.3335, 63.0867])
- R_value mean: **0.7371** (95% CI [0.4867, 0.9875])
- P_viol mean: **0.0000**
- Mean assignment coverage (epochs with ≥1 exec): **0.062**
- Mean wall / episode: **35.4 s**

## Paired ΔV vs matched Stage-0 baselines

| Baseline | Base V | POPL − Base | 95% CI |
|---|---:|---:|---|
| Direct | 46.0571 | +1.1530 | [-2.6536, +4.9597] |
| VAS | 38.0362 | +9.1739 | [+3.6496, +14.6982] |
| LAS | 36.2995 | +10.9106 | [+1.3865, +20.4347] |
| VLRO | 39.9785 | +7.2316 | [-0.3678, +14.8309] |
| LVM | 46.8173 | +0.3928 | [-0.8571, +1.6426] |

**Best matched baseline:** LVM (V=46.8173)
**Gain vs best matched:** 0.008389741663942633
**Gain vs Direct:** +1.1530
**Gain vs LVM:** +0.3928

## Per-seed completed value

| Seed | POPL | Direct | LVM | VAS | LAS | VLRO |
|---:|---:|---:|---:|---:|---:|---:|
| 82001 | 38.2612 | 39.8412 | 38.2612 | 32.5424 | 34.0848 | 34.2588 |
| 82002 | 52.4071 | 49.9532 | 50.8360 | 38.4196 | 34.0507 | 39.4703 |
| 82011 | 58.6936 | 58.6936 | 58.6936 | 49.9363 | 49.9363 | 56.0467 |
| 82012 | 39.4785 | 35.7404 | 39.4785 | 31.2464 | 27.1261 | 30.1382 |

## Deadline-valid completions vs Direct / LVM

| Seed | vs Direct gained/lost | vs LVM gained/lost |
|---:|---|---|
| 82001 | +0/−1 | +0/−0 |
| 82002 | +1/−0 | +1/−0 |
| 82011 | +0/−0 | +0/−0 |
| 82012 | +1/−0 | +0/−0 |

## Safety / runtime / coverage

| Seed | P_viol | wall (s) | frac epochs assigned | mean n_exec |
|---:|---:|---:|---:|---:|
| 82001 | 0.0000 | 39.3 | 0.050 | 0.12 |
| 82002 | 0.0000 | 32.4 | 0.070 | 0.20 |
| 82011 | 0.0000 | 33.3 | 0.080 | 0.21 |
| 82012 | 0.0000 | 36.5 | 0.050 | 0.14 |

## Decision

- Outcome code: `POSITIVE_BUT_INCONCLUSIVE_VS_DIRECT`
- No additional PPO updates, HPO, architecture changes, manuscript promotion, or final-test access were performed.
- P5 training-only CONDITIONAL PASS remains the training-status label; this report answers whether the frozen policy transfers on locked val.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\popl_p5_val`
