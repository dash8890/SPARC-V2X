# POPL-SPARC — Behavior-Preserving Inference Optimization Report

**Overall:** `PASS`
**Scope:** implementation-only speedup of frozen POPL; no PPO / HPO / architecture changes.

## Acceptance gates

| Gate | Status | Evidence |
|---|---|---|
| Action equivalence | `PASS` | pack mismatches=0 across 25 opportunity epochs |
| Outcome equivalence | `PASS` | max ‖ΔV‖=0.000e+00; P_viol unchanged=True |
| Latency ≤ 10 ms | `PASS` | P95=4.870 ms; max=6.063 ms; frac>10ms=0.000 |
| No algorithmic change | `PASS` | same checkpoint; batched scoring + lean action I/O only |

## Latency (candidate epochs, optimized)

| Stat | ms |
|---|---:|
| n | 25 |
| mean | 2.596 |
| P50 | 2.334 |
| P95 | 4.870 |
| P99 | 5.782 |
| max | 6.063 |

Pre-optimization audit (reference): P95=36.29 ms, max=108.11 ms.

## Per-seed outcomes (ref → opt)

| Seed | V_ref | V_opt | ΔV | packs match |
|---:|---:|---:|---:|:---:|
| 82001 | 38.2612 | 38.2612 | +0.0000e+00 | Y |
| 82002 | 52.4071 | 52.4071 | +0.0000e+00 | Y |
| 82011 | 58.6936 | 58.6936 | +0.0000e+00 | Y |
| 82012 | 39.4785 | 39.4785 | +0.0000e+00 | Y |

## Optimizations applied (implementation only)

1. Batched candidate scoring (one scorer forward per construction step).
2. Cached candidate feature matrix / id tables per epoch.
3. Incremental feasibility sets (no repeated full candidate feature rebuild).
4. Lean POPL action JSON (drop unused per-candidate audit dumps from the hot path).
5. `torch.inference_mode` on deterministic eval; one-time warmup before measured episodes.

TERMINATE semantics, network weights, governor, envelope, and baselines unchanged.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\popl_p5_inference_opt`
