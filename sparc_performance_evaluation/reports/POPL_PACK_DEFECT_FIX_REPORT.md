# POPL Pack-Defect Fix — One Closed-Loop Check

**Scope:** decode-only fixes on frozen POPL-P5 weights; nominal d35; seeds 82001/02/11/12.
**Not:** routers, RAO/W, threshold search, retraining.

## Fixes

1. **Early omit:** mask TERMINATE while any feasible add remains.
2. **Helper binding:** after task pick, rebind to min `finish_ms` helper among feasible adds for that task.

**Overall:** `FAIL`

| Method | mean V_done |
|---|---:|
| POPL_pack_fix | 38.755 |
| Direct | 46.057 |
| LVM | 46.817 |
| POPL_old | 47.210 |
| Δ vs best (POPL_old) | -8.455 |

## Per seed

| Seed | Fix | Direct | LVM | POPL_old | Δ vs best |
|---:|---:|---:|---:|---:|---:|
| 82001 | 34.259 | 39.841 | 38.261 | 38.261 | -5.582 |
| 82002 | 36.632 | 49.953 | 50.836 | 52.407 | -15.775 |
| 82011 | 52.503 | 58.694 | 58.694 | 58.694 | -6.191 |
| 82012 | 31.625 | 35.740 | 39.478 | 39.478 | -7.853 |

Seeds ≥ best baseline: **0**/4

## Verdict (once)

**FAIL.** Decode-only repair (no early TERMINATE + min-finish helper rebind) made nominal val *worse* than frozen POPL/Direct/LVM on all four seeds. This checkpoint’s win/loss pattern is not fixed by those two decode patches.

Further progress needs a **new training program** (fresh split)—not another gate/router/decode tweak on this frozen policy.

Artifacts: `sparc_performance_evaluation/results/raw/popl_pack_defect_fix`
