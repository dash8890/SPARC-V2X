# POPL-SPARC — Execution-Mechanism Audit (Training-Free)

**Scope:** frozen P5 checkpoint; existing locked-validation traces only.
**Not authorized:** PPO updates, HPO, architecture revision, manuscript promotion, final-test.

## 1. Decision latency

Denominator: controller epochs with `n_candidates > 0` (online decision opportunities).
Metric: `inference_ms + governor_ms` (sequential pack construction + TERMINATE + governor).
Decision budget reference: **10 ms**.

| Stat | Value (ms) |
|---|---:|
| n decision epochs (4 seeds) | 25 |
| mean | 17.586 |
| P50 | 7.938 |
| P95 | 36.287 |
| P99 | 91.095 |
| max | 108.111 |
| fraction > 10 ms | 0.480 (12/25) |

**Budget check:** P95=FAIL (P95=36.29 ms vs 10 ms).

### Per-seed latency (candidate epochs)

| Seed | n | mean | P95 | max | frac>10ms |
|---:|---:|---:|---:|---:|---:|
| 82001 | 5 | 27.601 | 90.197 | 108.111 | 0.400 |
| 82002 | 7 | 10.872 | 24.457 | 27.734 | 0.429 |
| 82011 | 8 | 17.487 | 35.592 | 37.214 | 0.500 |
| 82012 | 5 | 17.128 | 30.627 | 31.175 | 0.600 |

## 2. Assignment coverage (identical denominators)

Opportunity epoch = `n_candidates > 0`. Execution epoch = `n_executed > 0`.

| Method | opp. epochs | exec epochs | exec\|opp | mean n_exec\|opp | total asg |
|---|---:|---:|---:|---:|---:|
| POPL | 25 | 25 | 1.000 | 2.67 | 67 |
| Direct | 25 | 25 | 1.000 | 2.79 | 70 |
| LVM | 28 | 28 | 1.000 | 2.58 | 73 |

### Per-seed exec|opportunity

| Seed | POPL | Direct | LVM | POPL opp | Direct opp | LVM opp |
|---:|---:|---:|---:|---:|---:|---:|
| 82001 | 1.000 | 1.000 | 1.000 | 5 | 6 | 6 |
| 82002 | 1.000 | 1.000 | 1.000 | 7 | 6 | 7 |
| 82011 | 1.000 | 1.000 | 1.000 | 8 | 8 | 9 |
| 82012 | 1.000 | 1.000 | 1.000 | 5 | 5 | 6 |

**Interpretation:** Stage-0 / POPL opportunity counts are sparse (~6.2 opportunity epochs/episode of 100). The previously reported 0.062 frac-of-all-epochs reflects sparse arrivals, not under-admission relative to opportunity — POPL exec|opp = **1.000** vs Direct **1.000**.

## 3. Causal value attribution (POPL vs Direct)

| Seed | Δ completions | Gained tasks | Lost tasks | Value gained | Value lost | Pack-divergence epochs |
|---:|---:|---|---|---:|---:|---:|
| 82001 | +0/−1 | — | [0] | 0.000 | 1.580 | 5 |
| 82002 | +1/−0 | [19] | — | 2.454 | 0.000 | 7 |
| 82011 | +0/−0 | — | — | 0.000 | 0.000 | 8 |
| 82012 | +1/−0 | [1] | — | 3.738 | 0.000 | 6 |

### Decision-level events

- Seed **82001**:
  - `loss` task 0 (value≈1.5799639766464622): Direct completed; POPL never assigned; POPL assign@None None; Direct assign@0 {'task_id': 0, 'helper_id': 5, 'source_id': 0, 'payload_bytes': 61887, 'packet_count': 62}
- Seed **82002**:
  - `gain` task 19 (value≈2.4539326207635574): POPL completed; Direct assigned but did not complete; POPL assign@33 {'task_id': 19, 'helper_id': 2, 'source_id': 6, 'payload_bytes': 27218, 'packet_count': 28}; Direct assign@33 {'task_id': 19, 'helper_id': 2, 'source_id': 6, 'payload_bytes': 27218, 'packet_count': 28}
- Seed 82011: no completion set difference.
- Seed **82012**:
  - `gain` task 1 (value≈3.7381103978699155): POPL completed; Direct never assigned; POPL assign@33 {'task_id': 1, 'helper_id': 6, 'source_id': 0, 'payload_bytes': 21211, 'packet_count': 22}; Direct assign@None None

These gain/loss events are realized deadline-valid completion differences under matched exogenous seeds — consistent with the paired-outcome training target (ΔG from completed value), not an analytical surrogate. Concentration in few tasks remains a robustness concern.

**Mechanistic notes (consistent with paired-outcome training):**
- **82001 loss (task 0):** epoch-0 pack divergence — Direct includes `(0,5)`; POPL's pack omits task 0 entirely.
- **82002 gain (task 19):** both policies assign `(19,2)` at epoch 33; companion pack differs (`POPL` includes `(1,0)` vs Direct `(1,2)`). Completion difference is therefore a closed-loop interference / co-scheduling effect, not a different helper choice for task 19 itself.
- **82012 gain (task 1):** epoch-33 pack divergence — POPL admits task 1; Direct never schedules it.

## 4. Safety uncertainty (zero observed violations)

- Pooled valid safety windows: **400**
- Observed violations: **0** (P̂_viol = 0.0)
- Clopper–Pearson one-sided 95% upper bound on P_viol: **0.0075**
- Jeffreys 95% CI: **[0.0000, 0.0063]**
- Rule-of-3 upper (≈): **0.0075**

Zero violations on four seeds does **not** prove P_viol≡0; the upper bound quantifies residual risk.

## Audit verdict

1. **Latency:** FAIL vs 10 ms budget — P95=36.29 ms, max=108.11 ms (12/25 candidate-epochs over budget).
2. **Coverage:** Sparse opportunity epochs explain the 0.062 headline fraction; conditional on opportunity, POPL executes at a rate comparable to Direct/LVM.
3. **Attribution:** Net +2/−1 deadline-valid completions vs Direct are tied to specific pack divergences; value is realized, but thin-tailed.
4. **Safety:** No violations in pooled valid windows; residual P_viol upper bound ≈ 0.007 (95% one-sided).

**Status:** POPL remains the leading active research candidate; not manuscript-ready.
Final-test seeds remain sealed. No further training authorized by this audit.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\popl_p5_execution_audit`
