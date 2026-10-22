# SPARC-V2X Evidence Package Report

**Generated (UTC):** `2026-10-15T11:04:39.353901+00:00`
**Package status:** `PARTIAL_NOT_LOCKABLE` (campaign aggregate `DONE`; improve aggregate `IN_PROGRESS/PARTIAL`)

This report is assembled only from sealed/locked raw rows and audits. It separates **governor-matched** methods (SPARC, B1-B3, A0/A2-A6 under common 175 kB+PRR/P95) from **unshielded / differently admitted** methods (B4 Lagrangian 500 kB no governor; B5/ITORA governor disabled). Ranking claims must respect that split.

Companion audits: `reports/P0_P3_EVIDENCE_AUDIT.md`, `reports/B4_LAGRANGIAN_ELIGIBILITY_AUDIT.md`, `protocol/b5_itora_mapping_audit.md`, `protocol/LOCK_CRITERIA.md`

## 1. Claim hierarchy (faithful reading)

| Tier | Methods | Admission contract | How to use in manuscript |
|---|---|---|---|
| Governor-matched | SPARC A0, B1 VA, B2 MF, B3 Direct, A3/A5/A6 | Common SafetyGovernor: PRR>=0.95, P95<=50 ms, 175 kB envelope | Primary value/safety comparisons |
| Gate ablation | A1 | Envelope retained; PRR/P95/validity gates open | Isolates gate effect |
| Unshielded learned | B4 Lagrangian | No governor; physical_byte_cap=500 kB; soft dual kappa=0.05 | System-level alternative; **not** ranked vs Direct under SPARC shield |
| Unshielded heuristic | B5 ITORA-inspired | `gov=None` in eval (byte budget only) | Literature-inspired; report with P_viol |

**Governor-matched nominal ranking by mean V_done:**
- B3 Direct: **37.97**
- SPARC (A0): **35.81**
- B1 VA: **32.46**
- B2 MF: **32.16**

Direct leads governor-matched learned methods (Direct 37.97 vs SPARC 35.81, gap 2.16). B4 mean V=42.68 and ITORA=33.33 are **out of this ranking**.

SPARC VA-collapse seeds (exact V equality): **2**/5.
Ablation A5 residual-only mean V=38.68 (beats Direct) under the **common governor** - relevant to SPARC design, not a new baseline.
Ablation A6 random-init mean V=38.56 (beats Direct) under the common governor.

## 2. Lock status and completeness

Per `protocol/LOCK_CRITERIA.md`, status remains **PARTIAL / not lockable**.

### Present
- B1/B2/B3/B4/B5 nominal final-test (n=5)
- Ablations A0-A6 (A2/A4 reuse)
- Density E2 for VA/MF/SPARC/ITORA + separate Direct density_*
- Envelope E3 for VA/MF/SPARC
- Table III campaign merge + Table IV stub
- Seed seal n=5; campaign_aggregate_DONE
- P0 offered-byte backfill from bridge_audit

### Missing or partial (blocks LOCKED)
- E2 density does NOT include B4 Lagrangian at any density
- E2 main job excludes Direct (exists only as density_direct_d*)
- E4 workload/relative-deadline sensitivity - not run
- E5 Figs 7-8 radio diagnostics package - not assembled
- E6 Fig 9 ablation CIs / formal panels - numbers only
- E7 Fig 10 governor enforcement cost - not assembled
- E8 Fig 11 generalization - not assembled
- Simultaneous multi-contrast inference - not sealed
- protocol_hashes lock.status=LOCKED - not set
- G1/G4/G8/G9 still require/partial runtime audit entries in gate_results.json

### Interpretation holds from audits (do not contradict in tables)
- B4/B5 are unshielded or differently admitted; not governor-matched to SPARC/Direct
- At nominal d=35, PRR/P95 gates never fired; 175 kB envelope is binding (see P0_P3 audit)
- ITORA density V wins at rho=80 coincide with P_viol~1 (governor disabled)
- B4 eligibility audit: INCONCLUSIVE for replacing Direct (see B4 report)

### Verification gates (`verification/gate_results.json` + wilab audit)

| Gate | pass/status |
|---|---|
| G0 | True |
| G2 | True |
| G3 | True |
| G5 | True |
| G6 | True |
| G7 | True |
| G1 | REQUIRES_WILAB_RUNTIME_AUDIT |
| G4 | REQUIRES_WILAB_DELIVERY_TRACE_AUDIT |
| G8 | REQUIRES_SAFETY_DENOMINATOR_AUDIT |
| G9 | REQUIRES_PROVENANCE_INDEX_AFTER_RUNS |
| G1 (wilab audit) | True /  |
| G4 (wilab audit) | True / PASS_WITH_NOTE |
| G8 (wilab audit) | True /  |

## 3. Table II - parameters (sourced)

| Parameter | Value |
|---|---|
| density | 35 |
| controller_epochs | 100 |
| tasks_per_source | 3 |
| simulation_time | 12.0 |
| envelope_bytes | 175000 |
| safety_payload_bytes | 300 |
| episodes | 96 |
| rollouts_per_update | 4 |
| ppo_epochs | 6 |
| minibatch_size | 64 |
| lr | 0.0003 |
| logit_scale | 100.0 |
| residual_gain | 1.0 |
| entropy_coef | 0.02 |
| target_kl | 0.015 |
| va_relative | True |
| seed_base | 72000 |
| task_seed_base | 182000 |
| lr_final_frac | 0.2 |
| entropy_final_frac | 0.25 |
| eval_every_updates | 2 |
| output_root | results\wilab_residual_ppo_96ep_va_relative |
| checkpoint_out | checkpoints\wilab_residual_ppo_96ep_va_relative.pt |
| smoke | False |
| algorithm | Residual PPO + scaled logits + VA-relative GAE shift + LR/entropy decay |
| initialization | fresh; residual output exactly zero; seed 616161 |
| selection_rule | keep checkpoint with best mean completed_value delta vs corrected VA |
| logit_fixing | logits = 100 * (analytic + 1 * residual) |
| admission | 175-kB governor enforced before sampling; legacy action-count shield bypassed |
| task_process | three tasks/source spread uniformly across the episode; absolute deadlines |
| attribution | at most one selected task per source per epoch |
| counterfactual_replay | False |
| off_policy_replay | False |
| planned_updates | 24 |

Source: `tables/table_ii_parameters/table_ii.json`. Nominal envelope_bytes=175000; B4 FT uses physical_byte_cap=500000 (not this row).

## 4. Table III - nominal final-test (density=35, n=5 sealed seeds)

Seeds: sim `93000`-`93004`. `R_value` = completed_value / offered_value (task-value ratio). Offered **bytes** are backfilled from `bridge_audit.json` (P0 fix: trace key mismatch had previously forced zeros).

### 4.1 Governor-matched methods

| Method | V_done | R_value | sum offered B | mean epoch offered B | P_viol | mean PRR | mean P95 ms | invalid frac | gate-block epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPARC-V2X (A0) | 35.81 | 0.5504 | 628499 | 6285 | 0.0000 | 0.9843 | 21.32 | 0.0000 | 0.0 |
| B1 Corrected VA | 32.46 | 0.4951 | 707565 | 7076 | 0.0000 | 0.9841 | 23.56 | 0.0000 | 0.0 |
| B2 Min-Finish | 32.16 | 0.4908 | 695526 | 6955 | 0.0000 | 0.9846 | 23.46 | 0.0000 | 0.0 |
| B3 Direct PPO | 37.97 | 0.5893 | 630079 | 6301 | 0.0000 | 0.9853 | 21.61 | 0.0000 | 0.0 |

### 4.2 Unshielded / differently admitted (do not merge into matched ranking)

| Method | V_done | R_value | sum offered B | P_viol | Admission |
|---|---:|---:|---:|---:|---|
| B4 Lagrangian | 42.68 | 0.6578 | 791446 | 0.0000 | no governor; 500 kB physical cap; soft dual |
| B5 ITORA-inspired | 33.33 | 0.5062 | 732121 | 0.0000 | gov=None in run_paired_eval; see mapping audit |

At nominal density, B4/B5 also report P_viol=0, but that does **not** make them governor-matched: B4 admits ~26% more computation bytes than Direct and exceeds 175 kB on 40% of positive epochs (B4 eligibility audit).

### 4.3 Per-seed completed value

| sim_seed | VA | MF | SPARC | Direct | B4 Lag | ITORA |
|---:|---:|---:|---:|---:|---:|---:|
| 93000 | 34.29 | 37.28 | 34.29 | 35.23 | 42.23 | 34.29 |
| 93001 | 32.36 | 32.36 | 34.35 | 36.27 | 44.91 | 39.18 |
| 93002 | 28.39 | 28.39 | 28.39 | 35.95 | 37.79 | 28.39 |
| 93003 | 31.46 | 31.69 | 42.19 | 46.61 | 48.62 | 29.01 |
| 93004 | 35.81 | 31.07 | 39.83 | 35.81 | 39.83 | 35.81 |

### 4.4 Paired contrasts (completed value)

| Contrast | mean dV | min | max | wins | ties | losses | n | Tier |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SPARC - VA | 3.35 | 0.00 | 10.73 | 3 | 2 | 0 | 5 | matched |
| SPARC - MF | 3.65 | -2.99 | 10.50 | 3 | 1 | 1 | 5 | matched |
| SPARC - Direct | -2.16 | -7.56 | 4.02 | 1 | 0 | 4 | 5 | matched |
| Direct - SPARC | 2.16 | -4.02 | 7.56 | 4 | 0 | 1 | 5 | matched |
| Direct - VA | 5.51 | 0.00 | 15.14 | 4 | 1 | 0 | 5 | matched |
| B4 - Direct | 4.71 | 1.84 | 8.65 | 5 | 0 | 0 | 5 | cross-admission |
| B4 - SPARC | 6.87 | 0.00 | 10.56 | 4 | 1 | 0 | 5 | cross-admission |
| ITORA - SPARC | -2.48 | -13.18 | 4.82 | 1 | 2 | 2 | 5 | cross-admission |

## 5. Ablations (Fig. 9 family)

| ID | Configuration | mean V | mean R | P_viol | sum offered B | Admission |
|---|---|---:|---:|---:|---:|---|
| A0 | Full SPARC (reuse nominal) | 35.81 | 0.5504 | 0.0000 | 628499 | common_governor |
| | A1 without PRR/P95 gates (byte envelope retained) | 35.81 | 0.5504 | 0.0000 | 628499 | envelope_no_gates |
| | A3 without VA-relative PPO | 34.23 | 0.5253 | 0.0000 | 624450 | common_governor |
| | A5 residual-only logits + VA-relative advantages | 38.68 | 0.5979 | 0.0000 | 625611 | common_governor |
| | A6 random residual init | 38.56 | 0.5882 | 0.0000 | 661154 | common_governor |
| A2 | = B1 VA (reuse) | 32.46 | 0.4951 | 0.0000 | 707565 | common_governor |
| A4 | = B3 Direct (reuse) | 37.97 | 0.5893 | 0.0000 | 630079 | common_governor |

Note (P1): A1 equals A0 at nominal because PRR/P95 gates never fired at d=35; removing them is a no-op under these radio conditions.

## 6. Density sweep E2 (value + safety)

Main job methods: VA, MF, SPARC, ITORA. Direct from `density_direct_d*`. **B4 Lagrangian has no density results.** ITORA runs with governor disabled even inside the density job.

| density | VA V / P_viol | MF V / P_viol | SPARC V / P_viol | Direct V / P_viol | ITORA V / P_viol |
|---:|---:|---:|---:|---:|---:|
| 25 | 23.13 / 0.118 | 22.14 / 0.118 | 29.13 / 0.022 | 29.38 / 0.110 | 25.03 / 0.126 |
| 35 | 32.46 / 0.000 | 32.16 / 0.000 | 35.81 / 0.000 | 37.97 / 0.000 | 33.33 / 0.000 |
| 50 | 24.85 / 0.400 | 25.14 / 0.400 | 28.08 / 0.400 | 31.38 / 0.400 | 40.66 / 0.400 |
| 80 | 2.97 / 0.982 | 3.09 / 0.984 | 4.26 / 0.982 | 3.08 / 0.972 | 55.83 / 0.998 |
| 100 | 45.54 / 0.380 | 44.70 / 0.392 | 53.64 / 0.390 | 57.80 / 0.386 | 64.28 / 0.398 |

At rho=80, SPARC/VA/MF/Direct collapse (V~3-4) with P_viol~0.98 under PRR gating, while ITORA reports V=55.83 with P_viol~0.998 - not a feasible win (see `reports/P0_P3_EVIDENCE_AUDIT.md` P3).

## 7. Byte-envelope sensitivity E3

Methods: VA, MF, SPARC only (Direct/B4/ITORA not in envelope sweep).

| envelope B | VA V | MF V | SPARC V | SPARC P_viol |
|---:|---:|---:|---:|---:|
| 100000 | 33.57 | 31.79 | 34.57 | 0.0000 |
| 175000 | 32.46 | 32.16 | 35.81 | 0.0000 |
| 250000 | 32.46 | 32.16 | 35.74 | 0.0000 |
| 350000 | 32.46 | 32.16 | 37.56 | 0.0000 |

## 8. Safety and governor operation (nominal)

From P0/P1 audit of `bridge_audit.json`: at d=35, PRR_min stays above 0.95 and P95_max below 50 ms for governor-matched FT - **zero PRR/P95 interventions**. Binding restriction is the **175 kB envelope** (positive-action epochs reach ~174 kB; proposed==executed). `governor_interventions` undercounts gate activity when the actor already proposes empty after `admission_allowed=False`; use `governor_gate_block_epochs` from reason codes.

Offered-byte Table III zeros were a **reporting bug** (trace key `bytes` vs `computation_bytes_offered`), not phantom completions. Backfilled from audits; V_done unchanged.

## 9. B4 / B5 eligibility notes

- **B4:** Eligibility audit verdict **INCONCLUSIVE** for replacing Direct/SPARC. Implements soft dual (kappa=0.05), not hard SPARC constraints; +4.71 V vs Direct with +26% bytes and **lower** V/kB; no density FT. Details: `reports/B4_LAGRANGIAN_ELIGIBILITY_AUDIT.md`.
- **B5:** Mapping audit at `protocol/b5_itora_mapping_audit.md`. Eval forces `gov=None` for `itora_inspired`. Report value only with P_viol.

## 10. Post-improvement SPARC (isolated; does not mutate lock)

Success: mean V > frozen Direct **37.97** and non-collapse on >=4/5 seeds; P_viol=0. Escape I* = fixed 96; Convergence C*/X* = 96 -> auto +48 -> optional last +48 with gate.

| Variant | stage | n | mean V | vs SPARC | vs Direct | VA-collapse | P_viol | success |
|---|---|---:|---:|---:|---:|---:|---:|---|
| I1 high-beta (gain=3) | done | 5 | 31.51 | -4.30 | -6.46 | 4/5 | 0.0000 | no |
| I2 residual_only | done | 5 | 37.29 | 1.48 | -0.68 | 1/5 | 0.0000 | no |
| I3 changed-only | done | 5 | 32.21 | -3.60 | -5.76 | 4/5 | 0.0000 | no |
| I4 high-beta + changed-only | done | 5 | 32.46 | -3.35 | -5.51 | 5/5 | 0.0000 | no |

Live queue: `train_c1_slow_decay_ep96`

Best completed improve so far: **I2 residual_only** V=37.29 (vs Direct -0.68; success=False).

## 11. Training-curve diagnostics

| Run | n | mean return | roll16 end | half1 | half2 | zero-ep frac |
|---|---:|---:|---:|---:|---:|---:|
| SPARC locked | 96 | 31.13 | 35.52 | 29.39 | 32.88 | 0.17 |
| Direct locked | 96 | 31.02 | 35.13 | 29.74 | 32.29 | 0.17 |
| B4 Lagrangian | 96 | 39.28 | 40.99 | 39.16 | 39.39 | 0.00 |
| I1 | 96 | 31.44 | 29.12 | 34.98 | 27.89 | 0.14 |
| I2 | 96 | 31.81 | 26.01 | 32.33 | 31.28 | 0.16 |
| I3 | 96 | 27.72 | 32.22 | 26.49 | 28.95 | 0.23 |
| I4 | 96 | 28.23 | 28.51 | 26.58 | 29.87 | 0.25 |

## 12. Table IV - complexity / admission

| Method | trainable params | admission |
|---|---:|---|
| B1_VA | 0 | common governor |
| B2_MF | 0 | common governor |
| B3_Direct | 2307 | common governor |
| B4_Lagrangian | 2307 | unshielded; physical_byte_cap=500000; dual ascent |
| B5_ITORA_inspired | 0 | byte cap only; no SPARC PRR/P95 governor |
| SPARC_A0 | 2307 | common governor |

## 13. Artifact index

| Family | Path |
|---|---|
| Nominal VA/MF/SPARC | `results/raw/final_test_nominal_d35/` |
| Direct | `results/raw/final_test_direct_d35/` |
| B4 Lagrangian | `results/raw/final_test_b4_d35/` |
| B5 ITORA | `results/raw/final_test_itora_d35/` |
| Ablations | `results/raw/final_test_a*_d35/` |
| Density (+ Direct) | `results/raw/density_d*` / `density_direct_d*` |
| Envelope | `results/raw/envelope_b*` |
| Improve | `results/raw/improve_*_d35/` + `results/wilab_improve_*` |
| This report JSON | `results/aggregates/evidence_package_report.json` |
| Offered-byte backfill log | `results/aggregates/offered_bytes_backfill.json` |

## 14. Regeneration

```text
python -u sparc_performance_evaluation/scripts/write_evidence_package_report.py
python -u sparc_performance_evaluation/scripts/write_evidence_package_report.py --watch
```

Locked V_done outcomes are never rewritten by this script. Section 10 grows as improve FTs finish; `aggregate_improve_vs_frozen.py` also refreshes this report.
