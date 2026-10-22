# Stage 0 — Diagnostic Audit Report

**Overall verdict: PASS**

Substantial exploitable scheduling opportunity under common-governor validation replay; authorizes Stage 1 implementation proposal only.

## Protocol

- Validation seeds only (`82001/82002/82011/82012`); sealed final-test **not** used.
- All five methods under **common governor**, `B_max=175 kB`, dens=35.
- Identical-candidate disagreement via offline shadow re-scoring of logged candidates.
- STOP causality: observational low value **not** treated as evidence (D5=INCONCLUSIVE by protocol).
- Locked targets retained: ≥15% matched completed-value gain, `P_viol≤2%` (evaluation gates, not Stage 0 pass criteria alone).

## Outcome summary (matched governor)

| Method | mean V | mean R | mean P_viol | mean offered bytes |
|---|---:|---:|---:|---:|
| VAS | 38.0362 | 0.5941 | 0.0000 | 8653 |
| LAS | 36.2995 | 0.5662 | 0.0000 | 8207 |
| VLRO | 39.9785 | 0.6240 | 0.0000 | 9881 |
| SPARC | 46.0571 | 0.7181 | 0.0000 | 7698 |
| LVM | 46.8173 | 0.7312 | 0.0000 | 7973 |

## Diagnostics

### D1_candidate_competition: **PASS**

Epochs with |C_t|>=2 and mean candidate cardinality on VAS reference trajectories.

```json
{
  "P_ge2_candidates": 0.9032258064516129,
  "mean_n_candidates": 24.419354838709676,
  "n_epochs": 31
}
```

### D2_assignment_disagreement: **PASS**

Identical-candidate shadow greedy sets: SPARC vs VAS/LAS/VLRO/LVM.

```json
{
  "disagree_vs": {
    "VAS": 0.8709677419354839,
    "LAS": 0.8709677419354839,
    "VLRO": 0.8709677419354839,
    "LVM": 0.8387096774193549
  },
  "mean_vs_others": 0.8629032258064516,
  "n_shadow_epochs": 31
}
```

### D2b_materially_different_choices: **PASS**

Epochs with ≥2 materially different feasible schedules (not merely ≥2 candidates).

```json
{
  "P_materially_diverse": 0.9032258064516129,
  "n": 31
}
```

### D3_governor_override: **PASS**

Per-algorithm governor override rate and completed-value share from fallback actions.

```json
{
  "intervene_rate_by_method": {
    "VAS": 0.0,
    "LAS": 0.0,
    "VLRO": 0.0,
    "SPARC": 0.0,
    "LVM": 0.0
  },
  "fallback_value_share_by_method": {
    "VAS": 0.0,
    "LAS": 0.0,
    "VLRO": 0.0,
    "SPARC": 0.0,
    "LVM": 0.0
  }
}
```

### D4_deadline_uncertainty: **PASS**

Observational completion/miss among tasks; residual deadline outcome uncertainty.

```json
{
  "n_tasks": 84,
  "n_completed": 48,
  "n_deadline_miss": 12,
  "by_status": {
    "incomplete": 24,
    "completed": 48,
    "deadline_miss": 12
  },
  "completion_rate": 0.5714285714285714,
  "deadline_miss_rate": 0.14285714285714285
}
```

### D5_STOP_opportunity: **INCONCLUSIVE**

Protocol: low realized value alone does not prove STOP would help. Marginal admission effects require controlled paired replay.

```json
{
  "observational_low_value_admits": "not_used_as_causal_evidence"
}
```

### D6_helper_competition: **PASS**

Epochs with non-idle projected helper queue / finish state among candidates.

```json
{
  "P_helper_busy": 0.25806451612903225,
  "n_epochs": 31
}
```

### D7_sparc_vs_static_priority: **INCONCLUSIVE**

PASS requires both (i) assignments differ from VAS static ranking and (ii) ablating radio/deadline/helper features changes ≥10% of decisions.

```json
{
  "frac_assign_diff_vs_VAS": 0.8709677419354839,
  "feature_ablation_sensitivity": {
    "frac_changed_ablate_cbr": 0.0,
    "frac_changed_ablate_deadline": 0.032258064516129004,
    "frac_changed_ablate_helper_queue": 0.0,
    "n_ablate_epochs": 31
  },
  "uses_radio_deadline_helper_channels": false
}
```

## Authorization

Stage 0 PASS → preparation of a Stage 1 implementation proposal is authorized.

No architecture revision, PPO retraining, final-test access, or manuscript rewrite is authorized by this report.
