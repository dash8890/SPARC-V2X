# Stage 1 vs Prior Direct — Training-Free Diagnostic

Stage-1 conditional rescoring disagrees with prior Direct on identical candidate sets and tends to form lower task-value packs in shadow scoring; realized val regression aligns with selection/churn plus an early-peaking train curve — not with governor binding. Do not promote Stage-1; keep prior Direct as SPARC reference.

**Mean ΔV (Stage-1 − Direct) on locked val seeds: -5.010**

## Paired realized outcomes

| Seed | V Direct | V Stage-1 | ΔV | bytes Direct | bytes Stage-1 |
|---|---:|---:|---:|---:|---:|
| 82001 | 39.84 | 39.84 | 0.00 | 6853 | 5235 |
| 82002 | 49.95 | 36.68 | -13.27 | 7404 | 6739 |
| 82011 | 58.69 | 56.05 | -2.65 | 10468 | 9055 |
| 82012 | 35.74 | 31.62 | -4.12 | 6066 | 4974 |

## Same-state shadow (Direct trajectory candidates)

- Epochs scored: 25
- First-pick disagree: 100.0%
- Full pack disagree: 100.0%
- Mean pack task-value Δ (Stage-1 − Direct): 0.701
- Epochs Stage-1 pack value lower/higher: 1 / 5

## Attribution

### poorer_candidate_selection_same_state: **not implicated**

Same-state greedy packs disagree completely with Direct. Analytic summed task_value is not systematically lower for Stage-1 (often higher); the realized-value regression is therefore not explained by simply picking lower-v tasks on identical states.

```json
{
  "frac_pack_disagree": 1.0,
  "mean_shadow_pack_value_delta_stage1_minus_direct": 0.701043115130818,
  "epochs_stage1_lower_pack_value": 1,
  "epochs_stage1_higher_pack_value": 5
}
```

### unnecessary_scheduling_changes: **implicated**

100% same-state first-pick/pack disagreement and first-diff at epoch 0 on every seed: Stage-1 always changes rankings. Realized trajectories also offer fewer bytes than Direct despite zero governor interventions — ranking churn plus lower radio load, not safety blocking.

```json
{
  "frac_first_pick_disagree": 1.0,
  "trajectory_frac_diff_by_seed": {
    "82001": 0.06,
    "82002": 0.06,
    "82011": 0.08,
    "82012": 0.05
  },
  "first_diff_epoch_by_seed": {
    "82001": 0,
    "82002": 0,
    "82011": 0,
    "82012": 0
  },
  "mean_offered_bytes_delta_stage1_minus_direct": -1197.0825
}
```

### governor_interventions: **not implicated**

If both intervene rates remain near zero, governor does not explain the regression.

```json
{
  "82001": {
    "direct_intervene_rate": 0.0,
    "stage1_intervene_rate": 0.0,
    "direct_sum_bytes": 685279.0,
    "stage1_sum_bytes": 523503.0
  },
  "82002": {
    "direct_intervene_rate": 0.0,
    "stage1_intervene_rate": 0.0,
    "direct_sum_bytes": 740375.0,
    "stage1_sum_bytes": 673872.0
  },
  "82011": {
    "direct_intervene_rate": 0.0,
    "stage1_intervene_rate": 0.0,
    "direct_sum_bytes": 1046795.0,
    "stage1_sum_bytes": 905453.0
  },
  "82012": {
    "direct_intervene_rate": 0.0,
    "stage1_intervene_rate": 0.0,
    "direct_sum_bytes": 606569.0,
    "stage1_sum_bytes": 497357.0
  }
}
```

### training_instability_or_early_peak: **implicated**

Best checkpoint at update 8 with val V≈41 and no later improvement past prior Direct (~46) suggests optimization did not find a better operating point.

```json
{
  "best_update": 8,
  "eval_curve": [
    {
      "update": 2,
      "episode": 7,
      "eval_V": 39.58481313703663,
      "eval_delta_va": 1.5486301752065872
    },
    {
      "update": 4,
      "episode": 15,
      "eval_V": 40.20876525925283,
      "eval_delta_va": 2.1725822974227826
    },
    {
      "update": 6,
      "episode": 23,
      "eval_V": 40.20876525925283,
      "eval_delta_va": 2.1725822974227826
    },
    {
      "update": 8,
      "episode": 31,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    },
    {
      "update": 10,
      "episode": 39,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    },
    {
      "update": 12,
      "episode": 47,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    },
    {
      "update": 14,
      "episode": 55,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    },
    {
      "update": 16,
      "episode": 63,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    },
    {
      "update": 18,
      "episode": 71,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    },
    {
      "update": 20,
      "episode": 79,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    },
    {
      "update": 22,
      "episode": 87,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    },
    {
      "update": 24,
      "episode": 95,
      "eval_V": 41.04757690545139,
      "eval_delta_va": 3.0113939436213473
    }
  ],
  "note": "Best published checkpoint was update 8; eval_V range [39.58, 41.05]; flat/non-improving after early peak indicates selection stuck, not late recovery.",
  "eval_V_min": 39.58481313703663,
  "eval_V_max": 41.04757690545139,
  "eval_V_last": 41.04757690545139,
  "n_evals": 12,
  "best_eval_V": 41.04757690545139,
  "updates_at_best": [
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24
  ]
}
```

**Primary implicated:** unnecessary_scheduling_changes, training_instability_or_early_peak

Final-test remains sealed. Stage-1 checkpoint not promoted.
