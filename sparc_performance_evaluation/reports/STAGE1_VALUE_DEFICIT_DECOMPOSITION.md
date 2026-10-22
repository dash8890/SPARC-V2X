# Stage 1 Value Deficit Decomposition

**Stage 1 remains closed. Prior Direct retained. Final-test sealed.**

Mean ΔV (Stage-1 − Direct) = -5.010 is fully accounted for by asymmetric task completions: Stage-1 loses 5.010 value/seed on tasks Direct completes and only gains 0.000 on tasks Direct misses (net -5.010; reconcile err 5.440e-15). Dominant pathway for Direct-only completions under Stage-1: never_assigned_by_stage1 (2.641 value/seed). Admission: Δn_assigned=-0.75, Δtotal_payload=-21106, Δmean_payload/assign=715. Governor remains non-implicated. Stage 1 stays closed; prior Direct retained; final-test sealed.

## Headline: mean ΔV = -5.010 (deficit 5.010)

Identity (exact on this panel):

```
ΔV = value(Stage-1-only completions) − value(Direct-only completions)
   = 0.000 − 5.010 = -5.010
```

## Per-seed

| Seed | V Direct | V Stage-1 | ΔV | #assign D/S | payload D/S | Direct-only V | Stage-1-only V |
|---|---:|---:|---:|---:|---:|---:|---:|
| 82001 | 39.84 | 39.84 | 0.00 | 12/12 | 523503/523503 | 0.00 | 0.00 |
| 82002 | 49.95 | 36.68 | -13.27 | 17/15 | 740375/673872 | 13.27 | 0.00 |
| 82011 | 58.69 | 56.05 | -2.65 | 20/19 | 923376/905453 | 2.65 | 0.00 |
| 82012 | 35.74 | 31.62 | -4.12 | 12/12 | 497357/497357 | 4.12 | 0.00 |

## Where Direct-only completions go under Stage-1 (mean value/seed)

| Pathway | n (total) | value/seed | share of lost value |
|---|---:|---:|---:|
| never_assigned_by_stage1 | 3 | 2.641 | 52.7% |
| assigned_deadline_miss_stage1 | 3 | 2.369 | 47.3% |
| assigned_incomplete_stage1 | 0 | 0.000 | 0.0% |
| assigned_incomplete_no_radio_stage1 | 0 | 0.000 | 0.0% |
| other | 0 | 0.000 | 0.0% |

## Admission volume vs composition

```json
{
  "mean_n_assigned_direct": 15.25,
  "mean_n_assigned_stage1": 14.5,
  "delta_n_assigned_stage1_minus_direct": -0.75,
  "mean_payload_assigned_direct": 671152.75,
  "mean_payload_assigned_stage1": 650046.25,
  "delta_payload_assigned": -21106.5,
  "delta_mean_payload_per_assignment": 714.9876160990752,
  "mean_value_assigned_direct": 46.67056612600871,
  "mean_value_assigned_stage1": 44.02999454829438,
  "interpretation": "Lower offered bytes: fewer assignments; mean payload per assignment not much smaller; lower total assigned payload"
}
```

## Status counts (mean over seeds)

- Direct: completed=15.00, deadline_miss=0.25, incomplete=5.75
- Stage-1: completed=13.50, deadline_miss=1.00, incomplete=6.50

No retraining, architecture modification, or manuscript promotion authorized.
