# Direct PPO vs LVM — Value Deficit & Decision Opportunity Audit

**Training-free. Stage 1 remains closed. Final-test sealed. No implementation authorized.**

Direct mean V=46.057 vs LVM 46.817 (Δ=-0.760). Deficit vs LVM is exactly Direct-only (−) LVM-only completions: gained 0.788/seed, lost 1.548/seed. Dominant Direct failure on LVM-only tasks: never_assigned_by_direct (0.935/seed). Opportunity class: coverage. Same-state pack disagreement 84%. No implementation/training/final-test authorized.

## Five-method matched validation (locked seeds)

| Method | mean V | mean R | P_viol |
|---|---:|---:|---:|
| VAS | 38.0362 | 0.5941 | 0.0000 |
| LAS | 36.2995 | 0.5662 | 0.0000 |
| VLRO | 39.9785 | 0.6240 | 0.0000 |
| SPARC_Direct | 46.0571 | 0.7181 | 0.0000 |
| LVM | 46.8173 | 0.7312 | 0.0000 |

## Direct vs LVM: mean ΔV (Direct − LVM) = -0.760

```
ΔV = Direct-only completions − LVM-only completions
   = 0.788 − 1.548 = -0.760
```

## Per-seed

| Seed | V Direct | V LVM | ΔV | #assign D/L | Direct-only V | LVM-only V |
|---|---:|---:|---:|---:|---:|---:|
| 82001 | 39.84 | 38.26 | 1.58 | 12/11 | 1.58 | 0.00 |
| 82002 | 49.95 | 50.84 | -0.88 | 17/17 | 1.57 | 2.45 |
| 82011 | 58.69 | 58.69 | 0.00 | 20/20 | 0.00 | 0.00 |
| 82012 | 35.74 | 39.48 | -3.74 | 12/13 | 0.00 | 3.74 |

## Where LVM-only completions fail under Direct (mean value/seed)

| Pathway | n (total) | value/seed | share of LVM-only value |
|---|---:|---:|---:|
| never_assigned_by_direct | 1 | 0.935 | 60.4% |
| assigned_deadline_miss_direct | 1 | 0.613 | 39.6% |
| assigned_incomplete_direct | 0 | 0.000 | 0.0% |
| assigned_incomplete_no_radio_direct | 0 | 0.000 | 0.0% |
| helper_diff_among_lvm_only | 0 | 0.000 | 0.0% |
| other | 0 | 0.000 | 0.0% |

## Admission / governor

```json
{
  "admission": {
    "mean_n_assigned_direct": 15.25,
    "mean_n_assigned_lvm": 15.25,
    "delta_n_assigned": 0.0,
    "delta_payload": 10169.0
  },
  "governor": {
    "mean_intervene_direct": 0.0,
    "mean_intervene_lvm": 0.0
  }
}
```

## Same-state shadow (Direct trajectory candidates)

- Epochs: 25
- First-pick disagree: 80.0%
- Pack disagree: 84.0%
- Mean pack task-value Δ (LVM − Direct): 1.751
- Epochs LVM pack includes a task Direct omits: 11

## Decision opportunity (preserve Direct wins)

```json
{
  "mean_value_lvm_only_never_assigned_by_direct": 0.9345275994674789,
  "mean_value_lvm_only_deadline_miss_by_direct": 0.6134831551908894,
  "mean_value_lvm_only_incomplete_by_direct": 0.0,
  "mean_delta_n_assigned_direct_minus_lvm": 0.0,
  "seeds_under_assignment_vs_lvm": [
    82012
  ],
  "seeds_quality_failure_on_assigned": [
    82002
  ],
  "preserve_existing_wins": {
    "mean_value_direct_only_completions": 0.7877761936523849,
    "note": "Any fix must not erase Direct-only completions; report their value as the floor that must be preserved."
  },
  "opportunity_hypothesis": "coverage"
}
```

## Credible improvement opportunity (diagnostic only)

```json
{
  "identified": true,
  "gap_vs_lvm": 0.7602345610059871,
  "dominant_loss_pathway": "never_assigned_by_direct",
  "opportunity_class": "coverage",
  "must_preserve_direct_only_value_per_seed": 0.7877761936523849,
  "shadow_disagreement": 0.84,
  "implementation_authorized": false,
  "note": "A credible opportunity exists only as a *diagnostic finding*. Implementation requires a separate proposal with causal claim, cheap prerequisite test, and validation criterion. This audit does not authorize training or redesign."
}
```

Next step if pursued: a **separately justified** proposal naming one targeted modification, a cheap prerequisite test, and a validation criterion — not Stage-1 reopen, not STOP, not an unguided new PPO architecture.
