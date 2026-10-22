# SPARC-V2X seed / budget partition (manuscript)

Reduced design: keep statistical pairing without an unnecessarily large final-test panel.

| Partition | Seeds | Purpose |
|---|---|---|
| **Training** | 1 training run per learned configuration | Train policies (historical SPARC: sim=`72000+ep`, task=`182000+ep`, 96 episodes). Same budget rule for B3/A3/A5/A6/B4 unless stated. |
| **Validation** | **4** paired scenarios (already implemented) | Checkpoint and hyperparameter selection only. Sims `[82001,82002,82011,82012]` with tasks `[182001,182002,182011,182012]`. |
| **Final evaluation** | **5** paired seeds per operating point | All principal comparisons, ablations, and sensitivity experiments. Sealed in `final_test_seeds.json`. |
| **Calibration** | Reuse existing calibration data | No new calibration campaign required for lock. |

## Rules

1. Train / validation / final-test pools are **disjoint**.
2. Checkpoint selection uses **validation only** — never final-test.
3. Every operating point (density, envelope, workload, deadline multiplier, ablation, etc.) uses the **same five** final-test pairs unless a condition-specific extension is predeclared.
4. With a single training run per learned config, manuscript claims are **conditional on the selected checkpoint**; do not over-claim training-seed variability.
5. Older “30 final scenarios / 5 training seeds” figures are **superseded** by this partition.

## Final-test seal (n=5)

- Sims: `93000`–`93004`
- Tasks: `193000`–`193004`
- File: `protocol/final_test_seeds.json`
