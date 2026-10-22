# POPL-SPARC — P5 Learning Pilot PASS/FAIL Report

**Overall:** `PASS`
**Authorization:** one training-only pilot ≤64 pairs; locked PPO rule.
**Pairs executed:** 64

## Success criteria

| Check | Status | Evidence |
|---|---|---|
| Mathematical correctness | `PASS` | logπ invariants pass=100%; ratio invariants pass=100% |
| Optimization | `PASS` | finite_losses=True; nondegenerate=True |
| Outcome learning | `PASS` | init mean ΔG=0.31963471248910835; learned mean ΔG=1.1132258628128375; CI=0.10997283848496409,2.387758018582801 |
| Decision mechanism | `PASS` | Δlogπ beneficial=0.018332345145089284; harmful=-0.05810056413922991 |
| Feasibility | `PASS` | all_alt_feasible=True; matched_admission_blocks=12 |
| Runtime | `PASS` | total=4943.8s; mean pair=77.0s |

## Initial vs learned (train-only paired ΔG)

- Initial-policy pairs (first batch): n=8, mean ΔG=0.31963471248910835, 95% CI [0.0, 0.9589041374673251]
- Learned-policy pairs (last batch): n=8, mean ΔG=1.1132258628128375, 95% CI [0.10997283848496409, 2.387758018582801]
- Improvement (learned − init mean ΔG): 0.7935911503237292 (bootstrap 95% CI [-0.41932374800828853, 2.2014266665704927])

## Learning-rule invariants

- Sequential logπ (+TERMINATE) checks: 15/15
- Same-pack PPO ratio checks: 15/15

## Feasibility note

All 64 sampled alt packs satisfied packing constraints (\(F_t\): 175 kB, unique sources, deadline). Twelve pairs hit matched `prr_below_floor` admission blocks (both branches empty at fork; \(\Delta G=0\)); these are governor radio-state gates, not packing violations.

## Computational cost

- Paired rollouts: **64** (cap 64)
- Mean pair wall: **77.01 s**
- Total wall (pairs + updates): **4943.8 s**
- PPO updates: 8

## Boundaries respected

- Train seeds only (`72000+` / `182000+`); no 82001/02/11/12.
- No automatic extensions, restarts, or hyperparameter searches.
- Governor, 175 kB envelope, reward, simulator, frozen Direct preserved.
- Stage 1 / JIPI / RHO remain closed; Direct PPO remains SPARC reference.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\popl_p5_pilot`
