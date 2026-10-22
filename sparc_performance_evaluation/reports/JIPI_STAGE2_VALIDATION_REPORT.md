# JIPI-SPARC Stage-2 Validation Report

**Outcome:** `NO_IMPROVEMENT_OVER_DIRECT`
**Checkpoint:** `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\checkpoints\wilab_jipi_ppo_96ep.pt`
**Seeds:** [82001, 82002, 82011, 82012]

## Primary metrics

| Metric | JIPI mean | 95% CI |
|---|---:|---|
| $V_{\mathrm{done}}$ | 46.0571 | (29.6270, 62.4872) |
| $R_{\mathrm{value}}$ | 0.7181 | (0.4639, 0.9724) |
| $P_{\mathrm{viol}}$ | 0.0000 | (0.0000, 0.0000) |

Strongest matched competitor: **LVM** at 46.8173.
Gain vs best matched: **-1.62%** (15% target V ≈ 53.84).

## Paired ΔV by method

| Method | mean V | paired mean ΔV (JIPI−base) |
|---|---:|---:|
| VAS | 38.0362 | +8.0209 |
| LAS | 36.2995 | +9.7576 |
| VLRO | 39.9785 | +6.0786 |
| Direct | 46.0571 | +0.0000 |
| LVM | 46.8173 | -0.7602 |

## Edit frequencies (validation)

- KEEP: 0.998
- rebind: 0.003
- substitute: 0.000
- mean / max controller ms: 1.80 / 11.18

## Lost / gained completions vs Direct and LVM

### vs Direct
- seed 82001: gained 0 [], lost 0 []
- seed 82002: gained 0 [], lost 0 []
- seed 82011: gained 0 [], lost 0 []
- seed 82012: gained 0 [], lost 0 []

### vs LVM
- seed 82001: gained 1 [0], lost 0 []
- seed 82002: gained 1 [1], lost 1 [19]
- seed 82011: gained 0 [], lost 0 []
- seed 82012: gained 0 [], lost 1 [1]

## Per-seed JIPI

| Seed | V_done | R_value | P_viol | KEEP | rebind | sub | max ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 82001 | 39.8412 | 0.5927 | 0.0000 | 1.000 | 0.000 | 0.000 | 11.18 |
| 82002 | 49.9532 | 0.7519 | 0.0000 | 0.990 | 0.010 | 0.000 | 5.16 |
| 82011 | 58.6936 | 0.9307 | 0.0000 | 1.000 | 0.000 | 0.000 | 7.72 |
| 82012 | 35.7404 | 0.5973 | 0.0000 | 1.000 | 0.000 | 0.000 | 7.88 |

## Authorization boundary

One training run + one locked validation. No architecture revision, HPO, extra training, manuscript promotion, or final-test access.
