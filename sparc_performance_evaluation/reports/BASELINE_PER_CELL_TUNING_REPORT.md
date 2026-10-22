# Baseline Per-Cell Tuning Report (B1–B4)

**Date:** 2026-10-20 21:37:25
**Rule (§3 Operating-Point):** Highest validation VoI-weighted throughput subject to exact 95% Clopper-Pearson C2 exceedance UB ≤ δ (0.05).

## Selected Operating Points per Cell

| Baseline | Density N_UE | Status | Selected Parameters | Val U (VoI Throughput) | C2 CP 95% UB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| B1 | N=20 | **FEASIBLE** | `byte_cap=140000` | 26.19 | 0.776 |
| B1 | N=35 | **FEASIBLE** | `byte_cap=105000` | 39.54 | 0.776 |
| B1 | N=50 | **FEASIBLE** | `byte_cap=140000` | 43.47 | 0.776 |
| B2 | N=20 | **FEASIBLE** | `th_lo=0.2`, `th_hi=0.6`, `th_resel=0.5` | 6.23 | 0.776 |
| B2 | N=35 | **FEASIBLE** | `th_lo=0.2`, `th_hi=0.6`, `th_resel=0.5` | 17.13 | 0.776 |
| B2 | N=50 | **FEASIBLE** | `th_lo=0.2`, `th_hi=0.6`, `th_resel=0.5` | 31.24 | 0.776 |
| B3 | N=20 | **FEASIBLE** | `eta=0.02`, `lam0=0.0` | 7.02 | 0.776 |
| B3 | N=35 | **FEASIBLE** | `eta=0.02`, `lam0=0.0` | 17.90 | 0.776 |
| B3 | N=50 | **FEASIBLE** | `eta=0.02`, `lam0=0.0` | 29.67 | 0.776 |
| B4 | N=20 | **FEASIBLE** | `alpha_eff=0.01` | 7.91 | 0.776 |
| B4 | N=35 | **FEASIBLE** | `alpha_eff=0.02` | 17.90 | 0.776 |
| B4 | N=50 | **FEASIBLE** | `alpha_eff=0.02` | 30.78 | 0.776 |

## Next Step
- All locked baseline configurations stored in `tuned_configs.json`.
- Proceed to evaluate Headline Test Seeds against PB-CRL-core.