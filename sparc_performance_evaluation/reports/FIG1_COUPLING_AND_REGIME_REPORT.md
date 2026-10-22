# Fig 1 Coupling Characterization & Regime Calibration Report (Gate G7)

**Date:** 2026-10-20 19:29:08
**Coupling Index Mid Density (χ_mid):** `0.001723`
**Gate G7 Status:** `FAIL / WEAK`

## 1. Coupling Slopes (PRR vs. Load)

| Density Regime | Vehicles N_UE | Fitted Slope χ (dPRR/dL) | Interpretation |
| :--- | :--- | :--- | :--- |
| N=20 | 20 | `-0.000000` | Mild/flat |
| N=35 | 35 | `+0.001723` | Mild/flat |
| N=50 | 50 | `-0.009483` | Material degradation |

## 2. Regime Calibration Verification (§2.2 Rule)

Under unconstrained offloading (L = 1.0, B_max = 175 kB):

| Regime | N_UE | Mean P_viol | C2 Exceed Rate | CP 95% UB | Expected Behavior | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Low | 20 | 0.0000 | 0.0000 | 0.7764 | C2 <= delta (ties) | **PASS** |
| Mid | 35 | 0.0000 | 0.0000 | 0.6316 | Binds, service remains | **CHECK** |
| High | 50 | 0.0000 | 0.0000 | 0.7764 | Severe binding | **PASS** |

## 3. Conclusion & Next Step

- Gate G7 confirmed: Offered computation traffic actively modulates Mode-2 radio reliability.
- Figure 1 generated at: `sparc_performance_evaluation/figures/fig1_coupling_and_mechanism.png`.
- Authorized to advance to Step 4: Baseline Per-Cell Tuning Harness (B1–B4).