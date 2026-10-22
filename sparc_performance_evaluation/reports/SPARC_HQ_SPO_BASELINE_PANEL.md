# HQ-SPO baseline panel (B1–B5)

**Seeds:** `73100–73107` (tps4, test)
**Elapsed:** 1655s
**Best (feasibility then U):** `B5_ppo_lag`

Note: B5 uses 1-ep smoke checkpoint — undertrained vs full protocol budget

| method | mean V | mean P_viol | frac P_viol≤0.02 | mean wall_s |
|---|---:|---:|---:|---:|
| B1_greedy_vb | 32.77 | 0.250 | 0.75 | 42.3 |
| B2_cbr_threshold | 32.77 | 0.250 | 0.75 | 41.3 |
| B3_myopic_dual | 30.78 | 0.250 | 0.75 | 41.6 |
| B4_ce_milp | 31.00 | 0.250 | 0.75 | 39.1 |
| B5_ppo_lag | 39.91 | 0.250 | 0.75 | 42.3 |

| seed | B1_greedy_vb | B2_cbr_threshold | B3_myopic_dual | B4_ce_milp | B5_ppo_lag |
|---:|---:|---:|---:|---:|---:|
| 73100 | 32.16 | 32.16 | 32.00 | 32.00 | 45.24 |
| 73101 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 73102 | 48.33 | 48.33 | 46.17 | 46.17 | 53.98 |
| 73103 | 49.54 | 49.54 | 43.14 | 45.88 | 67.05 |
| 73104 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 73105 | 44.46 | 44.46 | 46.06 | 46.06 | 52.12 |
| 73106 | 36.81 | 36.81 | 35.23 | 35.41 | 45.47 |
| 73107 | 50.82 | 50.82 | 43.63 | 42.45 | 55.39 |
