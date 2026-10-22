# B5′ panel vs B5 (PB-CRL v3.1 foundations)

**B5′:** init weights on shared price-and-budget packer (not trained PPO yet).
**B5:** panel pack-logit values (locked).
**Seeds:** `73100–73107`
**Elapsed:** 734s

| | mean V | frac C2 (Z≤2) | frac P_viol≤0.02 |
|---|---:|---:|---:|
| B5′ init | 30.34 | 0.75 | 0.75 |
| B5 panel | 39.91 | (see audits) | 0.75 |

**mean ΔV (B5′−B5):** -9.57
**C2 cell CP UB P(Z>K):** 0.5996893891908329

| seed | B5′ V | B5 V | Δ | Z | C2_ok | P_viol |
|---:|---:|---:|---:|---:|---|---:|
| 73100 | 38.49 | 45.24 | -6.75 | 0 | True | 0.0 |
| 73101 | 0.00 | 0.00 | +0.00 | 100 | False | 1.0 |
| 73102 | 37.79 | 53.98 | -16.19 | 0 | True | 0.0 |
| 73103 | 37.76 | 67.05 | -29.29 | 0 | True | 0.0 |
| 73104 | 0.00 | 0.00 | +0.00 | 100 | False | 1.0 |
| 73105 | 48.00 | 52.12 | -4.12 | 0 | True | 0.0 |
| 73106 | 34.83 | 45.47 | -10.64 | 0 | True | 0.0 |
| 73107 | 45.83 | 55.39 | -9.56 | 0 | True | 0.0 |

## Headroom H_s = U(K+dK)−U(K) (hard pathwise)

| cell | U_K | U_K+dK | H_s | note |
|---:|---:|---:|---:|---|
| 73100 | 51.32 | 51.32 | +0.00 | ok |
| 73102 | 37.28 | 37.28 | +0.00 | ok |
| 73103 | 40.80 | 40.80 | +0.00 | ok |
| 73105 | 34.08 | 34.08 | +0.00 | ok |
