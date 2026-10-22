# Fig. 3 — training / validation (historical)

**Status:** HISTORICAL — not final-test. Direct PPO curve pending.

## Panel (a) validation mean ΔV_done (SPARC − VA)

| update | V_res | V_VA | ΔV |
|---:|---:|---:|---:|
| 0 | 38.036 | 38.036 | +0.000 |
| 2 | 36.948 | 38.036 | -1.088 |
| 4 | 38.779 | 38.036 | +0.743 |
| 6 | 39.393 | 38.036 | +1.357 |
| 8 | 41.082 | 38.036 | +3.046 |
| 10 | 40.420 | 38.036 | +2.384 |
| 12 | 44.338 | 38.036 | +6.302 |
| 14 | 46.817 | 38.036 | +8.781 |
| 16 | 45.729 | 38.036 | +7.693 |
| 18 | 46.817 | 38.036 | +8.781 |
| 20 | 46.204 | 38.036 | +8.168 |
| 22 | 45.589 | 38.036 | +7.553 |
| 24 | 45.589 | 38.036 | +7.553 |

Panel (b): 96 training episodes archived in `fig03_data.json` (returns, rolling-48, deviation, KL, entropy).
