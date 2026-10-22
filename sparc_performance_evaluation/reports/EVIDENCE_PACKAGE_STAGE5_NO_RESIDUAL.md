# Stage 5 Full Package — residual RL excluded

**Generated (UTC):** `2026-10-15T16:05:57.305590+00:00`
**Status:** `STAGE5_DATA_PARTIAL_NO_RESIDUAL`
**MATLAB / campaigns:** offline rebuild from locked + midnight catch-up artifacts (no new jobs started by this report script).
**PNG/PDF figures:** not rendered; **Fig.3–11 numerical data are embedded below.**

## Completeness ledger (read this first)

### Available now
- Table II/III/IV (no residual)
- Fig.3 Direct train/eval series
- Fig.4 density VA/MF/ITORA + Direct + B4 (25/35/50/80/100)
- Fig.5 operating points + VA/MF/Direct/B4/ITORA envelope
- Fig.6 E4 load + deadline sweeps (VA/MF/Direct)
- Fig.7a/b PRR/P95 vs density; Fig.7c delay hist; Fig.7d PRR-vs-distance + Direct@80 stress
- Fig.8 Direct closed-loop trace
- Fig.9 A2/A4/MF only (residual ablations excluded)
- Fig.10a/b offered bytes + governor reasons; Fig.10d candidate-set k sweep
- Fig.11 Direct density + full envelope (Fig.11a specialized pending)
- Additional: Governor ON vs OFF (when catchup_gov_* present)
- Additional: Iso-safety admission match V(baseline,E*) vs V(Direct,E_D) (when ANALYSIS present)
- Additional: Governor ON vs OFF (complete)
- Additional: Iso-safety admission (DATA_ONLY)

### Explicitly unavailable without a new campaign (not hidden)
- Fig.10c inference latency P50/P95 (not logged)
- Fig.11a condition-specialized Direct@50 (train/FT in midnight queue)
- Density levels 20/30/40/45/55/60 (rest branch in progress)

| Figure | data file | status |
|---|---|---|
| Fig.3 | `results/stage5_no_residual/figures/fig03_data.json` | DATA_ONLY |
| Fig.4 | `results/stage5_no_residual/figures/fig04_data.json` | DATA_ONLY |
| Fig.5 | `results/stage5_no_residual/figures/fig05_data.json` | DATA_ONLY |
| Fig.6 | `results/stage5_no_residual/figures/fig06_data.json` | DATA_ONLY |
| Fig.7 | `results/stage5_no_residual/figures/fig07_data.json` | DATA_ONLY |
| Fig.8 | `results/stage5_no_residual/figures/fig08_data.json` | DATA_ONLY |
| Fig.9 | `results/stage5_no_residual/figures/fig09_data.json` | DATA_ONLY_RESTRICTED |
| Fig.10 | `results/stage5_no_residual/figures/fig10_data.json` | DATA_ONLY_PARTIAL |
| Fig.11 | `results/stage5_no_residual/figures/fig11_data.json` | DATA_ONLY |
| Addl.GovernorONOFF | `results/stage5_no_residual/figures/addl_governor_on_off.json` | DATA_ONLY |
| Addl.IsoSafetyAdmission | `results/stage5_no_residual/figures/addl_iso_safety_admission.json` | DATA_ONLY |

## Scope

| Included | Excluded |
|---|---|
| B1 VA, B2 MF, B3 Direct, B4 Lagrangian, B5 ITORA | SPARC residual RL FT/train |
| A2(=VA), A4(=Direct) reuse | A0/A1/A3/A5/A6 residual-family ablations |
| Density Direct + VA/MF/ITORA | Residual density series |
| — | Improve residual variants |

Note: `final_test_direct_d35` rows carry `method="residual"` in the raw schema (Direct FT provenance quirk); treated as B3 Direct throughout this package.

## Claim-to-evidence audit

With SPARC residual excluded: does governor-matched Direct PPO yield higher deadline-valid completed value than the strongest safety-feasible non-learned comparator (VA/MF) on sealed final-test?

- Direct mean V=37.9711, VA=32.4616, MF=32.1578; all matched P_viol=0: True
- Paired Direct-VA dV mean=5.5094 CI95=[1.1562, 10.6487] (n=5)

Non-claims:
- Do not claim Direct beats B4 under matched admission (B4 V=42.67657849189164 uses 500 kB unshielded).
- Do not claim SPARC residual superiority (excluded from this package).
- Do not treat ITORA density V wins as feasible without P_viol.

## Table II

| Parameter | Value |
|---|---|
| density | 35 |
| envelope_bytes | 175000 |
| episodes | 96 |
| controller_epochs | 100 |
| simulation_time | 12.0 |
| safety_payload_bytes | 300 |
| lr | 0.0003 |
| logit_scale | 100.0 |
| target_kl | 0.015 |

SPARC residual training config retained for provenance; primary learned method in this Stage-5 package is B3 Direct PPO.

## Table III (no residual RL)

| Method | tier | V_done | R_value | sum offered B | P_viol | PRR | P95 | invalid | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1_VA | governor_matched | 32.4616 | 0.4951 | 707565 | 0.0000 | 0.9841 | 23.56 | 0.0000 | 5 |
| B2_MF | governor_matched | 32.1578 | 0.4908 | 695526 | 0.0000 | 0.9846 | 23.46 | 0.0000 | 5 |
| B3_Direct | governor_matched | 37.9711 | 0.5893 | 630079 | 0.0000 | 0.9853 | 21.61 | 0.0000 | 5 |
| B4_Lagrangian | unshielded | 42.6766 | 0.6578 | 791446 | 0.0000 | 0.9829 | 24.89 | 0.0000 | 5 |
| B5_ITORA | unshielded | 33.3346 | 0.5062 | 732121 | 0.0000 | 0.9846 | 22.31 | 0.0000 | 5 |

Strongest governor-matched: **B3_Direct**

### By seed (V)

| sim_seed | VA | MF | Direct | B4 | ITORA |
|---:|---:|---:|---:|---:|---:|
| 93000 | 34.29 | 37.28 | 35.23 | 42.23 | 34.29 |
| 93001 | 32.36 | 32.36 | 36.27 | 44.91 | 39.18 |
| 93002 | 28.39 | 28.39 | 35.95 | 37.79 | 28.39 |
| 93003 | 31.46 | 31.69 | 46.61 | 48.62 | 29.01 |
| 93004 | 35.81 | 31.07 | 35.81 | 39.83 | 35.81 |

### Simultaneous primary contrasts (paired, n=5 seeds)

| Contrast | tier | dV mean | CI95 lo | CI95 hi | W/T/L |
|---|---|---:|---:|---:|---:|
| direct_minus_va | governor_matched | 5.5094 | 1.1562 | 10.6487 | 4/1/0 |
| direct_minus_mf | governor_matched | 5.8133 | 1.2281 | 11.2426 | 4/0/1 |
| mf_minus_va | governor_matched | -0.3038 | -2.7955 | 1.7928 | 2/2/1 |
| b4_minus_direct | cross_admission | 4.7055 | 2.3477 | 7.0633 | 5/0/0 |
| itora_minus_direct | cross_admission | -4.6364 | -11.5748 | 1.1646 | 1/1/3 |

## Table IV

| Method | trainable | admission | train_episodes |
|---|---:|---|---:|
| B1_VA | 0 | common governor | 0 |
| B2_MF | 0 | common governor | 0 |
| B3_Direct | 2307 | common governor | 96 |
| B4_Lagrangian | 2307 | unshielded; physical_byte_cap=500000; dual ascent kappa=0.05 | 96 |
| B5_ITORA_inspired | 0 | gov=None | - |

Missing cost metrics: inference P50/P95 wall-clock, bridge overhead seconds, simulator CPU-hours

## Figure data (embedded)

PNG/PDF not rendered. Numerical panel data for each figure are below (same content as `results/stage5_no_residual/figures/figXX_data.json`).

### Fig.3 — Direct training (`DATA_ONLY`)

Excluded: SPARC residual training/validation series

Train return summary: n=96, mean=31.0181, roll16_end=35.1343

#### Panel (a) validation ΔV Direct−VA

| update | mean_delta_V | n |
|---:|---:|---:|
| 0 | 6.9655 | 12 |
| 2 | 3.9934 | 12 |
| 4 | 7.6889 | 12 |
| 6 | 7.8466 | 12 |
| 8 | 7.8466 | 12 |
| 10 | 7.6259 | 12 |
| 12 | 7.6259 | 12 |
| 14 | 8.0209 | 12 |
| 16 | 8.0209 | 12 |
| 18 | 8.0209 | 12 |
| 20 | 8.0209 | 12 |
| 22 | 8.0209 | 12 |
| 24 | 8.0209 | 12 |

#### Panel (b) train diagnostics (episode returns)

| episode | return | deviation_fraction | entropy | approx_kl |
|---:|---:|---:|---:|---:|
| 0 | 30.3340 | 0.0300 | — | — |
| 1 | 0.0000 | 0.0000 | — | — |
| 2 | 45.5424 | 0.0500 | — | — |
| 3 | 42.9594 | 0.0500 | 2.3435 | -0.0060 |
| 4 | 34.6872 | 0.0500 | — | — |
| 5 | 32.3022 | 0.0500 | — | — |
| 6 | 0.0000 | 0.0000 | — | — |
| 7 | 35.7804 | 0.0500 | 2.7140 | -0.0230 |
| 8 | 28.0972 | 0.0600 | — | — |
| 9 | 42.0711 | 0.0400 | — | — |
| 10 | 45.1584 | 0.0600 | — | — |
| 11 | 0.0000 | 0.0000 | 2.5862 | 0.0109 |
| 12 | 35.7404 | 0.0400 | — | — |
| 13 | 43.8520 | 0.0500 | — | — |
| 14 | 52.3838 | 0.0400 | — | — |
| 15 | 44.5132 | 0.0600 | 2.3576 | 0.0140 |
| 16 | 34.8374 | 0.0600 | — | — |
| 17 | 0.0000 | 0.0000 | — | — |
| 18 | 32.3100 | 0.0500 | — | — |
| 19 | 47.2652 | 0.0600 | 2.5682 | -0.0031 |
| 20 | 0.0000 | 0.0000 | — | — |
| 21 | 0.0000 | 0.0000 | — | — |
| 22 | 41.7764 | 0.0700 | — | — |
| 23 | 19.4514 | 0.0400 | 2.3032 | -0.0472 |
| 24 | 0.0000 | 0.0000 | — | — |
| 25 | 28.4562 | 0.0400 | — | — |
| 26 | 34.6350 | 0.0300 | — | — |
| 27 | 49.8156 | 0.0400 | 2.6303 | 0.0215 |
| 28 | 44.5975 | 0.0500 | — | — |
| 29 | 34.9459 | 0.0400 | — | — |
| 30 | 29.6331 | 0.0200 | — | — |
| 31 | 38.4148 | 0.0400 | 2.2245 | -0.0038 |
| 32 | 23.2952 | 0.0300 | — | — |
| 33 | 42.3417 | 0.0600 | — | — |
| 34 | 40.3426 | 0.0400 | — | — |
| 35 | 0.0000 | 0.0000 | 1.9161 | -0.0001 |
| 36 | 30.1953 | 0.0300 | — | — |
| 37 | 0.0000 | 0.0000 | — | — |
| 38 | 35.4271 | 0.0300 | — | — |
| 39 | 38.5821 | 0.0400 | 1.7559 | -0.0097 |
| 40 | 0.0000 | 0.0000 | — | — |
| 41 | 35.6039 | 0.0300 | — | — |
| 42 | 38.6237 | 0.0500 | — | — |
| 43 | 39.0610 | 0.0500 | 2.1122 | -0.0069 |
| 44 | 37.4219 | 0.0500 | — | — |
| 45 | 38.5069 | 0.0700 | — | — |
| 46 | 36.2667 | 0.0400 | — | — |
| 47 | 42.5165 | 0.0700 | 2.2948 | 0.0033 |
| 48 | 26.3913 | 0.0400 | — | — |
| 49 | 42.9253 | 0.0400 | — | — |
| 50 | 0.0000 | 0.0000 | — | — |
| 51 | 34.1010 | 0.0600 | 2.3055 | -0.0097 |
| 52 | 41.6532 | 0.0500 | — | — |
| 53 | 0.0000 | 0.0000 | — | — |
| 54 | 28.3967 | 0.0500 | — | — |
| 55 | 20.9334 | 0.0400 | 2.3823 | 0.0124 |
| 56 | 36.9408 | 0.0400 | — | — |
| 57 | 44.3387 | 0.0600 | — | — |
| 58 | 47.6085 | 0.0500 | — | — |
| 59 | 40.4313 | 0.0500 | 2.3704 | -0.0146 |
| 60 | 31.5540 | 0.0500 | — | — |
| 61 | 48.9850 | 0.0700 | — | — |
| 62 | 34.9797 | 0.0400 | — | — |
| 63 | 42.4431 | 0.0600 | 2.3699 | 0.0083 |
| 64 | 20.0097 | 0.0300 | — | — |
| 65 | 0.0000 | 0.0000 | — | — |
| 66 | 32.0671 | 0.0500 | — | — |
| 67 | 23.3691 | 0.0300 | 2.2292 | -0.0161 |
| 68 | 29.5728 | 0.0300 | — | — |
| 69 | 46.0313 | 0.0500 | — | — |
| 70 | 30.8130 | 0.0500 | — | — |
| 71 | 29.5616 | 0.0400 | 2.1792 | 0.0000 |
| 72 | 48.2970 | 0.0400 | — | — |
| 73 | 40.5616 | 0.0400 | — | — |
| 74 | 28.4788 | 0.0400 | — | — |
| 75 | 0.0000 | 0.0000 | 1.8848 | 0.0024 |
| 76 | 45.8270 | 0.0500 | — | — |
| 77 | 46.4863 | 0.0500 | — | — |
| 78 | 14.9524 | 0.0300 | — | — |
| 79 | 30.1310 | 0.0400 | 2.1891 | -0.0005 |
| 80 | 31.2330 | 0.0500 | — | — |
| 81 | 29.6826 | 0.0300 | — | — |
| 82 | 38.6380 | 0.0500 | — | — |
| 83 | 50.3267 | 0.0600 | 2.2387 | 0.0103 |
| 84 | 32.4156 | 0.0500 | — | — |
| 85 | 38.7394 | 0.0300 | — | — |
| 86 | 37.0220 | 0.0500 | — | — |
| 87 | 53.5743 | 0.0600 | 2.1140 | 0.0104 |
| 88 | 26.9444 | 0.0200 | — | — |
| 89 | 60.2992 | 0.0500 | — | — |
| 90 | 36.8483 | 0.0300 | — | — |
| 91 | 0.0000 | 0.0000 | 1.8296 | 0.0002 |
| 92 | 35.1682 | 0.0300 | — | — |
| 93 | 51.8802 | 0.0600 | — | — |
| 94 | 39.3769 | 0.1500 | — | — |
| 95 | 0.0000 | 0.0000 | 2.0612 | -0.0009 |

### Fig.4 — Density (`DATA_ONLY`)

Excluded methods: ['sparc_residual_A0']

#### Panel (a) R_value vs density

| density | value_aware | minfinish | direct | itora_inspired |
|---:|---:|---:|---:|---:|
| 20 | 0.5703 | 0.5528 | 0.6780 | 0.5324 |
| 25 | 0.4848 | 0.4685 | 0.6237 | 0.5207 |
| 30 | 0.5051 | 0.4975 | 0.6294 | 0.5178 |
| 35 | 0.4951 | 0.4908 | 0.5893 | 0.5062 |
| 40 | 0.3715 | 0.3667 | 0.4644 | 0.4688 |
| 45 | 0.3685 | 0.3672 | 0.4680 | 0.4592 |
| 50 | 0.2721 | 0.2740 | 0.3482 | 0.4409 |
| 55 | 0.1801 | 0.1837 | 0.2037 | 0.4146 |
| 60 | 0.2612 | 0.2620 | 0.3174 | 0.4351 |
| 80 | 0.0200 | 0.0209 | 0.0208 | 0.3805 |
| 100 | 0.2427 | 0.2370 | 0.3075 | 0.3496 |

#### Panel (b) P_viol vs density

| density | value_aware | minfinish | direct | itora_inspired |
|---:|---:|---:|---:|---:|
| 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 25 | 0.1180 | 0.1180 | 0.1100 | 0.1260 |
| 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 35 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| 45 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| 50 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| 55 | 0.5940 | 0.5940 | 0.5940 | 0.5980 |
| 60 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| 80 | 0.9820 | 0.9840 | 0.9720 | 0.9980 |
| 100 | 0.3800 | 0.3920 | 0.3860 | 0.3980 |

#### Full V / PRR / P95

| density | method | V | R | P_viol | PRR | P95 | n |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | value_aware | 22.1632 | 0.5703 | 0.0000 | 1.0000 | 38.24 | 5 |
| 20 | minfinish | 21.5300 | 0.5528 | 0.0000 | 1.0000 | 38.24 | 5 |
| 20 | itora_inspired | 20.9188 | 0.5324 | 0.0000 | 1.0000 | 38.15 | 5 |
| 20 | direct | 26.3502 | 0.6780 | 0.0000 | 1.0000 | 38.15 | 5 |
| 20 | b4_lagrangian | 27.1543 | 0.6977 | 0.0000 | 1.0000 | 38.24 | 5 |
| 25 | value_aware | 23.1279 | 0.4848 | 0.1180 | 0.9704 | 22.20 | 5 |
| 25 | minfinish | 22.1393 | 0.4685 | 0.1180 | 0.9705 | 21.50 | 5 |
| 25 | itora_inspired | 25.0350 | 0.5207 | 0.1260 | 0.9703 | 22.68 | 5 |
| 25 | direct | 29.3763 | 0.6237 | 0.1100 | 0.9706 | 20.73 | 5 |
| 25 | b4_lagrangian | 32.6532 | 0.6854 | 0.2720 | 0.9687 | 22.68 | 5 |
| 30 | value_aware | 28.4356 | 0.5051 | 0.0000 | 0.9920 | 23.12 | 5 |
| 30 | minfinish | 27.9165 | 0.4975 | 0.0000 | 0.9911 | 21.89 | 5 |
| 30 | itora_inspired | 29.1818 | 0.5178 | 0.0000 | 0.9911 | 22.94 | 5 |
| 30 | direct | 35.2273 | 0.6294 | 0.0000 | 0.9911 | 22.11 | 5 |
| 30 | b4_lagrangian | 38.8410 | 0.6942 | 0.0000 | 0.9908 | 23.61 | 5 |
| 35 | value_aware | 32.4616 | 0.4951 | 0.0000 | 0.9841 | 23.56 | 5 |
| 35 | minfinish | 32.1578 | 0.4908 | 0.0000 | 0.9846 | 23.46 | 5 |
| 35 | itora_inspired | 33.3346 | 0.5062 | 0.0000 | 0.9846 | 22.31 | 5 |
| 35 | direct | 37.9711 | 0.5893 | 0.0000 | 0.9853 | 21.61 | 5 |
| 35 | b4_lagrangian | 42.6766 | 0.6578 | 0.0000 | 0.9829 | 24.89 | 5 |
| 40 | value_aware | 27.1408 | 0.3715 | 0.2000 | 0.9709 | 21.56 | 5 |
| 40 | minfinish | 26.5603 | 0.3667 | 0.2000 | 0.9702 | 22.80 | 5 |
| 40 | itora_inspired | 35.0907 | 0.4688 | 0.2000 | 0.9691 | 22.34 | 5 |
| 40 | direct | 33.5300 | 0.4644 | 0.2000 | 0.9704 | 21.45 | 5 |
| 40 | b4_lagrangian | 49.2861 | 0.6637 | 0.2000 | 0.9688 | 25.37 | 5 |
| 45 | value_aware | 31.1022 | 0.3685 | 0.2000 | 0.9630 | 21.23 | 5 |
| 45 | minfinish | 30.9807 | 0.3672 | 0.2000 | 0.9630 | 21.45 | 5 |
| 45 | itora_inspired | 38.7886 | 0.4592 | 0.2000 | 0.9611 | 21.16 | 5 |
| 45 | direct | 38.8424 | 0.4680 | 0.2000 | 0.9635 | 20.00 | 5 |
| 45 | b4_lagrangian | 54.1340 | 0.6465 | 0.2000 | 0.9573 | 23.47 | 5 |
| 50 | value_aware | 24.8472 | 0.2721 | 0.4000 | 0.9656 | 21.57 | 5 |
| 50 | minfinish | 25.1365 | 0.2740 | 0.4000 | 0.9655 | 21.77 | 5 |
| 50 | itora_inspired | 40.6560 | 0.4409 | 0.4000 | 0.9646 | 23.04 | 5 |
| 50 | direct | 31.3802 | 0.3482 | 0.4000 | 0.9660 | 20.60 | 5 |
| 50 | b4_lagrangian | 58.5237 | 0.6390 | 0.4000 | 0.9636 | 25.30 | 5 |
| 55 | value_aware | 18.6614 | 0.1801 | 0.5940 | 0.9567 | 21.04 | 5 |
| 55 | minfinish | 18.9672 | 0.1837 | 0.5940 | 0.9573 | 20.59 | 5 |
| 55 | itora_inspired | 42.1498 | 0.4146 | 0.5980 | 0.9526 | 21.91 | 5 |
| 55 | direct | 21.0841 | 0.2037 | 0.5940 | 0.9565 | 19.95 | 5 |
| 55 | b4_lagrangian | 61.7236 | 0.6104 | 0.5980 | 0.9526 | 25.88 | 5 |
| 60 | value_aware | 28.5323 | 0.2612 | 0.4000 | 0.9284 | 19.40 | 5 |
| 60 | minfinish | 28.4920 | 0.2620 | 0.4000 | 0.9300 | 19.40 | 5 |
| 60 | itora_inspired | 48.3401 | 0.4351 | 0.4000 | 0.9285 | 20.47 | 5 |
| 60 | direct | 34.1717 | 0.3174 | 0.4000 | 0.9297 | 19.75 | 5 |
| 60 | b4_lagrangian | 68.3871 | 0.6173 | 0.4000 | 0.9260 | 22.90 | 5 |
| 80 | value_aware | 2.9668 | 0.0200 | 0.9820 | 0.9286 | 19.28 | 5 |
| 80 | minfinish | 3.0905 | 0.0209 | 0.9840 | 0.9286 | 19.28 | 5 |
| 80 | itora_inspired | 55.8289 | 0.3805 | 0.9980 | 0.9273 | 19.78 | 5 |
| 80 | direct | 3.0849 | 0.0208 | 0.9720 | 0.9287 | 19.28 | 5 |
| 80 | b4_lagrangian | 91.1015 | 0.6219 | 0.9960 | 0.9267 | 24.23 | 5 |
| 100 | value_aware | 45.5377 | 0.2427 | 0.3800 | 0.9521 | 19.90 | 5 |
| 100 | minfinish | 44.7016 | 0.2370 | 0.3920 | 0.9521 | 20.07 | 5 |
| 100 | itora_inspired | 64.2823 | 0.3496 | 0.3980 | 0.9511 | 20.07 | 5 |
| 100 | direct | 57.8035 | 0.3075 | 0.3860 | 0.9523 | 19.96 | 5 |
| 100 | b4_lagrangian | 106.9862 | 0.5836 | 0.3980 | 0.9504 | 25.25 | 5 |

### Fig.5 — Safety–value tradeoff (`DATA_ONLY`)

Excluded: SPARC residual operating point and envelope residual series

#### Operating points

| method | admission | V | R | P_viol | governor_matched |
|---|---|---:|---:|---:|---|
| B1_VA | common_governor_175kB | 32.4616 | 0.4951 | 0.0000 | True |
| B2_MF | common_governor_175kB | 32.1578 | 0.4908 | 0.0000 | True |
| B3_Direct | common_governor_175kB | 37.9711 | 0.5893 | 0.0000 | True |
| B4_Lagrangian | unshielded_500kB | 42.6766 | 0.6578 | 0.0000 | False |
| B5_ITORA | gov_disabled | 33.3346 | 0.5062 | 0.0000 | False |

#### Envelope sensitivity (VA/MF/Direct/B4/ITORA)

| envelope_B | method | V | R | P_viol |
|---:|---|---:|---:|---:|
| 100000 | value_aware | 33.5711 | 0.5113 | 0.0000 |
| 100000 | minfinish | 31.7933 | 0.4864 | 0.0000 |
| 100000 | direct | 33.0850 | 0.5124 | 0.0000 |
| 100000 | b4_lagrangian | 38.1350 | 0.5878 | 0.0000 |
| 100000 | itora_inspired | 32.3488 | 0.4918 | 0.0000 |
| 175000 | value_aware | 32.4616 | 0.4951 | 0.0000 |
| 175000 | minfinish | 32.1578 | 0.4908 | 0.0000 |
| 175000 | direct | 37.9711 | 0.5893 | 0.0000 |
| 175000 | b4_lagrangian | 40.9444 | 0.6310 | 0.0000 |
| 175000 | itora_inspired | 33.3346 | 0.5062 | 0.0000 |
| 250000 | value_aware | 32.4616 | 0.4951 | 0.0000 |
| 250000 | minfinish | 32.1578 | 0.4908 | 0.0000 |
| 250000 | direct | 38.6957 | 0.5962 | 0.0000 |
| 250000 | b4_lagrangian | 41.7175 | 0.6434 | 0.0000 |
| 250000 | itora_inspired | 33.3346 | 0.5062 | 0.0000 |
| 350000 | value_aware | 32.4616 | 0.4951 | 0.0000 |
| 350000 | minfinish | 32.1578 | 0.4908 | 0.0000 |
| 350000 | direct | 40.8279 | 0.6286 | 0.0000 |
| 350000 | b4_lagrangian | 42.6766 | 0.6578 | 0.0000 |
| 350000 | itora_inspired | 33.3346 | 0.5062 | 0.0000 |

Gaps: Do not infer Pareto frontier

### Fig.6 — Workload / deadlines (`DATA_ONLY`)

Excluded: residual RL (Direct catch-up rows labeled residual in raw files)

#### Panel (a) R vs computation load (`tasks_per_source`)

| tasks_per_source | method | V | R | P_viol | n |
|---:|---|---:|---:|---:|---:|
| 2 | value_aware | 22.5508 | 0.5080 | 0.0000 | 5 |
| 2 | minfinish | 22.2383 | 0.5009 | 0.0000 | 5 |
| 2 | direct | 25.8546 | 0.5850 | 0.0000 | 5 |
| 3 | value_aware | 32.4616 | 0.4951 | 0.0000 | 5 |
| 3 | minfinish | 32.1578 | 0.4908 | 0.0000 | 5 |
| 3 | direct | 37.9711 | 0.5893 | 0.0000 | 5 |
| 4 | value_aware | 40.7873 | 0.4683 | 0.0000 | 5 |
| 4 | minfinish | 40.3704 | 0.4622 | 0.0000 | 5 |
| 4 | direct | 49.0185 | 0.5629 | 0.0000 | 5 |
| 6 | value_aware | 63.6766 | 0.4938 | 0.0000 | 5 |
| 6 | minfinish | 62.1974 | 0.4817 | 0.0200 | 5 |
| 6 | direct | 76.0582 | 0.5899 | 0.0040 | 5 |

#### Panel (b) R vs deadline multiplier

| deadline_multiplier | method | V | R | P_viol | n |
|---:|---|---:|---:|---:|---:|
| 0.5 | value_aware | 10.1059 | 0.1551 | 0.0000 | 5 |
| 0.5 | minfinish | 10.1059 | 0.1551 | 0.0000 | 5 |
| 0.5 | direct | 13.1207 | 0.2006 | 0.0000 | 5 |
| 1.0 | value_aware | 32.4616 | 0.4951 | 0.0000 | 5 |
| 1.0 | minfinish | 32.1578 | 0.4908 | 0.0000 | 5 |
| 1.0 | direct | 37.9711 | 0.5893 | 0.0000 | 5 |
| 1.5 | value_aware | 49.5263 | 0.7534 | 0.0000 | 5 |
| 1.5 | minfinish | 49.1543 | 0.7498 | 0.0000 | 5 |
| 1.5 | direct | 52.8283 | 0.8026 | 0.0000 | 5 |
| 2.0 | value_aware | 57.1060 | 0.8652 | 0.0000 | 5 |
| 2.0 | minfinish | 57.1060 | 0.8652 | 0.0000 | 5 |
| 2.0 | direct | 59.4477 | 0.9008 | 0.0000 | 5 |

### Fig.7 — Radio safety (`DATA_ONLY`)

Excluded: SPARC residual density series

#### Panel (a) PRR vs density

| density | value_aware | minfinish | direct |
|---:|---:|---:|---:|
| 20 | 1.0000 | 1.0000 | 1.0000 |
| 25 | 0.9704 | 0.9705 | 0.9706 |
| 30 | 0.9920 | 0.9911 | 0.9911 |
| 35 | 0.9841 | 0.9846 | 0.9853 |
| 40 | 0.9709 | 0.9702 | 0.9704 |
| 45 | 0.9630 | 0.9630 | 0.9635 |
| 50 | 0.9656 | 0.9655 | 0.9660 |
| 55 | 0.9567 | 0.9573 | 0.9565 |
| 60 | 0.9284 | 0.9300 | 0.9297 |
| 80 | 0.9286 | 0.9286 | 0.9287 |
| 100 | 0.9521 | 0.9521 | 0.9523 |

#### Panel (b) P95 vs density

| density | value_aware | minfinish | direct |
|---:|---:|---:|---:|
| 20 | 38.24 | 38.24 | 38.15 |
| 25 | 22.20 | 21.50 | 20.73 |
| 30 | 23.12 | 21.89 | 22.11 |
| 35 | 23.56 | 23.46 | 21.61 |
| 40 | 21.56 | 22.80 | 21.45 |
| 45 | 21.23 | 21.45 | 20.00 |
| 50 | 21.57 | 21.77 | 20.60 |
| 55 | 21.04 | 20.59 | 19.95 |
| 60 | 19.40 | 19.40 | 19.75 |
| 80 | 19.28 | 19.28 | 19.28 |
| 100 | 19.90 | 20.07 | 19.96 |

#### Panel (c) delay CDF / hist (stress)

Source: `catchup_fig7d_direct_d80/episode_0001 last epoch`

```json
[
  0,
  1150,
  0,
  1072,
  339,
  198,
  1172,
  350,
  687,
  396,
  240,
  896,
  260,
  1426,
  260,
  1020,
  634,
  1525,
  594,
  291,
  801,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  7,
  8,
  0,
  8,
  0,
  110
]
```

#### Panel (d) PRR vs distance + Direct@80 stress

Stress Direct@80 (`catchup_fig7d_direct_d80`): V=3.0849, R=0.0208, P_viol=0.9720, n=5

status=`DATA_ONLY`; source=`catchup_fig7d_direct_d80`; reason=`None`

| distance_m | mean_prr | n |
|---:|---:|---:|
| 10.0 | 0.9652 | 500 |
| 20.0 | 0.9584 | 500 |
| 30.0 | 0.9485 | 500 |
| 40.0 | 0.9396 | 500 |
| 50.0 | 0.9287 | 500 |
| 60.0 | 0.9172 | 500 |
| 70.0 | 0.9065 | 500 |
| 80.0 | 0.8968 | 500 |
| 90.0 | 0.8861 | 500 |
| 100.0 | 0.8757 | 500 |
| 110.0 | 0.8666 | 500 |
| 120.0 | 0.8582 | 500 |
| 130.0 | 0.8506 | 500 |
| 140.0 | 0.8430 | 500 |
| 150.0 | 0.8356 | 500 |
| 160.0 | 0.8281 | 500 |
| 170.0 | 0.8212 | 500 |
| 180.0 | 0.8145 | 500 |
| 190.0 | 0.8084 | 500 |
| 200.0 | 0.8024 | 500 |
| 210.0 | 0.7960 | 500 |
| 220.0 | 0.7893 | 500 |
| 230.0 | 0.7827 | 500 |
| 240.0 | 0.7767 | 500 |
| 250.0 | 0.7707 | 500 |
| 260.0 | 0.7646 | 500 |
| 270.0 | 0.7583 | 500 |
| 280.0 | 0.7517 | 500 |
| 290.0 | 0.7449 | 500 |
| 300.0 | 0.7373 | 500 |
| 310.0 | 0.7285 | 500 |
| 320.0 | 0.7207 | 500 |
| 330.0 | 0.7122 | 500 |
| 340.0 | 0.7034 | 500 |
| 350.0 | 0.6940 | 500 |
| 360.0 | 0.6848 | 500 |
| 370.0 | 0.6753 | 500 |
| 380.0 | 0.6657 | 500 |
| 390.0 | 0.6564 | 500 |
| 400.0 | 0.6471 | 500 |
| 410.0 | 0.6376 | 500 |
| 420.0 | 0.6286 | 500 |
| 430.0 | 0.6199 | 500 |
| 440.0 | 0.6119 | 500 |
| 450.0 | 0.6038 | 500 |
| 460.0 | 0.5962 | 500 |
| 470.0 | 0.5889 | 500 |
| 480.0 | 0.5821 | 500 |

### Fig.8 — Closed-loop Direct trace (`DATA_ONLY`)

Source: `final_test_direct_d35/episode_0001`; sim_seed=93000; method=B3_Direct

Note: VA first-pick deviation N/A for Direct (no VA anchor).

| epoch | CBR | PRR | P95_ms | offered | proposed | executed | budget | intervened | reason | n_sel | reward |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 0 | 0.1192 | 0.9848 | 19 | 173468 | 173468 | 173468 | 175000 | False | none | 3 | 0 |
| 1 | 0.1187 | 0.9851 | 19 | 152456 | 152456 | 152456 | 175000 | False | none | 3 | 0 |
| 2 | 0.1187 | 0.9853 | 19 | 0 | 0 | 0 | 175000 | False | none | 0 | 0 |
| 3 | 0.1191 | 0.9848 | 19 | 0 | 0 | 0 | 175000 | False | none | 0 | 0 |
| 4 | 0.1191 | 0.9850 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0 |
| 5 | 0.1224 | 0.9852 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0 |
| 6 | 0.1246 | 0.9840 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 1.6108 |
| 7 | 0.1245 | 0.9842 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 7.0765 |
| 8 | 0.1258 | 0.9842 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 9 | 0.1274 | 0.9845 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 10 | 0.1306 | 0.9833 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 11 | 0.1345 | 0.9836 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 4.3387 |
| 12 | 0.1415 | 0.9838 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 13 | 0.1433 | 0.9839 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 14 | 0.1433 | 0.9840 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 15 | 0.1448 | 0.9841 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 16 | 0.1441 | 0.9830 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 17 | 0.1442 | 0.9832 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 18 | 0.1434 | 0.9832 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 19 | 0.1426 | 0.9834 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 20 | 0.1403 | 0.9836 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 21 | 0.1377 | 0.9838 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 22 | 0.1320 | 0.9839 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 23 | 0.1301 | 0.9840 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 24 | 0.1301 | 0.9842 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 25 | 0.1259 | 0.9842 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 26 | 0.1247 | 0.9832 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 27 | 0.1247 | 0.9834 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 28 | 0.1241 | 0.9834 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 29 | 0.1234 | 0.9836 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 30 | 0.1225 | 0.9837 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 31 | 0.1217 | 0.9839 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 32 | 0.1205 | 0.9840 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 33 | 0.1202 | 0.9841 | 20 | 135721 | 135721 | 135721 | 175000 | False | none | 3 | 0.0000 |
| 34 | 0.1202 | 0.9843 | 20 | 97472 | 97472 | 97472 | 175000 | False | none | 2 | 0.0000 |
| 35 | 0.1208 | 0.9843 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 36 | 0.1216 | 0.9835 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 37 | 0.1216 | 0.9837 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 38 | 0.1225 | 0.9838 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 39 | 0.1234 | 0.9835 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 40 | 0.1267 | 0.9826 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 41 | 0.1298 | 0.9828 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 3.3432 |
| 42 | 0.1382 | 0.9830 | 24 | 0 | 0 | 0 | 175000 | False | none | 0 | 6.3662 |
| 43 | 0.1410 | 0.9831 | 24 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 44 | 0.1410 | 0.9832 | 24 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 45 | 0.1483 | 0.9833 | 24 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 46 | 0.1493 | 0.9824 | 24 | 0 | 0 | 0 | 175000 | False | none | 0 | 4.9510 |
| 47 | 0.1493 | 0.9826 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 48 | 0.1493 | 0.9826 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 49 | 0.1487 | 0.9827 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 50 | 0.1461 | 0.9828 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 51 | 0.1430 | 0.9830 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 52 | 0.1350 | 0.9831 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 53 | 0.1319 | 0.9832 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 54 | 0.1319 | 0.9833 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 55 | 0.1230 | 0.9834 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 56 | 0.1209 | 0.9826 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 57 | 0.1209 | 0.9827 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 58 | 0.1201 | 0.9827 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 59 | 0.1198 | 0.9829 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 60 | 0.1192 | 0.9830 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 61 | 0.1190 | 0.9831 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 62 | 0.1186 | 0.9832 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 63 | 0.1187 | 0.9833 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 64 | 0.1187 | 0.9834 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 65 | 0.1191 | 0.9835 | 21 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 66 | 0.1192 | 0.9827 | 21 | 69620 | 69620 | 69620 | 175000 | False | none | 2 | 0.0000 |
| 67 | 0.1192 | 0.9829 | 21 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 68 | 0.1191 | 0.9821 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 69 | 0.1194 | 0.9822 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 70 | 0.1199 | 0.9815 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 71 | 0.1205 | 0.9816 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 3.5602 |
| 72 | 0.1215 | 0.9818 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 73 | 0.1218 | 0.9819 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 74 | 0.1218 | 0.9820 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 3.9800 |
| 75 | 0.1235 | 0.9820 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 76 | 0.1243 | 0.9813 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 77 | 0.1243 | 0.9815 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 78 | 0.1248 | 0.9815 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 79 | 0.1246 | 0.9816 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 80 | 0.1243 | 0.9817 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 81 | 0.1239 | 0.9819 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 82 | 0.1232 | 0.9820 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 83 | 0.1227 | 0.9820 | 22 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 84 | 0.1227 | 0.9821 | 21 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 85 | 0.1211 | 0.9822 | 21 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 86 | 0.1205 | 0.9815 | 21 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 87 | 0.1205 | 0.9817 | 21 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 88 | 0.1200 | 0.9817 | 21 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 89 | 0.1199 | 0.9818 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 90 | 0.1195 | 0.9819 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 91 | 0.1195 | 0.9820 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 92 | 0.1192 | 0.9821 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 93 | 0.1193 | 0.9822 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 94 | 0.1193 | 0.9823 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 95 | 0.1191 | 0.9823 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 96 | 0.1191 | 0.9817 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 97 | 0.1190 | 0.9818 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 98 | 0.1190 | 0.9818 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |
| 99 | 0.1190 | 0.9820 | 20 | 0 | 0 | 0 | 175000 | False | none | 0 | 0.0000 |

### Fig.9 — Ablations restricted (`DATA_ONLY_RESTRICTED`)

Excluded: ['A0 SPARC residual', 'A1', 'A3', 'A5', 'A6']
Included: ['A2(=B1 VA)', 'A4(=B3 Direct)', 'MF comparator']

| contrast | dV mean | CI lo | CI hi | W/T/L | dR mean |
|---|---:|---:|---:|---:|---:|
| direct_minus_value_aware | 5.5094 | 1.1562 | 10.6487 | 4/1/0 | 0.0942 |
| mf_minus_value_aware | -0.3038 | -2.7955 | 1.7928 | 2/2/1 | -0.0043 |

P_viol: value_aware=0.0000, minfinish=0.0000, direct=0.0000

Note: Residual-architecture ablations omitted per Stage-5 exclude-residual request.

### Fig.10 — Governor / cost (`DATA_ONLY_PARTIAL`)

Excluded: SPARC residual governor traces

#### Panel (a) offered-byte distribution

| method | mean | p50 | p90 | max | exceed_175kB | n_pos |
|---|---:|---:|---:|---:|---:|---:|
| B1_VA | 98273.0 | 102234.0 | 163288.0 | 173721.0 | 0 | 36 |
| B2_MF | 96600.9 | 109531.0 | 162965.0 | 171267.0 | 0 | 36 |
| B3_Direct | 116681.3 | 124365.0 | 166598.0 | 173468.0 | 0 | 27 |
| B4_Lagrangian | 158289.2 | 133583.0 | 296277.0 | 331897.0 | 10 | 25 |
| B5_ITORA | 101683.5 | 109531.0 | 165280.0 | 173721.0 | 0 | 36 |

#### Panel (b) governor reason counts

| method | reasons |
|---|---|
| B1_VA | `{"none": 500}` |
| B2_MF | `{"none": 500}` |
| B3_Direct | `{"none": 500}` |
| B4_Lagrangian | `{"disabled": 500}` |
| B5_ITORA | `{"disabled": 500}` |

#### Panel (c) inference latency: `{"status": "MISSING"}`

#### Panel (d) value vs candidate-set size *k*

status=`DATA_ONLY`

| k | method | V | R | P_viol | mean_offered_bytes | mean_n_candidates | n |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | direct | 34.3433 | 0.5287 | 0.0000 | 5269.8 | 0.51 | 5 |
| 8 | value_aware | 31.7699 | 0.4840 | 0.0000 | 5110.7 | 0.56 | 5 |
| 16 | direct | 37.4583 | 0.5748 | 0.0000 | 6046.2 | 0.75 | 5 |
| 16 | value_aware | 32.4616 | 0.4951 | 0.0000 | 6320.0 | 0.86 | 5 |
| 32 | direct | 39.3420 | 0.6078 | 0.0000 | 6310.3 | 1.09 | 5 |
| 32 | value_aware | 32.4616 | 0.4951 | 0.0000 | 6954.2 | 1.25 | 5 |
| 64 | direct | 37.9711 | 0.5893 | 0.0000 | 6300.8 | 1.38 | 5 |
| 64 | value_aware | 32.4616 | 0.4951 | 0.0000 | 7075.7 | 1.47 | 5 |

### Fig.11 — Robustness (`DATA_ONLY`)

Excluded: Nominal-trained SPARC residual vs condition-trained SPARC (residual excluded)

#### Panel (a) Direct across densities

| density | V | R | P_viol | PRR | P95 | n |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 26.3502 | 0.6780 | 0.0000 | 1.0000 | 38.15 | 5 |
| 25 | 29.3763 | 0.6237 | 0.1100 | 0.9706 | 20.73 | 5 |
| 30 | 35.2273 | 0.6294 | 0.0000 | 0.9911 | 22.11 | 5 |
| 35 | 37.9711 | 0.5893 | 0.0000 | 0.9853 | 21.61 | 5 |
| 40 | 33.5300 | 0.4644 | 0.2000 | 0.9704 | 21.45 | 5 |
| 45 | 38.8424 | 0.4680 | 0.2000 | 0.9635 | 20.00 | 5 |
| 50 | 31.3802 | 0.3482 | 0.4000 | 0.9660 | 20.60 | 5 |
| 55 | 21.0841 | 0.2037 | 0.5940 | 0.9565 | 19.95 | 5 |
| 60 | 34.1717 | 0.3174 | 0.4000 | 0.9297 | 19.75 | 5 |
| 80 | 3.0849 | 0.0208 | 0.9720 | 0.9287 | 19.28 | 5 |
| 100 | 57.8035 | 0.3075 | 0.3860 | 0.9523 | 19.96 | 5 |

#### Panel (b) envelope VA/MF/Direct/B4/ITORA

| envelope_B | method | V | R | P_viol |
|---:|---|---:|---:|---:|
| 100000 | value_aware | 33.5711 | 0.5113 | 0.0000 |
| 100000 | minfinish | 31.7933 | 0.4864 | 0.0000 |
| 100000 | direct | 33.0850 | 0.5124 | 0.0000 |
| 100000 | b4_lagrangian | 38.1350 | 0.5878 | 0.0000 |
| 100000 | itora_inspired | 32.3488 | 0.4918 | 0.0000 |
| 175000 | value_aware | 32.4616 | 0.4951 | 0.0000 |
| 175000 | minfinish | 32.1578 | 0.4908 | 0.0000 |
| 175000 | direct | 37.9711 | 0.5893 | 0.0000 |
| 175000 | b4_lagrangian | 40.9444 | 0.6310 | 0.0000 |
| 175000 | itora_inspired | 33.3346 | 0.5062 | 0.0000 |
| 250000 | value_aware | 32.4616 | 0.4951 | 0.0000 |
| 250000 | minfinish | 32.1578 | 0.4908 | 0.0000 |
| 250000 | direct | 38.6957 | 0.5962 | 0.0000 |
| 250000 | b4_lagrangian | 41.7175 | 0.6434 | 0.0000 |
| 250000 | itora_inspired | 33.3346 | 0.5062 | 0.0000 |
| 350000 | value_aware | 32.4616 | 0.4951 | 0.0000 |
| 350000 | minfinish | 32.1578 | 0.4908 | 0.0000 |
| 350000 | direct | 40.8279 | 0.6286 | 0.0000 |
| 350000 | b4_lagrangian | 42.6766 | 0.6578 | 0.0000 |
| 350000 | itora_inspired | 33.3346 | 0.5062 | 0.0000 |

#### Panel (c) Fig.11a specialized vs nominal Direct@50

specialized_train_done=`True`

| variant | V | R | P_viol | n |
|---|---:|---:|---:|---:|
| specialized_ft | 34.0982 | 0.3765 | 0.4000 | 5 |
| nominal_on_d50 | 31.3802 | 0.3482 | 0.4000 | 5 |

## Additional evidence (claims beyond Fig.3–11 core)

### Additional — Governor ON vs OFF ablation (`DATA_ONLY`)

**Claim:** The governor itself actively prevents unsafe outcomes

#### Experimental setup

- **Simulator:** WiLabV2Xsim closed-loop (run_paired_eval → train_wilab_closed_loop)
- **Mobility:** `synthetic_highway_rho_{density}_20s.txt under traces/mode2_external_validation`
- **Densities:** [35, 80] — 35: nominal sealed final-test density (matched-admission regime); 80: stress density where radio safety binds (high P_viol)
- **Methods:** VA+MF `['value_aware', 'minfinish']`; Direct `['Direct PPO (raw method label residual + --logit-mode direct)', 'checkpoint results/wilab_direct_ppo_96ep/checkpoint_best_delta_va.pt']`
- **Not included in this ablation:** ['B4 Lagrangian', 'ITORA-inspired', 'SPARC residual RL']
- **Seeds:** `sparc_performance_evaluation/protocol/final_test_seeds.json` (n=5; sim=[93000, 93001, 93002, 93003, 93004]; task=[193000, 193001, 193002, 193003, 193004])
- **Envelope:** 175000 bytes
- **Episode settings:** controller_epochs=100, simulation_time=12.0 s
- **Artifacts ON:** `catchup_gov_on_d{density}_{va_mf|direct}`
- **Artifacts OFF:** `catchup_gov_off_d{density}_{va_mf|direct}`

#### Protocol

- **Design:** paired ON vs OFF ablation; single factor flip
- **Factor:** application safety governor
- **ON condition:** SafetyGovernor with envelope_bytes=175000 (PRR/P95/validity gates + byte envelope enforced before sampling)
- **OFF condition:** --no-governor: no application governor; byte budget still via forced_byte_budget / OpenGates path (same envelope_bytes=175000) (CLI `--no-governor`)
- **Held fixed:**
  - sealed sim/task seeds
  - density
  - envelope_bytes=175000
  - method / checkpoint / logit-mode
  - tasks_per_source=3
  - controller_epochs / simulation_time
- **Procedure:**
  - 1. For each density in {35, 80}:
  - 2.   For each method group (VA+MF together; Direct alone):
  - 3.     Run sealed paired eval with governor ON -> catchup_gov_on_*.
  - 4.     Run identical eval with --no-governor -> catchup_gov_off_*.
  - 5. Compare mean V, R, P_viol, PRR, P95, and governor_interventions (ON vs OFF).
  - 6. Dual-verify artifact completeness (midnight_governor_VERIFY_1/2).
- **Primary contrast:** delta P_viol (OFF - ON) and delta V (OFF - ON) at matched seeds
- **Interpretation notes:**
  - Supports the claim if OFF increases unsafety (P_viol) and/or ON shows interventions that keep outcomes safer.
  - At dens 35, ON and OFF may coincide if policies never stress the gates.
  - At dens 80, OFF may inflate V while P_viol remains high - report both V and safety.

#### Results (paired means)

| pair | status | ON V | OFF V | ON P_viol | OFF P_viol | ΔP (off−on) | ON interventions | OFF interventions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d35_va_mf | OK | 32.3097 | 32.3097 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| d35_direct | OK | 37.9711 | 37.9711 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| d80_va_mf | OK | 3.0287 | 56.2478 | 0.9830 | 0.9980 | 0.0150 | 0.00 | 0.00 |
| d80_direct | OK | 3.0849 | 71.2032 | 0.9720 | 0.9600 | -0.0120 | 0.00 | 0.00 |

### Additional — Iso-safety admission match (`DATA_ONLY`)

**Claim:** Match baselines to Direct on safety (P_viol/PRR/P95) by reducing computation admission; compare value at E*

Protocol: dens=[35, 50, 80], E_Direct=175000, grid=[25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 300000, 350000], prr_tol=0.002, p95_tol_ms=1.0

Safety rule: `baseline ⪯ T iff P_viol<=T.P_viol AND PRR>=T.PRR-prr_tol AND P95<=T.P95+p95_tol_ms`

#### Direct anchors T_d

| dens | folder | E_D | V | R | P_viol | PRR | P95 | n |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 35 | `density_direct_d35` | 175000 | 37.9711 | 0.5893 | 0.0000 | 0.9853 | 21.61 | 5 |
| 50 | `density_direct_d50` | 175000 | 31.3802 | 0.3482 | 0.4000 | 0.9660 | 20.60 | 5 |
| 80 | `density_direct_d80` | 175000 | 3.0849 | 0.0208 | 0.9720 | 0.9287 | 19.28 | 5 |

#### E* and value at matched safety

| key | status | E* | E_D | baseline V | Direct V | ΔV (base−Direct) | base P_viol | Direct P_viol | base PRR | Direct PRR | base P95 | Direct P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d35_value_aware | OK | 150000 | 175000 | 32.4616 | 37.9711 | -5.5094 | 0.0000 | 0.0000 | 0.9845 | 0.9853 | 22.32 | 21.61 |
| d35_minfinish | OK | 150000 | 175000 | 32.1578 | 37.9711 | -5.8133 | 0.0000 | 0.0000 | 0.9858 | 0.9853 | 21.58 | 21.61 |
| d50_value_aware | OK | 175000 | 175000 | 24.8472 | 31.3802 | -6.5329 | 0.4000 | 0.4000 | 0.9656 | 0.9660 | 21.57 | 20.60 |
| d50_minfinish | OK | 150000 | 175000 | 25.1365 | 31.3802 | -6.2437 | 0.4000 | 0.4000 | 0.9655 | 0.9660 | 20.18 | 20.60 |
| d80_value_aware | OK | 350000 | 175000 | 3.7592 | 3.0849 | 0.6742 | 0.9700 | 0.9720 | 0.9284 | 0.9287 | 19.28 | 19.28 |
| d80_minfinish | OK | 350000 | 175000 | 3.4944 | 3.0849 | 0.4094 | 0.9700 | 0.9720 | 0.9283 | 0.9287 | 19.28 | 19.28 |

E* eligibility notes:

- `d35_value_aware`: E*=150000, eligible=[25000, 50000, 75000, 100000, 125000, 150000], note=At this dens Direct P_viol==0; E* is driven by PRR/P95 match
- `d35_minfinish`: E*=150000, eligible=[25000, 50000, 75000, 100000, 125000, 150000], note=At this dens Direct P_viol==0; E* is driven by PRR/P95 match
- `d50_value_aware`: E*=175000, eligible=[25000, 50000, 75000, 100000, 125000, 150000, 175000]
- `d50_minfinish`: E*=150000, eligible=[25000, 50000, 75000, 100000, 125000, 150000]
- `d80_value_aware`: E*=350000, eligible=[250000, 300000, 350000]
- `d80_minfinish`: E*=350000, eligible=[300000, 350000]

## Artifact paths

- Bundle: `results/stage5_no_residual/`
- Completeness ledger: `results/stage5_no_residual/COMPLETENESS_LEDGER.json`
- Claim audit: `results/stage5_no_residual/claim_to_evidence_audit.json`
- Figure JSON (duplicate of embedded tables): `results/stage5_no_residual/figures/`
