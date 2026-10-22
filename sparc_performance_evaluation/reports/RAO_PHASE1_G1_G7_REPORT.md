# RAO-SPARC — Phase-1 G1–G7 Prerequisite Report

**Overall:** `FAIL`
**G4 (decisive):** `FAIL`
**Scope:** frozen Direct vs frozen POPL; causal selector only; train-dev seeds `72100+/182100+`.

## Gate summary

| Gate | Status | Evidence |
|---|---|---|
| G1_causal_inputs | `PASS` | leakage checks on 5 samples; forbidden seeds excluded |
| G2_complementarity | `FAIL` | disagree=45; frac+=0.09; frac-=0.22 |
| G3_predictability | `FAIL` | bacc=0.269 vs best_comp=0.654; acc=0.500 |
| G4_net_value | `FAIL` | ΔvsD=+0.000 CI[+0.000,+0.000]; ΔvsP=-1.154 CI[-4.398,+0.743] |
| G5_calibration | `FAIL` | ECE=0.400; monotone=False |
| G6_cost | `FAIL` | P95=9.51ms max=12.01ms |
| G7_fairness | `PASS` | governor=SafetyGovernor horizon_envelope_only; envelope=175kB; density=35; unchanged baselines |

## G4 — Realized completed value (held-out development)

- n disagreeing held-out pairs: **14**
- Mean \(G^{\mathrm{sel}}-G^{\mathrm{D}}\): **0.0** (95% CI [0.0, 0.0])
- Mean \(G^{\mathrm{sel}}-G^{\mathrm{P}}\): **-1.1540582611976091** (95% CI [-4.398027208073834, 0.7432677109877527])
- Always-Direct mean G: 24.952740466191155
- Always-POPL mean G: 26.106798727388767
- Selector mean G: 24.952740466191155
- Oracle mean G: 26.44965819327024

### Per-pair held-out (disagreeing)

| pair_idx | seed | G_D | G_P | ΔG | choose | G_sel | vs D | vs P |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 32 | 72132 | 0.000 | 0.000 | +0.000 | POPL | 0.000 | +0.000 | +0.000 |
| 33 | 72133 | 0.000 | 0.000 | +0.000 | POPL | 0.000 | +0.000 | +0.000 |
| 34 | 72134 | 0.000 | 0.000 | +0.000 | POPL | 0.000 | +0.000 | +0.000 |
| 35 | 72135 | 43.767 | 40.298 | -3.469 | Direct | 43.767 | +0.000 | +3.469 |
| 36 | 72136 | 26.992 | 26.992 | +0.000 | Direct | 26.992 | +0.000 | +0.000 |
| 37 | 72137 | 41.450 | 41.450 | +0.000 | POPL | 41.450 | +0.000 | +0.000 |
| 39 | 72139 | 50.571 | 50.571 | +0.000 | Direct | 50.571 | +0.000 | +0.000 |
| 40 | 72140 | 45.563 | 45.563 | +0.000 | POPL | 45.563 | +0.000 | +0.000 |
| 41 | 72141 | 0.000 | 0.000 | +0.000 | Direct | 0.000 | +0.000 | +0.000 |
| 42 | 72142 | 44.854 | 44.854 | +0.000 | Direct | 44.854 | +0.000 | +0.000 |
| 44 | 72144 | 44.904 | 43.572 | -1.331 | Direct | 44.904 | +0.000 | +1.331 |
| 45 | 72145 | 32.833 | 32.833 | +0.000 | POPL | 32.833 | +0.000 | +0.000 |
| 46 | 72146 | 0.000 | 0.000 | +0.000 | Direct | 0.000 | +0.000 | +0.000 |
| 47 | 72147 | 18.405 | 39.362 | +20.957 | Direct | 18.405 | +0.000 | -20.957 |

## Collection summary

- Indices collected: 48
- Disagreeing pairs: 45
- Ties / no-opportunity: 3
- Fit disagreeing: 31
- Held-out disagreeing: 14

## Mechanism decision

**CLOSED.** G4 FAIL → dual-policy Direct↔POPL causal selector rejected.
No corrective retuning, no Phase-2 adaptive-actor PPO on this path.
SPARC-V2X goal unchanged. Flag: `RAO_PHASE1_DUAL_POLICY_SELECTOR_CLOSED.flag`.

## Boundaries

- No RAO adaptive-actor PPO; no extra collection/tuning; no architecture/baseline changes.
- Locked val `82001/02/11/12` unused for fit/threshold/selection; final-test sealed.
- G4 FAIL → dual-policy selector mechanism closed (not the SPARC-V2X goal).

Artifacts: `sparc_performance_evaluation/results/raw/rao_phase1`
