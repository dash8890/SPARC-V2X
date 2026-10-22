# SPARC MDP Phase-1 — Headroom & Commitment Diagnostic

**Overall:** `PASS`
**Swing (A):** `PASS`
**Lift (B):** `PASS`

Training-free. Inputs: `rao_phase1/pairs.json`. Fit 0–31 / hold 32–47.
No PPO. No new MATLAB collection. Locked val / final-test unused for fitting.

## A — Structural swing (Direct vs POPL packs)

- n disagreeing: 45
- frac |ΔG|>0.1: **0.311**
- mean(oracle − Direct): **0.636**
- mean(oracle − POPL): **0.459**
- mean |ΔG| (nonzero): **3.519**
- frac +/−/0: 0.089 / 0.222 / 0.689
- Rule: `frac(|dG|>0.1)>=0.25 OR mean(oracle-Direct)>=1.0 OR mean(|dG||nonzero)>=2.0`
- Result: **PASS**

## B — Commitment predictive lift

- n fit / hold: 31 / 14
- Δ bacc (F_both − F_base): **+0.077**
- Δ Spearman_|pred|,|ΔG|: **+0.211**
- Rule: `bacc(F_both)>=bacc(F_base)+0.02 OR spearman_abs(F_both)>=spearman_abs(F_base)+0.05`
- Result: **PASS**

| Set | bacc | acc | Spearman(|pred|,|ΔG|) | Spearman(pred,ΔG) |
|---|---:|---:|---:|---:|
| F_base | 0.346 | 0.643 | -0.336 | 0.081 |
| F_commit | 0.462 | 0.857 | 0.481 | -0.262 |
| F_both | 0.423 | 0.786 | -0.125 | -0.051 |

## C — Audit remaining-return (report only)

- status: ok
- detail: {"n_epochs": 61, "n_episodes_sampled": 12, "spearman_meanfeat_vs_Gt": 0.0007932310946589105, "note": "report-only; not a hard gate"}

## Decision

Phase-1 **PASS**. CA-PPO / Phase-2 path remains **eligible** pending user `AUTHORIZE Phase 2a`.
Does not authorize training by itself.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\mdp_phase1`
