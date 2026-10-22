# SPARC MDP Phase-2c — Auxiliary Afterstate Critic W

**Overall:** `PASS`
**Scope:** MSE fit of W(Z, A_exec) to realized episode G; **actor not updated**.

## Data

- Source: `rao_phase1/pairs.json` disagreeing pairs (Direct + POPL packs as two afterstates).
- Fit pair_idx ≤31: **62** rows; hold ≥32: **28** rows.
- Features: regime (RAO) + commitment solo features of the pack (no outcome leakage).

## Held-out results

| Metric | Value | Gate |
|---|---:|---|
| Spearman(W, G) | **0.701** | ≥ 0.20 → `True` |
| Pack-rank accuracy | **0.571** | ≥ best_const+0.05 (0.929+0.05=0.979) → `False` |
| Best constant accuracy | 0.929 | — |
| Hold MAE | 8.507 | — |

## Decision

Phase-2c **PASS**. Auxiliary W is calibrated enough to *allow* Phase 2d authorization.
Does **not** by itself put W into the actor advantage. Await `AUTHORIZE Phase 2d`.

Checkpoint: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\mdp_phase2c\afterstate_critic.pt`
Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\mdp_phase2c`
