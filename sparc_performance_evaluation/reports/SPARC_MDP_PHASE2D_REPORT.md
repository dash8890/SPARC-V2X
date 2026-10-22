# SPARC MDP Phase-2d — W in Advantage

**Overall:** `FAIL`
**Scope:** advantage-mode wiring + offline W-greedy audit. No MATLAB retrain (no budget in auth).

## Code

- `build_ppo_steps` / `ppo_update`: `advantage_mode ∈ {gae, w_minus_v, blend}` using per-step `w_estimate`.
- `train_wilab_residual_ppo.py`: `--advantage-mode`, `--w-blend`.
- Wiring check: `True` (gae vs w_minus_v differ; blend@1 ≡ w).

## Offline W-greedy (held-out pairs 32–47)

- n: 14
- mean G_sel / Direct / POPL: **24.858** / 24.953 / 26.107
- mean(G_sel−G_Direct): **-0.095**
- mean(G_sel−G_POPL): **-1.249**
- Spearman(W_P−W_D, ΔG): -0.138
- Beat always-Direct: `False` (offline PASS rule)
- Beat always-POPL: `False`

## Closed-loop V_done vs Phase-2b policy

- Status: `NOT_RUN` — Phase 2d auth had no explicit episode/seed budget; HANDOFF allows offline audit
- Runbook’s mean-V_done comparison requires a **budgeted** train/eval; deferred to Phase 3 auth with explicit episode budget.

## Decision

Phase-2d **FAIL**. Do not authorize Phase 3 W-driven training on this evidence.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\mdp_phase2d`
