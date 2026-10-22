# SPARC MDP Phase-2a — γ=1 + Terminal Drain

**Overall:** `PASS`
**Scope:** training-objective alignment only; same actor architecture; no Phase 2b–3.

## Changes

1. `controllers/rl/residual_ppo.py`: `build_ppo_steps` / `ppo_update` default `gamma=1.0` (legacy `0.99` still passable).
2. `train_wilab_residual_ppo.py`: `--gamma` (default 1.0), `--gae-lambda` (default 0.95); logged on updates.
3. `train_wilab_closed_loop.py`: terminal drain via `set_epoch(controller_epochs)` after the episode loop; credits queued deadline-valid completions into `total` / last reward / trace.

## Verification

| Check | Pass | Detail |
|---|---|---|
| gamma defaults | `True` | steps=1.0, ppo_update=1.0, CLI=True |
| GAE gamma effect | `True` | G0 γ1=0.9024999737739563, γ0.99=0.8845402598381042 |
| Drain unit (TaskLayer) | `True` | credited=3.25 |
| Historical audit gap | `ok` | mean V−∑r=-5.329070518200751e-16 |

## Non-regression

- Frozen Direct/POPL **inference** does not use `gamma`; actor weights untouched → closed-loop `V_done` unchanged by this phase alone.
- No retrain authorized (no episode budget in Phase-2a auth). Retrain under `gamma=1` is a later, separately budgeted step if desired.

## Decision

Phase-2a **PASS**. Next requires user `AUTHORIZE Phase 2b` (full-pack logπ hygiene).

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\mdp_phase2a`
