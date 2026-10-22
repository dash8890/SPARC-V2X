# SPARC MDP Phase-2b — Full-Pack Proposed logπ Hygiene

**Overall:** `PASS`
**Scope:** proposed-pack joint likelihood for PPO; governor inside transition; no Phase 2c/3.

## Changes

1. After `sample_action`, freeze `proposed_assignments`, `proposed_selected_candidate_indices`, `proposed_logp` before `governor.apply`.
2. Trace / PPO `old_logp` uses **proposed** joint logπ; executed pack stored separately.
3. `_sequence_logp` prefers `proposed_selected_candidate_indices`.

## Verification

| Check | Pass | Detail |
|---|---|---|
| joint_logp | `True` | `{"n_selected": 4, "logp": -14.028541564941406, "joint": true}` |
| governor_preserves_proposed_logp | `True` | `{"intervened": true, "reason": "over_budget_or_invalid_assignment", "packs_differ": true, "stored_proposed_logp": -5.054` |
| sequence_logp_prefers_proposed | `True` | `{"recomputed": -3.043614387512207, "original": -3.043614387512207}` |

## Decision

Phase-2b **PASS**. Next requires `AUTHORIZE Phase 2c` (auxiliary W critic; actor not driven by W).

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\mdp_phase2b`
