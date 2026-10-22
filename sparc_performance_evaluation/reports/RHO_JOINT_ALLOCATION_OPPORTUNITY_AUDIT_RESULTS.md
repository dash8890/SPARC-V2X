# RHO Joint-Allocation Opportunity Audit — Results

**Verdict:** `INCONCLUSIVE`

Analytical pack-score opportunities are reported; realized completed-value and causal p̂ ranking quality are not yet verified by closed-loop replay.

## Offline (non-RL mid profile)

Beam width=8, max expansions=200, profile=`dm_hm_bm`.

| Reference | n states | mean ΔE[V] | frac improved | mean / max search ms |
|---|---:|---:|---:|---:|
| Direct | 5 | +2.5997 | 1.000 | 11.06 / 15.66 |
| LVM | 5 | +0.3230 | 0.600 | 11.40 / 14.50 |

## Authorization boundary

- No RHO PPO training.
- No final-test.
- Closed-loop confirmation of promising packs: **not run** (needs separate approval).

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\rho_joint_allocation_opportunity_audit`.
