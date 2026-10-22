# Controlled Replay — Direct Helper Swap (seed 82002)

**Authorized:** seed 82002, epoch 33, task 19, helper 2→0 only.
**Not authorized:** training, architecture, STOP, 82012 substitution, final-test, implementation.

## Locked intervention

- Applied: **True**
- Epoch-33 assignment after swap: `{'task_id': 19, 'helper_id': 0, 'source_id': 6, 'payload_bytes': 27218, 'packet_count': 28}`
- Matched prefix through epoch 32: **True** (first state diff=None, first assign diff=33)

## Required outcomes

| Metric | Direct ref | Helper-swap |
|---|---:|---:|
| \(V_{\mathrm{done}}\) | 49.9532 | 52.4071 |
| ΔV | — | +2.4539 |
| Task 19 status | deadline_miss | completed |
| \(P_{\mathrm{viol}}\) | 0.0 | 0.0 |
| Lost Direct completions | — | none |
| Gained completions | — | [19] |

**Hypothesis result:** `positive_feasibility_evidence_local`

## Preservation / downstream

- Preservation OK (no lost Direct completions): **True**
- Lost value: 0.0000; gained value: 2.4539
- Post-epoch assignment diffs after 33: **0**

## Decision boundary

This result is local feasibility evidence on one validation trajectory. It does **not** authorize policy implementation, training, or a claim that SPARC surpassed LVM.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\direct_helper_swap_replay_82002\runs\episode_0001` vs reference `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\stage0_val_sparc_direct\runs\episode_0002`.
