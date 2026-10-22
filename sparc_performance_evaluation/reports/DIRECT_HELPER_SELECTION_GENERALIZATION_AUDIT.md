# Helper-Selection Generalization Audit (Direct Validation Panel)

**Local feasibility:** PASS (82002 helper-swap replay).
**Generalization:** NOT ESTABLISHED.
**Reference:** prior Direct PPO. Stage 1 closed. Final-test sealed.
**Not authorized:** implementation, training, architecture, additional replay, STOP.

## Question

How often do all four conditions coincide on locked Direct validation decisions,
including assignments Direct already completes?

## Primary thresholds (observable)

- Tight chosen slack: `< 5.0 ms`
- Material alternate gain: `Δslack ≥ 5.0 ms`
- Preserve pack/bytes: same-task helper rebind only
- Plausibility screen: alt slack > 0 and destination helper not already in pack

## Aggregate (primary)

| Quantity | Count |
|---|---:|
| Assignment slots | 70 |
| With ≥2 helper options | 69 |
| C1∧C2 (tight + better alt) | 16 |
| All four conditions | 16 |
| … of which deadline_miss | 1 |
| … of which completed | 15 |
| Completed with C1∧C2 only | 15 |

Seeds with any four-condition hit: `[82001, 82002, 82011, 82012]`
Seeds with miss-class four-hit: `[82002]`

## Per-seed

| Seed | V_done | Assignments | Multi-helper | C1∧C2 | Four | Miss-four | Completed-four | Completed C1∧C2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 82001 | 39.84 | 16 | 16 | 3 | 3 | 0 | 3 | 3 |
| 82002 | 49.95 | 17 | 17 | 3 | 3 | 1 | 2 | 2 |
| 82011 | 58.69 | 23 | 22 | 5 | 5 | 0 | 5 | 5 |
| 82012 | 35.74 | 14 | 14 | 5 | 5 | 0 | 5 | 5 |

## Four-condition events (primary)

| Seed | Epoch | Task | Status | Helper | Slack | →Helper | →Slack | Δ | Value | Class |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 82001 | 0 | 0 | completed | 5 | 2.04 | 2 | 8.64 | 6.60 | 1.580 | success_at_risk_if_changed |
| 82001 | 33 | 4 | completed | 9 | 2.38 | 2 | 74.34 | 71.96 | 4.447 | success_at_risk_if_changed |
| 82001 | 66 | 5 | completed | 11 | 0.65 | 2 | 20.69 | 20.04 | 1.542 | success_at_risk_if_changed |
| 82002 | 33 | 7 | completed | 7 | 0.52 | 0 | 25.56 | 25.04 | 2.156 | success_at_risk_if_changed |
| 82002 | 33 | 19 | deadline_miss | 2 | 3.11 | 0 | 9.53 | 6.42 | 2.454 | miss_recovery_candidate |
| 82002 | 66 | 17 | completed | 11 | 2.55 | 0 | 54.03 | 51.48 | 3.302 | success_at_risk_if_changed |
| 82011 | 0 | 15 | completed | 4 | 1.37 | 9 | 36.53 | 35.15 | 2.165 | success_at_risk_if_changed |
| 82011 | 0 | 6 | completed | 4 | 1.61 | 9 | 29.06 | 27.45 | 2.891 | success_at_risk_if_changed |
| 82011 | 0 | 3 | completed | 1 | 2.78 | 9 | 13.88 | 11.10 | 1.257 | success_at_risk_if_changed |
| 82011 | 33 | 16 | completed | 5 | 0.60 | 9 | 38.80 | 38.20 | 3.272 | success_at_risk_if_changed |
| 82011 | 66 | 8 | completed | 8 | 0.29 | 9 | 52.57 | 52.28 | 1.527 | success_at_risk_if_changed |
| 82012 | 0 | 9 | completed | 10 | 4.25 | 6 | 50.59 | 46.34 | 2.981 | success_at_risk_if_changed |
| 82012 | 0 | 15 | completed | 4 | 4.77 | 6 | 39.87 | 35.10 | 2.830 | success_at_risk_if_changed |
| 82012 | 0 | 6 | completed | 2 | 3.10 | 6 | 14.23 | 11.13 | 4.494 | success_at_risk_if_changed |
| 82012 | 33 | 13 | completed | 0 | 4.30 | 6 | 78.41 | 74.12 | 1.482 | success_at_risk_if_changed |
| 82012 | 66 | 17 | completed | 2 | 1.96 | 6 | 13.66 | 11.70 | 4.120 | success_at_risk_if_changed |

## Harm surface: successful assignments with C1∧C2

These are Direct completions that a naive tight-slack rebind rule might also touch.

| Seed | Epoch | Task | Helper | Slack | →Helper | →Slack | Δ | Four? | Dest contended? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 82001 | 0 | 0 | 5 | 2.04 | 2 | 8.64 | 6.60 | True | False |
| 82001 | 33 | 4 | 9 | 2.38 | 2 | 74.34 | 71.96 | True | False |
| 82001 | 66 | 5 | 11 | 0.65 | 2 | 20.69 | 20.04 | True | False |
| 82002 | 33 | 7 | 7 | 0.52 | 0 | 25.56 | 25.04 | True | False |
| 82002 | 66 | 17 | 11 | 2.55 | 0 | 54.03 | 51.48 | True | False |
| 82011 | 0 | 15 | 4 | 1.37 | 9 | 36.53 | 35.15 | True | False |
| 82011 | 0 | 6 | 4 | 1.61 | 9 | 29.06 | 27.45 | True | False |
| 82011 | 0 | 3 | 1 | 2.78 | 9 | 13.88 | 11.10 | True | False |
| 82011 | 33 | 16 | 5 | 0.60 | 9 | 38.80 | 38.20 | True | False |
| 82011 | 66 | 8 | 8 | 0.29 | 9 | 52.57 | 52.28 | True | False |
| 82012 | 0 | 9 | 10 | 4.25 | 6 | 50.59 | 46.34 | True | False |
| 82012 | 0 | 15 | 4 | 4.77 | 6 | 39.87 | 35.10 | True | False |
| 82012 | 0 | 6 | 2 | 3.10 | 6 | 14.23 | 11.13 | True | False |
| 82012 | 33 | 13 | 0 | 4.30 | 6 | 78.41 | 74.12 | True | False |
| 82012 | 66 | 17 | 2 | 1.96 | 6 | 13.66 | 11.70 | True | False |

## Sensitivity

| Band | Tight | Δslack | C1∧C2 | Four | Miss-four | Completed-four | Completed C1∧C2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary_T5_D5 | 5.0 | 5.0 | 16 | 16 | 1 | 15 | 15 |
| T10_D5 | 10.0 | 5.0 | 26 | 26 | 1 | 25 | 25 |
| T5_D3 | 5.0 | 3.0 | 18 | 18 | 1 | 17 | 17 |
| T10_D3 | 10.0 | 3.0 | 28 | 28 | 1 | 27 | 27 |

## Proposed causal trigger (not implemented)

**Name:** `tight_slack_helper_rebind`

At decision time, after Direct forms its pack:

- Task t is admitted in the current Direct pack.
- Predicted slack of chosen helper h* is < 5.0 ms.
- Another feasible same-task helper h' exists with slack(h') - slack(h*) >= 5.0 ms.
- Rebind changes only helper_id for t; remaining pack members and byte totals unchanged.
- Destination helper h' is not already used by another assignment in the same pack (offline harm screen; closed-loop still required).

**Action:** Replace (t, h*) with (t, h') where h' maximizes predicted slack among feasible helpers.

This is an observable state predicate, not a seed-specific exception.
The 82002/task19 replay is one confirming instance under this predicate.

## Bounded replay protocol (not authorized now)

1. Freeze Direct PPO checkpoint, governor ON, B_max=175 kB, val seeds only.
1. Enumerate all primary-threshold four-condition events from this audit.
1. For each event, run one matched closed-loop helper-rebind intervention (same pre-decision prefix; alter only that assignment's helper).
1. Compare vs frozen Direct reference: ΔV, task deadline outcome, lost/gained completions, P_viol, post-epoch assignment divergence.
1. PASS gate for generalization evidence: majority of miss-class events improve or tie V_done with preservation_ok and no safety deterioration; AND completed-class trigger firings do not produce systematic negative ΔV.
1. Do not train, change architecture, touch final-test, or promote implementation from this protocol alone.

Primary four-condition candidate events enumerated: **16** (miss-class 1, completed-class 15).

## Decision

| Gate | Result |
|---|---|
| Local feasibility (82002) | **PASS** |
| Generalization across panel | **NOT ESTABLISHED** |
| Repeatable pattern signal | `predicate_fires_panel_wide_but_mostly_on_successes` |
| Implementation / training | **Not authorized** |
| Additional replay | **Not authorized** (protocol specified only) |

15 successful Direct assignments also show tight chosen slack with a materially better alternate under primary C1∧C2; 15 of those also pass C3∧C4. Any rule must be stress-tested against these success cases first.

Offline predicted slack ≠ realized completion. Destination-contention filter is a necessary but not sufficient harm screen. Closed-loop replay remains required before implementation. Origin-helper sharing does not uniquely identify the confirmed miss (several completions share it).

Audit only. No implementation, training, architecture change, STOP, final-test, or additional replay authorized by this report.
