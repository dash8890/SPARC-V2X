# Direct vs LVM — Offline Matched-State Substitution Audit

**Diagnostic only.** Stage 1 closed. Prior Direct retained. Final-test sealed.
**Not authorized:** training, architecture change, STOP, implementation, controlled replay (until separately approved).

## Purpose

Establish whether the coverage deficit vs LVM can be converted into additional completed value via feasible single-task additions/substitutions that preserve Direct's successful assignments — before designing another learning algorithm.

## Answers to the three diagnostic questions

### 1. Candidate omission — why does Direct exclude LVM-completed tasks?

Only **two** LVM-only completed-value cases exist across the four val seeds:

| Seed | Task | Pathway | Offline cause |
|---|---:|---|---|
| 82012 | 1 | never_assigned_by_direct | Byte envelope: cannot add without displacing another Direct pack member |
| 82002 | 19 | deadline_miss | Already in Direct's pack (not an omission); fails deadline under chosen helper |

No pure ranking-only omission (absent from candidates / never considered) was found. The coverage pathway is a **byte-envelope displacement** problem, not source uniqueness or empty helper sets.

### 2. Replacement cost — what must be displaced to include an LVM-only task?

- **82012 / task 1:** Cheapest feasible single-task substitution displaces Direct task **13** (completed value ≈1.48) to free bytes for task 1 (value ≈3.74). Analytic net ≈**+2.26**, but this **sacrifices a Direct win**. Classified as conditional only — not a preserve-win candidate.
- Free additions under the 175 kB envelope: **0**.

### 3. Deadline quality — are better helpers available for missed tasks?

- **82002 / task 19:** Direct assigns helper **2** (slack ≈3.11 ms) while helper **0** is available with slack ≈9.53 ms and a higher analytic score. This is a **preserve-task-set helper_swap** candidate (no task displacement).

Analytic slack improvement ≠ realized completed value; closed-loop replay is required.

## Aggregate

- LVM-only completion cases: **2**
- Feasible add/substitute under Direct state constraints: **1**
- Of which additions: **0**, substitutions: **1**
- Promising & preserve Direct completed tasks: **1**
- Promising but would displace a Direct win: **1**

### Omission / constraint causes

```json
{
  "already_in_direct_pack": 1,
  "byte_envelope_requires_displacement": 1
}
```

## Replay candidates (offline filter only)

| Seed | Epoch | Mode | Detail | Analytic net | Preserves Direct |
|---|---:|---|---|---:|---|
| 82002 | 33 | helper_swap | task 19: helper 2→0 (slack 3.11→9.53) | 2.454 | True |

### Conditional (displaces a Direct-completed task)

```json
[
  {
    "sim_seed": 82012,
    "matched_direct_epoch": 33,
    "mode": "substitution",
    "add_task": 1,
    "add_helper": 2,
    "displace_task": 13,
    "displace_value": 1.4822216247549505,
    "analytic_net": 2.255888773114965,
    "task_value": 3.7381103978699155,
    "preserves_direct_completed": false,
    "note": "Offline net positive only if displaced Direct completion is valued less than added task; replay required."
  }
]
```

## Per-seed detail

### Seed 82001 — V Direct 39.84 vs LVM 38.26 (Δ=1.58)

Direct-only value 1.580 (tasks [0]); LVM-only value 0.000 (tasks []).


### Seed 82002 — V Direct 49.95 vs LVM 50.84 (Δ=-0.88)

Direct-only value 1.571 (tasks [1]); LVM-only value 2.454 (tasks [19]).

- Task **19** (v=2.454, pathway=deadline_miss): feasible=False, mode=None, cause=already_in_direct_pack, net_analytic=None
  - Helper audit: alt_more_slack=True, chosen={'helper_id': 2, 'finish_ms': 87.97307689151205, 'deadline_ms': 91.0867617731827, 'slack_ms': 3.1136848816706504, 'analytic_score': 0.02789413183927536, 'was_chosen': True}, best_alt={'helper_id': 0, 'finish_ms': 81.55569555447562, 'deadline_ms': 91.0867617731827, 'slack_ms': 9.531066218707082, 'analytic_score': 0.030089041218161583, 'was_chosen': False}

### Seed 82011 — V Direct 58.69 vs LVM 58.69 (Δ=0.00)

Direct-only value 0.000 (tasks []); LVM-only value 0.000 (tasks []).


### Seed 82012 — V Direct 35.74 vs LVM 39.48 (Δ=-3.74)

Direct-only value 0.000 (tasks []); LVM-only value 3.738 (tasks [1]).

- Task **1** (v=3.738, pathway=never_assigned_by_direct): feasible=True, mode=substitution, cause=byte_envelope_requires_displacement, net_analytic=2.255888773114965

## Deadline-helper opportunities

```json
[
  {
    "sim_seed": 82002,
    "task_id": 19,
    "task_value": 2.4539326207635574,
    "alt_exists": true,
    "chosen": {
      "helper_id": 2,
      "finish_ms": 87.97307689151205,
      "deadline_ms": 91.0867617731827,
      "slack_ms": 3.1136848816706504,
      "analytic_score": 0.02789413183927536,
      "was_chosen": true
    },
    "best_alt": {
      "helper_id": 0,
      "finish_ms": 81.55569555447562,
      "deadline_ms": 91.0867617731827,
      "slack_ms": 9.531066218707082,
      "analytic_score": 0.030089041218161583,
      "was_chosen": false
    }
  }
]
```

## Decision

| Gate | Result |
|---|---|
| Offline preserve-win opportunity found? | **Yes** — 1 helper_swap (82002 / task 19) |
| Offline coverage add without displacing Direct wins? | **No** — only byte-envelope substitution that displaces task 13 |
| Controlled replay (82002 helper_swap)? | **Done** — see `DIRECT_HELPER_SWAP_REPLAY_82002.md` |
| Implementation / training / STOP / final-test? | **Not authorized** |
| 82012 substitution replay? | **Not authorized** |

### Controlled replay result (82002 only)

| Metric | Direct ref | Helper-swap |
|---|---:|---:|
| \(V_{\mathrm{done}}\) | 49.953 | 52.407 |
| ΔV | — | **+2.454** |
| Task 19 | deadline_miss | **completed** |
| Lost Direct completions | — | none |
| \(P_{\mathrm{viol}}\) | 0% | 0% |

Local positive feasibility evidence for helper selection on this trajectory only. Does **not** authorize implementation, training, or SPARC>LVM claims. Repeatability and LVM-beating remain separate gates.
