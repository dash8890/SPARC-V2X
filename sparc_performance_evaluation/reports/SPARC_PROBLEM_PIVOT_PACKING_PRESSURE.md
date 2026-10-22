# Problem pivot — high packing pressure

**Authorized by user:** `change the problem` (2026-10-19)  
**Replaces:** nominal d35 / tasks_per_source=3 residual-on-LVM program (CLOSED).

## Old problem (dead)

Beat LVM on **nominal** SPARC (density 35, 3 tasks/source).  
Evidence: little room above LVM. Residual learning on LVM is closed.

## New problem

Beat LVM on **high packing pressure**:

| Knob | Value |
|---|---|
| density | 35 |
| tasks_per_source | **4** |
| deadline × | 1.0 |
| helper CPU | [2e8, 1e9] |
| envelope / governor | unchanged (175 kB, SafetyGovernor) |
| objective | same \(J=V_{\mathrm{done}}\); \(P_{\mathrm{viol}}\le0.02\); ≤10 ms later |

Why this problem: it is the only robustness cell where a frozen pack policy beat LVM on all four locked-val seeds (POPL − LVM ≈ +1.33). That does **not** count as a win yet — those seeds are contaminated for fitting. The new work starts by checking whether the gap is real on **fresh** seeds.

## What we are not doing

- Claiming the old nominal problem is solved by moving the goalposts quietly.
- Training a new learner before the fresh-seed LVM gap is confirmed.
- Touching final-test `9300x` or fitting on `8200x`.

## Immediate gate (this phase)

Run matched **LVM / Direct / POPL** on fresh tps4 seeds.  
Pass only if POPL (or another frozen comparator) shows a **stable positive mean gap vs LVM** under the bars in the seed/protocol file.  
Fail ⇒ this pivot dies too; no new algorithm story.
