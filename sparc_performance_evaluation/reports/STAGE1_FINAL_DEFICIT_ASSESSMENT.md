# Stage 1 — Final Deficit Assessment

**No implementation or experiment is authorized by this assessment.**

Mean realized completed-value deficit per validation seed: **−5.01**

| Source | Value/seed | Share |
|---|---:|---:|
| Never assigned | 2.64 | 52.7% |
| Deadline misses | 2.37 | 47.3% |

Evidence basis: `STAGE1_VALUE_DEFICIT_DECOMPOSITION.md` (exact identity reconcile on locked val panel).

---

## What is now established

1. **Under-assignment is the largest contributor.** Stage 1 schedules fewer tasks than Direct PPO and loses completed value Direct captures. Lower offered bytes come from **fewer assignments**, not smaller average payloads.
2. **Deadline outcomes account for almost the entire remaining deficit.** Some tasks Direct completes are assigned by Stage 1 but miss deadlines.
3. **No compensating gain.** Stage 1 produces no Stage-1-only completed value in the reported comparison. Every reported difference favors Direct.
4. **The governor is not the cause.** Neither policy experiences governor intervention.

---

## Implication for the algorithm

Stage 1 altered scheduling behavior without improving realized task selection. Higher analytical pack value did not translate into additional deadline-valid completions.

This does **not** establish that the same-state scorer was intrinsically worse: policies diverged from epoch 0, so later queue/radio states may differ. The decomposition explains **where** value was lost, not every causal pathway producing those losses.

---

## Development decision (locked)

| Item | Status |
|---|---|
| Stage 1 | **Closed** (failed validation of conditional rescoring) |
| SPARC reference | **Prior Direct PPO** |
| Final-test set | **Sealed** |
| Further Stage-1 training | **Not authorized** |
| STOP action | **Not authorized** (poorly motivated: largest deficit is fewer assignments) |
| Architectural revision | **Not authorized** on this evidence alone |
| Manuscript promotion of Stage-1 scheduler | **Not authorized** |
| Implementation / new experiment | **Not authorized** by this assessment |

### Remaining technical question (future, separately justified only)

Whether existing Direct PPO can be improved to preserve productive assignment coverage while increasing deadline-valid completion probability — requires a **new** proposal and controlled validation, **not** reopening Stage 1.
