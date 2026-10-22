# Stage 2 / JIPI — Closure

**Verdict:** JIPI-SPARC did not improve scheduling performance. Do **not** reopen Stage 1 or JIPI. Do **not** promote the JIPI checkpoint.

| Metric | Result |
|---|---|
| JIPI mean \(V_{\mathrm{done}}\) | 46.06 |
| Direct (matched) | 46.06 |
| Strongest matched competitor (LVM) | 46.82 |
| JIPI vs Direct | **0.00** |
| JIPI vs LVM | **−1.62%** |
| \(P_{\mathrm{viol}}\) | 0% |
| Validation KEEP rate | ≈ 99.8% |
| Training | 96/96 episodes (authorized once) |
| Selected checkpoint | Update 4 (train-only) |
| Final-test status | **Sealed** |

## What this establishes

1. **Editing a completed Direct pack is not an effective new policy** under the tested configuration: the learned editor collapsed to KEEP.
2. Safety and radio thresholds were not the bottleneck (\(P_{\mathrm{viol}}=0\); PRR ≈ 0.988–0.990; P95 ≈ 21–29 ms).
3. Separate closed-loop radio traces for JIPI vs Direct differ slightly; they do **not** demonstrate a radio advantage because completed tasks are identical.
4. Prior diagnostic evidence still holds: helper decisions have pack-level externalities (82002 benefit; 82011/helper-9 harm; 82012 substitution tradeoff).

## What this does not establish

It does **not** prove that opportunity-cost reasoning is useless. It shows that a KEEP/rebind/substitute editor over frozen Direct failed to convert that reasoning into higher realized \(V_{\mathrm{done}}\).

## Decision

- Close Stage 2 / JIPI as a **failed matched validation** of pack-edit RL.
- Do **not** open sealed final-test.
- Do **not** revise the manuscript to claim JIPI as an improvement.
- **Do not reopen** Stage 1 or JIPI architecture/training.

**Authorized next direction (design only until further approval):** RHO-SPARC — see `RHO_SPARC_ALGORITHM_SPECIFICATION.md`. Next concrete work is the joint-allocation opportunity audit and causal \(\widehat p\) prerequisite — **not** a new 96-episode training run.
