# RHO — Joint-Allocation Opportunity Audit

**Status:** Protocol specified; offline audit authorized as development work  
**Not authorized:** RHO PPO training; final-test; fitting \(\widehat p\) to locked validation labels; manuscript claims

## Purpose

Decide whether a **non-learned** joint pack optimizer can identify assignment packs that dominate Direct and/or LVM reference packs using only information available at scheduling time.

This is the decisive prerequisite before any RHO training authorization.

## Inputs (development artifacts only)

| Source | Use |
|---|---|
| Direct Stage-0 / training episode audits + `candidate_epochs.json` | Reference packs \(A^{\mathrm{Direct}}\) and candidate tables |
| LVM matched Stage-0 audits | Reference packs \(A^{\mathrm{LVM}}\) |
| Existing helper/queue projection semantics | Pack-dependent finish / load estimates |
| Train-seed episodes (72000+) when needed for \(\widehat p\) calibration | Calibration **only** — never val/final-test labels |

Locked validation seeds (82001/02/11/12) may appear as **held-out reporting** states for the audit, but their realized outcomes must **not** be used to fit \(\widehat p\) or profile levels.

## Non-learned planner under test

- Fixed mid-level planning profile (or a small fair grid chosen on **train** only).  
- Beam search with pre-declared width / expand limit / latency budget.  
- Interim \(\widehat p\): pack-projected deadline feasibility softened by a train-calibrated slack map (documented); binary projection alone is insufficient for a PASS on the \(\widehat p\) gate.

## Recorded quantities (per decision state)

- Reference vs joint predicted pack value \(\sum v_i\widehat p_i\)  
- Byte use; helper incremental loads \(L_h\)  
- Feasibility flags  
- Search latency (ms); beam stats  
- Assignment diffs vs Direct / LVM  
- Where separately authorized: matched closed-loop Δ\(V_{\mathrm{done}}\), \(P_{\mathrm{viol}}\)

## Verdict criteria

| Result | Criterion |
|---|---|
| **PASS** | Demonstrated positive net completed-value opportunities attributable to joint allocation; no unacceptable safety or timing regression |
| **FAIL** | Mainly rearranges assignments without improving realized value, or requires unavailable future information |
| **INCONCLUSIVE** | Analytical opportunity exists, but closed-loop benefit or causal \(\widehat p\) quality remains unverified |

Benchmark: strongest matched competitor (LVM 46.82). Ambition: ≈ 53.84 (+15%). Smaller gains judged on merit.

## Explicit exclusions

- No RHO PPO training  
- No architecture search / HPO beyond the pre-declared audit grid  
- No final-test  
- No selective beam widening on favorable validation seeds  
