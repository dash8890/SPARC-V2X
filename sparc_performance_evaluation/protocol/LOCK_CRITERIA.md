# Evidence-package lock criteria (manuscript)

**Rule:** The SPARC-V2X performance evidence package must **not** be marked `LOCKED` / shipped for the manuscript until every item below is complete with provenance. Partial historical or validation-only results do not satisfy lock.

## Hard blockers (cannot lock without)

### Principal baselines
- [ ] **B1** Corrected VA + governor — final-test complete
- [ ] **B2** Min-Finish + governor — final-test complete
- [ ] **B3** Direct PPO + governor — trained, checkpoint selected on validation only, final-test complete
- [ ] **B4** Unshielded Lagrangian PPO — **implemented** (`train_wilab_lagrangian_ppo.py`); train+final-test via `campaign_queue.py`
- [ ] **B5** ITORA-inspired — **mapping audit frozen** + **scoring implemented**; final-test via `campaign_queue.py`

### Ablations (Fig. 9)
- [ ] **A0** Full SPARC (reuse final-test)
- [ ] **A1** Without safety gates (frozen SPARC; byte/physical limit retained)
- [ ] **A2** ≡ B1 (reuse, do not duplicate)
- [ ] **A3** Without VA-relative PPO (separate train)
- [ ] **A4** ≡ B3 (reuse)
- [ ] **A5** Residual-only logits + VA-relative advantages (separate train)
- [ ] **A6** Random residual init (separate train)

### Experiment families / figures
- [ ] **E2 / Fig. 4** Density sweep (value + safety) for all six methods
- [ ] **E3 / Fig. 5** Safety–value tradeoff + byte-envelope sensitivity
- [ ] **E4 / Fig. 6** Workload and relative-deadline sensitivity (+ archived safety)
- [ ] **E5 / Figs. 7–8** Radio diagnostics and closed-loop admission traces
- [ ] **E6 / Fig. 9** Ablation panels with CIs
- [ ] **E7 / Fig. 10** Governor enforcement and controller cost
- [ ] **E8 / Fig. 11** Generalization / robustness
- [ ] **E9 / Table IV** Complexity and B5 adaptation audit
- [ ] **Table III** Final-test six-method comparison (not validation-only)
- [ ] **Fig. 3** Includes Direct PPO validation curve alongside SPARC

### Statistics and seals
- [x] Final-test seeds sealed and disjoint (**n=5** per operating point; see `SEED_PARTITION.md`)
- [ ] Simultaneous (or predeclared) inference across five principal contrasts on those 5 seeds
- [ ] Safety feasibility procedure applied; invalid windows reported separately
- [ ] G0–G9 accepted with evidence paths
- [ ] `protocol_hashes.json` re-sealed after all campaigns; `lock.status = LOCKED`

Budget reminder: 1 train run / learned config · 4 validation pairs · **5 final-test pairs per condition** · reuse calibration. Do not expand final-test back to 30 without a protocol amendment.

## Explicit non-lock states

| Status | Meaning |
|---|---|
| `SPEC_ONLY` | Protocol written; campaigns incomplete |
| `PARTIAL` | Some offline/historical/final-test waves running or done |
| `LOCKABLE` | All hard blockers checked |
| `LOCKED` | Hashes sealed; no further retuning from final-test |

**Current status: `PARTIAL` — not lockable.**

## Ownership note

Gaps are expected mid-campaign. They are **not optional for manuscript lock**. Do not ship Tables II–IV / Figs. 3–11 as final evidence until this checklist is complete.
