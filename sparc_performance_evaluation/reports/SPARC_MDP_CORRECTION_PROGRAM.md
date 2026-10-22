# SPARC-V2X — MDP Correction Program (locked ladder)

**Purpose:** Prevent chat amnesia. Work proceeds only through Phases 0→1→2→3 with explicit authorization and stop rules.  
**Read first:** `SPARC_MDP_HANDOFF.json` (machine next-action)  
**How to execute (specifics):** `SPARC_MDP_EXECUTION_RUNBOOK.md`  
**Status:** `SPARC_MDP_PROGRAM_STATUS.md`  
**Closed paths:** RAO dual-policy Direct↔POPL selector (`RAO_PHASE1_DUAL_POLICY_SELECTOR_CLOSED.flag`). Do not reopen.

This file is the ladder overview. **Commands, inputs, feature lists, PASS numbers, and exit patches live in the runbook.** If the runbook and this file disagree, the runbook wins for execution detail; STATUS/HANDOFF win for what is authorized now.

## Hard rules (all agents)

1. **Read `SPARC_MDP_PROGRAM_STATUS.md` before any SPARC RL / MATLAB / training work.**
2. **Do only the current phase.** Completing Phase *k* does not authorize Phase *k+1*.
3. **Advance only by updating STATUS** after the phase deliverable exists and the user has authorized the next phase (or the phase’s own stop rule closes the program).
4. **No skipping.** Phase 2 training is forbidden until Phase 1 PASS. Phase 3 forbidden until Phase 2d PASS.
5. **No dual-policy selector retuning.** No final-test. No manuscript promotion from this program alone.
6. **Frozen baselines** (Direct, LVM, VAS, LAS, VLRO), governor, 175 kB envelope remain unless a later authorized phase explicitly changes them.

---

## Phase 0 — Lock the scientific ask

**Goal:** Freeze utility, decision problem, and success criterion in writing. No MATLAB. No training.

**Deliverable:** `SPARC_MDP_PHASE0_UTILITY_LOCK.md` containing exactly:
1. Utility thesis (default candidate: undiscounted deadline-valid `V_done`; state whether safety is constraint-only).
2. Decision problem: one executed pack per decision epoch; governor in transition; actor `log π` on **proposed** pack.
3. Success criterion for later phases (beat locked LVM/Direct under matched envelope; 15% is aspirational, not a Phase-1 gate).
4. Explicit closure note: RAO selector path closed.

**Exit:** User accepts Phase-0 doc → STATUS: Phase 0 DONE; Phase 1 becomes `AUTHORIZED` or `PENDING_USER_AUTH` per user word.

**Does not authorize:** diagnostics, collection, CA-PPO, γ changes.

---

## Phase 1 — Headroom & commitment diagnostic (training-free)

**Prerequisite:** Phase 0 DONE + Phase 1 authorized.

**Goal:** Discriminate credit/MDP misspec vs little headroom **before** any new actor training.

**Work:**
1. Define commitment / post-decision features from pre-radio information only (no outcome leakage).
2. Prefer reuse of existing audits (POPL / RAO Phase-1). Any new rollouts only if separately authorized and budgeted.
3. Tests:
   - Do commitment features predict held-out remaining `G_t` / `V_done` better than candidate-only features?
   - What fraction of decision epochs show pack-choice moving `V_done` by >ε? Report structural headroom proxies.

**Deliverable:** `SPARC_MDP_PHASE1_HEADROOM_REPORT.md` with PASS/FAIL and stop-rule outcome.

**PASS (proceed to Phase 2 only after user auth):** commitment features add predictive lift **and** material pack-choice swing exists under the locked env.  
**FAIL (close CA-PPO path; do not train):** no lift and/or negligible swing → environment/utility ceiling; pivot requires new authorized problem redesign, not Phase 2.

**Does not authorize:** PPO updates, γ/actor changes, CA-PPO.

---

## Phase 2 — Staged MDP hygiene (only if Phase 1 PASS)

**Prerequisite:** Phase 1 PASS + Phase 2 authorized. One step at a time; update STATUS after each step.

| Step | Change | Deliverable | Pass if |
|------|--------|-------------|---------|
| **2a** | `γ=1` + drain/terminal accounting only | `SPARC_MDP_PHASE2A_REPORT.md` | Eval `V_done` not worse; train objective matches report |
| **2b** | Full-pack joint `log π`; ratio on proposed; execute via governor | `SPARC_MDP_PHASE2B_REPORT.md` | Well-defined learning signal; no behavior regression vs 2a |
| **2c** | Auxiliary `W(Z, A_exec)` on realized returns; **do not** drive actor | `SPARC_MDP_PHASE2C_REPORT.md` | Held-out calibration / pack ranking |
| **2d** | Use `W` in advantage only if 2c PASS (else stop) | `SPARC_MDP_PHASE2D_REPORT.md` | Beats 2b on authorized train-dev; then locked val if authorized |

**Stop:** 2c FAIL → do not run 2d; mark CA-PPO advantage path closed.

**Does not authorize:** unbounded training, final-test, manuscript claims.

---

## Phase 3 — Bounded commitment-aware training

**Prerequisite:** Phase 2d PASS + Phase 3 authorized with explicit episode/pair budget.

**Goal:** Bounded CA-PPO (internal label) or equivalent under frozen governor/envelope/baselines.

**Deliverable:** `SPARC_MDP_PHASE3_REPORT.md` — matched `V_done` vs Direct/LVM; no final-test unless separately authorized.

---

## Relationship to CA-PPO note

The commitment-aware / post-decision critic idea is the **intended Phase 2–3 mechanism**. It is **not** authorized by this document alone. Phase 1 is the gate that decides whether that mechanism is worth training.

## Anti-amnesia checklist (paste into agent turns)

```
[ ] Read SPARC_MDP_PROGRAM_STATUS.md
[ ] Confirm current_phase and authorization
[ ] Do only that phase’s deliverable
[ ] Do not start next phase
[ ] Update STATUS when phase completes or fails stop-rule
```
