# HQ-SPO — Helper-Queue Sequential Pack Offloading

**Status:** ACTIVE BUILD (not a kill-gate theater)  
**Date:** 2026-10-19  
**Replaces as active novel claim:** radio-endogenous EC-SPO (`SPARC_NOVEL_RL_NECESSARY_PROBLEM.md`) — **closed** after G9 FAIL×2.

---

## 1. Problem (one sentence)

Choose a sequence of feasible computation packs under NR-V2X Mode-2 + 175 kB governor + \(P_{\mathrm{viol}}\le 0.02\) so that **helper-queue commitments** from today’s pack change which future tasks can finish before deadline — maximize expected deadline-valid completed value \(J=\mathbb E[V_{\mathrm{done}}]\).

## 2. Why this is sim-honest (unlike EC-SPO radio)

WiLab task layer already updates `helper.ready_time_ms` / `queue_cycles` when work is delivered (`task_layer.apply_radio_outcomes`). G9 showed pack→CBR is ~null here; pack→**helper backlog** is the endogenous channel we actually have.

## 3. Why not “weak pack-search paper”

| Weak (rejected) | This |
|---|---|
| Anecdote edits / epoch-0 panel as the paper | Causal premise on queue→future \(V\), then a **deployable** policy |
| Claim radio congestion without evidence | Claim queue opportunity cost with audit |
| PackSwap router with `n_Direct=0` | Search teacher → gated policy that actually fires |

## 4. Premise audit (running)

Causal EMPTY vs CONTENDED (same-helper-heavy) pack @ epoch 0, LVM residual after; measure Δ\(V_{\mathrm{done}}\), Δ max/mean helper ready-time, Δ later completions.  
Script: `run_hq_spo_premise_audit.py`  
Pass bar: mean \(|\Delta V|\) materially > 0 on seeds with real packs **or** clear ready-time↑ with downstream miss rate↑.

## 5. Contingency (pre-committed — not a stop)

| Premise | Next work (no idle) |
|---|---|
| **PASS** | Build queue-aware search / short-horizon rollout teacher + deployable policy; eval flat win vs LVM & Direct on fresh tps4 |
| **FAIL** | Drop HQ-SPO novelty claim; **immediately** expand tps4 Search-RPI (more forks/seeds/families) → distill/gate a policy that fires on \(\Delta^\star>0\) states; eval flat win vs LVM |

Either branch ends in **publishable method numbers**, not another hard stop.

## 6. Success

Flat win vs the **best feasible** of the locked five-baseline suite at \(P_{\mathrm{viol}}\le 0.02\), with decision latency reported. Ties / few defensible losses OK.

**Suite (authoritative):** `SPARC_HQ_SPO_BASELINE_SUITE.md` — greedy \(v/b\), CBR-threshold, myopic dual (LVM-class), one-step CE MILP, PPO-Lagrangian.

**Not the success definition:** informal bake-offs against only frozen LVM/Direct outside that suite.
