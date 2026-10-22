# LVM-RPI-Distill — PATH CLOSED (failure)

**Date:** 2026-10-19  
**Status:** CLOSED. Do not authorize C0, critic, actor, or distill for this design.  
**Flag:** this file + Canon §0 + HANDOFF `closed.lvm_rpi_distill`.

---

## 1. What failed

**LVM-RPI-Distill** (residual / rollout-PI learning on top of LVM) is dropped as a failure case under the locked SPARC environment and success criterion \(J>J_{\mathrm{LVM}}\).

Reason in one line: the design assumes harvestable LVM mistakes; existing evidence already shows that mass of material LVM-relative headroom is too thin to justify building the residual learner.

---

## 2. Evidence used (no new C0 campaign)

| Fact | Source | Implication for RPI |
|---|---|---|
| Direct∪POPL oracle ≈ +1.9% vs Direct; ~69% disagreeing packs \(\Delta G=0\) | RAO Phase-1 | Tiny complementarity among strong generators |
| POPL vs LVM nominal / locked-val class | POPL robustness / val | Not a clear, robust LVM beat |
| Pack-defect decode “fix LVM omit/bind” | POPL pack-defect FAIL (−8.5) | Local LVM-mistake patches destroy value |
| RAO / routers / W-advantage | CLOSED flags | Switching or reweighting frozen packs ≠ win |
| Only clear POPL>LVM regime | `workload_tps4` (+1.33) | Niche, not nominal-panel residual program |

C0 was the plan’s kill gate. It was not required as a ritual after this evidence already answers the kill question the same way: **do not proceed to distill.**

---

## 3. Agent process failure (why this plan was locked anyway)

1. Diagnosis already required: *if swing vs LVM tiny → stop RL superiority claim* (`SPARC_RESEARCH_DIAGNOSIS_AND_OBJECTIVE.md` §5).
2. Agent locked a coherent teacher/deploy story (D1) **before** enforcing that kill with LVM as anchor.
3. C0 was used to defer the kill into a future measurement, which is algorithm-first with extra steps.
4. When challenged, agent hedged instead of closing the path.

That process error is part of this close record.

---

## 4. What remains true (do not throw away)

- Objective \(J=V_{\mathrm{done}}\) with safety/runtime constraints.
- LVM is the right **comparator**, not an easy baseline to avoid.
- Simulator-grounded labels beat RHO-style surrogates *when* headroom exists.
- Frozen forbids: no final-test, no weakening LVM, no metric swap, no another router/decode loop on the same stack.

---

## 5. Non-claims

- This is **not** “RL can never help V2X.”
- This is **not** a completed fresh-seed C0 distribution paper.
- This **is** a stop on residual-on-LVM as the program’s next build under the locked nominal problem.
