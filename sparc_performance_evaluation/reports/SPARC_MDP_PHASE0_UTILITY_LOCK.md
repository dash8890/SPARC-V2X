# SPARC MDP Phase-0 Utility Lock

**Authorized:** 2026-10-19  
**Status:** LOCKED  
**Sources:** `SPARC_MDP_CANON.md` §3; math in `SPARC_MDP_REFORMULATION_MATH.md`  
**Does not authorize:** MATLAB, PPO, Phase 1 diagnostics, CA-PPO, RAO retuning, final-test.

---

## 1. Utility thesis

- **Primary scalar:** undiscounted deadline-valid completed value \(V_{\mathrm{done}}\) / episode return \(G_0\) with \(\gamma=1\).
  \[
  V_{\mathrm{done}}=\sum_i v_i\,\mathbf{1}\{\text{task }i\text{ deadline-valid done}\},\quad
  G_t=\sum_{k=t}^{T_{\mathrm{end}}} r_k,\quad r_k=\Delta V_{\mathrm{done},k}.
  \]
  \(T_{\mathrm{end}}\) includes locked drain so late assignments still credit if they complete validly. Full statement: `SPARC_MDP_REFORMULATION_MATH.md` §1.
- **Safety:** **constraint-only** — SafetyGovernor + 175 kB envelope + feasible set \(\mathcal{F}(O_t)\). Not a soft penalty in \(r_t\) under this lock.
- **Forbidden as training reward:** RHO-style analytic \(E[V]\), slack-only scores, invented completion probabilities \(\hat p\), RAO dual-policy selector retuning objectives.

---

## 2. Decision problem

Canonical math: `SPARC_MDP_REFORMULATION_MATH.md` §§3–4. Locked summary:

- **Decision epoch:** controller epoch (with candidate work when scheduling is possible).
- **Action:** one complete feasible assignment pack \(A_t\in\mathcal{F}(O_t)\).
- **Executed action:** \(A_t^{\mathrm{exec}}=g(A_t,O_t)\) (governor unchanged; inside the transition).
- **Policy likelihood:** joint \(\log\pi_\theta(A_t\mid H_t)\) on the **proposed** pack (full pack; sequential decoder is internal only).
- **Post-decision state:** \(S_t^+=\Phi(S_t,A_t^{\mathrm{exec}})\) — deterministic resource commitments only; no assumed radio success.
- **Stochastic step:** \(S_{t+1}\sim P(\cdot\mid S_t^+)\).
- **Critics (Phase 2+ only, not authorized now):** \(V(Z_t)\); auxiliary \(W(Z_t,A_t^{\mathrm{exec}})\) on realized \(G_t\).

One line: choose pack → land in commitment state \(S_t^+\) → optimize \(\mathbb{E}[G_0]\) at \(\gamma=1\).

---

## 3. Success criterion (later phases)

| Phase | Gate |
|-------|------|
| **1** | Commitment-feature predictive lift **and** material pack-choice swing (thresholds in `SPARC_MDP_EXECUTION_RUNBOOK.md` / Canon §5). **Not** +15%. |
| **2a–2d** | Staged MDP hygiene gates; 2c must calibrate before 2d uses \(W\) in advantages. |
| **3** | Matched episode \(V_{\mathrm{done}}\) vs frozen Direct and LVM under an explicit user budget. |

**Comparators (frozen):** Direct, LVM; retain VAS, LAS, VLRO. Governor + 175 kB unchanged.

**Aspirational:** \(V\approx 53.84\) vs LVM \(\approx 46.82\) (~15%) is recorded, **not** a Phase-1 or Phase-2a acceptance gate.

---

## 4. Closed paths

- **RAO Stage-3 dual-policy Direct↔POPL causal selector:** CLOSED after G4 FAIL.  
  Flag: `RAO_PHASE1_DUAL_POLICY_SELECTOR_CLOSED.flag`  
  Report: `RAO_PHASE1_G1_G7_REPORT.md`  
  Do not reopen corrective tuning on that mechanism.
- **Final-test seeds** (`9300x`): sealed.
- **Locked val** `82001/02/11/12`: forbidden for fitting / threshold / model selection in this program.

---

## 5. Explicit non-goals of Phase 0

No MATLAB, no PPO, no new pair collection, no CA-PPO implementation, no manuscript promotion.

**Next (not started):** user must say `AUTHORIZE Phase 1` before any headroom diagnostic script runs. Phase-1 contract is locked in `SPARC_MDP_CANON.md` §5.
