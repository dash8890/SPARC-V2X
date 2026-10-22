# CPI-SPARC vs LVM-Rollout PI — Independent Opinion

**Date:** 2026-10-19  
**Status:** Load-bearing opinion. Updates the win path in `SPARC_CPI_EVALUATION_AND_ROADMAP.md`.  
**Not:** implementation authorization.

---

## 1. What the rollout proposal actually claims

**LVM-Rollout Policy Improvement + RL Distillation:**

1. Offline: at state \(s\), form candidate set \(\mathcal C\) including \(A^{\mathrm{LVM}}\), optionally VLRO/Direct/learned edits.
2. Score each \(A\in\mathcal C\) by **simulator** \(Q_H^{\mathrm{LVM}}(s,A)\) (LVM continuation / bootstrap after horizon \(H\)).
3. Teacher: \(A^\star=\arg\max_{\mathcal C} Q_H^{\mathrm{LVM}}\). With exact eval, \(Q(s,A^\star)\ge Q(s,A^{\mathrm{LVM}})\).
4. Distill \(\pi_\theta\approx A^\star\); fit \(\hat Q_\phi\approx Q_H^{\mathrm{LVM}}\).
5. Deploy: execute learned alt only if \(\hat\Delta=\hat Q(A)-\hat Q(A^{\mathrm{LVM}})>\tau\); else LVM.

---

## 2. Independent verdict

### Agree — this has a stronger *training-label* basis than free PPO or CPI-as-surrogate-first

| Point | Assessment |
|---|---|
| Never invent a scheduler from scratch; LVM is base | **Agree** — same as CPI, matches map |
| Teacher from **actual** sim downstream \(V\), not RHO-style pack score | **Agree** — strongest correction to RHO |
| Always keep \(A^{\mathrm{LVM}}\) in \(\mathcal C\) ⇒ offline PI over that set | **Agree** *when evaluation is exact* |
| Multi-step \(H\) can capture coordinated future packs | **Agree** — real advantage over one-step CPI labels |
| VLRO as **generator**, not anchor | **Agree** — VLRO mean ≪ LVM |
| \(\Delta=0\) states add no false “improvement” signal | **Agree** |
| Distill + runtime LVM fallback | **Agree** as deploy shape |

### Disagree / qualify — do not take at face value

**1. “Fundamentally different from CPI” is overstated.**  
CPI’s paired label \(\Delta G = G(A\rightarrow\pi_{\mathrm{LVM}})-G(A_{\mathrm{LVM}}\rightarrow\pi_{\mathrm{LVM}})\) *is* simulator-grounded LVM-relative advantage (one deviation + LVM continuation). Rollout PI generalizes that to **search over a candidate set with horizon \(H\)** and uses the **argmax as a teacher**. That is a real upgrade to *how labels/teachers are produced*, not a different scientific target.

**2. Exact sim is only offline.**  
At deployment you again trust \(\hat\Delta>\tau\) — a learned surrogate. The sentence “chosen because the real simulator says it beats LVM” applies to **dataset construction**, not to the shipped controller. Runtime risk is the same class as CPI certification: false accepts break the LVM floor.

**3. Guarantee is only over the candidate set.**  
\[
Q(s,A^\star)\ge Q(s,A^{\mathrm{LVM}})
\]
holds for \(A^\star\in\mathcal C\). If the good pack is never proposed, PI cannot find it. Coverage of \(\mathcal C\) (edit operators + VLRO/Direct) is the search problem; it does not disappear.

**4. Cost scales as \(\lvert\mathcal C\rvert\times\) (rollout length).**  
Multi-\(H\) and many candidates multiply MATLAB load. \(H=T-t\) is “optimal over \(\mathcal C\)” only if you can afford full residual episodes per candidate. Practical \(H=1,2,3\) needs a terminal \(V^{\mathrm{LVM}}(s_{t+H})\) — another approximator unless you roll to episode end.

**5. Headroom still unknown.**  
Same as CPI: ~0.84% POPL−LVM on contaminated val seeds ≠ enough mass for a robust mean win or 15%. Rollout PI does not create headroom; it **measures and harvests** it if present.

**6. Distillation can wash out rare gems.**  
If \(A^\star\neq A^{\mathrm{LVM}}\) is rare, BC on \(A^\star\) is mostly “imitate LVM,” and the interesting tail may be poorly fit unless training explicitly upweights improving states.

---

## 3. Comparison table

| Dimension | CPI-SPARC (prior note) | LVM-Rollout PI + distill |
|---|---|---|
| Anchor | LVM | LVM |
| Offline label | Paired \(\Delta G\), usually one-step deviation | Sim \(Q_H\) over full candidate set; \(A^\star=\arg\max\) |
| Improvement theorem | If \(\underline A\le A\) for accepts | Exact PI **offline** over \(\mathcal C\); deploy approx |
| Multi-epoch coordination | Weak unless extended | Explicit via \(H\) |
| Deploy | Certify \(\underline A_B>0\) | \(\hat\Delta>\tau\) else LVM |
| Surrogate risk | Critic ensemble | \(\hat Q\) / \(\hat\Delta\) |
| MATLAB cost | 2 branches × pairs | \(\lvert\mathcal C\rvert\) branches × horizon |
| Fit to diagnosis | Strong | **Stronger on teacher quality** |

---

## 4. Revised architectural commitment

**Commit to: LVM-anchored improvement with simulator-verified teachers, then conservative distill+gate at deploy.**

Concrete synthesis (not two competing projects):

1. **Offline teacher (Rollout-PI core):** Build \(\mathcal C_t=\{A^{\mathrm{LVM}},A^{\mathrm{VLRO}},A^{\mathrm{Direct}},\mathrm{edits}\}\); evaluate with simulator under LVM continuation (start with \(H=1\) full residual return to episode end if budget allows; only then consider \(H>1\) short-horizon + bootstrap).
2. **Record** \(A^\star\), \(\Delta_H=Q(A^\star)-Q(A^{\mathrm{LVM}})\).
3. **Learn** (a) \(\hat Q\) or \(\hat\Delta\) regressor, (b) optional policy/edit model imitating \(A^\star\) on states with \(\Delta_H>\varepsilon\).
4. **Deploy CPI-like gate:** execute non-LVM only if \(\hat\Delta>\tau\) with \(\tau\) / calibration set so held-out false-accept rate ≤ α; else LVM.

Call this **LVM-RPI-Distill** (rollout policy improvement + distillation). Treat “CPI-SPARC” as the deploy+certification pattern; treat “rollout PI” as the **label/teacher engine**. Do not run two parallel forks.

---

## 5. Updated win roadmap (replaces algorithm-first CPI training)

Phases still sequential; each needs authorize. Locked val `8200x` not for fitting; final-test sealed.

| Phase | Deliverable | Pass / fail |
|---|---|---|
| **D1** | Formulation lock: LVM-RPI-Distill math + objectives + non-claims | User accept |
| **C0** | Consequentiality: on **fresh** seeds, fixed \(\mathcal C\) editors+LVM+VLRO+Direct; sim-eval \(\Delta\); report mass of \(\Delta>0\), \(\mathbb{E}[\Delta_+]\), cost | Predeclared mass/magnitude; else **STOP** |
| **C0b** | Horizon study on fit only: \(H=1\) full residual vs short \(H\) — pick one protocol | Freeze \(H\) before scaling |
| **C1** | Deploy-gate calibration: \(\hat\Delta\) or conservative bound; false-accept ≤ α on held-out forks | Else **STOP** |
| **C2** | Scale paired/rollout dataset under frozen C0 protocol | Hygiene report |
| **C3** | Fit \(\hat Q\)/ \(\hat\Delta\); optional imitate \(A^\star\) on improving states | Held-out ranking / capture |
| **C4** | Closed-loop controller on **dev** seeds | \(J>J_{\mathrm{LVM}}\), safety, ≤10 ms |
| **C5** | Robustness panel | Win vs LVM with allowed ties/defensible losses |
| **C6** | Release decision / final-test only if C4–C5 pass | Separate auth |

**Win definition unchanged:** \(J>J_{\mathrm{LVM}}\), \(P_{\mathrm{viol}}\le0.02\), \(T_{\mathrm{decision}}\le10\,\mathrm{ms}\). 15% stretch only.

---

## 6. Immediate next action

Same as before, with clarified method name:

`AUTHORIZE Phase D1 — LVM-RPI formulation lock`

Then C0 consequentiality with **simulator-scored candidates** (rollout-PI teacher), not a learned critic first.

---

## 7. One-sentence opinion

**Rollout-PI is the right way to *create* LVM-relative teachers; CPI-style gating is the right way to *deploy* them; neither escapes measuring headroom (C0) or false accepts (C1); do not train a free PPO, and do not promise 15%.**
