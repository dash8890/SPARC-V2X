# Phase D1 — LVM-RPI-Distill Formulation Lock

**Status: SUPERSEDED — PATH CLOSED (FAIL).**  
**Date locked:** 2026-10-19  
**Closed:** 2026-10-19 — see `SPARC_LVM_RPI_PATH_CLOSED.md`. Do not run C0/distill. Historical formulation only.  
**Next:** `SPARC_NEXT_ROADMAP.md`.

This replaces CPI-SPARC as a standalone training method. CPI is retained only as the **deployment gating principle**. The offline learning target is produced by **simulator-evaluated rollout policy improvement relative to LVM**.

---

## 1. Scientific objective

The system objective remains

\[
\boxed{
J(\pi)
=
\mathbb E_{\pi}
\left[
\sum_{i\in\mathcal T}
v_i\,
\mathbf 1\{T_i^{\rm done}\le d_i\}
\right]
}
\]

subject to

\[
P_{\rm viol}(\pi)\le 0.02,
\qquad
B_t\le 175\,{\rm kB},
\qquad
A_t\in\mathcal F(s_t),
\qquad
T_{\rm decision}\le10{\rm ms}.
\]

The primary comparison is against **LVM**. VLRO, Direct PPO, and the other frozen baselines remain in evaluation.

The 15% target is stretch only. A valid win is a reproducible

\[
\boxed{J(\pi_{\rm RPI})>J(\pi_{\rm LVM})}
\]

with safety and runtime satisfied.

---

## 2. Core design decision

The new controller does **not** learn a scheduler from scratch.

LVM is the base policy

\[
\pi_B\equiv\pi_{\rm LVM}.
\]

At each decision state \(s_t\), we ask:

\[
\boxed{
\text{Is there a feasible complete scheduling action whose actual downstream return exceeds LVM's?}
}
\]

This is the only quantity the learning system is allowed to call an improvement.

For the LVM action

\[
A_t^B=\pi_B(s_t),
\]

define the one-deviation LVM continuation value

\[
Q^B(s_t,A)
=
\mathbb E
\left[
\sum_{\tau=t}^{T_{\rm end}} r_\tau
\;\middle|\;
s_t,\,
A_t=A,\,
A_{\tau>t}=\pi_B(s_\tau)
\right].
\]

The corresponding LVM-relative advantage is

\[
\boxed{
\Delta^B(s_t,A)
=
Q^B(s_t,A)-Q^B(s_t,A_t^B).
}
\]

This is the central quantity for the entire method.

It is **not** predicted slack, predicted completion probability, value density, RHO pack score, PPO advantage, or any other proxy.

---

## 3. Reward

The only environment reward relevant to policy improvement is realized deadline-valid completed value:

\[
r_t
=
\sum_{i\in\mathcal C_t}v_i,
\]

where \(\mathcal C_t\) contains tasks that become newly completed by their deadline during physical epoch \(t\).

Each task contributes once.

The rollout return is undiscounted:

\[
G_t=\sum_{\tau=t}^{T_{\rm end}}r_\tau.
\]

Therefore the teacher's optimization target and the paper's evaluation metric are exactly aligned.

---

## 4. State / observation

The deployable policy may use only causally available information.

Represent the observation as

\[
o_t=
\left(
W_t,
H_t,
X_t,
R_t,
B_t,
\tau_t
\right),
\]

where:

| Component | Meaning |
|---|---|
| \(W_t\) | waiting tasks: value, payload, cycles, source, remaining deadline |
| \(H_t\) | helper capabilities and observable queue/work commitments |
| \(X_t\) | observable outstanding communication/computation commitments |
| \(R_t\) | causally available radio indicators such as CBR and relevant Mode-2 state exposed to the controller |
| \(B_t\) | remaining computation-admission capacity under the frozen governor accounting |
| \(\tau_t\) | remaining episode / scheduling horizon |

Future channel realizations, future arrivals, future successful receptions, or hidden simulator randomness are prohibited from deploy-time features.

---

## 5. Candidate action set

At state \(s_t\),

\[
\mathcal C_t=
\{
A_t^{\rm LVM},
A_t^{\rm VLRO},
A_t^{\rm Direct},
A_{t,1}^{\rm edit},
\dots,
A_{t,K}^{\rm edit}
\}.
\]

Every candidate is a **complete task-helper scheduling pack**, not an independent pair score.

The edit generator operates around the LVM pack and may create structurally distinct feasible packs through task omission, substitution, helper re-binding, addition of previously excluded tasks, and combinations of these operations.

Every resulting candidate must be passed through exactly the same feasibility rules and frozen governor semantics as the corresponding controller would experience.

LVM is always included:

\[
A_t^{\rm LVM}\in\mathcal C_t
\quad\forall t.
\]

VLRO remains a **diversity generator** even though its reported aggregate value is lower than LVM.

---

## 6. Offline rollout teacher

For every candidate \(A\in\mathcal C_t\), clone the same pre-decision simulator state and preserve common stochastic realizations wherever the simulator permits paired replay.

Execute \(A_t=A\) for the current epoch. Thereafter use LVM: \(A_{\tau}=\pi_B(s_\tau)\) for \(\tau>t\). Run until the actual terminal/drain boundary.

Measure \(G_t(A\rightarrow B)\) and \(G_t(B\rightarrow B)\).

\[
\boxed{
\Delta_t(A)
=
G_t(A\rightarrow B)-G_t(B\rightarrow B).
}
\]

\[
\boxed{
A_t^\star
=
\arg\max_{A\in\mathcal C_t}
G_t(A\rightarrow B).
}
\qquad
\boxed{
\Delta_t^\star
=
G_t(A_t^\star\rightarrow B)
-
G_t(B\rightarrow B).
}
\]

Since \(A_t^B\in\mathcal C_t\), \(\Delta_t^\star\ge0\) for an exact deterministic replay of the candidate set.

**Offline labels can never designate a candidate worse than the observed LVM action as the teacher optimum.**

---

## 7. Multi-step coordination

**Locked interpretation of \(H\):**

\[
H=1
\]

means **one alternative physical scheduling action followed by LVM for the entire remaining episode**.

It does **not** mean that evaluation stops after one epoch. This gives an exact downstream label for one-decision deviation without a terminal-value approximator.

Larger \(H\) is only for genuinely coordinated multi-epoch alternatives (not authorized until \(H=1\) C0 establishes need).

---

## 8. Why this addresses known SPARC failures

Causal chain:

\[
\text{candidate pack}
\rightarrow
\text{actual radio outcome}
\rightarrow
\text{helper workload}
\rightarrow
\text{future LVM decisions}
\rightarrow
\text{deadline-valid completed value}.
\]

| Prior failure | Correction |
|---|---|
| RHO surrogate ranking | Teacher = realized \(G\) |
| Free PPO / Stage-1 noise | Only \(\Delta\) from paired LVM continuation |
| JIPI KEEP | Fallback = LVM; improve only when evidence exists |
| POPL full-policy generalization | Distill sparse improving deviations |
| Companion / collateral pack effects | Complete pack + full residual return |

Narrow supervised question:

\[
\boxed{
\text{Which concrete deviation from LVM has already demonstrated positive downstream value?}
}
\]

---

## 9. Distillation target

**Advantage estimator**

\[
\hat\Delta_\phi(o_t,A,A_t^B)\approx\Delta_t(A),
\]

\[
L_\Delta(\phi)
=
\frac{1}{N}
\sum_n
w_n
\left[
\hat\Delta_\phi(o_n,A_n,A_n^B)
-
\Delta_n(A_n)
\right]^2,
\]

\[
w_n=
1+\lambda_+
\mathbf1\{\Delta_n>\epsilon_{\rm imp}\}.
\]

**Optional edit/policy model** \(\pi_\theta(A\mid o,A^B)\) trained primarily on states with \(\Delta^\star>\epsilon_{\rm imp}\):

\[
L_\pi
=
-
\sum_{n:\Delta_n^\star>\epsilon_{\rm imp}}
w_n^\star
\log
\pi_\theta(A_n^\star\mid o_n,A_n^B).
\]

The actor is **not** predominantly trained to reproduce LVM.

---

## 10. Deploy-time policy

\[
A^B=\pi_B(o),\quad
\tilde{\mathcal C}=\{A^B,A_1,\ldots,A_K\},
\quad
\hat\Delta_k=\hat\Delta_\phi(o,A_k,A^B).
\]

\[
\boxed{
\pi_{\rm RPI}(o)
=
\begin{cases}
A_{k^\star},
&
\mathrm{LCB}_{1-\alpha}
\left[\Delta(o,A_{k^\star})\right]>\tau,
\\[2mm]
A^B,
&
\text{otherwise}.
\end{cases}}
\]

\[
\boxed{\text{No calibrated positive advantage } \Rightarrow \text{ execute LVM}.}
\]

Uncertainty architecture is secondary; the decision rule is primary.

---

## 11. Gate criterion

\[
FA=\{\hat{\Delta}>\tau,\ \Delta\le0\},
\qquad
\boxed{
P(FA\mid \hat\Delta>\tau)
\le \alpha.
}
\]

Complementary: \(\mathrm{Capture}=P(\hat\Delta>\tau\mid \Delta>\epsilon_{\rm imp})\).

Desired: low false acceptance and nontrivial capture.

---

## 12. Formal improvement statement

For an exact rollout teacher over \(\mathcal C_t\),

\[
Q^B(s_t,A_t^\star)
\ge
Q^B(s_t,A_t^B).
\]

For the deployable distilled controller, improvement is **statistical** because \(\hat\Delta\neq\Delta\).

**Legitimate claim:** The offline teacher is monotonically non-worse than LVM over the searched candidate set; the deployed model is designed to inherit only calibrated positive deviations while falling back to LVM elsewhere.

**Illegitimate claim:** Unconditional mathematical dominance of the learned deployment policy before calibration/evaluation.

---

## 13. Relationship to VLRO

VLRO: frozen baseline + candidate generator. Aggregate \(V_{\rm VLRO}\ll V_{\rm LVM}\) ⇒ not the anchor. Occasional superior packs can still be harvested by rollout evaluation.

---

## 14. Literature (mechanisms only)

Use transferable mechanisms (feasibility vs learned improvement; masked structured actions; commitment/temporal state; relative/counterfactual credit). Do **not** import COMA/DAG/transformers unless the workload requires them.

---

## 15. Locked non-claims

LVM-RPI-Distill does **not** claim 4%, 15%, or any numerical gain a priori.

It does **not** claim the LVM floor is mathematically guaranteed at deployment unless the gate bound is valid.

It does **not** claim candidate-set optimality outside \(\mathcal C_t\).

It does **not** claim VLRO is superior to LVM.

It does **not** claim offline simulator access at runtime.

It does **not** change the frozen NR-V2X simulator, governor, envelope, safety metric, baselines, or final-test protocol.

---

## 16. Final locked algorithm

\[
\boxed{\textbf{LVM-RPI-Distill}}
\]

**Offline:**
\[
s_t
\rightarrow
\mathcal C_t
\rightarrow
\text{paired simulator rollouts}
\rightarrow
\Delta_t(A)
\rightarrow
A_t^\star
\rightarrow
\{\hat\Delta_\phi,\pi_\theta\}
\]

**Online:**
\[
s_t
\rightarrow
A_t^{\rm LVM}
+
\text{small learned candidate set}
\rightarrow
\hat\Delta
\rightarrow
\text{calibrated positive-advantage gate}
\rightarrow
A_t^{\rm exec}
\]

with \(\boxed{\text{fallback}=A_t^{\rm LVM}}\).

Source of improvement:

\[
\boxed{
\text{Harvest simulator-demonstrated LVM mistakes,
generalize those mistakes across causal states,
and refuse unconfident deviations.}
}
\]

---

## 17. Agent acceptance notes (binding for next phases)

**Opinion: LOCK this formulation.** It is the first SPARC design that is simultaneously (a) LVM-anchored, (b) label-aligned with \(J\), (c) honest about offline vs deploy guarantees, and (d) matched to the success/fail map.

| Strength | Why it matters |
|---|---|
| \(H=1\) = full residual + LVM continuation | Exact label; no bogus \(V(s_{t+1})\) bootstrap |
| \(A^{\rm LVM}\in\mathcal C\) ⇒ \(\Delta^\star\ge0\) offline | Teacher cannot prefer worse-than-LVM |
| Deploy claim statistical; FA is the gate metric | Avoids CPI overclaim of “guarantee” |
| Actor trained on \(\Delta^\star>\epsilon\) only | Avoids “imitate LVM forever” |
| Complete packs + same governor | Matches companion/collateral evidence |

| Residual risk (not a design reopen — bind C0/C1) | Mitigation |
|---|---|
| Headroom may be tiny outside contaminated val | **C0** on fresh seeds before any distill |
| \(\lvert\mathcal C\rvert\) may miss the good pack | Predeclare edit operators; report coverage |
| False accepts destroy LVM floor | **C1** FA ≤ α before closed-loop claims |
| Paired replay / common RNG hygiene | Same as RAO lessons; force_match reporting |
| Latency: LVM + \(K\) + ensemble ≤ 10 ms | Explicit C4/C5 latency gate |

**Next authorize phrase:** `AUTHORIZE Phase C0 — LVM-RPI consequentiality`  
(Protocol + numeric pass bars must be fixed *before* collection; see `SPARC_CPI_VS_ROLLOUT_PI_OPINION.md` / roadmap.)
