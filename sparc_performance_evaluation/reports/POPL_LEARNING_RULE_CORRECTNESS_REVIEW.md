# POPL-SPARC — Learning-Rule Correctness Review (Pre-P5)

**Review of:** four mandatory requirements for using \(\Delta G\) with PPO  
**Harness status:** P1–P4 PASS (accepted)  
**P5 / weight training:** still **not authorized** by this review alone  

---

## Verdict

| Requirement | Verdict |
|---|---|
| R1 Correct action likelihood | **PASS** (conditional on locked factorization below) |
| R2 Valid policy-gradient estimator | **PASS** (conditional on on-policy \(A^{\mathrm{alt}}\) + Direct baseline) |
| R3 Correct PPO importance ratios | **PASS** (conditional on ratios over the **same** sampled pack) |
| R4 No outcome leakage | **PASS** (conditional on protocol below) |
| **Overall learning-rule design** | **CONDITIONAL PASS** |
| **Authorize P5 pilot?** | **Not by this document** — rule is cleared for *design*; P5 needs separate authorization |

**Rejected design:** Treating \(\Delta G\) as a drop-in PPO advantage without stating \(\beta\), sequential \(\log\pi\), and on-policy (or IS) sampling — as warned in `POPL_PPO_DELTA_G_CORRECTNESS_NOTE.md` and in the overly loose sketch in feasibility-spec §5.2.

---

## R1 — Correct action likelihood

**Requirement:** Stored log-probability must reflect the complete sampled action sequence, including termination.

**Locked factorization (required for any future P5):**

\[
\log\pi_\theta(A\mid s)
=
\sum_{j=1}^{m}
\log\pi_\theta(a_j\mid s,a_{<j})
+
\log\pi_\theta(\mathrm{TERMINATE}\mid s,a_{1:m}).
\]

| Check | Status |
|---|---|
| Pack is sequential; TERMINATE is an action | Required |
| Partial credit only on first differing assignment | **Forbidden** |
| Log-prob at sample time stored for PPO \(\rho\) | Required |

**R1:** PASS under this factorization; FAIL if P5 omits TERMINATE or truncates the product.

---

## R2 — Valid policy-gradient estimator

**Requirement:** Direct-reference return must function as an action-independent baseline under stated sampling assumptions.

**Locked sampling assumption for PPO-clip + \(\hat A=\Delta G\):**

\[
A^{\mathrm{alt}}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),
\quad
A^{\mathrm{ref}}=A^{\mathrm{Direct}}\ \text{(frozen; independent of \(\theta\))}.
\]

Then for fixed \(s_t\),

\[
\mathbb E\bigl[\nabla_\theta\log\pi_\theta(A^{\mathrm{alt}}\mid s)\,G^{\mathrm{ref}}\bigr]
=
\mathbb E[G^{\mathrm{ref}}]\,
\mathbb E\bigl[\nabla_\theta\log\pi_\theta(A^{\mathrm{alt}}\mid s)\bigr]
=
0,
\]

hence

\[
\mathbb E\bigl[\nabla_\theta\log\pi_\theta(A^{\mathrm{alt}}\mid s)\,\Delta G\bigr]
=
\mathbb E\bigl[\nabla_\theta\log\pi_\theta(A^{\mathrm{alt}}\mid s)\,G^{\mathrm{alt}}\bigr].
\]

So \(\Delta G=G^{\mathrm{alt}}-G^{\mathrm{ref}}\) is a **valid score-function multiplier** (state-matched baseline), not merely a heuristic, **iff** \(A^{\mathrm{alt}}\) is on-policy as above.

| Check | Status |
|---|---|
| Direct \(G^{\mathrm{ref}}\) independent of \(\theta\) | Yes (frozen Direct) |
| \(A^{\mathrm{alt}}\sim\pi_{\theta_{\mathrm{old}}}\) | **Mandatory** for \(\hat A=\Delta G\) + PPO-clip |
| Explorer / mixture \(\beta\neq\pi_{\theta_{\mathrm{old}}}\) | Must use IS or preference loss — **not** unweighted PPO-clip |

**R2:** PASS for the locked on-policy rule; FAIL for naive “any alt pack + \(\Delta G\) into PPO.”

---

## R3 — Correct PPO importance ratios

**Requirement:** Ratios use the probability of the **same** sampled action under current and behavior policies.

**Locked ratio:**

\[
\rho
=
\frac{\pi_\theta(A^{\mathrm{alt}}\mid s)}{\pi_{\theta_{\mathrm{old}}}(A^{\mathrm{alt}}\mid s)},
\]

with both numerator and denominator using the **R1** joint pack likelihood (including TERMINATE). The clipped surrogate applies only to this \(A^{\mathrm{alt}}\).

| Check | Status |
|---|---|
| Ratio on \(A^{\mathrm{ref}}\) (Direct) | Not used as a \(\pi_\theta\) action in default design |
| Ratio mixing different packs in num/den | **Forbidden** |
| Recomputing \(\log\pi_{\theta_{\mathrm{old}}}\) from stored joint log-prob | Required |

**R3:** PASS under this definition.

---

## R4 — No outcome leakage

**Requirement:** Matched-rollout results may enter training targets, never inference-time observations.

| Channel | Allowed? |
|---|---|
| \(\Delta G\), \(G^{\mathrm{alt}}\), \(G^{\mathrm{ref}}\) as training targets / advantages | Yes (train seeds only) |
| Critic targets from realized returns | Yes |
| Feeding \(\Delta G\), future completions, or branch labels into actor observations at inference | **No** |
| Fitting \(\beta\) or hyperparameters on locked validation labels | **No** |
| Final-test seeds | **Prohibited** |

**R4:** PASS under this protocol.

---

## Locked update (summary)

**On-policy POPL–PPO (only form cleared by this review):**

1. Sample \(A^{\mathrm{alt}}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)\); set \(A^{\mathrm{ref}}\) = Direct.  
2. Run paired rollouts (P1–P4 harness).  
3. \(\hat A \leftarrow \mathrm{normalize}(\Delta G)\) (optional batch baseline on \(\Delta G\), train-only).  
4. PPO-clip with \(\rho=\pi_\theta(A^{\mathrm{alt}})/\pi_{\theta_{\mathrm{old}}}(A^{\mathrm{alt}})\) using **full** sequential log-prob.  
5. Actor inputs at step time: causal state only.

**Preference-loss alternative** (also cleared): Bradley–Terry / logistic on \(\mathrm{sign}(\Delta G)\) with scores \(s_\theta=\log\pi_\theta\); does not claim \(\Delta G\) is a PPO advantage.

---

## Relation to P1–P4

P1–P4 show the **simulator contrast** \(\Delta G\) is trustworthy for the reported cases. They do **not** by themselves make a learning rule correct. This review separates those concerns and locks the estimator assumptions.

---

## Authorization

| Activity | Decision |
|---|---|
| P1–P4 harness | Accepted PASS |
| This learning-rule correctness review | **CONDITIONAL PASS** (locked rule above) |
| P5 learning pilot | **Not authorized** — requires separate approval citing this locked rule |
| PPO weight training / full-scale train | Not authorized |
| Stage 1 / JIPI / RHO reopen | Forbidden |
| Final-test | Sealed |

Direct PPO remains the SPARC reference.
