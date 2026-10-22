# POPL — PPO / \(\Delta G\) Learning-Rule Correctness Note

**Status:** Pre-P5 mathematical review (no weight training)  
**Formal gate review:** `POPL_LEARNING_RULE_CORRECTNESS_REVIEW.md` → **CONDITIONAL PASS** (locked on-policy rule)  
**Conclusion:** Naively substituting \(\Delta G_t\) into PPO is **not** automatically valid. Under on-policy \(A^{\mathrm{alt}}\) and full sequential \(\log\pi\), Direct’s return is an action-independent baseline and \(\Delta G\) is a legitimate score-function multiplier.

---

## 1. What standard PPO assumes

Canonical PPO (Schulman et al.) uses trajectories from the **current** (or frozen old) policy \(\pi_{\theta_{\mathrm{old}}}\):

\[
\hat A_t
=
\text{GAE}(r_t,V_{\theta_{\mathrm{old}}}),
\quad
L^{\mathrm{CLIP}}(\theta)
=
\mathbb E_{a_t\sim\pi_{\theta_{\mathrm{old}}}}
\Bigl[
\min\bigl(\rho_t\hat A_t,\,\mathrm{clip}(\rho_t)\hat A_t\bigr)
\Bigr],
\quad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
\]

Unbiasedness (up to baseline/GAE bias–variance choices) relies on:

1. Actions in the expectation being drawn from \(\pi_{\theta_{\mathrm{old}}}\) (or IS-corrected).  
2. \(\log\pi\) matching the **same** factorization used at sampling.  
3. Advantages being functions of the return of **that** sampled trajectory (or a valid control variate).

---

## 2. What POPL paired sampling actually does

A paired sample compares:

- \(a^{\mathrm{ref}}=A_t^{\mathrm{Direct}}\) (frozen behavior, **not** \(\pi_\theta\)),  
- \(a^{\mathrm{alt}}\) proposed by some behavior law \(\beta\) (may be \(\pi_{\theta_{\mathrm{old}}}\), a mixture with Direct, or an explorer).

Both are rolled out under matched exogenous noise with common Direct continuation. The scalar

\[
\Delta G_t=G_t(a^{\mathrm{alt}})-G_t(a^{\mathrm{ref}})
\]

is a **realized contrast**, not a one-trajectory GAE advantage under \(\pi_{\theta_{\mathrm{old}}}\).

### Failure mode of the naive rule

If one sets \(\hat A\leftarrow\Delta G\) and applies PPO’s clip ratio only to \(a^{\mathrm{alt}}\) as if \(a^{\mathrm{alt}}\sim\pi_{\theta_{\mathrm{old}}}\) while:

- \(a^{\mathrm{alt}}\) was forced / mixed / edited from Direct, or  
- \(\log\pi(a^{\mathrm{alt}})\) ignores sequential TERMINATE / partial-pack factors, or  
- the baseline ignores that \(\Delta G\) is already a **difference of two returns**,

then the gradient is generally a **biased** estimator of \(\nabla_\theta\mathbb E_{\pi_\theta}[G_0]\).

---

## 3. Conditions under which \(\Delta G\) *can* enter a valid update

### 3.1 Preference / pairwise ranking view (recommended primary)

Treat each pair as a preference label:

\[
y=\mathbf{1}\{\Delta G_t>0\}.
\]

A valid objective is pairwise logistic / Bradley–Terry on policy scores of the two packs (or REINFORCE on the preferred pack with IS), e.g.

\[
L_{\mathrm{pref}}
=
-\,y\log\sigma\bigl(s_\theta(a^{\mathrm{alt}})-s_\theta(a^{\mathrm{ref}})\bigr)
-(1-y)\log\sigma\bigl(s_\theta(a^{\mathrm{ref}})-s_\theta(a^{\mathrm{alt}})\bigr),
\]

with \(s_\theta=\log\pi_\theta\) (sum of sequential decisions). This does **not** claim \(\Delta G\) is a PPO advantage; it uses only the **sign** (or a monotone transform) of the contrast. Magnitude weighting by \(|\Delta G|\) is optional and must be stated as a heuristic or derived from a specific ranking loss.

### 3.2 On-policy REINFORCE on the alternative only

If and only if \(a^{\mathrm{alt}}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)\) (strict on-policy proposal):

\[
\nabla_\theta\mathbb E[G^{\mathrm{alt}}]
\approx
\mathbb E\bigl[\nabla_\theta\log\pi_\theta(a^{\mathrm{alt}}\mid s)\,(G^{\mathrm{alt}}-b)\bigr].
\]

Using \(\Delta G=G^{\mathrm{alt}}-G^{\mathrm{ref}}\) as the multiplier is valid as a **control variate** when \(G^{\mathrm{ref}}\) is independent of \(\theta\) (Direct frozen) and \(a^{\mathrm{alt}}\) was sampled from \(\pi_{\theta_{\mathrm{old}}}\):

\[
\mathbb E\bigl[\nabla\log\pi(a^{\mathrm{alt}})\,G^{\mathrm{ref}}\bigr]
=
\mathbb E[G^{\mathrm{ref}}]\,\mathbb E[\nabla\log\pi(a^{\mathrm{alt}})]
=
0
\]

under the usual score-function identity (for fixed \(s_t\)). Thus

\[
\mathbb E\bigl[\nabla\log\pi(a^{\mathrm{alt}})\,\Delta G\bigr]
=
\mathbb E\bigl[\nabla\log\pi(a^{\mathrm{alt}})\,G^{\mathrm{alt}}\bigr]
\]

— so \(\Delta G\) is a **valid advantage-like multiplier** for the alt action **under on-policy sampling of \(a^{\mathrm{alt}}\)**, with Direct providing a state-matched baseline return.

**PPO clip** may then be applied to \(\rho=\pi_\theta(a^{\mathrm{alt}})/\pi_{\theta_{\mathrm{old}}}(a^{\mathrm{alt}})\) with \(\hat A=\Delta G\) **only under that sampling assumption**.

### 3.3 Off-policy / mixed proposals

If \(a^{\mathrm{alt}}\sim\beta\neq\pi_{\theta_{\mathrm{old}}}\), require importance weights

\[
w=\frac{\pi_{\theta_{\mathrm{old}}}(a^{\mathrm{alt}}\mid s)}{\beta(a^{\mathrm{alt}}\mid s)}
\]

(or train with a preference loss §3.1). Unweighted PPO-clip on such pairs is **not** justified.

### 3.4 Sequential pack likelihoods

For auto-regressive pack construction,

\[
\log\pi(A\mid s)
=
\sum_{j=1}^{m}\log\pi(a_j\mid s,a_{<j})
+
\log\pi(\mathrm{TERMINATE}\mid s,a_{1:m}).
\]

Partial credit to only the first differing assignment without the joint likelihood biases the update (Stage-1-style credit errors). P5 implementations must use the **full** pack log-prob.

---

## 4. Checklist before authorizing P5

| Item | Required |
|---|---|
| Declare behavior law \(\beta\) for \(a^{\mathrm{alt}}\) | Yes |
| If \(\beta=\pi_{\theta_{\mathrm{old}}}\): document on-policy sampling | Yes |
| If \(\beta\neq\pi_{\theta_{\mathrm{old}}}\): IS or preference loss | Yes |
| Full sequential \(\log\pi(A)\) including TERMINATE | Yes |
| Direct return used only as baseline / preference contrast | Yes |
| No analytical \(\widehat p/J\) in \(\hat A\) | Yes |
| Train-seed pairs only | Yes |

**Bottom line:** \(\Delta G_t\) is a legitimate **paired contrast**. It becomes a PPO-style advantage multiplier only when \(a^{\mathrm{alt}}\) is on-policy (or correctly IS-weighted) and pack likelihoods are complete. Do not equate “we have \(\Delta G\)” with “standard PPO is unbiased.”

---

## 5. Authorization

This note does **not** authorize P5 or weight training. It is a prerequisite correctness review for any future learning-pilot design.
