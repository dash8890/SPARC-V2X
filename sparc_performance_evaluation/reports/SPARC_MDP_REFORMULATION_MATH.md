# SPARC-V2X — Locked MDP Reformulation (math)

**Status:** Canonical formulation for the MDP correction program.  
**Not authorization to implement or train.** Implementation follows `SPARC_MDP_HANDOFF.json` + `SPARC_MDP_EXECUTION_RUNBOOK.md` only when authorized.

Any agent working on SPARC scheduling RL must treat this file as the mathematical source of truth for the *intended* decision problem. Do not silently revert to candidate-score-only MDP language.

---

## 1. Reported objective (unchanged physically)

Deadline-valid completed value:
\[
V_{\mathrm{done}}=\sum_i v_i\,\mathbf{1}\{\text{task }i\text{ completed by its deadline}\}.
\]
Per-epoch reward in the closed loop remains the increment
\[
r_t=\Delta V_{\mathrm{done},t}
\]
(including delayed `advance_reward` when queued work finishes).  
**Optimization / reporting discount:** \(\gamma=1\).
\[
G_t=\sum_{k=t}^{T_{\mathrm{end}}} r_k,\qquad
\text{maximize }\mathbb{E}[G_0].
\]
\(T_{\mathrm{end}}\) includes locked drain so assignments near the last decision epoch still receive credit if they complete validly.

Safety / envelope: hard constraints via feasible set and governor (not a soft term in \(r_t\) unless a later authorized lock changes that).

---

## 2. What is wrong with the implicit old MDP

- Action often treated as sequential micro-choices without one joint environment action.
- Governor rewrite sits outside \(\log\pi\).
- Critic / features do not represent **post-decision commitments**.
- Direct PPO used \(\gamma=0.99\) and offered-normalized \(\tilde r\) while eval reports \(\gamma=1\) \(V_{\mathrm{done}}\).

---

## 3. Reformulated process

### 3.1 Physical state
\[
S_t=(\mathcal{W}_t,\mathcal{X}_t,\mathcal{Q}_t,\mathcal{R}_t,\mathcal{H}_t,\mathcal{B}_t,t)
\]

| Symbol | Meaning |
|--------|---------|
| \(\mathcal{W}_t\) | Waiting tasks (value, payload, cycles, release, deadline, …) |
| \(\mathcal{X}_t\) | In-flight transmissions / unresolved delivery–execution commitments |
| \(\mathcal{Q}_t\) | Helper queues (pending jobs, residual cycles, ready times) |
| \(\mathcal{R}_t\) | Radio / reservation state (sim-internal; may be partially hidden) |
| \(\mathcal{H}_t\) | Helper capabilities / availability |
| \(\mathcal{B}_t\) | Remaining capacity under the real byte-envelope rule |
| \(t\) | Decision epoch / remaining horizon |

### 3.2 Observation (no leakage of hidden future radio)
\[
H_t=(O_0,A_0^{\mathrm{exec}},r_0,\ldots,O_t),\quad
Z_t=f_\eta(H_t).
\]
Policy and critics may use \(Z_t\) or \(O_t\) only — never hidden simulator RNG or future outcomes.

### 3.3 Action = one feasible pack
\[
A_t=\{(i,h)\}\in\mathcal{F}(O_t).
\]
Joint likelihood (sequential decoder is internal):
\[
\log\pi_\theta(A_t\mid H_t)=\sum_{m=1}^{M_t}\log\pi_\theta(a_{t,m}\mid H_t,a_{t,<m}).
\]

### 3.4 Governor inside the effective transition
\[
A_t^{\mathrm{exec}}=g(A_t,O_t).
\]
- **Actor ratio / \(\log\pi\):** proposed pack \(A_t\).
- **Environment / afterstate / \(W\):** executed pack \(A_t^{\mathrm{exec}}\).

### 3.5 Post-decision state (deterministic commitments)
\[
S_t^+=\Phi(S_t,A_t^{\mathrm{exec}}).
\]
\(\Phi\) applies only commitments knowable at decision time (selected tasks, committed bytes, source/task locks, helper commitments knowable pre-radio, reduced remaining \(\mathcal{B}\), etc.).

\(\Phi\) must **not**: assume successful reception; prematurely mark `done`; enqueue work that requires radio delivery.

### 3.6 Stochastic transition
\[
S_{t+1}\sim P_{\mathrm{radio+compute}}(\cdot\mid S_t^+).
\]

### 3.7 Diagram
\[
S_t \xrightarrow{A_t\sim\pi} A_t \xrightarrow{g} A_t^{\mathrm{exec}}
\xrightarrow{\Phi} S_t^+ \xrightarrow{P} (S_{t+1},r_t).
\]

---

## 4. Learning objects (intended; Phase 2+)

State value:
\[
V_\psi(Z_t)\approx\mathbb{E}_\pi[G_t\mid Z_t].
\]

Post-decision / action-conditioned value:
\[
W_\phi(Z_t,A_t^{\mathrm{exec}})\approx\mathbb{E}_\pi[G_t\mid Z_t,A_t^{\mathrm{exec}}].
\]

Fit \(W\) to **realized** returns (MSE). Forbidden targets: analytic pack \(E[V]\), slack scores, invented \(\hat p\).

Optional advantage (only after calibration gate):
\[
\widehat{A}_t=\widehat{W}_\phi(Z_t,A_t^{\mathrm{exec}})-\widehat{V}_\psi(Z_t).
\]
Lower-risk default: keep realized-return GAE (\(\gamma=1\)) for the actor; train \(W\) as auxiliary first.

---

## 5. One-line summary

**Scheduling is choosing a pack that lands the system in a commitment state \(S_t^+\); value is expected remaining deadline-valid completed value from that afterstate under \(\gamma=1\).**

---

## 6. Pointers

| File | Role |
|------|------|
| `SPARC_MDP_HANDOFF.json` | What is authorized *now* |
| `SPARC_MDP_EXECUTION_RUNBOOK.md` | How to execute each phase |
| `SPARC_MDP_PROGRAM_STATUS.md` | Human-readable status |
| This file | The math — does not expire when chat ends |
