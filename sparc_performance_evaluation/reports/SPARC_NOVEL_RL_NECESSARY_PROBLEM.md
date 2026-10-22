# Novel Problem — When RL is Required for Safe NR-V2X Mode 2 Offloading

**Status:** BLOCKED / RADIO PREMISE FAIL — formulation kept for record; not an active path  
**Date:** 2026-10-19  
**G9 (sim contract):** FAIL twice — first audit ΔCBR≈0.001; re-audit with DCC ON + envelope HEAVY + CBR@epochs 10–100 → `premise_supported=false` (mean ΔCBR≈−0.0003; ΔPRR≈−0.002, below bar). See `SPARC_G9_ENDOGENOUS_LOAD_AUDIT.md`.  
**Instruction (historical):** No RL until G9; G9 failed → do not train on EC-SPO radio-endogenous claim in this WiLab stack.

---

## 0. Why prior attacks failed (thread summary)

| Attempt | Failure mode |
|---|---|
| Free / residual PPO on packs | Treated a near-myopic packing score as if it needed deep RL |
| RHO analytic surrogates | Optimized the wrong scalar |
| Routers among frozen generators | No new sequential decision; selection among myopic packs |
| Decode / PackSwap heuristics | Local edits without solving sequential opportunity cost |
| “Beat LVM on nominal” as the problem | Anchored novelty to beating a strong **myopic** heuristic on a problem that *is* largely myopic |

**Diagnosis:** We often posed a problem that **MINLP / Lagrangian packing can already solve well one step at a time**. On that problem, RL is unnecessary and loses.  
**Workflow correction:** Redefine the problem until classical methods are provably intractable or structurally suboptimal; **only then** apply RL.

---

## 1. Novel problem (name + one sentence)

### Name
**Endogenous-Congestion Sequential Pack Offloading (EC-SPO)** under NR-V2X Mode 2 with chance-constrained safety.

### One sentence
Choose, over a finite horizon, a sequence of feasible computation packs whose offered load **feeds back into Mode-2 congestion and helper queues**, so as to maximize expected deadline-valid completed value subject to a **closed-loop chance constraint** on safety QoS — a sequential stochastic control problem with combinatorial actions and action-dependent radio dynamics.

### What is novel vs common formulations
1. **Endogenous radio:** computation offers change CBR / sensing / delivery statistics → future reception probabilities depend on past packs (not an exogenous fading trace independent of \(A_t\)).  
2. **Sequential chance constraint:** \(P_{\mathrm{viol}}(\pi)\le\alpha\) is a constraint on the **closed-loop** process, not a per-slot hard cut that makes the problem static.  
3. **Pack action + multi-step commitments:** helper queues and in-flight packets couple epochs; one-shot set-packing is not the episode problem.  
4. **RL necessity claim** is part of the contribution (below) — not “we trained PPO on a simulator.”

Baselines (LVM, greedy, online MINLP, CE-MILP, etc.) are **to be implemented after** this formulation is accepted — they are competitors on EC-SPO, not the definition of the problem.

---

## 2. Utility, objective, constraints

### 2.1 Primitive quantities (epoch \(t=0,\ldots,T-1\))

- Tasks \(i\in\mathcal W_t\): value \(v_i>0\), workload, deadline \(d_i\), source.  
- Helpers \(h\in\mathcal H\): service rate, queue state.  
- Pack action \(A_t\subset\mathcal W_t\times\mathcal H\): complete feasible assignment set.  
- Mode-2 / safety state \(R_t\) (CBR, PRR windows, pending SPS, …).  
- Commitments \(X_t\): in-flight computation bytes, helper residual work.

### 2.2 Instantaneous feasibility (hard, deterministic)

\[
A_t\in\mathcal F(s_t)
\quad\Leftrightarrow\quad
\begin{aligned}
&\text{unique task, unique source},\\
&\sum_{(i,h)\in A_t} b_i \le B_{\max}\quad(175\,\mathrm{kB}),\\
&\text{deadline-feasible under current helper queues},\\
&\text{governor / invalid-radio rules}.
\end{aligned}
\]

### 2.3 Episode utility (primary)

Deadline-valid completed value along the closed loop:

\[
U(\tau)
=
\sum_{i\in\mathcal I(\tau)}
v_i\,
\mathbf 1\{T_i^{\mathrm{rx}}<\infty,\ T_i^{\mathrm{done}}\le d_i\},
\]

where \(\tau\) is a full trajectory of radio, mobility, arrivals, and decisions.

### 2.4 Objective (risk-constrained expected utility)

\[
\boxed{
\begin{aligned}
\max_{\pi}\quad
&J(\pi)=\mathbb E_{\pi}[U(\tau)] \\
\text{s.t.}\quad
&P_{\mathrm{viol}}(\pi)\le\alpha\quad(\alpha=0.02),\\
&A_t=\pi(o_t)\in\mathcal F(s_t)\ \forall t,\\
&T_{\mathrm{decision}}\le 10\,\mathrm{ms}\ \text{(deploy)}.
\end{aligned}
}
\]

\(P_{\mathrm{viol}}\): fraction of valid safety windows violating PRR/P95 thresholds (same operational definition as the stack), under the **policy-induced** load process.

### 2.5 Why this utility is not “myopic pack score”

\(U\) depends on deliveries and completions **after** \(t\), through queues and through **load-dependent** Mode-2 outcomes. A one-step analytic score \(r(s_t,A_t)\) is at best a surrogate; it is not \(J\).

---

## 3. MDP / CMDP formulation

### 3.1 State / observation

**Physical state** (simulator):

\[
s_t=(W_t,H_t,X_t,R_t,\tau_t),
\]

waiting tasks, helper state, commitments, radio/safety, remaining horizon.

**Agent observation** \(o_t=\omega(s_t)\): causal features only (no future arrivals/channels). This is a **POMDP** in general; for design we may work with a belief or a marked observation MDP \(\tilde s_t=o_t\).

### 3.2 Action

\[
a_t=A_t\in\mathcal F(s_t)\subset 2^{\mathcal E_t},
\]

\(\mathcal E_t\): feasible task–helper edges at \(t\). Action space is **combinatorial**.

### 3.3 Transition (structural form)

\[
s_{t+1}=f(s_t,A_t,\xi_{t+1}),
\]

where \(\xi\) includes Mode-2 randomness **and** congestion feedback from offered load in \(A_t\) (endogenous \(R_{t+1}\)), arrivals, and compute completions.  
**Key:** law of \(\xi_{t+1}\) depends on \(A_t\) through load — not an open-loop exogenous process.

### 3.4 Reward (undiscounted, \(\gamma=1\))

Per-epoch reward = newly realized deadline-valid completions in that epoch (or accounting-consistent increment of \(U\)). Episode return \(G_0=U(\tau)\).

### 3.5 Constrained MDP (CMDP)

Safety as expectation constraint on cumulative violation indicators:

\[
\max_\pi\ \mathbb E_\pi\Big[\sum_t r(s_t,A_t)\Big]
\quad
\text{s.t.}
\quad
\mathbb E_\pi\Big[\sum_t c_{\mathrm{viol}}(s_t)\Big]\le\tilde\alpha.
\]

Equivalent operational form: \(P_{\mathrm{viol}}(\pi)\le\alpha\).

---

## 4. Why classical methods are unsuitable or suboptimal

We need **both** intractability (can't solve exactly) **and** structural suboptimality of the usual practical substitutes (myopic MINLP / CE).

### 4.1 One-shot pack selection is already hard

Even with fixed \(s_t\), choosing \(A_t\in\mathcal F(s_t)\) to maximize a linear value under byte knapsack + unique-source matching is a **multiple-choice / multidimensional knapsack / set-packing** hybrid:

- Special cases reduce to **0-1 knapsack** (NP-hard) or **set packing** (NP-hard).  
- With \(n\) candidates, enumerating maximal feasible packs is exponential in worst case.

So even the per-slot oracle is combinatorial. That alone does **not** justify RL (approx MILP might do). Endogenous sequential structure does.

### 4.2 Finite-horizon exact DP is intractable

Let \(N_t=|\mathcal E_t|\). Then \(|\mathcal A(s_t)|\le 2^{N_t}\). With continuous or finely discretized radio/queue components, \(|\mathcal S|\) is astronomical.  
Exact finite-horizon DP:

\[
V_t(s)=\max_{A\in\mathcal F(s)}\ \mathbb E[r+\gamma V_{t+1}(s')\mid s,A]
\]

is **computationally intractable** at operational \(n,T\) (curse of dimensionality + combinatorial actions).  
**Claim A (complexity):** EC-SPO as exact DP is intractable for realistic \(n,T\).

### 4.3 Open-loop / scenario MINLP is misspecified under endogenous congestion

Standard approach: build a scenario tree for channels and arrivals, solve a large MINLP for \(\{A_t\}\).

**Failure under EC-SPO:** Mode-2 statistics depend on offered load. Scenarios that treat \((R_t)\) as **exogenous** omit the map \(A_t\mapsto \mathrm{Law}(R_{t+1}\mid A_t)\). The optimized open-loop plan is optimal for the **wrong measure**.  
**Claim B (model misspecification):** Any open-loop MINLP that assumes exogenous radio is solving a different problem than EC-SPO; its solution is not optimal for \(J(\pi)\) under endogenous congestion.

### 4.4 Certainty-equivalent online MINLP is structurally suboptimal

**Practical classical method:** at each \(t\), plug in \(\mathbb E[R_{t+1}]\) or a point forecast, solve

\[
A_t^{\mathrm{CE}}\in\arg\max_{A\in\mathcal F(s_t)}\ \tilde r(s_t,A;\hat R),
\]

optionally with a one-step safety cut.

This is **certainty-equivalent (CE) control**. For stochastic dynamic programs with non-linear dynamics / constraints, CE policies are **not** optimal in general (gap between EV solution and true recourse / DP optimum — stochastic programming EEV vs RP; sequential CE vs stochastic DP).

With a **chance constraint** on the closed loop, CE with a point forecast does not enforce \(P_{\mathrm{viol}}(\pi)\le\alpha\); it enforces a heuristic surrogate.  
**Claim C (suboptimality of CE-myopic):** There exist EC-SPO instances (endogenous load → PRR; delayed completions; \(\alpha\)-chance constraint) where the CE online MINLP policy achieves strictly lower \(J\) than the CMDP-optimal policy, or violates the chance constraint while a constrained-optimal policy does not.

See §4.7 for an explicit instance.

### 4.5 Pure myopic Lagrangian / dual packing (LVM-class) is a heuristic for another problem

LVM-class methods optimize a **static** (or slowly adapted) dualized packing score. They can be excellent for **Claim-A’s one-shot** problem. They are not solvers of the CMDP with endogenous congestion and chance constraints.  
**Claim D:** Optimality of myopic dual packing for one-shot set-packing **does not** transfer to optimality for EC-SPO.

### 4.6 Toy theorem — CE myopic strictly suboptimal (G8)

**Instance (2 epochs, deterministic congestion map, stochastic delivery).**  
Epochs \(t\in\{0,1\}\). Byte cap allows at most one “Heavy” pack or one “Light” pack per epoch.  
Tasks: \(H\) (value \(10\), heavy load) available only at \(t=0\); \(L\) (value \(9\)) available at \(t=0\) and \(t=1\); \(S\) (value \(100\)) arrives at \(t=1\) and needs good radio.

Endogenous rule:
- If Heavy offered at \(t=0\), then at \(t=1\) delivery of \(S\) succeeds w.p. \(0.1\).  
- If Light or empty at \(t=0\), then \(S\) succeeds w.p. \(0.95\).  
- \(L\) always completes if offered. Heavy completes if offered at \(t=0\).  
Chance constraint: \(\mathbb E[\mathbf 1\{S\text{ offered and fails}\}]\le 0.2\) (proxy safety/reliability).

**CE online rule at \(t=0\):** maximize expected immediate value using mean success forecast ignoring future endogenous effect → picks Heavy (\(10>9\)). At \(t=1\) offers \(S\); success w.p. \(0.1\).  
\(J_{\mathrm{CE}}=10+0.1\cdot 100=20\), violation prob \(0.9>0.2\) (infeasible for CMDP) or if forced not to offer \(S\), \(J=10\).

**Optimal constrained policy:** Light at \(t=0\), then \(S\) at \(t=1\).  
\(J^\star=9+0.95\cdot 100=104\), violation prob \(0.05\le 0.2\).

**Corollary:** \(J^\star > J_{\mathrm{CE}}\) (and CE may be constraint-infeasible). Hence CE-myopic is not optimal for this EC-SPO instance. ∎

(The same qualitative gap holds if “Heavy” is a large multi-task pack and “Light” a small pack under Mode-2 load→PRR maps.)

### 4.7 Why RL (safe / constrained RL) is the fitting tool class

| Requirement | Classical | RL / constrained RL |
|---|---|---|
| Combinatorial \(A_t\) online at ms scale | Heavy MINLP each slot | Learned policy amortized |
| Endogenous stochastic dynamics | Misspecified if open-loop | Learns closed-loop from interaction |
| Chance constraint on trajectory | Hard in CE | CMDP / Lagrangian / constrained policy opt |
| Long-horizon opportunity cost | Needs DP / tree | Bootstraps / returns |

**Claim E (method class):** Among scalable closed-loop methods for EC-SPO, constrained RL (or RL with a safety filter that preserves the CMDP constraint) is the appropriate primary solver class; online CE-MINLP and myopic dual packing are baselines expected to be suboptimal under Claims B–D.

We do **not** claim “no algorithm except PPO exists.” We claim: **exact DP/MINLP over the true endogenous CMDP is intractable or misspecified; the standard engineering substitutes are structurally suboptimal; learning a closed-loop policy is the coherent attack.**

---

## 5. Redefinition checklist (workflow gate — must pass before RL)

| # | Gate | Status |
|---|---|---|
| G1 | Problem stated without reference to beating a named frozen baseline | **PASS** (EC-SPO) |
| G2 | Utility = realized \(U\); constraint = closed-loop \(P_{\mathrm{viol}}\le\alpha\) | **PASS** |
| G3 | MDP/CMDP with endogenous radio dependence written | **PASS** |
| G4 | NP-hard / intractable exact DP argument | **PASS** (A) |
| G5 | CE / open-loop MINLP misspecification + suboptimality argument | **PASS draft** (B,C) — needs fully specified toy proof |
| G6 | Statement why myopic dual packing ≠ EC-SPO optimum | **PASS** (D) |
| G7 | RL method class justified without promising a numeric % | **PASS** (E) |
| G8 | Toy theorem: explicit 2-epoch instance CE < optimal | **PASS** (§4.6) |
| G9 | Simulator contract implements endogenous load→CBR feedback measurably | **FAIL** (DCC-on re-audit `72600–72607`; radio-endogenous EC-SPO unsupported here) |
| G10 | Apply RL | **BLOCKED** — G9 FAIL; do not apply RL to this problem as written |

---

## 6. What we will *not* do next

- Train PPO “to beat LVM” as the problem definition.  
- Another router among frozen packs.  
- Surrogate pack scores as \(J\).  
- Declare the entire NR-V2X RL field dead because myopic packing was hard to beat.

---

## 7. Immediate next technical steps (still pre-RL)

1. **Write G8 toy theorem** with numbers (2 epochs, 2 packs, endogenous PRR).  
2. **Audit simulator** for measurable \(A_t\to R_{t+1}\) dependence; if weak, strengthen problem coupling (load sensitivity) so EC-SPO is not secretly exogenous.  
3. Only after G8–G9: specify constrained RL algorithm (e.g. Lagrangian PPO / CPO-style) and **then** implement baselines (CE-MINLP, myopic dual, greedy) for fair comparison.

---

## 8. Success criterion (after RL is allowed)

Aligned with user: **flat-out win vs the best implemented baseline** on the EC-SPO panel (ties / few defensible losses OK), with \(P_{\mathrm{viol}}\le\alpha\) and deploy latency.  
Baselines TBD after formulation lock — not used to define the problem.
