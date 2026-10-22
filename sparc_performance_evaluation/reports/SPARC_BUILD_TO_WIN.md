# BUILD TO WIN — Problem + Method

**User:** proceed under strong problem/method (2026-10-19).  
**Not:** kill-gate theater with frozen weak probes.

---

## 1. Strong problem (locked for this branch)

**Name:** Contention-aware pack scheduling under high packing pressure.

**Environment:** density 35, **tasks_per_source = 4**, deadline×1, same 175 kB governor / Mode-2 stack.

**Why this can beat LVM (evidence, not hope):**

On fresh tps4 seeds `72300–72307`, mean LVM still led — but **Direct beat LVM on multiple seeds** (+3.22 on 72301, +6.47 on 72305). So LVM is not an episode-wise ceiling. The failure of mean(POPL) vs LVM only killed a **weak probe**, not this problem.

**Question the method must answer:**

> At packing-pressure states, which feasible complete packs raise realized \(V_{\mathrm{done}}\) vs LVM when the rest of the episode follows LVM?

**Success (method, later):** paired \(J > J_{\mathrm{LVM}}\) on fresh tps4, \(P_{\mathrm{viol}}\le0.02\), ≤10 ms deploy.  
**15%:** stretch only.

**Explicit non-goal:** winning nominal tps=3 by residual-on-LVM (closed).

---

## 2. Strong method — Search-RPI (not frozen POPL)

**Offline teacher:** large **search** over complete packs at \(s_t\):

\[
\mathcal C_t =
\{A^{\mathrm{LVM}}, A^{\mathrm{Direct}}\}
\cup
\mathrm{EditFamilies}(A^{\mathrm{LVM}})
\cup
\mathrm{RandomFeasible}(s_t)
\]

Score each \(A\) by simulator \(G(A\rightarrow\pi_{\mathrm{LVM}})\) (H=1 full residual).  
Teacher: \(A^\star=\arg\max \Delta^{\mathrm{safe}}\).

**Learn:** propose/score packs that match \(A^\star\) on states with \(\Delta^\star>\varepsilon\); deploy with LVM fallback + calibrated gate.

**Why stronger than failed paths:**

| Failed | This |
|---|---|
| Probe = frozen POPL/Direct mean | Probe = **search** for packs that beat LVM |
| Close problem if POPL loses | Keep problem if search finds \(\Delta^\star>0\) |
| Residual from scratch / RPI without search mass | Only distill **demonstrated** improvements |
| Nominal tps3 | tps4 where episodic gaps already observed |

---

## 3. First experiment — signal hunt (not a FAIL trap)

**Goal:** find whether searchable one-step packs beat LVM under tps4.

**Seeds:** fresh `72300+` (reuse gate seeds; prioritize 72301, 72305 where Direct already won the episode).

**Per selected decision state:** evaluate LVM (ref), Direct pack, and edit families (DROP/REBIND/SUB/ADD/DROP+ADD/REBIND+ADD), capped for compute.

**Report:** \(P(\Delta^\star>0)\), \(E[\Delta^\star\mid>0]\), \(M_e\), winning families, top before/after packs.  
**Continue if:** repeated material \(\Delta^\star>0\) with force_match.  
**Do not:** declare problem dead because mean POPL < LVM.

Script: `run_tps4_search_signal.py`  
Raw: `results/raw/tps4_search_signal/`
