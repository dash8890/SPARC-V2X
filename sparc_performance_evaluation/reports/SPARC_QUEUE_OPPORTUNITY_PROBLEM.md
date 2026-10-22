# Sequential Pack Offloading under Helper-Queue Opportunity Cost

**Status:** ACTIVE after G9 (radio-endogenous EC-SPO premise **failed** in WiLab).  
**G9 evidence:** `SPARC_G9_ENDOGENOUS_LOAD_AUDIT.md` — mean ΔCBR(heavy−empty)≈0.001; premise_supported=false.

## Problem (sim-backed)

At each epoch choose a feasible computation pack. Assignments **occupy helper queues** and block or delay other tasks’ completions under stochastic Mode-2 delivery. Maximize expected deadline-valid completed value subject to \(P_{\mathrm{viol}}\le 0.02\) and byte envelope.

Radio may be treated as (approximately) exogenous given G9; **endogeneity is through helper queues / commitments** — which the WiLab task layer already implements.

## Why not one-shot MINLP / myopic value packing

Myopic max \(\sum v_i\) (or dualized packing) ignores that admitting a long job on helper \(h\) increases finish times of future jobs on \(h\).  
CE one-step with current queues is still suboptimal for multi-epoch stochastic arrivals (standard sequential knapsack / scheduling gap).

Toy: two tasks on one helper — large value slow job now vs two medium jobs across epochs; myopic takes slow job and loses.

## Method class

Constrained / residual RL or short-horizon rollout search over packs with queue-aware continuation — vs myopic packer and CE one-step MILP baselines (to implement).

## Success

Flat win vs best classical baseline on matched panel; safety held.
