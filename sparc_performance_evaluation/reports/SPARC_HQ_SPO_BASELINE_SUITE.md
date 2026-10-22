# HQ-SPO — Official baseline suite (user-specified)

**Status:** LOCKED 2026-10-19  
**Success:** flat win on \(U=V_{\mathrm{done}}\) vs **best feasible** of these five at \(P_{\mathrm{viol}}\le\alpha=0.02\), with decision latency reported.  
**Not success:** beating only frozen LVM/Direct outside this suite.

Notation map: user \(o_t,W_t,H_t,A_t,B_{\max},\alpha\) ↔ stack state / task_layer candidates / assignments / 175 kB envelope / SafetyGovernor + \(P_{\mathrm{viol}}\).

Shared interface assumptions accepted unless marked:

| User | Stack |
|------|--------|
| \(v_i,b_i,d_i,w_i\) | `task_value`, `payload_bytes`, `deadline_ms`, cycles |
| \(queue_h, rate_h\) | `helper.ready_time_ms` / `queue_cycles`, `cpu_cycles_s` |
| `edge_ok` / finish | `task_layer.candidates` deadline filter + `finish_ms` |
| `best_helper(i)` | **[assumption]** argmin `finish_ms` among edges for task \(i\) |
| unique source | `unique_sources=True` |
| `feasible(A)` | governor + envelope + existing \(F\) |
| \(U\) | episode `V_done` / completed_value |
| \(P_{\mathrm{viol}}\) | existing safety window fraction |
| `resel_t` | **[gap]** not exposed as first-class feature today — need instrument or proxy |

---

## Five baselines

| # | Name | Spec (user) | Stack status |
|---|------|-------------|--------------|
| 1 | Greedy value-density | sort \(v_i/b_i\); pack under \(B_{\max}\) + unique source; repair | **MISSING** as exact mode (`value_aware` uses \(v/\mathrm{finish}\), not \(v/b\)) |
| 2 | CBR-threshold admission | \(B_{\mathrm{eff}}(CBR,resel)\); then baseline 1 | **PARTIAL** — `cbr_rule` scales pick count by CBR, not byte cap + resel; needs rewrite + tune |
| 3 | Myopic dual packing (LVM-class) | \(u_i=v_i-\lambda c_i\); knapsack DP; dual on viol | **PARTIAL** — frozen LVM checkpoint ≈ this class; exact DP+dual-update form may differ — audit vs spec |
| 4 | One-step CE MILP | logistic viol model → \(B_{\mathrm{cut}}\); knapsack \(\lambda=0\) | **MISSING** |
| 5 | PPO-Lagrangian | GAE reward/cost, dual \(\lambda\), same obs/mask as method | **PARTIAL** — `lagrangian_ppo.py` + training scripts exist; must match protocol (γ, cost def, seed budget) |

## Eval protocol (locked)

- Tune on **val** seeds; report on **disjoint test** seeds (not locked `8200x`; not sealed `9300x` for fitting).
- Report: \(U\) + CI, \(P_{\mathrm{viol}}\) + Clopper–Pearson, decision latency median/p99.
- Compare only at \(P_{\mathrm{viol}}\le\alpha\); violators = infeasible.
- Report hindsight upper bound for context.

## Method (SPEC — PB-CRL v3.1)

**PB-CRL v3.1** — `SPARC_HQ_SPO_PB_CRL.md` (**not run**).  
Core: \((\lambda,\beta,g)\) + shared price-and-budget packer; fair compare = **B5′** on that packer; original B5 remains a panel row.  
**Not method:** pack-logit PPO / `train_hq_spo_method_converge.py`.

## Next work (when authorized)

1. Instrument C2/\(K\) on WiLab; rebuild B5′; matched-C2 headroom.  
2. Proxy fidelity gate; then fine-tune per v3.1.  
3. Pre-registered fork: PB-CRL-core vs B5′+shield (±1% margin).
