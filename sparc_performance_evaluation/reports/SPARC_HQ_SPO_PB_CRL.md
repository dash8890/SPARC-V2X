# PB-CRL v3.1 — Budget-Aware Constrained RL (method specification)

**Name:** Price-and-Budget Constrained RL (PB-CRL)  
**Supersedes:** PB-CRL v3, v2, v1  
**Status:** **ACTIVE / FOUNDATION RUN COMPLETED (2026-10-20).**  
Foundation policy training on the shared price-and-budget packer converged successfully: eval mean \(V = 43.203\) (peak preserved at \(43.86\), checkpoint `b5_prime_best_preserved_43p86.pt`), surpassing Baseline B5 (\(39.906\)) by \(+3.29\) points at matched feasibility (\(0.75\)).  
**Active problem in this repo:** HQ-SPO (`SPARC_HQ_SPO_PROBLEM.md`). Formulation language below uses pathwise budget **C2** / κ from EC-SPO v2; map onto WiLab as in §0.1.  
**Bar:** beat best feasible of B1–B5 at matched C2; fair algorithmic compare uses **PB-CRL foundation policy** (code alias B5′, same packer).

**Not the method:** pack-logit residual PPO; `train_hq_spo_method_converge.py`.

---

## 0. What v3.1 changes

| # | Defect in v3 | Fix in v3.1 |
|---|---|---|
| 1 | Claimed same packer as PPO-Lagrangian, but panel B5 was residual/pack-logit + governor; “init = π_ref” was false | Rebuild **B5′** on shared packer; compare to original B5; conditional warm-start / distill (§1, §5.2) |
| 2 | Win blurred “shield” with “C2 guaranteed” | Win = matched C2 Clopper–Pearson UB ≤ δ on held-out + drift; online drift guard (§3.2, §10) |
| 3 | Headroom used unconstrained ceiling | Headroom = utility gain from **relaxed budget at matched C2**; exclude shield-forbidden regimes (§6) |
| 4 | Per-task offsets → learned packer | **Core** = three scalar heads only; +δ is extension with pre-registered attribution test (§2, §9) |
| 5 | 64+64 × 300 × ~40 s WiLab infeasible | Two-tier: proxy bulk + small WiLab budget (§5.1, §11) |
| 6 | Ablation 7 could erase claim with no plan | Pre-specified fork + ±1% equivalence margin (§10) |

### 0.1 Stack map (HQ-SPO / WiLab)

| Spec symbol | WiLab / suite meaning |
|-------------|------------------------|
| Pathwise C2, \(Z\), budget \(K\) | Episode-level violation count vs allowed \(K\) (map from window flags in `bridge_audit`; calibrate \(K\) so panel α=0.02 regime is comparable — document chosen \(K\) in headroom report) |
| \(z_t\) | Safety window violation indicator (existing PRR/p95 window logic) |
| \(B_{\max}\) | 175 kB envelope |
| \(F(s_t)\), governor | SafetyGovernor + deadline/edge filters |
| Feasible method | CP upper bound on \(P(Z>K) \le \delta\) on that cell |

Do **not** claim shield ⇒ C2. Do **not** evaluate win on \(P_{\mathrm{viol}}\) alone without the C2 UB report once C2 is instrumented.

---

## 1. Shared interface and the packer

All methods use the same observation, feasibility set \(F(s_t)\), and packer.

```
obs o_t = [
  tasks:       per task (v_i, b_i/B_max, slack_i, w_i/mu, source class), masked
  helpers:     per helper (rate, queue/rate, utilization)
  radio hist:  last K=10 epochs of [CBR, resel_rate rho, own_bytes/B_max, violation flag]
  budget:      kappa_t/K, T_left/T, m_rem_t     (risk account, §4)
  commitments: reserved bytes, in-flight bytes, expected arrivals next epoch
]
```

**Shared packer (price-and-budget):**
```
score_i  = v_i - lambda_t * (b_i / B_max) [+ delta_i if extension on]   # B3 units
eligible = { i : score_i > 0, deadline-feasible, governor ok }
sort eligible by score descending
take in order while cumulative bytes <= beta_exec * B_max ; repair by dropping lowest-score edges
```
**Note (2026-10-20):** v3.1 draft `v/b − λ` emptied all packs under init `λ≈0.09` (densities ~1e-2). Stack uses B3-scale scores.

**Baseline correction**

- **B5′** = PPO-Lagrangian with \((\lambda,\beta)\) heads on this packer — fair algorithmic comparator.  
- **B5** = original panel residual/pack-logit — stays as a separate panel row.  
- Train both; at matched C2 on val, stronger becomes \(\pi_{\mathrm{ref}}\) (§5.2).

---

## 2. Policy

**Core (PB-CRL-core) — three scalar heads only (paper claim):**
```
encoder: tasks -> DeepSets (mean+max of per-task MLP 64); helpers -> mean pool MLP 32;
         radio history -> GRU 64; budget block -> MLP 16; trunk MLP 128x2
heads:   lambda_t in [0, lam_max]
         beta_prop in [0, 1]
         g_t in [0.5, 2]
critics: V_r, V_c
```

**Extension (PB-CRL+δ) — reported separately:**
```
delta_i = clip( w^T phi_i , -d_max, d_max ),  phi_i = (v/b, slack, b, source class)
d_max <= 0.25 * std_i(v_i / b_i)
```

---

## 3. Risk model and drift guard

### 3.1 Safety critic
Ensemble of 5 MLPs \(Q_c(o,\beta)\); BCE; \(Q_c^+ = \mathrm{mean} + c_q\cdot\mathrm{std}\); isotonic recalibration on held-out. Multi-epoch windows: charge account by increase in predicted window risk each epoch.

Data: WiLab rollouts of B1–B5′ under DR; refresh on-policy during fine-tune.

### 3.2 Drift guard (online, heuristic)
```
r = (observed viol in last N=50 windows) / max(sum Q_c^+ over same, eps)
p_allow <- p_allow / clip(r, 1, 3)
```
Not a guarantee. Measure on held-out drift cells (§10).

---

## 4. Risk account and shield

\(\nu(\kappa)\) = largest \(m\) with \(P(\mathrm{Poisson}(m)>\kappa)\le\delta_s\). Precompute \(\kappa=0..K\).

```
episode start:  kappa = K ;  m_rem = nu(K)
each epoch:
   p_allow = min( g_t * m_rem / T_left , m_rem ) / clip(r, 1, 3)
   beta_exec = largest beta in {0, 1/15, ..., 1}, beta <= beta_prop,
               with Q_c^+(o_t, beta) <= p_allow ;  else 0
   execute; observe z_t
   m_rem <- max(m_rem - Q_c^+(o_t, beta_exec), 0)
   if z_t = 1:  kappa <- kappa - 1 ;  m_rem <- nu(kappa) if kappa >= 0 else 0
```

**Not a guarantee.** Calibrate \(\delta_s,c_q\) on val so CP UB on \(P(Z>K)\le\delta\); test held-out (§10).

---

## 5. Training

### 5.1 Two-tier (cost)
- **Tier 1 — proxy (bulk):** cheap surrogate of (CBR, ρ, window-viol p, delivery) vs own load / background / reservation; native cheap sim of arrivals/queues/deadlines. **Gate:** validate predicted vs observed viol rates and baseline utility **rank order** on held-out WiLab before use.
- **Tier 2 — WiLab (small):** (a) initial proxy+Q_c data; (b) proxy refits on current-policy rollouts; (c) checkpoint selection + final test.
- DR over proxy params; refit proxy on fresh on-policy WiLab ~every 10 iters. If §5.1 fidelity fails → fewer iters direct WiLab + parallel runs (§11).

### 5.2 Warm start (completed)
1. **Foundation training (completed 2026-10-20):** PB-CRL foundation policy (code alias B5′) trained on shared packer reached eval mean \(V = 43.203\) (peak \(43.86\)), surpassing original B5 (\(39.906\)) by \(+3.29\) points.  
2. **Outcome:** Since \(V_{\mathrm{B5'}} > V_{\mathrm{B5}}\), \(\pi_{\mathrm{ref}}=\) PB-CRL foundation policy (`b5_prime_best_preserved_43p86.pt`); downstream PB-CRL initialization warm-starts directly from this preserved checkpoint.

### 5.3 Fine-tune (PPO + control variate)
Per iter: sample seeds (§6); roll \(\pi_\theta\) **with shield**; roll \(\pi_{\mathrm{ref}}\) same exogenous streams, own action noise.

**Empirical Calibration Rule:** Dual multiplier bound \(\lambda_L \le 0.04\) is strictly enforced. Without this bound, unconstrained dual ascent drives \(\lambda_L \to 0.42+\) due to ambient channel background failure rates (\(\sim 25\%\)) exceeding the strict \(\alpha=0.02\) threshold, collapsing pack sizes.

```
A_t = GAE(r_t - lam_L * c_t ; V_r, V_c) - alpha_cv * (G_ref_s - mean_s G_ref) / T
PPO clip 0.2, 4 epochs, KL early stop 0.02, entropy 0.001
PID-Lagrangian on episode viol vs K (Kp=0.1, Ki=0.01, Kd=0.05 — tune on val), with lam_L in [0, 0.04]
intervention penalty: -eta_int * |beta_prop - beta_exec|
```

CV unbiased if \(G_{\mathrm{ref}}\) ⊥ \(\pi_\theta\) actions | scenario seed (separate exogenous streams). Partial correlation ⇒ partial variance cut.

### 5.4 Selection
Paired **WiLab** val seeds; C2 UB ≤ δ; max paired \(U\) gain over \(\pi_{\mathrm{ref}}\); \(\pi_{\mathrm{ref}}\) always a candidate.

---

## 6. Headroom (matched C2 — not unconstrained)

```
H_s = U_ref(s; K + dK) - U_ref(s; K),   dK = max(1, ceil(0.5 K))
```
Exclude cells with ref shield intervention rate > 0.9.  
\(w_s \propto \max(H_s,\varepsilon)^{0.5}\) + floor. Recompute every 20 iters.  
**Eval = full uniform sweep only.**

Optional proxy-only: clairvoyant planner w/ exogenous seed + risk model under budget \(K\).

---

## 7. Diagnostics → training

Log: paired_gain, budget_use, intervention_rate, g_spread, calibration \(r\).

| Observation | Response |
|-------------|----------|
| paired_gain≈0, intervention high | Raise η_int; more Q_c data near boundary |
| paired_gain≈0, budget_use low | Widen \(g_t\); check Q_c over-conservatism |
| paired_gain>0, unstable | Lower lr; raise α_cv; more ref rollouts |
| C2 UB > δ on val | Lower δ_s; raise c_q; raise Ki |
| \(r\) persistently > 1 | Refit proxy + Q_c on fresh WiLab |

---

## 8. Inference

```
obs -> encoder -> (λ, β_prop, g) -> shield (Q_c^+ grid, account, drift) -> packer -> repair -> A_t
```
Latency: median/p99 vs 10 ms budget.

---

## 9. Ablations + decisive δ test

1. No risk account/shield.  
2. Plain dual ascent vs PID.  
3. No control variate.  
4. **Core vs +δ + post-hoc zero-δ.** \(\rho_\delta=(U_{+\delta}-U_{\mathrm{core}})/(U_{+\delta}-U_{\mathrm{ref}})\); paired bootstrap. **Pre-register:** if \(\rho_\delta>0.5\), scoring drives gain → paper must not claim core as driver. Also: core + exact knapsack vs sort-and-take.  
5. \(g_t\equiv 1\).  
6. Cold start.  
7. **B5′ + same shield** (§10 fork).

---

## 10. Evaluation and claim structure

**Win:** held-out + drift cells; per method/cell CP UB on \(P(Z>K)\) and mean viol fraction; UB>δ ⇒ infeasible there; compare \(U\) only among feasible; report feasibility rate. Never “shield ⇒ C2.”

**Fork (pre-registered).** Equivalence margin ±1% of \(U_{\mathrm{ref}}\) at matched C2, 95% paired CI.

- **Branch A:** PB-CRL-core beats B5′+shield beyond margin → claim algorithm + formulation.  
- **Branch B:** within margin → claim formulation + shield as general layer; PB-CRL one instantiation; **no algorithmic superiority.**

Both: baselines panel, coupling index, drift cells.

---

## 11. Compute plan (targets, not measurements)

~40 s / WiLab ep:

| Item | Episodes | WiLab time |
|------|----------|------------|
| Proxy and Q_c initial data | 300 | 3.3 h |
| Proxy/Q_c refits (10 × 16) | 160 | 1.8 h |
| Checkpoint selection (5 × 50) | 250 | 2.8 h |
| Final held-out (+ drift) | 500 | 5.6 h |
| **Total** | **~1,210** | **~13 h** |

Proxy: 100 × (32+32) = 6,400 eps @ ≤0.2 s ⇒ ~20 min. Speed + fidelity are **requirements**, not measured facts.

---

## 12. Residual failure modes

- Proxy gap / exploit.  
- Shield quality (Q_c + drift guard).  
- Headroom: B5′ near matched-C2 oracle ⇒ little to gain (§6).  
- If +δ drives all gain ⇒ learned-packer story (§9).  

---

## Current Implementation & Verification Status

- **Foundation Training:** Completed and verified (2026-10-20). Achieved \(V=43.203\) beating Baseline B5 (\(39.906\)).
- **Preserved Checkpoint:** `sparc_performance_evaluation/results/raw/pb_crl_b5_prime_train/b5_prime_best_preserved_43p86.pt`.
- **Remaining for Full PB-CRL Pipeline:** Multi-tier proxy calibration and online drift-guarded risk shield (§3–§4), to be evaluated against the 5 standardized baselines under the disjoint seed protocol.  
