# B4 Lagrangian PPO — Eligibility Audit for SPARC-V2X

**Date:** 2026-10-15  
**Mode:** Read-only (existing artifacts + implementation). No new training or simulations.  
**Question:** Does B4’s higher completed value make it a legitimate candidate to replace Direct PPO / represent SPARC-V2X safety+value?

---

## Verdict: **INCONCLUSIVE**

B4 correctly implements the **protocol’s unshielded Lagrangian baseline**, and on **nominal density=35** it achieves higher deadline-valid completed value with \(P_{viol}=0\).  
However, existing evidence does **not** establish that it learns or guarantees the paper’s **hard** safety constraints under relevant regimes, and its +4.71 \(V\) advantage over Direct is primarily **extra admission headroom** (500 kB, no governor)—not superior scheduling efficiency under matched constraints.

Therefore B4 is **not cleared** to replace Direct PPO as the strongest governor-matched comparator, nor to stand in as a SPARC-V2X-equivalent safety architecture, until density (or matched-envelope) evidence exists.

| Criterion | Result |
|---|---|
| Controller matches B4 contract | Yes |
| Hard SPARC PRR/P95+175 kB governor | **No** (by design) |
| Nominal FT \(P_{viol}=0\) | Yes (d=35 only) |
| Safety shown at ρ∈{25,50,80,100} | **No data** |
| Advantage under matched admission | **No** — more bytes, lower \(V\)/byte |
| Eligible to replace Direct | **No (insufficient / unfair)** |

---

## Step 1 — Controller correctness

### What B4 implements

| Element | Implementation | Source |
|---|---|---|
| Actor | `ResidualTaskPolicy`, `logit_mode=direct` (no VA additive) | `train_wilab_lagrangian_ppo.py` |
| Reward | Normalized completed-value return (`normalized_reward`) | `lagrangian_ppo.py` / episode trace |
| Safety cost | Per-epoch **union** indicator: PRR&lt;0.95 or P95&gt;50 ms on **valid** windows; invalid → cost **0** | `episode_violation_costs` |
| Advantage | \(\widehat A^r - \lambda \widehat A^c\), then normalize | `build_lagrangian_steps` |
| Cost critic | `policy.cost_value` MSE to cost return | `lagrangian_ppo_update` |
| Dual ascent | \(\lambda \leftarrow \max(0,\lambda+\eta(J_c-\kappa))\), \(\kappa=\)`cost_limit`**0.05**, \(\eta=0.05\), \(\lambda_0=0.1\) | `dual_ascent` |
| Admission | **No** SPARC governor; `forced_byte_budget` / physical cap **500000**; unique sources | trainer + FT `--no-governor` |
| Empty action | Empty assignment list (no positive sample) when budget/choices empty | `sample_action` |
| Checkpoint rule | `best_delta_va` via `evaluate()` **with** SafetyGovernor @ 175 kB | selection ≠ FT admission |

### Does it learn the paper’s hard safety constraints?

**No — it optimizes a soft average violation-rate budget of 5% of epochs**, not hard \(P_{viol}=0\).

Training evidence (`results/wilab_lagrangian_ppo_96ep/training_metrics.json`):

- 24 PPO updates; **13/24 (54%)** had `mean_epoch_violation_rate > 0.05`
- Violation rates repeatedly hit 0.25–0.50 mid-training
- \(\lambda\) rose 0.1 → ~0.24 (dual responding, not pinning cost to 0)
- Final update batch mean violation rate = 0.0 (lucky/end state, not a hard guarantee)

Protocol language matches this: B4 is *“never [to be] interpret[ed] … as guaranteed compliance.”*

**Conclusion (Step 1):** Correct as **B4 unshielded baseline**. Incorrect as an implementation of SPARC’s **hard** application-layer safety governor.

---

## Step 2 — Safety across densities

### Available artifacts

| Density | B4 present? |
|---|---|
| 35 (nominal FT) | **Yes** — `final_test_b4_d35/` |
| 25, 50, 80, 100 | **No** B4 density jobs exist |
| Direct density | Yes — `density_direct_d*` (for comparison only) |

Campaign queue never scheduled Lagrangian density sweeps (only nominal FT with `--no-governor --envelope-bytes 500000`).

### Nominal d=35 (only available)

| Seed | B4 \(P_{viol}\) | Direct \(P_{viol}\) |
|---:|---:|---:|
| 93000–93004 | 0.0 | 0.0 |

Mean PRR/P95 remain feasible at d=35 for both (same regime as P1 audit: PRR gates idle).

**Conclusion (Step 2):** Zero nominal violations **do not generalize from evidence**—there is nothing to extract at hard densities. Safety generalization is **unknown**.

---

## Step 3 — Source of performance advantage vs Direct

Paired final-test seeds; bytes from `bridge_audit.json` (post-P0 backfill).

| Seed | Direct \(V\) | Direct sum B | Direct \(V\)/kB | B4 \(V\) | B4 sum B | B4 \(V\)/kB | Δ\(V\) | Δ bytes | B4 epochs &gt;175kB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 93000 | 35.23 | 628737 | 0.0560 | 42.23 | 799733 | 0.0528 | +7.00 | +170996 | 2 |
| 93001 | 36.27 | 708081 | 0.0512 | 44.91 | 1061852 | 0.0423 | +8.65 | +353771 | 3 |
| 93002 | 35.95 | 690730 | 0.0521 | 37.79 | 829087 | 0.0456 | +1.84 | +138357 | 3 |
| 93003 | 46.61 | 682080 | 0.0683 | 48.62 | 811899 | 0.0599 | +2.02 | +129819 | 2 |
| 93004 | 35.81 | 440767 | 0.0812 | 39.83 | 454658 | 0.0876 | +4.02 | +13891 | 0 |
| **Mean** | **37.97** | **630079** | **0.0618** | **42.68** | **791446** | **0.0576** | **+4.71** | **+161367 (+26%)** | **40% of pos. epochs** |

Additional structure:

- Mean selected assignments/episode: Direct **14.8** vs B4 **17.8**
- Positive-offer epochs: Direct 5.4 vs B4 5.0 (B4 packs more bytes per offer)
- Direct max epoch bytes ≤175k always; B4 max observed in FT episode_0001: **301056** B

**Conclusion (Step 3):** Advantage is **primarily more admission** (larger physical cap, no PRR/P95 block). Scheduling **efficiency** (\(V\) per offered kB) is **worse** for B4 on average (0.0576 vs 0.0618). Not a pure “smarter scheduler under the same shield” win.

---

## Step 4 — Physical feasibility (same system?)

| Component | Same as Direct/SPARC? | Notes |
|---|---|---|
| WiLab Mode-2 bridge / radio | **Yes** | Same `run_episode` / MATLAB path |
| TaskLayer deadlines / helpers / uniqueness | **Yes** | Shared task model |
| Sealed final-test seeds | **Yes** | sim `93000`–`93004` |
| Safety payload / sim time / density (nominal) | **Yes** | 300 B, 12 s, ρ=35 |
| Application governor (PRR/P95 + 175 kB) | **No** | B4: disabled |
| Byte capacity | **No** | B4: 500 kB physical cap vs Direct 175 kB envelope |
| Checkpoint selection regime | **Mismatch** | Best ckpt chosen under **governed 175 kB** `evaluate()`; FT deployed **ungoverned 500 kB** |

**Conclusion (Step 4):** Same physical radio/task system; **not** the same admission operating point. Higher \(V\) is achieved under a **looser** admission contract.

---

## Step 5 — Architecture decision

### Against SPARC-V2X / Direct replacement criteria

| Decision criterion | Assessment |
|---|---|
| **PASS** — correct hard safety + compliance in relevant regimes + higher \(V\) under same physical **constraints** | **Not met** (soft κ=0.05; no density evidence; unmatched admission) |
| **FAIL** — advantage from violating required safety / wrong accounting / incompatible physics | **Not met** at nominal FT (\(P_{viol}=0\); accounting OK; same radio) |
| **INCONCLUSIVE** — implementation OK as B4, evidence insufficient beyond nominal | **Met** |

### Practical recommendation

1. **Keep Direct PPO** as the strongest **governor-matched** learned baseline for the SPARC manuscript claim.
2. **Keep B4** as a disclosed **unshielded / learned-soft-constraint** reference (Table IV admission row), with the +4.71 gap labeled as **not comparable** under the common 175 kB+PRR/P95 shield.
3. Do **not** authorize B4 as SPARC replacement on present evidence.
4. If eligibility is to be reopened later (out of scope here), minimum new evidence would be: (i) B4 density FT with reported \(P_{viol}\), and/or (ii) a matched-envelope B4 (175 kB ± governor) vs Direct.

---

## Artifact index

| Item | Path |
|---|---|
| B4 FT | `results/raw/final_test_b4_d35/` |
| Direct FT | `results/raw/final_test_direct_d35/` |
| B4 train | `results/wilab_lagrangian_ppo_96ep/` |
| Dual / cost code | `controllers/rl/lagrangian_ppo.py` |
| Trainer | `experiments/wilab_closed_loop/train_wilab_lagrangian_ppo.py` |
| Contracts | `protocol/controller_contracts.md`, `protocol/evaluation_protocol.md` §B4 |
| Density B4 | **absent** |

---

## One-line summary

**INCONCLUSIVE:** B4 is a valid unshielded Lagrangian baseline with a nominal \(V\) win driven by a larger admission allowance; it is **not** evidenced as a safety-feasible SPARC/Direct replacement.
