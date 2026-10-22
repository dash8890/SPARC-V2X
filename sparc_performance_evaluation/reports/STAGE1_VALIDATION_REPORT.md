# Stage 1 — Validation Report

**Training:** complete (96 episodes, train pool `72000+` / `182000+`).  
**Checkpoint:** `results/wilab_stage1_context_direct_96ep/checkpoint_best_delta_va.pt` (selected at update 8 by `best_mean_v`).  
**Val panel:** dens=35, \(B_{\max}=175\) kB, common governor, seeds `82001/82002/82011/82012`.  
**Final-test:** not used / not authorized.

Baselines (VAS, LAS, VLRO, LVM, prior Direct SPARC) are the Stage-0 matched common-governor replay on the same seeds. New SPARC is context-conditioned Direct with sequential context recompute.

---

## Matched outcomes

| Method | mean \(V_{\mathrm{done}}\) | mean \(R\) | \(P_{\mathrm{viol}}\) | Offered bytes (mean) |
|---|---:|---:|---:|---:|
| VAS | 38.0362 | 0.5941 | 0.000 | 8653 |
| LAS | 36.2995 | 0.5662 | 0.000 | 8207 |
| VLRO | 39.9785 | 0.6240 | 0.000 | 9881 |
| SPARC (prior Direct, Stage 0) | 46.0571 | 0.7181 | 0.000 | 7698 |
| **LVM (best matched)** | **46.8173** | 0.7312 | 0.000 | 7973 |
| **SPARC Stage-1 (context Direct)** | **41.0476** | 0.6405 | **0.000** | 6500 |

Per-seed Stage-1 SPARC \(V\): 39.84, 36.68, 56.05, 31.62.

\[
G
=
\frac{41.0476-46.8173}{46.8173}\times 100\%
=
\mathbf{-12.3\%}.
\]

Engineering benchmark \(1.15\times V_{\mathrm{LVM}}\approx 53.84\) was **not** approached.

---

## Dual decisions (`STAGE1_EVALUATION_FRAMEWORK.md`)

### Decision A — Engineering target (\(G\ge 15\%\))?

**No.** \(G=-12.3\%$ vs strongest matched competitor (LVM).

### Decision B — Scientifically meaningful scheduling improvement worth developing further?

**No — not as a value-superior scheduler on this panel.**

| Factor | Assessment |
|---|---|
| Effect size | \(G\le 0\%\) → **no demonstrated value advantage** over LVM; also **−10.9%** vs prior Direct SPARC |
| Safety | **Held:** \(P_{\mathrm{viol}}=0\) (mandatory OK) |
| Fairness | Matched governor / envelope / seeds (OK) |
| Mechanism (D7) | Trained policy remains responsive on probes: deadline 32%, helper 79%, CBR 64% (gate spirit OK) |
| Consistency | One strong seed (56.0) cannot offset weak seeds; mean below LVM and prior SPARC |
| Overhead | Larger scorer + sequential recompute without a value win |

Mechanism improved relative to Stage-0 D7 failure, but **did not translate into matched completed-value gains**.

---

## Hard stops

| Check | Result |
|---|---|
| \(P_{\mathrm{viol}}\le 2\%\) | Pass (0%) |
| Unequal admission explaining a *gain* | N/A (no gain); offered bytes lower than LVM (6500 vs 7973) |
| Post-train D7 negligible | No — responsiveness retained |
| Ignore strongest comparator | Avoided — primary \(G\) vs LVM |

---

## Classification

| Label | Applies? |
|---|---|
| Target achieved (\(G\ge15\%\)) | No |
| Promising (\(5\le G<15\)) | No |
| Marginal (\(0<G<5\)) | No |
| **No demonstrated value advantage (\(G\le0\))** | **Yes** |

---

## Authorization

- **Sealed final-test:** still **not** authorized.
- **Manuscript architecture claim** for Stage-1 SPARC as superior: **not** authorized.
- Next step (if any) requires an explicit revise/retain decision — e.g. diagnose why context conditioning underperforms prior Direct (selection/init/capacity/sequential likelihood), not silent governor retuning.

## Artifacts

- Train: `results/wilab_stage1_context_direct_96ep/`
- Val SPARC: `sparc_performance_evaluation/results/raw/stage1_val_sparc_context/`
- Framework: `STAGE1_EVALUATION_FRAMEWORK.md`
