# RAO-SPARC — Phase-1 Prerequisite Gate (Direct vs POPL Selector)

**Status:** Locked design for Stage-3 Phase 1.  
**Authorized now:** Protocol verification and documentation.  
**Not authorized by this document:** New paired closed-loop collection, selector fitting, RAO PPO, baseline edits, locked-val fitting, manuscript claims, final-test access.

**Parent specs:** `RAO_SPARC_ALGORITHM_SPECIFICATION.md`, `POPL_P5_RESEARCH_DECISION.md`  
**Seed file:** `protocol/rao_phase1_seeds.json`

---

## 0. Purpose

Test whether **observable regime and pack features** can predict, *before execution*, which of two frozen generators (Direct PPO vs POPL-P5) yields higher matched remaining completed value — and whether acting on that prediction improves episode-level \(V_{\mathrm{done}}\) over always-Direct and always-POPL.

This is a **feasibility** test for adaptive selection. A Direct/POPL switcher is **not** assumed sufficient for the ≥15% project target; it is the first falsifiable scientific obstacle for RAO.

---

## 1. Frozen constituents (no weight updates)

| Role | Artifact |
|---|---|
| Reference \(\pi_D\) | `results/wilab_direct_ppo_96ep/checkpoint_best_delta_va.pt` (`logit_mode=direct`) |
| Alternative \(\pi_P\) | `sparc_performance_evaluation/results/raw/popl_p5_pilot/popl_p5_checkpoint.pt` (optimized inference path) |
| Governor / envelope | `SafetyGovernor(horizon_envelope_only)`, \(B_{\max}=175\,\mathrm{kB}\) |
| Simulator | Existing WiLab Mode-2 closed loop; unchanged |

Both policies run **deterministic** inference for Phase-1 labels (reproducible packs). Stochastic exploration is out of scope for the prerequisite gate.

---

## 2. Candidate-state sampling (when collection is separately authorized)

### 2.1 Seed pool

| Pool | Rule |
|---|---|
| Collection | `sim = 72100+i`, `task = 182100+i`, \(i=0,\ldots,N-1\), \(N\le 48\) |
| Fit split | pairs \(i\in[0,31]\) |
| Held-out development | pairs \(i\in[32,47]\) |
| **Forbidden for fitting** | locked val `82001/02/11/12` |
| **Forbidden entirely** | final-test `93000–93004` |

Historical train episodes `72000–72095` are **not** reused for Phase-1 pair collection (avoid recycling Direct/POPL training trajectories as selector labels).

### 2.2 Environment (nominal Phase-1)

Density 35; deadline×1.0; tasks/source 3; default helper CPU range \([2\times10^8,10^9]\); 100 controller epochs; 12 s radio; safety payload 300 B.

*(Off-nominal axes are reserved for later robustness of a frozen RAO controller — not Phase-1 fitting.)*

### 2.3 Fork protocol (matched \(\Delta G\))

For each episode index \(i\):

1. Run a **probe** episode under Direct (or capture at first opportunity) to obtain causal state \(s_{t^\*}\) and candidate set \(\mathcal C_{t^\*}\) at fork epoch \(t^\*\).
2. **Fork epoch:** first epoch with `n_candidates > 0`; if none, epoch 0.
3. At \(t^\*\), compute  
   - \(A^D=\pi_D(s_{t^\*})\) (deterministic),  
   - \(A^P=\pi_P(s_{t^\*})\) (deterministic).
4. If `pack_sig(A^D)=pack_sig(A^P)`, record a **tie sample** (no value label for selection; used only for complementarity / coverage stats) and **do not** spend a second rollout.
5. If packs differ and both are feasible under \(\mathcal F_{t^\*}\):
   - Episode REF: `force_pack` \(A^D\) at \(t^\*\); continuation = frozen Direct for \(t>t^\*\).
   - Episode ALT: `force_pack` \(A^P\) at \(t^\*\); continuation = frozen Direct for \(t>t^\*\).
6. Matched exogenous: identical `(sim_seed, task_seed)`, governor, envelope, density, MATLAB trajectory prefix for \(t<t^\*\).

**Budget (when approved):** ≤48 episodes for probes + ≤2×(number of disagreeing forks) forced completions. Cap disagreeing paired rollouts at **32** (fit) + **16** (held-out) = **48** pairs. No automatic extension.

Reuse of the POPL-P5 paired harness (`popl_paired_rollout_harness.py` / `force_pack` path) is preferred over new simulator APIs.

---

## 3. Outcome labels and matched-return calculation

### 3.1 Episode returns

Let \(G^{\mathrm{D}}\) / \(G^{\mathrm{P}}\) be **episode completed value** \(V_{\mathrm{done}}\) from the REF / ALT branches (same definition as Stage-0 / POPL-P5).

Optionally also store return-to-go from fork:

\[
G_{\ge t^\*}=\sum_{e\ge t^\*} r_e,\quad r_e=\text{audit reward at epoch }e.
\]

**Primary label** uses full-episode \(V_{\mathrm{done}}\) (consistent with POPL-P5 pilot). Report fork RTG as secondary.

### 3.2 Selection label

\[
\Delta G = G^{\mathrm{P}}-G^{\mathrm{D}},\qquad
y=\mathbf 1\{\Delta G>0\}.
\]

Ties \(\Delta G=0\): exclude from BCE training; count in calibration / net-value as “indifferent.”

### 3.3 Action-independent baseline assumption

Frozen Direct \(G^{\mathrm{D}}\) is independent of selector parameters \(\phi\). Continuation policy after the fork is identical (Direct) on both branches — same locking assumption as `POPL_LEARNING_RULE_CORRECTNESS_REVIEW.md`.

---

## 4. Selector features (causal only)

Feature builder: target module `controllers/rl/rao/regime_features.py`.

### 4.1 Allowed inputs (G1)

Built **only** from information available at decision time \(t^\*\):

| Block | Examples |
|---|---|
| Radio / safety | CBR, PRR, P95, validity flags, pending packets, cumulative safety counters |
| Workload / deadline | #pending tasks, mean/min slack, offered value pending, tasks released vs horizon fraction |
| Helper | #helpers, mean/std ready-time, mean/std CPU rate, queue pressure |
| Envelope | remaining / used bytes under 175 kB, #unique sources eligible |
| Pack \(A^D\), \(A^P\) | sizes, total bytes, mean task value, mean finish_ms, helper-id histograms, source sets |
| Pack **difference** | symmetric difference of `(task,helper)` pairs; bytes Δ; value Δ of exclusive tasks; helpers newly loaded vs freed; #deadline-feasible tasks admitted in one pack but not the other |

### 4.2 Forbidden inputs (leakage)

| Forbidden | Reason |
|---|---|
| \(\Delta G\), \(G^{\mathrm{D}}\), \(G^{\mathrm{P}}\), future rewards | Outcome leakage |
| Branch identity as a free label feature at inference | Trivial |
| Locked-val seed IDs or “known intervention” tags (82002 task-19 lore) | Overfitting / non-causal |
| Post-governor delivered/completed task sets from the future | Future |
| Analytical surrogate scores used **as training targets** | RHO failure mode (features may include finish_ms of **proposed** tasks as state description only) |

### 4.3 Selector model (Phase-1 default)

- Binary classifier \(g_\phi(x)=\widehat P(y=1\mid x)\) (logistic regression or small MLP ≤2×64).
- Loss: BCE on fit split only.
- Deployment rule (for G4 closed-loop):  
  \[
  \text{choose POPL iff } g_\phi(x)\ge \tau,\quad \tau=0.5
  \]
  unless packs identical → Direct. No threshold tuning on locked val. Optional calibration plot on held-out **development** pairs only (G5).

---

## 5. Matched closed-loop selection evaluation (G4)

On the held-out development pairs (and, if collection approved, a fresh replay of those forks):

For each disagreeing fork state:

1. Selector picks \(A\in\{A^D,A^P\}\) from features only.
2. Realized return \(G^{\mathrm{sel}}\) is the **already collected** branch return for that pack (no third rollout required if both branches cached).
3. Compare:

| Comparator | Definition |
|---|---|
| Always-Direct | \(G^{\mathrm{D}}\) |
| Always-POPL | \(G^{\mathrm{P}}\) |
| Oracle | \(\max(G^{\mathrm{D}},G^{\mathrm{P}})\) |
| Selector | \(G^{\mathrm{sel}}\) |

**Net value (decisive):**

\[
\Delta_{\mathrm{vs\ Direct}}=\mathbb E[G^{\mathrm{sel}}-G^{\mathrm{D}}],\quad
\Delta_{\mathrm{vs\ POPL}}=\mathbb E[G^{\mathrm{sel}}-G^{\mathrm{P}}].
\]

G4 **PASS** requires both means \(>0\) on the held-out development set, with bootstrap 95% CI for \(\Delta_{\mathrm{vs\ Direct}}\) reported (CI may include 0 — report honestly; PASS still requires positive point estimates on **both** vs Direct and vs POPL).

If only classification accuracy rises but \(\Delta_{\mathrm{vs\ Direct}}\le 0\) or \(\Delta_{\mathrm{vs\ POPL}}\le 0\), G4 **FAIL**.

---

## 6. Fixed G1–G7 acceptance protocol

| ID | Test | Required evidence | PASS rule |
|---|---|---|---|
| **G1** | Causal inputs | Static audit of feature code + automated test: features computable from `(state, cands, A^D, A^P)` alone; unit test rejects tensors that include \(\Delta G\) / future rewards | No leakage channel |
| **G2** | Complementarity | On fit+heldout disagreeing pairs: fraction with \(\Delta G>0\) and fraction with \(\Delta G<0\) both \(\ge 0.15\); mean \(|\Delta G|\) on nonzero pairs reported | Both sides contribute ≥15% of disagreeing pairs |
| **G3** | Predictability | Held-out accuracy / balanced accuracy of \(g_\phi\) vs always-Direct (\(y=0\) prior) and vs majority-class; also vs fixed rules: (a) always POPL, (b) “prefer larger pack”, (c) “prefer higher analytic sum” | Selector balanced accuracy **>** best of {always-Direct label accuracy, always-POPL, rules (b)(c)} |
| **G4** | Net value | §5: \(\mathbb E[G^{\mathrm{sel}}-G^{\mathrm{D}}]>0\) and \(\mathbb E[G^{\mathrm{sel}}-G^{\mathrm{P}}]>0\) on held-out | Both point estimates positive |
| **G5** | Calibration | Bin held-out \(\widehat p\) into ≤5 bins; plot mean \(\widehat p\) vs empirical \(P(y=1)\); ECE reported | ECE ≤ 0.15 **or** monotone reliability curve (document which) |
| **G6** | Cost | Microbench: feature build + selector forward + both pack proposes (Direct+POPL) + governor.apply, P95 and max on candidate epochs | P95 ≤ 10 ms and max ≤ 10 ms after warmup |
| **G7** | Fairness | Config checksum: same governor class, envelope, candidate generator, traces, seeds | No silent baseline/simulator edits |

**Overall Phase-1 PASS:** G1 ∧ G2 ∧ G3 ∧ G4 ∧ G6 ∧ G7; G5 recorded (soft FAIL → “PASS_WITH_CALIBRATION_WARNING” only if G4 strong and ECE≤0.20).

**G4 is decisive.** Accuracy without net value → do not proceed to Phase 2.

---

## 7. Runtime budget (G6 detail)

Measured path (online):

```
candidates → Direct.propose → POPL.propose → regime_features
         → selector.choose → governor.apply
```

| Component | Notes |
|---|---|
| Direct | Existing residual sampling (deterministic) |
| POPL | Optimized pack policy (P5 inference-opt; already ≤6 ms alone on val) |
| Selector | Must be cheap (logistic / tiny MLP) |
| Features | Pure CPU/numpy or tiny torch; no second closed-loop |

Warmup before measurement (same as POPL inference-opt). Budget includes **complete** controller decision, not actor-only.

---

## 8. Leakage checklist (mandatory before any fit)

1. Feature dump on one synthetic state contains no keys matching `/delta_G|G_alt|G_ref|future|realized/`.
2. Selector training script refuses if any fit seed ∈ `{82001,82002,82011,82012}` or final-test set.
3. Labels \(y\) joined by pair_id only after feature freeze for that row.
4. No peeking at held-out for threshold search (τ fixed at 0.5 for G4).

Automated tests (target paths when implementation is authorized):

- `sparc_performance_evaluation/tests/test_rao_invariants.py` — G1 feasibility / leakage
- `sparc_performance_evaluation/tests/test_rao_learning_signal.py` — label construction / ΔG sign
- `sparc_performance_evaluation/tests/test_rao_controller_runtime.py` — G6 microbench harness

---

## 9. Baselines for Phase-1 reporting (not full RAO bake-off)

| Method | Role in Phase 1 |
|---|---|
| Always-Direct | Lower bound / reference |
| Always-POPL | Constituent alternative |
| Oracle (best of D/P per fork) | Upper bound on switcher value |
| Feature rules (b)(c) | Non-learned switching |
| Fitted selector | Gate subject |

VAS/LAS/VLRO/LVM are **not** required for Phase-1 G1–G7; they remain mandatory for any later full RAO validation (§8 of the algorithm spec).

---

## 10. STOP conditions (Phase 1)

| If | Then |
|---|---|
| G2 fails (no complementarity on train-dev) | Reject adaptive selection; do not invent val-seed complementarity |
| G3 fails | Do **not** rescue using 82001/02/11/12 |
| G4 fails | Do **not** start Phase-2 adaptive-actor PPO |
| G6 fails | Fix implementation cost with **behavior-preserving** optimization before any training; no silent policy change |
| G1 fails | Halt until leakage removed |

These reject an unsuccessful **mechanism**, not the project goal.

---

## 11. Authorization boundary (this document)

| Activity | Decision |
|---|---|
| Publish / lock this gate protocol | **Done** |
| Implement feature/selector **code stubs + unit tests** without MATLAB pairs | Allowed as engineering prep |
| Collect ≤48 train-dev disagreeing pairs + fit selector | **Requires separate human approval** citing this gate |
| RAO adaptive-actor PPO (Phase 2) | Not authorized |
| Tune on locked validation seeds | Forbidden |
| Final-test / manuscript superiority | Forbidden |
| Modify VAS/LAS/VLRO/LVM/Direct/POPL weights | Forbidden |

---

## 12. Deliverable checklist (spec completeness)

| Item | Location |
|---|---|
| Candidate-state sampling | §2, `rao_phase1_seeds.json` |
| Outcome labels / matched returns | §3 |
| Selector features | §4 |
| Leakage checks | §4.2, §8 |
| Runtime budget | §7 / G6 |
| Fixed G1–G7 protocol | §6 |
| STOP / phase gating | §10, algorithm spec §3 |

---

## 13. Status line

**Phase-1 protocol is locked.** The first scientific question for RAO-SPARC is whether a controller can recognize *when* to use Direct vs POPL **before** the outcome is known, and whether that recognition improves realized completed value on held-out **training-development** pairs.

Until a separate collection/fitting authorization is granted, no new paired rollouts or selector training will be run.
