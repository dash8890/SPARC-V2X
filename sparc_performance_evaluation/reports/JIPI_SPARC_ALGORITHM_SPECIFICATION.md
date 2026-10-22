# JIPI-SPARC — Algorithm Specification

**Full name:** Joint-Impact Pack-Edit Inference for SPARC  
**Status:** Specification + implementation + pre-training correctness gate  
**Training / validation eval / final-test / manuscript:** still **not authorized**

---

## 0. Positioning

| Item | Decision |
|---|---|
| Relationship to Stage 1 | **New candidate.** Not a modification or reopening of the failed Stage-1 conditional scorer |
| Base scheduler | Frozen, previously validated Direct PPO (immutable checkpoint) |
| What is learned | Edit-selection policy over KEEP / helper rebind / task substitution |
| What is not learned | Direct logits, governor, radio, delivery predictors |
| Primary scientific hypothesis | Learning whether a bounded pack edit is worth its **opportunity cost** (displaced value + projected helper contention) can raise deadline-valid completed value without abandoning Direct’s proven schedule |

**Diagnostic lineage (evidence motivating this design):**

| Finding | Implication for JIPI |
|---|---|
| Stage 1 wholesale rescoring lost value | Keep Direct pack available via KEEP; edit at most one decision |
| 82002/task19 helper rebind: local +2.454 | Rebind can recover deadline-valid value when externalities are benign |
| Universal `tight_slack_helper_rebind`: FAIL (3/15 harm) | Must represent helper-queue externalities; never hard-code “more slack ⇒ apply” |
| 82011→helper9 collateral loss of task 9 | Joint-impact / queue-risk features are mandatory regression content |
| 82012 task1 vs task13 analytic +2.256 | Substitution allowed if net value features expose displacement cost |

Prior Direct PPO remains the SPARC reference until a future matched validation promotes otherwise. Stage 1 stays closed. Final-test remains sealed.

---

## 1. Design decisions

| Component | Specification |
|---|---|
| Base scheduler | Frozen Direct PPO |
| New learning component | Joint-impact edit-selection policy |
| RL algorithm | PPO |
| Control level | Application-layer task–helper scheduling |
| Radio mode | NR-V2X Mode 2, unchanged |
| Safety governor | Existing governor, unchanged |
| Computation byte envelope | 175,000 bytes |
| Reward | Realized deadline-valid completed task value |
| Permitted edits | KEEP, helper rebind, task substitution |
| Edits per decision epoch | At most one |
| Explicit task-admission STOP | Not introduced |
| Candidate generation | Existing common feasibility rules |
| Primary comparator | Strongest of all five matched methods |
| Final-test access | Prohibited during development |

---

## 2. Core scheduling mechanism

At each scheduling epoch:

1. Observe the existing system state.
2. Generate feasible task–helper candidates (existing generator).
3. Let **frozen** Direct PPO construct its normal assignment pack \(A_t^0\).
4. Enumerate feasible modifications to that pack.
5. Estimate each modification’s effect on the whole scheduling state, including helper-queue externalities.
6. Use a learned PPO policy to choose **KEEP** or **exactly one** modification.
7. Submit the chosen pack to the existing safety governor.
8. Execute admitted assignments through the unchanged Mode 2 simulator.
9. Train the editing policy using actual deadline-valid completed value.

Step 6 is the distinguishing decision: learn **whether an alternative is worth its opportunity cost**, not merely pick the highest individual score.

```text
Observed system state
          |
          v
Common feasible task–helper candidates
          |
          v
Frozen Direct PPO scheduler
          |
          v
Reference assignment pack A0
          |
          v
Enumerate feasible edits
     /       |        \
   KEEP    REBIND   SUBSTITUTE
     \       |        /
          |
          v
Whole-pack impact features
          |
          v
Learned edit-selection PPO
          |
          v
Choose ONE edit or KEEP
          |
          v
Existing safety governor
          |
          v
NR-V2X Mode 2 transmission
          |
          v
Helper execution / deadline outcome
          |
          v
Realized completed-value reward
          |
          v
PPO policy and critic update
```

---

## 3. Mathematical formulation

### 3.1 State and reference action

Let \(s_t\) be the causally available system state at epoch \(t\) (tasks, sizes/values/deadlines, radio observations, helper rates/ready times/workloads, remaining byte capacity, governor observations).

Frozen Direct produces

\[
A_t^0=\pi_{\mathrm{Direct}}(s_t),
\]

with assignments \((i,h)\) and

\[
\sum_{(i,h)\in A_t^0}b_i\le B_{\max},\qquad B_{\max}=175000.
\]

Task uniqueness, source restrictions, connectivity, deadlines, and existing helper-assignment restrictions are unchanged.

### 3.2 Feasible edit space

\[
\mathcal{E}_t=\{\mathrm{KEEP}\}\cup\mathcal{E}_t^{H}\cup\mathcal{E}_t^{S}.
\]

An edit \(e\) yields pack \(A_t(e)\).

**KEEP.** \(A_t(\mathrm{KEEP})=A_t^0\).

**Helper rebind.** For \((i,h)\in A_t^0\) and feasible alternative \(h'\):

\[
A_t(e)=A_t^0\setminus\{(i,h)\}\cup\{(i,h')\}.
\]

Task set and admitted byte total unchanged; all feasibility constraints must hold.

**Task substitution.** For \((j,h_j)\in A_t^0\) and new feasible \((i,h_i)\):

\[
A_t(e)=A_t^0\setminus\{(j,h_j)\}\cup\{(i,h_i)\},
\quad
\sum_{(k,h)\in A_t(e)}b_k\le 175000.
\]

Displacement of a currently successful reference task is **not** automatically prohibited; displaced value and downstream costs must appear in decision inputs.

**At most one edit per epoch** in this initial version.

### 3.3 Editing policy

Let \(z_t(e)\) be observable joint-impact features of edit \(e\). The PPO actor selects

\[
e_t\sim\pi_\theta(e\mid s_t,A_t^0,\mathcal{E}_t)
=
\frac{\exp(f_\theta(z_t(e)))}
{\sum_{e'\in\mathcal{E}_t}\exp(f_\theta(z_t(e')))}.
\]

KEEP has its own learnable logit. Proposal \(\widetilde{A}_t=A_t(e_t)\); executed pack

\[
A_t^{\mathrm{exec}}=g(\widetilde{A}_t,s_t)
\]

via the **unchanged** governor \(g\).

---

## 4. Joint-impact feature design

Universal rebinding failed by scoring the target task’s slack gain without adequate harm accounting. Features must describe effects on the **whole pack and affected helper queues**.

### 4.1 Task-value features

- Value of newly included / rebound task  
- Value of any displaced task  
- Added − displaced value difference (e.g. 82012: \(3.738-1.482\approx 2.256\) as **input**, not guaranteed gain)  
- Existing total assigned value; remaining opportunities  
- Change in assigned bytes; envelope utilization fraction  

### 4.2 Deadline features

Predicted slack \(\sigma_{i,h,t}=D_i-\widehat{F}_{i,h,t}\) using existing simulator timing conventions:

- Original / alternative slack and finish times; Δslack  
- Remaining deadline; computation demand; helper rate  

No realized future completion information.

### 4.3 Helper-externality features

Address the helper-9 failure. For every affected helper (existing queue semantics only):

- Ready time before/after; Δ projected workload  
- Pending task count and pending value  
- Min remaining slack among affected pending tasks  
- Count/value of pending tasks whose predicted deadline feasibility deteriorates  

Projected deadline-risk cost:

\[
C_{\mathrm{queue}}(e)
=
\sum_{j\in\mathcal{Q}_{\mathrm{affected}}}
v_j\,
\mathbf{1}\!\left[
\widehat{F}_j(A_t^0)\le D_j
\land
\widehat{F}_j(A_t(e))>D_j
\right].
\]

Risk indicator, not an oracle. Must not invent a new queue discipline. If the simulator cannot expose enough causal queue state, the implementation must **report that limitation**—not substitute future outcomes.

### 4.4 Radio-state features

Only already-available observations (CBR, safety radio metrics, link quality, payload change, connectivity). No separately trained packet-success predictor in this version.

### 4.5 Normalization

Fit on **training data only**; freeze before validation. No validation-seed outcomes, event IDs, or final-test observations as inputs.

---

## 5. Explicit treatment of helper contention

Known facts from harm screen: a move to higher predicted slack can complete the target yet cause another miss; net ΔV can be negative.

Interpretable diagnostic (not PPO reward, not action selector):

\[
H(e)=\Delta V_{\mathrm{nominal}}(e)-C_{\mathrm{queue}}(e).
\]

**Regression-test content:** 82011/helper-9 states. Feature extractor must expose relevant helper workload and deadline relationships if causally observable. The model must **not** be given the historical fact that task 9 failed.

---

## 6. PPO architecture

### 6.1 Frozen reference policy

Load original Direct PPO checkpoint. Freeze weights, config, observation normalization, and candidate-generation interface. Do not retrain, fine-tune, or replace.

### 6.2 Editing actor

Compact shared scorer:

```text
Joint-impact feature vector
        → Linear(64) → Tanh → Linear(64) → Tanh → Linear(1) → edit logit
```

Same network scores all edits; categorical distribution with validity mask (invalid → zero probability). KEEP always available.

### 6.3 Critic

\(V_\phi(s_t,A_t^0)\) estimates expected remaining completed value under the editing policy. No future outcomes or validation labels.

### 6.4 Training objective

Reward unchanged:

\[
r_t=\sum_{i\in\mathcal{I}_t^{\mathrm{new}}}v_i.
\]

Standard PPO+GAE on the **edit-selection** log-probabilities (not frozen Direct). Governor interventions logged via actual transitions/reward. No analytical bonus for substitutions, slack, or LVM agreement.

### 6.5 Initialization

Moderate initial KEEP preference with nonzero exploration over feasible edits. No safety/performance guarantee from initialization alone.

---

## 7. Inference algorithm (semantics)

```python
def jipi_schedule(observation, direct_policy, edit_policy, governor):
    candidates = build_candidates(observation)
    reference_pack = direct_policy.propose(observation, candidates)

    edits = [KeepEdit()]
    edits += enumerate_helper_rebinds(observation, reference_pack, candidates)
    edits += enumerate_task_substitutions(observation, reference_pack, candidates)
    edits = [e for e in edits
             if feasible(apply_edit(reference_pack, e), observation, byte_cap=175000)]

    features = [compute_joint_impact_features(observation, reference_pack, e)
                for e in edits]
    selected_edit, logprob = edit_policy.sample(
        observation, reference_pack, edits, features)
    proposal = apply_edit(reference_pack, selected_edit)
    executed_pack = governor.apply(proposal, observation)
    return executed_pack, {...}
```

Names are proposed interfaces; repository conventions take precedence while preserving these semantics.

---

## 8. Implementation boundaries (when later authorized)

### Proposed new modules (filenames provisional)

| Module | Responsibility |
|---|---|
| `jipi_edit_enumerator.py` | KEEP, feasible rebinds, feasible substitutions |
| `jipi_impact_features.py` | Causal whole-pack / helper-externality features |
| `jipi_actor_critic.py` | Editing actor + critic |
| `jipi_controller.py` | Compose frozen Direct, edit selection, governor |
| `train_jipi.py` | Train editing policy only |
| `evaluate_jipi.py` | Paired matched-condition validation |
| `test_jipi_invariants.py` | Deterministic correctness / regression tests |

### Must remain unchanged

NR-V2X Mode 2; mobility/channel; safety messages; task arrivals/deadlines/values; helper capacities; safety governor; feasibility rules; Direct checkpoint; VAS/LAS/VLRO/LVM; seed/randomness framework; completed-value and safety metric definitions.

No baseline may receive different information, altered constraints, or reduced tuning solely to benefit JIPI-SPARC.

---

## 9. Pre-training correctness requirements (T1–T12)

| ID | Test | PASS requirement |
|---|---|---|
| T1 | KEEP equivalence | KEEP proposal = Direct pack |
| T2 | Feasibility | Every edited pack satisfies existing constraints |
| T3 | Byte conservation | Pack bytes ≤ 175,000 |
| T4 | Rebind consistency | Changes exactly one task’s helper |
| T5 | Substitution consistency | Removes one assignment; inserts one feasible assignment |
| T6 | Queue projection | Matches existing helper execution/queue semantics |
| T7 | Causality | No future/realized outcomes in actor features |
| T8 | Frozen reference | Direct parameters/outputs unchanged |
| T9 | PPO correctness | Stored vs recomputed masked logprobs agree |
| T10 | Governor parity | Identical governor logic vs matched comparators |
| T11 | Zero-edit equivalence | Forced KEEP reproduces Direct trajectory under paired randomness |
| T12 | Runtime | Total controller within 10 ms application decision budget |

---

## 10. Cheap prerequisite evidence (before full PPO training)

| Test | Purpose |
|---|---|
| A — Known beneficial edit | 82002/task19: enumerator offers helper 2→0; features report slack/externalities (correctness, not new generalization) |
| B — Known harmful edits | 82011/helper-9: features expose projected queue consequences if causally observable |
| C — Capacity substitution | 82012 task1 vs task13: represent value-increasing substitution with explicit displacement/bytes; not hard-coded beneficial |
| D — No-edit equivalence | Forced KEEP episodes match Direct (assignments, deliveries, \(V\), safety) within documented tolerances |
| E — Runtime / action-space cost | Edit counts, feature/policy/total ms; deterministic pruning only if needed (KEEP retained; no outcome-dependent pruning) |

These tests do **not** auto-authorize training. Results require separate review.

---

## 11. Training and validation protocol (when later authorized)

**Training:** edit actor/critic only; existing training-seed pool; Direct frozen; existing horizon and realized reward; start from documented PPO hyperparameter reference; pre-limited budgets; save all checkpoints/curves; no budget expansion or repeated HPO without separate decision.

**Validation seeds (locked):** 82001, 82002, 82011, 82012.

**Comparators (matched common-governor):** JIPI-SPARC, prior Direct PPO, VAS, LAS, VLRO, LVM. Primary competitor = strongest matched method, not VAS alone.

**Primary metrics:** \(V_{\mathrm{done}}\), \(R_{\mathrm{value}}\), \(P_{\mathrm{viol}}\). Also: paired ΔV by seed; CIs; edit counts/types; KEEP rate; value gained/lost via edits; deadline outcomes; queue effects; runtime; assignment/byte differences.

---

## 12. Success and continuation criteria

**Engineering target (ambitious, not automatic reject):**

\[
V_{\mathrm{JIPI}}\ge 1.15\,V_{\mathrm{best\ matched}}.
\]

With current matched LVM validation mean ≈ 46.82, the corresponding panel benchmark is ≈ **53.84**. Not a predicted JIPI result.

**Scientific PASS can be <15% if:** positive paired improvement vs strongest matched method; acceptable uncertainty; \(P_{\mathrm{viol}}\le 2\%\) with uncertainty; mechanism explains gains; no unacceptable runtime; robustness beyond one seed.

**Do not promote if:** underperforms Direct without compensating benefit; gains vanish vs strongest matched baseline; repeatedly destroys successful assignments; mechanism unexplained; safety violated; inference over budget; validation memorization / outcome leakage.

---

## 13. Required logging

Per epoch: seed, epoch, reference Direct pack, candidate/eligible edit counts, selected edit type and IDs (task/helpers/displaced), reference vs proposed pack value/bytes, predicted slack change, predicted helper-queue cost, affected pending value, edit logprob, critic estimate, governor flags, executed pack, realized new completed value, deadline-miss count, safety window valid/violation, Direct/feature/policy/total controller ms.

Task-level: assignments, queue entry/service, radio delivery, completion time, deadline, earned value.

Logs must attribute net ΔV to individual edits and downstream effects.

---

## 14. Explicitly excluded changes

- Reopening/retraining Stage-1 conditional scorer  
- Modifying frozen Direct checkpoint  
- VAS-based residual anchor  
- Task-admission STOP action  
- Separately learned delivery predictor  
- Analytical pack-score reward  
- Governor or 175 kB envelope changes  
- Weakening/replacing baselines  
- Opening final-test seeds  
- Manuscript revision before validation  

A different mechanism requires a separate design decision.

---

## 15. Authorization boundary and next step

| Action | Status |
|---|---|
| Algorithm specification (this document) | **Authorized / done** |
| Implementation of JIPI modules | **Not authorized** |
| Pre-training correctness tests T1–T12 / Tests A–E | **Not authorized** (require separate approval) |
| Training edit PPO | **Not authorized** |
| Validation / final-test | Validation locked to four seeds when authorized; **final-test sealed** |
| Manuscript revision | **Not authorized** |

**Immediate recommendation:** Implement JIPI-SPARC **only after** correctness and causal-feature prerequisites are separately approved.

Essential innovation: **learn the opportunity cost of modifying an existing feasible Direct pack**, accounting for displaced task value, deadlines, and projected helper contention—a concrete, testable hypothesis, not an assumed improvement.

---

## Related reports

- `DIRECT_LVM_SUBSTITUTION_AUDIT.md` — coverage vs LVM; 82012 substitution analytic  
- `DIRECT_HELPER_SWAP_REPLAY_82002.md` — local beneficial rebind PASS  
- `DIRECT_HELPER_SELECTION_GENERALIZATION_AUDIT.md` — universal trigger not established  
- `DIRECT_HELPER_REBIND_HARM_SCREEN.md` — universal rebind FAIL (3/15 damage)  
- `STAGE1_CLOSURE` / Stage-1 reports — wholesale scorer path closed  
