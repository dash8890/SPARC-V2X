# SPARC MDP Correction — Execution Runbook (how, not slogans)

**Amnesia rule:** A new chat does **not** need prior conversation. It only needs:
1. `SPARC_MDP_HANDOFF.json` ← open first  
2. `SPARC_MDP_PROGRAM_STATUS.md`  
3. This runbook § for `current_phase`

If HANDOFF says `authorized_now: false`, the only allowed reply is to wait / ask for the next authorize phrase. Do not invent work.

---

## Phase 0 — Utility lock (docs only)

### Inputs
- Existing reports: `POPL_P5_VAL_REPORT.md`, `RAO_PHASE1_G1_G7_REPORT.md`, `RAO_PHASE1_DUAL_POLICY_SELECTOR_CLOSED.flag`
- Code facts (cite, do not change):
  - Reward: `experiments/wilab_closed_loop/train_wilab_closed_loop.py` (`reward = advance_reward + Δcompleted_value`)
  - `completed_value`: `experiments/wilab_closed_loop/task_layer.py`
  - Direct PPO γ: `controllers/rl/residual_ppo.py` (`gamma=0.99`)

### Exact work (when user says `AUTHORIZE Phase 0`)
Create **one** file:

`sparc_performance_evaluation/reports/SPARC_MDP_PHASE0_UTILITY_LOCK.md`

**Required sections (fill concretely, not vaguely):**

```markdown
# SPARC MDP Phase-0 Utility Lock

## 1. Utility thesis
- Primary scalar: ...
- Safety: constraint-only via SafetyGovernor + 175kB | OR in-objective (choose one)
- Not used as training reward: analytic slack, RHO E[V], selector ΔG retuning

## 2. Decision problem
- Decision epoch: controller epoch with candidates
- Action: complete assignment pack A
- Executed action: A_exec = governor(A, obs)
- Policy likelihood: log π(A_proposed | obs) full-pack joint
- Critic may condition on A_exec features (Phase 2c+ only)

## 3. Success criterion (later phases)
- Metric: matched episode V_done (γ=1 accounting)
- Comparators: frozen Direct + LVM (VAS/LAS/VLRO retained)
- Phase-1 gate: headroom/commitment diagnostic PASS (not +15%)
- Aspirational: ~53.84 vs LVM~46.82 is NOT a Phase-1/2a gate

## 4. Closed paths
- RAO Direct↔POPL selector: CLOSED (cite flag)

## 5. Explicit non-goals this lock
- No MATLAB, no PPO, no new collection in Phase 0
```

### Exit procedure
1. Write the file above.  
2. Patch `SPARC_MDP_PROGRAM_STATUS.md`: `phase0=DONE`, `phase1=PENDING_USER_AUTH`.  
3. Patch `SPARC_MDP_HANDOFF.json`:
   - `current_phase`: 0  
   - `authorized_now`: false  
   - `next_user_phrase_to_start`: `"AUTHORIZE Phase 1"`  
   - `next_agent_action_when_authorized` → Phase-1 block from §Phase1 below  
4. **STOP.**

### Time / tools
- ~15–30 minutes, editor only. Zero MATLAB.

---

## Phase 1 — Headroom & commitment diagnostic (training-free)

### Prerequisite
- `phase0=DONE` in STATUS  
- User says `AUTHORIZE Phase 1`  
- HANDOFF `authorized_now` set true for phase 1 only

### What “this” means in code

**Create script** (does not exist until Phase 1 starts):

`sparc_performance_evaluation/scripts/run_mdp_phase1_headroom.py`

**Optional helper module:**

`controllers/rl/mdp_diag/commitment_features.py`

### Primary data (already on disk — no new collection by default)

| Input | Path | Use |
|-------|------|-----|
| Matched Direct/POPL pairs | `results/raw/rao_phase1/pairs.json` | Pack A vs B, `G_Direct`, `G_POPL`, `delta_G`, fork features |
| Episode audits | `results/raw/rao_phase1/runs/episode_*/bridge_audit.json` | Per-epoch rewards → remaining return `G_t` |
| Seed rules | `protocol/rao_phase1_seeds.json` | Fit 0–31 / hold 32–47; forbid 82001/02/11/12 |

**Do not use** locked val `82001/02/11/12` for fitting thresholds. **Do not** touch final-test `9300x`.

### Commitment features (v1 — must be pre-radio / no outcome leakage)

Built from `(obs-at-fork, pack)` only. Forbidden: `delta_G`, future rewards, `G_*`, delivered flags.

Minimum vector (implement exactly these names):

| Feature | Definition |
|---------|------------|
| `exec_bytes` | sum payload_bytes in pack / 175000 |
| `envelope_residual` | 1 − exec_bytes |
| `n_assign` | \|pack\| / 10 |
| `n_helpers_used` | unique helper_id / 12 |
| `max_helper_bytes` | max bytes on one helper / 175000 |
| `helper_bytes_hhi` | Herfindahl of bytes across helpers |
| `n_sources` | unique source_id / 20 |
| `mean_payload` | mean payload_bytes / 1e5 |
| `pack_jaccard_vs_alt` | if paired: \|A∩B\|/\|A∪B\| else 0 |
| `byte_diff_vs_alt` | \|bytes_A − bytes_B\| / 175000 |
| `helper_overlap_vs_alt` | \|H_A ∩ H_B\| / max(1, \|H_A ∪ H_B\|) |

Plus existing RAO regime keys already in `pairs.json` → `features` as **baseline (candidate/regime) set**.

### Analyses the script must run

**A. Structural swing (headroom between two frozen packs)**  
Already partially known; recompute and freeze numbers:

- On disagreeing pairs: frac `\|ΔG\|>0.1`, mean `oracle−Direct`, mean `oracle−POPL`
- **PASS threshold for “material swing”:**  
  `frac(|ΔG|>0.1) ≥ 0.25` **OR** `mean(oracle − max(G_D,G_P wait no))` — use:  
  `mean(oracle − G_Direct) ≥ 1.0` **OR** `mean(|ΔG| | ΔG≠0) ≥ 2.0`  
  (If both fail → headroom FAIL.)

**B. Commitment predictive lift (offline, no new policy)**  
For each disagreeing pair with packs A=Direct, B=POPL:

- Label: `y = 1{G_POPL > G_Direct}` (strict); also regression target `ΔG`
- Fit on pair_idx 0–31; evaluate 32–47  
- Models: L2 logistic (classification) + ridge (ΔG)  
- Feature sets:  
  - `F_base` = keys in `pairs[i]["features"]` (existing regime/pack-diff)  
  - `F_commit` = commitment features above for both packs (concat or diff)  
  - `F_both` = concat  
- **PASS lift:** held-out balanced accuracy or Spearman(|pred|,|ΔG|) for `F_both` **strictly >** `F_base` by ≥0.02 bacc **or** ≥0.05 Spearman  
- Leakage unit test: reject any feature name matching `delta_g|g_direct|g_popl|reward|delivered`

**C. Epoch-level remaining return (audit)**  
For a sample of Direct-branch episodes in `rao_phase1/runs`:

- `G_t = sum(reward[e] for e ≥ t)`  
- Correlate commitment features of `selected_assignments` at epochs with `n_candidates>0` against `G_t`  
- Report only; not a hard gate if pair-level B is decisive

### Exact command

```text
python -u sparc_performance_evaluation/scripts/run_mdp_phase1_headroom.py --i-authorize-mdp-phase1
```

Writes:

- `results/raw/mdp_phase1/summary.json`  
- `results/raw/mdp_phase1/lift_table.json`  
- `reports/SPARC_MDP_PHASE1_HEADROOM_REPORT.md`

### Phase-1 overall gate

| Result | Meaning |
|--------|---------|
| **PASS** | Swing PASS **and** lift PASS |
| **FAIL** | Either fails → write `SPARC_MDP_CA_PPO_PATH_CLOSED.flag`; STATUS phase2/3 = CLOSED; **do not** authorize Phase 2 |

### Exit procedure
Update STATUS + HANDOFF: `phase1=PASS|FAIL`, `authorized_now=false`, next phrase `AUTHORIZE Phase 2a` only if PASS. **STOP.**

### Time / tools
- ~1–3 hours coding + run (CPU only if no new MATLAB).  
- New MATLAB **not** in default Phase 1. If audits insufficient, STOP and ask user before any collection.

---

## Phase 2a — γ=1 + drain/terminal (only if Phase 1 PASS + auth)

### Exact code touch points
1. `controllers/rl/residual_ppo.py` — `build_ppo_steps(..., gamma=1.0)` via CLI/flag default change **only when authorized**; keep ability to run old γ for A/B.  
2. Training entry: `experiments/wilab_closed_loop/train_wilab_residual_ppo.py` (or successor used for SPARC Direct) — add `--gamma 1.0` and document drain: episode return includes post-decision completions already in `completed_value()` after last epoch (verify in `run_episode`; if gap, fix accounting in one place and unit-test).  
3. **Do not** change actor architecture.

### Eval command pattern (budget to be set at auth time)
Reuse existing paired eval harness, e.g. pattern from POPL val scripts / `run_paired_eval.py`, seeds **train-dev only** unless user unlocks locked val.

### Pass
Mean `V_done` on authorized eval seeds ≥ baseline checkpoint − 0.5 (non-regression) **and** training code path logs `gamma=1.0`.

### Deliverable
`reports/SPARC_MDP_PHASE2A_REPORT.md` + STATUS step → `2b` pending auth. **STOP.**

---

## Phase 2b — Full-pack joint log-prob hygiene

### Exact work
1. Ensure residual/Direct or pack actor stores **joint** `log π(A_proposed)` for the full pack (POPL already sequential; residual path must match).  
2. PPO ratio uses **proposed** pack log-prob; environment applies governor → `A_exec`.  
3. Unit test: governor rewrite does not change stored `logp_old` identity for the proposed pack.

### Deliverable
`SPARC_MDP_PHASE2B_REPORT.md`. **STOP** for `AUTHORIZE Phase 2c`.

---

## Phase 2c — Auxiliary post-decision critic W (no actor use)

### Exact work
1. Module: `controllers/rl/mdp_diag/afterstate_critic.py` (or `controllers/rl/cao/`)  
2. Input: commitment features of `A_exec` + obs encoding (no future leakage)  
3. Target: realized `G_t` from rollouts (γ=1)  
4. Loss: MSE only; **actor frozen or updated with old GAE only**  
5. Fit/hold split on authorized train-dev seeds  
6. Pass: held-out Spearman(W, G_t) ≥ 0.2 **or** pack-ranking accuracy vs Direct/POPL pairs ≥ best constant baseline + 0.05

### Deliverable
`SPARC_MDP_PHASE2C_REPORT.md`. If FAIL → close 2d path. **STOP.**

---

## Phase 2d — W in advantage (only if 2c PASS)

### Exact work
`Â = W − V` or blend with GAE; single bounded run; compare to Phase 2b checkpoint on same seeds.

### Pass
Mean `V_done` > Phase 2b on authorized hold-out train-dev.

### Deliverable
`SPARC_MDP_PHASE2D_REPORT.md`. **STOP** for Phase 3 auth.

---

## Phase 3 — Bounded CA-PPO training

Budget, seed list, and wall-clock cap **must appear in the user authorize message**.  
Without an explicit budget number, refuse.

Deliverable: `SPARC_MDP_PHASE3_REPORT.md` vs frozen Direct/LVM.

---

## Session start template (copy into agent)

```
1. Read SPARC_MDP_HANDOFF.json
2. If authorized_now is false: tell user current_phase and waiting phrase; STOP
3. Else execute next_agent_action_when_authorized.do exactly
4. Write deliverable
5. Update HANDOFF + STATUS; set authorized_now=false
6. STOP
```

That is how Phase 1 is remembered after Phase 0: it is the **next_agent_action** blob in HANDOFF.json, not chat history.
