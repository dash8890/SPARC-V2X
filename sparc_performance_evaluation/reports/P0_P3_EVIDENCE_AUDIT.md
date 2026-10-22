# P0–P3 Evidence Audit (read-only artifacts)

**Date:** 2026-10-15  
**Scope:** Existing `bridge_audit.json`, `rows.json`, aggregation scripts, and controller contracts. **No new training or simulations.**  
**Verdict summary:** P0 = reporting bug (fixed by backfill). P1–P3 = scientifically consequential interpretation issues; locked \(V\) outcomes unchanged.

---

## P0 — Offered-byte accounting (zero in Table III)

### Exact question
Why does Table III report zero offered bytes when tasks are being completed?

### Cause (reporting bug — not a phantom-completion bug)

1. `run_paired_eval.py` computed
   `mean_offered_bytes` from the **episode return-trace** field
   `computation_bytes_offered`.
2. `run_episode(..., return_trace=True)` stores that quantity under the key
   **`bytes`**, not `computation_bytes_offered`  
   (`train_wilab_closed_loop.py` trace append).
3. Therefore every row got `mean_offered_bytes = 0.0` even though
   `runs/episode_*/bridge_audit.json` already logged correct nonzero
   `computation_bytes_offered` / `wilab_enqueued_bytes`.

### Evidence that radio offers exist

On `final_test_nominal_d35` (15 audits):

| Method | mean per-epoch offered bytes | mean episode **sum** offered bytes |
|---|---:|---:|
| VA | ~7076 | ~707565 |
| MF | ~6955 | ~695526 |
| SPARC | ~6285 | ~628499 |

Sample epoch with selections: `computation_bytes_offered == wilab_enqueued_bytes` for the computation payload (e.g. 137595 B / 139 packets). Row `completed_value` matches audit reward sums seed-by-seed (map check passed).

Delayed credit is expected: reward often lands on later epochs with `num_selected=0` after prior deliveries/queue finish (`TaskLayer.apply_radio_outcomes` + `set_epoch` advance). Prior gate G4 already noted `rx_delta>0` with zero same-epoch offer as delayed delivery, not phantom value.

### Scientific validity
**No evidence of crediting completions without corresponding offers in the audit trail.**  
\(V_{done}\) / \(R\) / safety columns were never derived from the broken byte field.

### Actions taken
- Fixed `run_paired_eval.py` to prefer `bridge_audit.json` (fallback: trace `bytes`).
- Backfilled all existing `rows.json` via
  `scripts/backfill_offered_bytes_from_audits.py` (no re-sims).
- Regenerated Table III / evidence package report.

**Affected:** tables/reports only. **Not affected:** raw episode outcomes.

---

## P1 — Governor operation (PRR/P95 vs byte envelope)

### Exact question
Does the PRR/P95 governor actually intervene, or is the byte envelope the only effective restriction?

### Findings at nominal density=35 (locked final-test)

| Dataset | epochs | `governor_intervened` | PRR&lt;0.95 | P95&gt;50 | reason |
|---|---:|---:|---:|---:|---|
| nominal (VA/MF/SPARC) | 1500 | **0** | **0** | **0** | all `none` |
| Direct | 500 | 0 | 0 | 0 | all `none` |
| A1 (gates open) | 500 | 0 | 0 | 0 | all `none` |
| B4 Lagrangian | 500 | 0 | 0 | 0 | all `disabled` |

Radio stays comfortably feasible at d=35: PRR mean ≈0.984 (min≈0.952), P95 mean≈22 ms (max≈39).

**Byte envelope is the binding constraint under the common governor:**

- Positive-action epoch payload p50/p90/p100 (nominal): ~110k / ~165k / **174k** (cap 175k).
- `proposed_assignments` bytes **always equal** executed bytes (`prop_ne_exec = 0`) → governor never had to rewrite over-budget proposals.
- Policies are also masked by `forced_byte_budget` / sampling byte budget **before** `governor.apply`, so the envelope often binds inside the actor, leaving `governor_intervened=False`.

### Important metric caveat
`governor_interventions` counts only `governor_intervened=True`.  
When `admission_allowed(state)` is already False, the policy proposes empty and
`apply` returns `([], False, "prr_below_floor")` — **gate active, intervention flag false**.  
Backfill now also records `governor_gate_block_epochs` from `governor_reason ∉ {none, disabled}`.

### Density contrast (gates become consequential)
At density **80**, SPARC/VA/MF logs ~491/500 epochs with reason `prr_below_floor` and nearly zero completions; ITORA has governor **disabled** and keeps offering (~1.5 MB sum/episode) with \(P_{viol}≈0.998\).

### Scientific validity / manuscript implication
At **nominal final-test**, zero \(P_{viol}\) does **not** demonstrate that PRR/P95 gates caused safety preservation — those gates never fired.  
The **175 kB envelope** (plus in-policy masking) is the effective admission restriction.  
PRR/P95 contribution is evidenced in denser regimes (E2), not in Table III nominal.

**Recommendation:** Retain locked nominal results; strengthen manuscript language to attribute nominal safety to the envelope + naturally feasible radio, and point to density sweeps for PRR-gate consequence.

---

## P2 — Lagrangian PPO +4.71 vs Direct (500 kB confound)

### Exact question
Is B4’s completed-value advantage over Direct primarily due to the larger 500 kB admission allowance?

### Evidence (same final-test seeds, from audits)

| Method | mean \(V\) | \(P_{viol}\) | mean sum offered bytes | mean positive-epoch bytes | frac epochs &gt;175 kB | max epoch bytes | governor |
|---|---:|---:|---:|---:|---:|---:|---|
| B3 Direct | **37.97** | 0 | ~630k | ~117k | **0%** | 173k | 175 kB |
| B4 Lagrangian | **42.68** | 0 | ~791k | ~158k | **40%** | **332k** | disabled; 500 kB physical cap |

- B4 admits **~25% more** computation bytes per episode and frequently exceeds the SPARC/Direct envelope (40% of positive epochs).
- If B4 positive-epoch bytes are capped at 175k in a counterfactual mean, they drop to ~121k — nearly Direct’s ~117k — indicating the extra headroom is material.
- Protocol already classifies B4 as an **unshielded / different admission regime** (`evaluation_protocol.md`, `controller_contracts.md`).

### Scientific validity / manuscript implication
**Yes — the +4.71 \(V\) gap is not an apples-to-apples governor-matched win.**  
Do **not** claim Direct is the strongest *overall* method while presenting B4’s 42.68 without the admission caveat.  
Correct claim: **Direct is the strongest governor-matched learned baseline (B1–B3 / SPARC shield).** B4 is a different system-level safety mechanism and must be reported with its 500 kB / no-PRR-P95 contract (Table IV).

**No rerun required for this conclusion** from existing audits. A matched-envelope B4 (175 kB + governor) would be a *new* experiment if desired later.

---

## P3 — Density-sweep ITORA at ρ=80 / 100

### Exact question
Why does ITORA substantially outperform other controllers at densities 80 and 100?

### Evidence

**Workloads are comparable:** all methods share the same sealed seeds and the same `offered_value` per density (e.g. d80 offered \(V\)=146.5 for every method). This is **not** an incomparable task-generation bug.

**Admission mechanisms are not comparable:**

| Density | Method | mean \(V\) | sum offered bytes | PRR-fail epochs (/100) | \(P_{viol}\) | governor |
|---|---|---:|---:|---:|---:|---|
| 80 | SPARC / VA / MF | 4.3 / 3.0 / 3.1 | ~97k / 97k / 68k | ~98 | ~0.98 | 175 kB + PRR/P95 |
| 80 | **ITORA** | **55.8** | **~1.50M** | ~100 (radio) | **~0.998** | **disabled** (`method==itora_inspired` ⇒ `gov=None`) |
| 100 | SPARC / VA / MF | 53.6 / 45.5 / 44.7 | ~0.91–1.14M | ~39 | ~0.39 | enabled |
| 100 | **ITORA** | **64.3** | **~1.76M** | ~40 | **~0.40** | disabled |

Code: `run_paired_eval.py` forces `gov = None` whenever `method == "itora_inspired"` (also used inside density jobs that otherwise enable the SPARC governor for VA/MF/residual).

### Interpretation
At ρ=80, SPARC/VA/MF are **almost fully gated by PRR&lt;0.95**, so they barely admit; ITORA ignores that gate and keeps pushing ~15× more computation bytes, scoring higher \(V\) while **failing the safety criterion**.  
At ρ=100 the PRR gate is less saturated; SPARC recovers, but ITORA still admits more bytes.

### Scientific validity / manuscript implication
Density “ITORA wins on \(V\)” is **internally consistent** with different admission rules, **not** a workload bug.  
It is **not** evidence that ITORA is a stronger *safety-feasible* controller.  
Figures/tables must either (a) mark ITORA as unshielded / non-feasible at high density, or (b) report value **conditional on** \(P_{viol}=0\) (ITORA would drop out at ρ=80).

**Affected experiments for claim language:** E2 density panels including ITORA.  
**Not required:** re-sim of SPARC/VA/MF/Direct locked nominal.

---

## Overall decision tree

| Issue | Type | Affects locked \(V\)? | Action |
|---|---|---|---|
| P0 zero offered bytes | Aggregation bug | No | **Fixed** (backfill + script patch); regenerate tables |
| P1 nominal governor | Interpretation | No | Keep results; clarify envelope binds, PRR/P95 idle at d=35 |
| P2 Lagrangian vs Direct | Comparison fairness | No (values OK) | Keep both; **do not** rank B4 above Direct without admission caveat |
| P3 ITORA density | Comparison fairness | No (values OK) | Keep; label unshielded / report with \(P_{viol}\) |

### Manuscript path
**Retain locked evidence and proceed with a Direct-PPO–centric governor-matched claim**, with explicit B4/B5/ITORA admission disclaimers.  
No simulation inconsistency found that forces nominal/Direct reruns.

### Regenerated artifacts
- `scripts/backfill_offered_bytes_from_audits.py` (+ `results/aggregates/offered_bytes_backfill.json`)
- Patched `scripts/run_paired_eval.py`
- Updated `tables/table_iii_comparison/table_iii_final_test.*`
- Refresh `reports/EVIDENCE_PACKAGE_REPORT.md`
