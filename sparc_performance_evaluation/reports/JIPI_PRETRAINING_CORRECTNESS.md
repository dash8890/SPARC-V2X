# JIPI-SPARC Pre-Training Correctness Report

**Gate:** `PASS_WITH_SKIPPED_CLOSED_LOOP`
**PASS/FAIL/SKIPPED:** 15/0/2
**Training authorized:** No (correctness review only).

| ID | Status | Detail |
|---|---|---|
| T1 | **PASS** | KEEP proposal matches Direct pack (n=4) |
| T2 | **PASS** | all 4 enumerated edits feasible |
| T3 | **PASS** | no edit exceeds 175000 bytes |
| T4 | **PASS** | rebind changed tasks=[19] |
| T5 | **PASS** | sub SubstituteEdit(remove_task_id=13, remove_helper_id=0, add_task_id=1, add_helper_id=2, edit_type='SUBSTITUTE') |
| T6 | **PASS** | \|proj_finish-cand_finish\|=0.000ms (tol 5ms) |
| T7 | **PASS** | feature keys ok; banned=[] |
| T8 | **PASS** | Direct parameters unchanged and frozen |
| T9 | **PASS** | logp=-0.375762 recomputed=-0.375762 |
| T10 | **PASS** | governor reason=none, intervened=False |
| T11 | **SKIPPED** | Forced-KEEP full paired Direct trajectory requires MATLAB closed-loop; not run in default gate. |
| T12 | **PASS** | worst_controller_ms=4.70, n_edits=3, rebinds=2, subs=0 (budget 10ms) |
| A | **PASS** | rebind_available=True, delta_slack=6.417, C_queue=0.000, H=0.000 |
| B | **PASS** | helper9 exposures=[(15, True, 2.646857184400823, 0.0795441605375229), (6, True, 2.646857184400823, 0.08310669361439252), (3, True, 2.646857184400823, 0.14018... |
| C | **PASS** | feasible=True, added=3.738, displaced=1.482, nominal_delta=2.256, C_queue=0.000 |
| D | **SKIPPED** | No-edit equivalence episode match requires MATLAB; not run in default gate. |
| E | **PASS** | worst_offline_decision_ms=2.79; rows=[{'seed': 82001, 'epoch': 0, 'n_edits': 4, 'n_rebind': 2, 'n_sub': 1, 'enum_ms': 0.929300207644701, 'feat_ms': 0.5952999... |

T11/D closed-loop KEEP equivalence deferred (MATLAB). Offline Tests A–C exercise locked Direct validation decision states.

Next: human review of this gate before any training authorization.
