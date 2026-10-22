# SPARC-V2X — Revised Performance Evaluation Protocol and Evidence Package

**Status:** aligned with the definitive deadline-feasible, byte-governed, VA-residual PPO framework. This is an experimental specification, not a claim that experiments have been executed. All unspecified simulator settings must be copied from the actual frozen implementation/configuration, not inferred.

**Scope:** Performance Evaluation only. System-model and mechanism figures are excluded. Retain nine evaluation figures (Figs. 3–11, 23 panels) and three tables (Tables II–IV), but replace obsolete contents. **No cloned-state counterfactual training, residual regression loss, MF deployment gate, cardinality shield, shield-coefficient risk prediction, or variable-delay execution model is part of SPARC-V2X.**

## 1. Research question, estimands, and decisive criterion

**Objective:** maximize total value of tasks delivered and completed by both absolute deadline and control horizon:

\[
V_{\mathrm{done}}=\sum_{i\in\mathcal I^\star}v_i,\qquad R_{\mathrm{value}}=V_{\mathrm{done}}/V_{\mathrm{off}},\quad V_{\mathrm{off}}=\sum_{i\in\mathcal I}v_i.
\]

**Safety:** with valid measurement epochs \(\mathcal T_{\mathrm{valid}}\), report \(P_{\mathrm{viol}}=|\mathcal T_{\mathrm{valid}}|^{-1}\sum_{t\in\mathcal T_{\mathrm{valid}}}\mathbf1[\mathrm{PRR}^{\mathrm{safe}}_t<\Gamma\lor D^{\mathrm{safe}}_{95,t}>D^{\max}]\), separately reporting the fraction of epochs with invalid measurements. A run with no valid safety windows cannot be classified as compliant. Report PRR-floor and P95-ceiling failures separately in addition to their union.

**Primary test:** in a *predeclared nominal operating scenario* on untouched final-test seeds, compare SPARC against the highest-performing safety-feasible **principal competitor**; success requires a positive paired \(\Delta R_{\mathrm{value}}\) with the stated multiplicity-controlled inference, and statistically supported satisfaction of the safety criterion. Report all five individual contrasts even if the primary gate fails. Do not redefine the main metric or competitor after seeing results. Identifying the strongest feasible competitor on final-test data requires either a prespecified simultaneous-inference procedure across all five competitors or an independently selected competitor on validation data. Prefer simultaneous paired intervals for all five contrasts, combined with a separate safety-feasibility procedure, to avoid winner selection bias.

**Research questions:** RQ1, comparative completed value; RQ2, safety-service effect of byte-governed admission; RQ3, benefit of VA residual initialization and VA-relative PPO; RQ4, behavior under workload/deadline/density shifts; RQ5, controller cost and reproducibility; RQ6, sensitivity of admission and learning to operating regimes.

**Non-claims:** The byte envelope is a deterministic admission restriction, not a proof of an episode-level safety probability bound. PPO convergence or positive superiority cannot be presumed. No separate safety theorem is required in the evaluation section.

## 2. Common environment, fairness contract, and execution semantics

All controllers use the **same** native WiLabV2Xsim NR-V2X Mode-2 radio, same Python task/helper layer, same MATLAB bridge, safety beacons, reservations and interference, mobility, and common task-generation configuration. Methods choose application-level offers, **never physical subchannels**. Within each paired master seed, match exogenous initial scenario and task/mobility inputs; after different decisions, radio reservations, pending packets, helper queues, and future endogenous states may diverge naturally. Do not force outcome equality.

At epoch \(t\), application time is \(t\Delta\). For task \(i\) released at epoch \(r_i\), absolute deadline \(D_i=r_i\Delta+d_i\). For helper \(h\) with rate \(\mu_h\) and ready time \(\tau_h\), use the identical finish estimator for **all** policies:

\[
\widehat F_{i,h,t}=\max\{(t+1)\Delta,\tau_h\}+c_i/\mu_h.
\]

The shared candidate set \(\mathcal C_t\) contains only released, unfinished, not-yet-queued pairs with \(\widehat F_{i,h,t}\le D_i\), and any common contact/helper eligibility requirements actually implemented. At most one pair per task and **at most one offered task per source per epoch**. An admitted pair earns no value unless packet delivery occurs during the radio interval and helper completion occurs by both its absolute deadline and the horizon. On successful delivery, commit the task with the same finish/ready-time rule across controllers; failed packets do not queue or earn value. No phantom completion for merely selected tasks. Do not substitute an old transmission-plus-queue-plus-CPU estimator.

The byte-governed policies share the same external function \(g\): validity of radio measurements, measured safety PRR/P95 thresholds and the hard offered-byte budget \(\sum b_i\le B_{\max}\). Invalid/malformed/over-envelope proposals are handled by the **same** deterministic deadline-feasible VA fallback; blocked safety gates admit nothing. A missing measurement and an observed safety violation must be separately logged. Keep the timing and interpretation of the envelope exactly consistent with the implemented `horizon_envelope_only` path; audit whether the remaining budget is per epoch or a horizon-persistent state before freezing experiments, and make prose/equations match the actual code. The supplied theoretical framework defines a per-epoch inequality; do **not** silently reinterpret it as a global episode budget.

**Envelope audit (frozen implementation):** under definitive SPARC training/eval, `forced_byte_budget=B_max` is applied inside each epoch's `sample_action` with a fresh remaining-byte counter, and `SafetyGovernor.max_bytes` likewise validates each proposal independently. With `horizon_envelope_only=True`, next-window risk certification is not required; PRR/P95/validity gates and the per-epoch byte bound remain. **Conclusion: \(B_{\max}\) is a per-epoch offered-byte budget, not a cumulative episode budget.**

Every deployable controller receives identical causal information and the same candidate table; no controller sees future arrivals, unseen radio outcomes, or reserved simulator hidden states. The policy may use its own encoder of these features but not a larger information set. Record candidate count, offered bytes, admitted bytes, actual deliveries, task status, governor action, and realized safety observables by epoch.

## 3. Five principal baselines — implementation contract

**B1 — Corrected VA + common governor (analytical anchor).** For each deadline-feasible candidate use \(s^{\mathrm{VA}}_{i,h,t}=v_i/\max\{\widehat F_{i,h,t}-t\Delta,\varepsilon\}\). Rank descending, break ties deterministically, greedily select respecting unique task/source and remaining bytes. Pass proposal through the identical governor. No learning and no MF gate. **Question:** Does deployed SPARC improve its own correct analytical anchor?

**B2 — Minimum-Finish + common governor.** Rank by ascending \(\widehat F_{i,h,t}\), same candidate eligibility, tie rule declaration, unique task/source, byte budget, and governor. Reuse exactly the same CPU-ready-time accounting. **Question:** Does learning outperform shortest predicted completion time rather than merely exploiting a weak value/CPU comparator?

**B3 — Direct PPO + common governor.** The actor produces \(\ell_\psi(i,h,t)=\alpha q_\psi(x_{i,h,t})\), excluding the additive VA score. Use the *same candidate encoder architecture*, feature information, categorical sequence/masks, separate value critic, GAE, clipped PPO, entropy, KL stopping, rollout count, optimizer schedule, and admission governor as SPARC. Match actor capacity and training-budget opportunities as closely as possible; disclose any differences. Do **not** use SPARC's VA-relative advantage shift. If the direct actor has no VA zero-initialization property, use its ordinary stated initialization. **Question:** Are VA anchoring and VA-relative learning better than a comparably trained PPO scheduler? Component ablations disentangle the two.

**B4 — Unshielded Lagrangian PPO (learned safety management).** Same task/helper candidate feasibility and causal radio observations, but no SPARC PRR/P95 application governor and no deterministic VA fallback. The policy may choose an empty action via an explicit STOP decision and is permitted to vary admitted task count, subject to a disclosed *physical interface capacity* common to any unshielded competitor (not SPARC's safety envelope). Optimize realized normalized completed value with a cost critic for the union safety-violation indicator and nonnegative Lagrange multiplier \(\lambda\), using \(\widehat A^r-\lambda\widehat A^c\). Define invalid-window handling and multiplier updates before training. Report its actual offered bytes and violation rate, and never interpret the Lagrangian formulation as guaranteed compliance. **Question:** How do learned safety penalties compare with application-layer hard admission?

**B5 — ITORA-inspired Mode 2 adaptation (literature).** Reproduce the *actual published task-selection/value-learning mechanism*, retaining as much original decision structure as the common task/helper state permits; translate unsupported infrastructure-based radio allocation to autonomous Mode 2. Enforce the common task and source uniqueness and deadline accounting. Its application admission may use a validation-selected fixed **byte** cap or a faithfully adapted original admission mechanism; the choice must be frozen before final test. Do not use the SPARC governor or pretend a newly invented heuristic is an exact ITORA reproduction. Produce a method-mapping audit (original equation/function → adapted equation/function → rationale) from the paper before programming B5. If the paper cannot be reproduced faithfully, identify B5 as **ITORA-inspired**, not an exact baseline. **Question:** Does SPARC outperform a legitimate literature-informed value-aware offloader in the shared-radio setting?

**Important fairness distinction:** B1–B3 isolate scheduling under the common governor; B4–B5 test system-level alternative safety/admission mechanisms. A method that violates the safety criterion may score higher in value but cannot be called a stronger *feasible* competitor.

## 4. Seven mechanism ablations

All ablations preserve the common task/radio model and the correct completion objective. Reuse B1/B2 results where exactly identical; do not count duplicate runs as new independent data.

| ID | Policy | Controlled difference vs full SPARC | Interpretation |
|---|---|---|---|
| A0 | Full SPARC | Corrected VA + residual actor + VA-relative PPO + byte governor | Proposed method |
| A1 | SPARC without safety gates | Same frozen actor but remove PRR/P95 and validity admission blocking; retain task/source/deadline constraints and a declared physical packet limit | Safety-gate effect; may be infeasible |
| A2 | SPARC without residual | Identical VA + governor (B1) | Total benefit over analytic anchor |
| A3 | SPARC without VA-relative adjustment | Same VA-residual actor, PPO and governor; ordinary GAE only | Incremental training-signal effect |
| A4 | SPARC without VA anchor | Direct PPO + governor (B3), matched capacity | Combined anchor/initialization effect |
| A5 | Residual-only scores with VA-relative PPO | Remove additive VA from logits but retain the same VA-relative PPO advantages, matched critic/architecture and governor | Isolates the analytical score contribution while preserving the learning signal |
| A6 | SPARC without residual zero-init | Same network and VA-relative PPO, but standard random residual initialization | Initialization effect |

These are **seven distinct configurations A0–A6**; A2 is identical to B1 and A4 is identical to B3 and their results must be reused, not duplicated. A5 is trained separately: the VA reference remains in the advantage calculation but is removed only from the actor logits. Since its initialized logits need not reproduce VA, its behavioral deviation is measured rather than presumed. Figure 9 retains two panels and seven configuration bars.

For A1, specify whether removing PRR/P95 blocking while retaining a byte limit creates only a *partial* no-governor ablation; label it "without safety gates" rather than "without governor". Compare frozen-policy behavior but acknowledge changes to safety-state trajectories. For each ablation report paired \(\Delta R_{\mathrm{value}}\), \(P_{\mathrm{viol}}\), offered bytes, and fraction of altered executed proposals. Train separately where training logic or initialization changes; only safety-gate removal is the designated frozen-checkpoint intervention.

## 5. Frozen experiment families and figure inventory

Keep nine numbered evaluation figures and 23 panels (2+2+1+2+4+4+2+4+2). Figures may be combined/reduced at manuscript-layout time; the underlying numerical evidence is still retained.

| Figure | Panels | Required evidence | Experiment |
|---|---:|---|---|
| **Fig. 3** Training convergence and VA-relative learning | 2 | (a) validation paired \(\Delta V_{\mathrm{done}}\) SPARC–VA against PPO update (plus Direct PPO); (b) first-pick VA-deviation fraction and trajectory-relative advantage diagnostics versus update | E1 |
| **Fig. 4** Completed value and safety versus density | 2 | (a) \(R_{\mathrm{value}}\); (b) \(P_{\mathrm{viol}}\), threshold line | E2 |
| **Fig. 5** Safety–value tradeoff | 1 | Paired \((P_{\mathrm{viol}},R_{\mathrm{value}})\) operating points for six policies and one-dimensional **byte-envelope** sensitivity (with safety gates held fixed); do not infer a Pareto frontier absent nondominance | E3 |
| **Fig. 6** Sensitivity to workload and deadlines | 2 | (a) \(R_{\mathrm{value}}\) versus computation load; (b) versus relative-deadline multiplier; archive safety results at each point | E4 |
| **Fig. 7** Safety-radio outcomes | 4 | (a) safety PRR versus density; (b) P95 delay versus density; (c) delivered-packet delay CDF at preselected stressful case; (d) PRR versus distance with eligible-reception denominator | E2/E5 |
| **Fig. 8** Closed-loop admission and scheduling trace | 4 | (a) CBR; (b) safety PRR/P95 with thresholds; (c) **proposed/offered bytes plus budget** and blocked/fallback flags; (d) task offers, successful deliveries, and VA first-pick deviations over epochs | E5 |
| **Fig. 9** Ablations | 2 | (a) paired \(\Delta R_{\mathrm{value}}\) from corrected VA (with CI); (b) \(P_{\mathrm{viol}}\); include six independent variants and optionally MF as separately labeled comparator | E6 |
| **Fig. 10** Governor enforcement and controller cost | 4 | (a) offered-byte distribution and envelope exceedance count; (b) admission-block/fallback frequencies split by validity, PRR, P95, malformed/over-budget causes; (c) inference latency P50/P95 as candidate set grows; (d) completed value versus candidate-set size or throughput-cost scatter from actual measured runs | E7 |
| **Fig. 11** Generalization/robustness | 2 | (a) nominal-trained SPARC versus condition-trained SPARC on independent unseen regimes; (b) paired completed value and safety under alternate byte-envelope values or **gate-threshold** sensitivity, one factor at a time | E8 |

**E0 correctness:** independent tests of tasks/releases/deadlines, radio-delivery-to-queue commitment, byte counts, source/task masks, malformed proposal handling, PRR/P95 measurement validity and block/fallback behavior, initial residual zero property, sequence log probabilities under masks, PPO data provenance, and correct seed separation.

**E1 training:** archive all on-policy episode returns, validation SPARC/VA/MF/Direct PPO curves, VA relative baseline estimate and shift, KL, entropy, critic loss, VA-deviation rate, blocked offers and selected checkpoint. Distinguish training trajectories (used for gradients) from fixed validation trajectories (used for model selection). **No counterfactual branch reward or one-step \(\widehat A^{CF}\) plot.**

**E2 density:** all six methods, same task-generation rule and physically meaningful density parameter. Do not conflate the implementation's reported nominal density `35` with units of vehicles/km until confirmed. Preserve nominal as one explicit point or report it separately. Evaluate the same density levels across all methods.

**E3 admission sensitivity:** vary \(B_{\max}\) only, with nonzero and safety-gated conditions, and replay each method on matched scenarios; do not vary an obsolete shield coefficient \(\kappa\) or MF margin. Retuning the policy at each envelope must be explicitly marked as specialized retraining, not confused with fixed-checkpoint sensitivity.

**E4 task arrival/deadline:** vary task volume and multiply *relative* deadlines \(d_i\), always recomputing \(D_i=r_i\Delta+d_i\) and the common feasible set; preserve source uniqueness. Distinguish saturation from no task opportunities. Safety results must accompany all workload points.

**E5 radio diagnostics:** safety-only zero-computation reference and representative shielded/unshielded policies, frozen preselected case; log PRR eligible opportunities, latency of successful packets only with explicit reception caveat, CBR, offered bytes, governor decisions, and task completions. A safety-only infeasible scenario is identified rather than misattributed to offloading.

**E6 ablations:** matched final seeds, pre-registered variants and training budget; no MF deployment gate, residual regression loss, or cloned-state comparisons.

**E7 governor/runtime:** verify zero accepted rule violations deterministically (per code contract); quantify admitted versus proposed bytes and legitimate governor interventions. Measure **actual scheduler wall-clock inference and bridge call time separately**. The definitive application model uses fixed epochs and does not implement a continuous delay-and-expiry mechanism, so do not fabricate an injected-delay or discarded-assignment experiment. Optional extra-latency stress testing would require a separately documented environment extension and must not be presented as current SPARC behavior.

**E8 robustness:** frozen nominal-trained policy on unseen task/density regimes; compare condition-specialized policies only when separately trained and selected within the correct data split. Sweep envelope/gate *settings* separately as interventions. If a new envelope changes the valid candidate masks, ensure evaluation recomputes every mask and its governor state; no stale masks.

**E9 complexity:** count actor/critic parameters, training episodes, policy decisions, PPO gradient updates, total simulator time, validation calls, controller inference cost, Python↔MATLAB communication overhead. No nonexistent cloned reference simulation cost.

## 6. Primary simulation and learning parameter table

**Table II** records *actual* WiLab config and application/learning values. Already reported by definitive implementation: nominal density 35 (unit must be verified); 100 controller epochs; 3 tasks/source released across horizon; 12 s radio simulation; safety payload 300 B; byte budget 175 kB in `horizon_envelope_only` governor path; up to one task/source/epoch; SPARC training 96 episodes, 4 rollouts/update = 24 PPO updates; residual head zero; \(\alpha=100\), \(\beta=1\); learning rate \(3\times10^{-4}\) with decay, entropy coefficient 0.02 with decay, six PPO epochs/update, minibatch 64, KL target 0.015. These values belong **only** in performance evaluation, not System Model or Proposed Method. Source the physical radio, SCS, mobility, helper rates, task-size/cycle/deadline/value distributions, safety PRR/P95 thresholds, measurement windows, warmup, discount/GAE lambda, clipping threshold, VA-shift weights, and CPU/GPU metadata from checked configuration/logs. Do not invent omitted values.

See also `parameter_manifest.yaml` in this directory for sourced fields and explicit `UNVERIFIED` / `TBD` markers.

**Table III** nominal six-method comparison: \(V_{\mathrm{done}}\), \(R_{\mathrm{value}}\), completion ratio, \(P_{\mathrm{viol}}\), PRR, P95 safety delay, mean offered/admitted bytes, invalid-window rate, and safety feasibility. Include paired \(\Delta R_{\mathrm{value}}\) and \(\Delta V_{\mathrm{done}}\) versus corrected VA and strongest feasible baseline below table. Do **not** include discarded-assignment fraction as a metric of the definitive model.

**Table IV** baseline implementation and computational cost: admission policy, ranking/training, trainable parameters, training budgets, measured inference P50/P95, simulator calls/seconds, bridge overhead, and adaptation audit for B5. Report B4's admission conditions explicitly to avoid a misleading unfair comparison.

## 7. Seed partitions, training, and statistics

**Seed partition (locked for manuscript cost control):** see `SEED_PARTITION.md`.

| Partition | Budget |
|---|---|
| Training | **1** training run per learned configuration |
| Validation | **4** paired scenarios (sims `82001,82002,82011,82012`) for checkpoint/hyperparameter selection only |
| Final evaluation | **5** paired seeds per operating point (`final_test_seeds.json`) for all principal comparisons, ablations, and sensitivity experiments |
| Calibration | Reuse existing calibration data |

The definitive historical SPARC training uses `sim=72000+episode` and `task=182000+episode`; validation seeds above are **already-used** and never final-test. Claims from a single training run are **conditional on the selected checkpoint**. The older 30-final / 5-training-seed targets are **superseded** by this partition.

For each environment seed \(s\), calculate paired differences \(D_{b,s}=R_{\mathrm{SPARC},s}-R_{b,s}\). Resample independent **scenario-seed clusters**, not packets/epochs, for CIs. With multiple training seeds, use training-run nesting or hierarchical bootstrap and make clear whether comparisons condition on a selected checkpoint or integrate training variability. For the five primary baselines, construct simultaneous 95% confidence intervals (e.g., seed-cluster bootstrap calibrated for familywise error) or prespecify one validation-selected strongest comparator before opening final results; keep all nominal contrasts visible. For safety feasibility, use an upper one-sided CI for mean run-level \(P_{\mathrm{viol}}\) against \(\epsilon\); if it crosses \(\epsilon\), classify empirical compliance as uncertain rather than proven. Report a separately defined rule for invalid measurement windows. If no baseline qualifies as safety-feasible, do not declare an unqualified best-feasible-baseline victory.

Freeze before final test: workload distributions, physical scenarios, hard envelope, safety thresholds, measurement rule, pretrained checkpoints and checkpoint-selection function, baseline implementations, learning budgets, hypothesis, statistical procedure, and final seed registry. Record every tuning decision and its validation source. Do not substitute alternative metrics or favorable subsets after observing final-test outcomes.

## 8. Correctness and provenance gates

| Gate | Required acceptance evidence |
|---|---|
| G0 | Same seed reproduces exogenous source tasks, mobility setup, and initial radio state |
| G1 | Native Mode 2 reservation, sensing, collision and delivery path actually used; no analytic shortcut in one baseline |
| G2 | Governor enforces valid measurement / PRR / P95 gates, offered-byte bound, blocked state, and deterministic fallback, including edge cases |
| G3 | Candidate deadline calculation matches commit calculation; source/task uniqueness; no duplicate earned value |
| G4 | All earned completions require successful packet delivery and execution within horizon/deadline |
| G5 | Zero-initialized residual recovers deterministic corrected VA top choices; normalized logits/masks correct |
| G6 | Old/current PPO sequence probabilities use same masks; on-policy updates, advantage normalization, and critic targets match logs |
| G7 | Training, checkpoint-selection validation, and untouched final-test seeds disjoint; best-ΔVA checkpoint chosen only on validation |
| G8 | Safety denominators include failed receptions; invalid windows accounted separately; correct P95 measurement definition |
| G9 | Every table cell/figure point traceable to seed, controller, checkpoint, config hash, output file, aggregation script |

Never replace failed scientific gates by post hoc retuning on final-test results. Correct implementation errors on training/validation data before final-test opening; if a previously declared final-test run is invalid, document and handle by a predeclared invalid-run policy.

## 9. Log schema and evidence files

Minimum record groups:

- `run_metadata`: method, scenario/seed pair, training seed, checkpoint ID, config and code hashes, hardware, experiment ID.
- `task_log`: release epoch, source, payload, cycles, value, absolute deadline, candidate/selection history, delivered flag, queue/finish time, success/expiry reason.
- `decision_log`: candidate IDs/features, VA scores, actor logits, categorical or argmax ordered proposal, masks, first-pick deviation, controller inference time.
- `governor_log`: observed safety metrics/validity, admission/block reasons, incoming proposal, fallback flag, byte budget, proposed/offered bytes, executed pair IDs.
- `radio_safety_log`: CBR, safety-window validity, reception opportunities, delivered packets, PRR, delivery samples, P95, receiver-distance eligibility.
- `training_log`: policy/value losses, entropy, approx KL, GAE, adjusted advantages, VA baseline estimate, completed value, checkpoint selection, updates, RNG.
- `run_summary`: \(V_{\mathrm{off}},V_{\mathrm{done}},R_{\mathrm{value}}\), completion ratio, safety violations, valid/invalid window rates, PRR/P95, byte offers, drops, runtime.

**Do not create** `counterfactual_log`, `gate_log`, `shield_prediction_coefficients`, or `discarded_assignments` as required outputs for this architecture.

Directory layout (this package):

```
sparc_performance_evaluation/
  protocol/{evaluation_protocol.md,controller_contracts.md,parameter_manifest.yaml,
            experiment_matrix.csv,seed_registry.yaml,protocol_hashes.json}
  verification/{mode2,task_deadline,governor,ppo_masking,provenance}/
  checkpoints/{sparc,direct_ppo,lagrangian_ppo,itora_adapted,ablations}/
  results/{raw,aggregates,paired_statistics,safety_audit}/
  figures/{fig03_training,fig04_density,fig05_tradeoff,fig06_workload,
           fig07_radio_safety,fig08_closed_loop,fig09_ablation,
           fig10_governor_cost,fig11_robustness}/
  tables/{table_ii_parameters,table_iii_comparison,table_iv_complexity}/
  reports/{baseline_fairness,controller_verification,training_ablations,
           safety_compliance,main_claim_evidence}
```

Each figure has a script, input manifest, scenario IDs, plotted aggregates and uncertainty files. Preserve raw event-level logs for independent recomputation.

## 10. Execution order and failure handling

**Stage 1 (correctness; cheap):** pass G0–G9 and verify baseline/controller parity on hand-constructed cases. Measure initial zero-head VA equivalence. No training if task reward or safety denominator is wrong.

**Stage 2 (science feasibility; cheap):** run paired validation scenarios comparing deployed SPARC checkpoint, corrected VA, MF, Direct PPO where trained, plus safety-only and one unshielded stress reference. Check exact immutable project goal: positive value advantage over the **strongest safety-feasible comparison**, not merely VA, and no safety budget failure. Record the strongest reason *not* to launch the main campaign if validation shows SPARC substantially behind MF or comparable direct PPO.

**Stage 3 (training and ablation):** run controlled independent training seeds and variants, normalize training access fairly, choose all checkpoints only on validation data, and freeze the protocol. Never count test outcomes toward VA-relative checkpoint selection.

**Stage 4 (final):** execute all declared paired final scenarios and stress conditions, including unfavorable outcomes. Preserve independent final-test seal until gates 1–3 pass; never retune from final data.

**Stage 5 (analysis):** create Figs. 3–11 and Tables II–IV, simultaneous primary contrasts, one-sided safety uncertainty, and claim-to-evidence audit. Publish failures or uncertain results accurately.

## 11. Manuscript structure (Performance Evaluation)

A. Experimental configuration and comparator implementations — Tables II and IV.  
B. Training convergence and learned decision behavior — Fig. 3.  
C. Completed value, safety compliance and operating regimes — Figs. 4–7 and Table III.  
D. Governor dynamics and component ablations — Figs. 8–9.  
E. Computational cost, sensitivity and generalization — Figs. 10–11.

The central scientific acceptance question remains: **Does the definitive closed-loop, safety-governed VA-residual PPO yield a statistically supported increase in deadline-valid completed value over the strongest safety-feasible competitor in genuinely held-out Mode 2 scenarios?** Every mechanism and robustness claim is subordinate to that answer.
