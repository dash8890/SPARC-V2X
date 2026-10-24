# SPARC-V2X — Dual-track safety and RL superiority: consolidated report

Date: 2026-10-24
Frozen commit: `d01ea0d1` (`main`, clean, pushed to `origin`)
Scope: Track 1 (zero-violation, backlog-aware safety) and Track 2 (standards-grounded
safety evaluation) on the SC-RLGMI line.

**Sealed test seeds 73400–73407 were never opened.** No baseline implementation, workload,
mobility, radio configuration, completion-only reward, or trained checkpoint was modified.
The frozen B1–B5 campaign has since been located (`evaluation/benchmark_panel.py`, seeds
73100–73103, density 35, TPS 4) and spot-checked under the corrected contract; see the
update below. The full multi-seed B1–B5 comparison is deferred pending the queue/injection
and helper-geometry fixes.

> **Corrected-contract update (2026-10-24).** A payload-faithful completion
> contract has since been implemented and run. Under it, a task is credited only
> when every required payload unit reaches the assigned helper and computation
> finishes by the deadline. On the same seed, policy, and radio, SC-RLGMI goes
> from legacy 38.82 to **0.00**, and the original B1–B5 panel is **0.00 for every
> method** at seed 73100. The cause is a single-packet queue that drops 96 % of
> injected payloads, plus a helper-geometry mismatch (assigned helpers are often
> 400–600 m away, true PRR 0.00–0.20). The Track 2 PRR/timeliness figures below
> are **beacon-level** and remain valid as radio metrics; they are not
> payload-faithful utility. See
> [`CORRECTED_CONTRACT_RESULTS.md`](CORRECTED_CONTRACT_RESULTS.md). Every number
> in this report that depends on `V_done` is a legacy-accounting result.

---

## 0. Headline

| Track | Outcome | Evidence |
| :--- | :--- | :--- |
| Track 2 — standards evaluation | **Succeeds.** The 10 Hz / 300 B awareness workload is served at **99.79 % packet reliability** and **99.79 % timely delivery within 100 ms** across 4 724 opportunities; the legacy "1.25 % violating windows" and the packet-level result describe the same run. | §3 |
| Track 1 — zero violation at preserved utility | **Fails, and fails for a structural reason.** At density 15 the measured net service is 16–17 packets/epoch while the mandatory beacon load is ~15. Safety-reserved capacity therefore leaves 1–3 packets/epoch of headroom, while every task is 29–109 packets. The specified inequality admits **no task at all**: smoke run `V_done = 0.00`, `admitted_packets = 0` on every epoch. | §4 |

The two outcomes are consistent: Track 1 asks for a *window-level* guarantee that the
physics at this operating point cannot fund, while Track 2 measures the *packet-level*
reliability that the system actually delivers.

---

## 1. What was executed

| Phase | Status | Artifact |
| :--- | :--- | :--- |
| 1 — Freeze evidence, map the application | done | `docs/dual_track/PHASE1_FREEZE_AND_APPLICATION_MAPPING.md` |
| 2 — Standards measurement | done | `scripts/evaluate_standards_safety.py`, `results/aggregates/standards_safety/` |
| 3 — Backlog-aware guard | done | `controllers/rl/backlog_governor.py`, simulator queue instrumentation |
| 4 — Training-seed comparison | **not executable** (floor artifact, F2) | — |
| 5 — Dev evaluation of Track 1 config | **not executable** (guard admits nothing, F5) | `results/raw/backlog_guard/smoke_h2_r08/` |
| 6 — Freeze safety contract | not reached | — |
| 7 — Sealed test | **not started, not authorised** | — |

### 1.1 Frozen facts about the simulated service

| Quantity | Value |
| :--- | :--- |
| Radio | NR-V2X Mode 2, 10 MHz, SCS 15 kHz, half duplex, MCS 11 |
| Safety traffic | **300 B every 100 ms per vehicle** (10 Hz cooperative awareness) |
| Range | `Raw = 150 m`; distance accounting to 483 m |
| Computation traffic | payload-faithful stream, 1000 B packets |
| Network | 15 vehicles, 12 s episodes, 100 control epochs of 100 ms |
| Checkpoint | `checkpoints/sc_rlgmi_best.pt` sha256 `230ea2a8…92cdaf7` |
| Trace | `synthetic_highway_rho_15_20s.txt` sha256 `88cf8d66…71bdb0f0` |

The workload is a **periodic awareness service**, not pre-crash sensing or platooning.
That is why the evaluation is anchored on TS 22.185 (100 ms) rather than on the
10 ms / 99.99 % cooperative-collision-avoidance class of TS 22.186.

### 1.2 What was added to the simulator

Additive instrumentation only — no change to radio behaviour:

* `mainV2X.m` — at every controller boundary, **before** that epoch's admission is
  injected, the state now carries `pending_packets` (Mode-2 queue depth),
  `scheduled_attempts`, and `active_vehicles`.
* `wilab_closed_loop_callback.m` — propagates those three fields to the Python controller.

Verified live in the smoke run: `pending_packets = 1.0`, `active_vehicles = 15.0`,
`drained_last_epoch = 15.0`.

---

## 2. Track 2 — method

```
R_timely(D_max) = Σ_i # safety packets decoded with latency ≤ D_max in window i
                  ───────────────────────────────────────────────────────────────
                  Σ_i # decode opportunities in window i   (rx_i + err_i)
```

Computed from the frozen per-epoch audits by differencing the cumulative counters that
WiLab already writes (`safety_rx_cumulative`, `safety_error_cumulative`,
`safety_delay_hist_cumulative` at 1 ms resolution, `prr_vs_distance`).

Reported alongside:

* cumulative packet PRR,
* per-window PRR and window P95,
* PRR versus distance (0–50 / 50–150 / 150–300 / 300–500 m),
* longest run of consecutive loss windows (explicit window-level proxy for the packet
  inter-reception interval — WiLab exports no per-packet receive timestamps),
* the **unchanged** legacy metric `P_window = Pr[PRR_100ms < 0.95 ∨ P95 > 50 ms]`.

Reproduce:

```bash
python scripts/evaluate_standards_safety.py \
  --run sc_rlgmi=sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval \
  --run scql=sparc_performance_evaluation/results/raw/scql_training/eval \
  --run candidate_c=sparc_performance_evaluation/results/raw/stage_b_candidates/Candidate_C_Surgical \
  --run candidate_d=sparc_performance_evaluation/results/raw/stage_b_candidates/Candidate_D_RiskCalibrated
```

## 3. Track 2 — results

### 3.1 Primary packet-level metrics (development seeds 73300–73303)

| Run | Opportunities | Packet PRR | R_timely 20 ms | R_timely 50 ms | R_timely 100 ms | Worst window | Longest loss run |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **SC-RLGMI** | 4 724 | **99.788 %** | 91.152 % | 99.788 % | **99.788 %** | 87.500 % | 1 |
| SCQL | 4 708 | 99.788 % | 91.461 % | 99.788 % | 99.788 % | 87.500 % | 1 |
| Candidate C (surgical throttle) | 4 599 | **99.913 %** | 93.694 % | 99.913 % | 99.913 % | 87.500 % | 1 |
| Candidate D (risk-calibrated) | 4 779 | 99.707 % | 90.061 % | 99.707 % | 99.707 % | 87.500 % | 1 |

Two facts follow directly:

1. **`R_timely(50 ms) = R_timely(100 ms) = packet PRR`.** Every packet that is received
   arrives within 50 ms. The residual failures are **losses, not late deliveries**. Against
   the 100 ms awareness requirement, the method is bounded only by loss.
2. **`R_timely(20 ms) ≈ 91 %.** If the represented service were a 20 ms pre-crash class,
   the same run would be at 91 %. The requirement class must therefore be stated with the
   number, not after it.

### 3.2 Per-seed utility / legacy safety (same runs)

| Run | 73300 | 73301 | 73302 | 73303 | Mean V | Windows failed | Pooled packet PRR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **SC-RLGMI** | 38.82 (4) | 38.16 (1) | 52.13 (0) | 50.71 (0) | **44.96** | **5 / 400** | 99.788 % |
| SCQL | 36.02 (4) | 50.49 (1) | 48.08 (0) | 45.08 (0) | 44.92 | 5 / 400 | 99.788 % |
| Candidate C | 30.04 (2) | 30.41 (0) | 36.30 (0) | 40.69 (0) | 34.36 | 2 / 400 | 99.913 % |
| Candidate D | 41.97 (4) | 42.55 (3) | 56.94 (0) | 60.93 (0) | 50.60 | 7 / 400 | 99.707 % |

(Values are `V_done`; parentheses are the number of 100 ms windows below the legacy
threshold.)

**The entire legacy "safety problem" is 10 lost packets out of 4 724.** Those 10 losses
occur in 5 windows of ~14 packets each, so two lost packets in one window is already a
14 % window-PRR drop and an automatic violation. The window metric is
denominator-fragile; the packet metric is not.

### 3.3 PRR versus distance

| Run | 0–50 m | 50–150 m | 150–300 m | 300–500 m |
| :--- | ---: | ---: | ---: | ---: |
| SC-RLGMI | no attempts | 99.76–100 % | 98.47–99.60 % | 83.83–90.46 % |
| Candidate C | no attempts | 99.88–100 % | 98.59–99.59 % | 82.66–89.87 % |
| Candidate D | no attempts | 99.76–100 % | 97.74–99.92 % | 83.75–89.71 % |

Reliability inside `Raw = 150 m` is ≥ 99.7 % in every run. Beyond 300 m it degrades to
~84–90 %, which is expected for a 150 m configured range and must be quoted with the
evaluation range attached (TR 37.885 style).

No transmissions are attempted in the 0–50 m band during these episodes, so that cell is
empty rather than perfect.

### 3.4 The 2 % ceiling, correctly labelled

The legacy 2 % ceiling is **not** a standards quantity and is not used as the primary
criterion here. For completeness, its statistical resolution on this sample:

* observed 5 / 400 → 1.25 %, Wilson 95 % CI ≈ **0.3 %–3.2 %**;
* a hypothetical 0 / 400 would still only bound the rate at **< 0.75 %**.

With ~14 packets per window, the ceiling cannot separate methods that are 99.7 % reliable
from methods that are 99.9 % reliable. This is a property of the metric, not of the
controllers.

---

## 4. Track 1 — mechanism, implementation, and result

### 4.1 Implemented rule

`controllers/rl/backlog_governor.py` implements exactly the specified inequality, per
future window rather than as an aggregate budget:

```
L_backlog(t:t+H) + L(M, t:t+H)  ≤  Ĉ(t:t+H) − S(t:t+H)
```

with causal estimators only:

| Symbol | Estimator | Source |
| :--- | :--- | :--- |
| `Q_t` | total Mode-2 queue at the boundary, before injection | simulator (`pending_packets`) |
| `Q^c_t` | `max(0, Q_t − active_vehicles)` — compute-attributable backlog | derived |
| `Ĉ_t` | EWMA(α = 0.3) of **measured** net drain `Q_{t−1} + A_{t−1} + veh − Q_t` | measured, causal |
| `S_t` | `reserve_frac · active_vehicles` (one beacon per vehicle per 100 ms) | measured |
| envelope | `packets_max = max(0, H·max(0, Ĉ − S) − Q^c)` | — |

The proposal is truncated by the policy's own analytic ranking, never replaced wholesale,
so the learned matching survives wherever it fits. The `θ_RL`, beam search, reward,
workload and baselines are untouched.

### 4.2 Measured capacity (the decisive number)

Net drain per 100 ms epoch, reconstructed from the audits:

| Seed | median | mean | p10–p90 |
| :--- | ---: | ---: | ---: |
| 73300 | 17 | 21.7 | 12–35 |
| 73301 | 16 | 18.8 | 12–32 |
| 73302 | 17 | 23.0 | 14–42 |
| 73303 | 16 | 21.8 | 12–38 |

The periodic beacon load alone is **~15 packets/epoch** (15 vehicles × 1 beacon / 100 ms).
Median headroom is therefore **1–3 packets per epoch**, and every task in the workload is
**29–109 packets**. Payloads are atomic at admission, so:

### 4.3 Result: the envelope admits nothing

Smoke run, seed 73200, `H = 2`, `reserve_frac = 0.8`:

```
V_done = 0.00     P_viol = 20.00 % (20/100)     admitted_packets = 0 (every epoch)
governor: packets_max = 4–6 (mean 6.0, never above 6 for the whole episode)
          capacity_ewma = 15.0 (constant), reserve = 12, headroom = 3
          reasons: 35 backlog_envelope · 20 prr_below_floor · 45 nothing proposed
```

The 20 % is **not** caused by the guard: seed 73200 has a 20 / 100 violating-window
**background floor with zero computation at all** (§6, F2), and those are exactly the
20 epochs on which the pre-existing PRR floor fires. The guard added nothing and
prevented nothing, because there was nothing it could admit: the envelope never rose
above 6 packets.

The failure is structural, not a tuning miss: for any `H` consistent with the observed
1–3 epoch spillover, `H·(Ĉ − S)` is 3–9 packets, i.e. one order of magnitude below the
smallest task.

### 4.4 What this means

At this operating point the safety-reserved capacity is **fully consumed by mandatory
awareness traffic**. There is no packet budget that can be handed to computation without
eating into beacon service. Zero *window* violations therefore implies ~zero offloading —
which is consistent with the earlier Stage B observation that all three strict candidates
score 27–28 % on 73200 regardless of their thresholds, and with Candidate C reaching
0.50 % only by collapsing `V_done` from 44.96 to 34.36.

This is a negative result for Track 1 as specified, and it is reportable as such.

---

## 5. Measurement contract: what `V_done` actually counts

`wilab_closed_loop_write_result.m` marks a task delivered when **any** positive receive
entry exists for its source:

```matlab
if any(stationManagement.pckReceived(:,ix) > 0)
    result.delivered_task_ids(end+1) = round(s);
```

`task_layer.apply_radio_outcomes` then completes the task, and `completed_value()` credits
its full value. A 29-packet / 28 kB task is credited after packet #1; a 80-packet / 79 kB
task likewise.

Consequences:

1. `V_done` is **not payload-faithful**. It is "admitted tasks with ≥ 1 decoded packet and
   a feasible CPU finish". All methods share this contract, so internal comparisons are
   like-for-like, but the absolute number must not be presented as delivered payload.
2. It creates a systematic incentive to inject large bursts: the value is collected
   immediately while the queue keeps interfering for several epochs. That is precisely the
   Stage A failure mechanism.
3. It is why an injection-pacing guard would look free: pacing delays the remaining packets
   but not the first one, so `V_done` would barely move while the radio load disappears.
   Any pacing result must be reported as a **shaper under the legacy accounting**, never as
   preserved utility.

The contract was kept frozen as instructed. It is disclosed here because it bounds every
utility claim in the manuscript.

---

## 6. Findings register

| # | Finding | Impact |
| :--- | :--- | :--- |
| **F1** | The frozen B1–B5 values (39.20 / 34.10 / 36.50 / 28.30 / 39.91) exist nowhere in the repository; the only dev-seed evaluations belong to SCQL, SC-RLGMI, Candidate C and D. | Baseline comparison deferred by decision; Track 2 currently covers the SC-RLGMI line only. |
| **F2** | Background floor is seed-dependent: zero-offload 73300 = 0/100, 73301 = 0/100, but **73200 = 20/100** (window PRR 0.938). No zero-offload run exists for 73201 (guarded runs show 2/100, so floor ≤ 2 %). | Training-seed calibration is invalid; the 20 % floor swamps any guard effect. Safety must be quoted as Δ over the same seed's floor. |
| **F3** | The legacy metric is denominator-fragile: 10 lost packets in 4 724 attempts (99.79 %) become 5 violating windows (1.25 %) because windows hold ~14 packets. | 2 % cannot separate 99.7 % from 99.9 % methods; packet-level is the primary metric. |
| **F4** | `V_done` credits a task after one received packet. | Utility is not payload-faithful; pacing would exploit it. See §5. |
| **F5** | The specified inequality admits no atomic task at density 15 (headroom 1–3 packets vs task sizes 29–109). | Track 1 negative result; requires a different actuation or a different capacity definition. |
| **F6** | WiLab's `blocked` column is not a reliability denominator: `mainCV2XttiEnds.m:143` adds *all neighbours* of a half-duplex-blocked receiver (dev 73300: 5 188 blocked vs 2 643 correct). | The attempt-based PRR the project uses is the defensible one; distance PRR uses `correct/(correct+errors)`. |

---

## 7. Possible directions

Ordered by expected scientific value per unit of risk. None of these was executed; each
changes something the current freeze protects, so each needs an explicit decision.

### A. Publish Track 2 as primary, legacy as secondary (no new runs)

* Claim: 99.79 % packet reliability and 99.79 % timely delivery at 100 ms for a 10 Hz
  awareness service, with PRR-vs-distance and loss-run distributions; the 2 % window
  ceiling reported as a historical diagnostic.
* Cost: none. Risk: the utility claim still depends on F4.
* Effect: converts the current 1.25 % "safety failure" into an explicit, standards-anchored
  statement — the honest form of the existing result.

### B. Payload-faithful completion accounting

* Require all (or a stated fraction of) packets to arrive before crediting a task; re-run
  SC-RLGMI, candidates, and baselines.
* Cost: full re-run; changes the reward, so it needs sign-off.
* Effect: removes the burst incentive, makes `V_done` defensible, and likely *raises* the
  relative value of the guard (queue spillover would finally cost utility).

### C. Two-regime capacity estimator (most promising technical fix for Track 1)

* The current `Ĉ` is the equilibrium drain (~16–17), which is beacon-dominated. During
  bursts the channel actually serves 20–42 packets/epoch (p90 of the measured drain).
  Estimate capacity conditioned on queue state (e.g. `Ĉ_burst` from epochs with
  `Q > active_vehicles`) and use `Ĉ_burst − S` as headroom.
* Effect: headroom rises to ~5–25 packets/epoch, so small/medium tasks become admissible
  while 80–109-packet bursts are still excluded. This is the only variant that can plausibly
  land in the 39.20–44.96 band with fewer violations.
* Risk: `Ĉ_burst` is measured *because* the system was already congested, so it partly
  encodes the damage being prevented. Requires an explicit identification argument.

### D. Per-window subchannel reservation instead of packet reservation

* Mode 2 capacity is subchannel/TTI structured; reserve `k` subchannels per 100 ms window
  for beacons and admit computation only into the remainder. This is a more faithful reading
  of "reserved resources per future radio window" than a packet count.
* Effect: protects specific windows rather than average load; may admit a burst while
  bounding its worst window. Needs simulator-level knowledge of the allocation grid.

### E. Queue-aware learning rather than queue-aware guarding

* Retrain the advantage network with `Q_t`, `Ĉ_t` in the decision state so the policy itself
  prefers small/medium payloads under load. Keeps the guard fixed and puts the adaptation
  where the plan wants the credit ("attributable to learning").
* Cost: retraining + re-validation; `θ_RL` is currently frozen.
* Effect: the only route that can claim the safety gain is learned rather than imposed.

### F. Injection pacing (traffic shaping)

* Admit the task, release its packets over future windows under a token bucket sized by
  `Ĉ − S`.
* Effect: implements the per-window inequality literally and would show large safety gains.
* **Do not publish without B**: under the current first-packet credit the value is nearly
  free, so the result would be an accounting artifact.

### G. Diagnose the 73200 background floor

* 20 / 100 violating windows with zero offloading is a ~6–7 % beacon loss rate that no
  controller can fix. Check trace geometry, beacon generation phase, and controller-boundary
  alignment (the failures are periodic, every 5th epoch).
* Effect: if it is a boundary/phase artifact, the training regime becomes usable and Phase 4
  calibration becomes possible. If it is real geometry, training seeds must be excluded from
  safety claims.

### H. Statistical power before any zero-violation claim

* At ~100 windows/seed, a 0 % observation only bounds the rate at < 0.75 %; 5/400 is
  indistinguishable from 2 %. A credible zero-violation claim needs ≥ 150–300 clean windows
  *per configuration* (rule of three) plus the packet-level metric.
* Effect: changes the evaluation budget, not the algorithm.

### I. Recover or re-run B1–B5 for Track 2

* Deferred by decision. If the manuscript needs the comparison table, the frozen baselines
  must be re-run unchanged on 73300–73303 to obtain per-epoch audits.

### J. Accept the constraint as the result

* State that at density 15 the mandatory awareness load saturates Mode 2 service, so
  zero-window-violation operation is incompatible with the offloading utility that produces
  the RL advantage. Frame the contribution as "99.8 % packet reliability at 45.0 completed
  value" rather than "zero violations".
* Effect: turns the Track 1 failure into a quantified operating-envelope statement.

**Recommended combination:** A + G + H immediately (cheap, removes three confounds), then
C or E if Track 1 is to be pursued further, with B as the prerequisite for any pacing or
utility claim.

---

## 8. Status and open items

| Item | State |
| :--- | :--- |
| Track 2 measurement + results | complete for SC-RLGMI, SCQL, Candidate C, Candidate D |
| Track 2 for B1–B5 | deferred (no artifacts, F1) |
| Track 1 guard implementation | complete, tested, negative |
| Track 1 dev evaluation | not run (guard admits nothing; a dev run would add no information) |
| Phase 6 freeze | not reached |
| Sealed seeds 73400–73407 | **untouched** |
| Git | all new work **uncommitted** on `main` @ `d01ea0d1` |

### Reproduction

```bash
# Track 2 (all reported tables)
python scripts/evaluate_standards_safety.py \
  --run sc_rlgmi=sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval \
  --run scql=sparc_performance_evaluation/results/raw/scql_training/eval \
  --run candidate_c=sparc_performance_evaluation/results/raw/stage_b_candidates/Candidate_C_Surgical \
  --run candidate_d=sparc_performance_evaluation/results/raw/stage_b_candidates/Candidate_D_RiskCalibrated

# Track 1 smoke (negative result)
python scripts/evaluate_backlog_guard.py --seeds 73200 --horizon 2 --reserve-frac 0.8 \
  --label smoke_h2_r08 --base-eid 96000
```

Runtime: one 12 s episode ≈ 110–150 s wall on MATLAB R2026a.

---

## 9. Limitations

1. `R_timely` counts decode attempts (`NcorrectlyTxBeaconsTOT + NerrorsTOT`). Transmissions
   never attempted because the receiver was itself transmitting are not represented.
2. Packet inter-reception intervals are approximated by consecutive-loss windows; WiLab does
   not export per-packet receive timestamps.
3. Distance bands beyond 300 m are outside the configured `Raw = 150 m` service range and are
   reported for completeness only.
4. Standards numeric targets are inherited from the supplied review; the normative PDFs
   could not be re-fetched in this environment and must be pinned to a spec version before
   publication.
5. All utility values are subject to the first-packet delivery contract (§5).
