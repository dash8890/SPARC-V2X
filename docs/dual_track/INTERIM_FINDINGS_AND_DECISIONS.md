# Interim findings and decisions required

This records what Track 1 and Track 2 produced so far, and the four issues that
have to be settled before Phase 4–7 because they materially change the claim
structure. Nothing here modifies the frozen baselines, workload, reward, or
simulator.

> **Update — corrected-contract run.** See
> [`CORRECTED_CONTRACT_RESULTS.md`](CORRECTED_CONTRACT_RESULTS.md) for the
> controlled `38.82 → 0.00` comparison (same seed, policy, and radio; only the
> accounting differs), the queue-overflow artifact, and the helper-geometry
> mismatch. F2, F4, and F5 below are superseded in detail by that document.

---

## 1. What is now implemented

| Artifact | Path |
| :--- | :--- |
| Standards-grounded measurement (Track 2) | `scripts/evaluate_standards_safety.py` |
| Track 2 results for existing runs | `sparc_performance_evaluation/results/aggregates/standards_safety/` |
| Causal queue instrumentation (simulator) | `mainV2X.m`, `wilab_closed_loop_callback.m` |
| Backlog-aware governor (Track 1) | `controllers/rl/backlog_governor.py` |
| Track 1 evaluation driver | `scripts/evaluate_backlog_guard.py` |
| Phase 1 freeze / application mapping | `docs/dual_track/PHASE1_FREEZE_AND_APPLICATION_MAPPING.md` |

The simulator now reports, at every controller boundary and before the epoch's
admission is injected: `pending_packets` (Mode-2 queue depth), `scheduled_attempts`,
and `active_vehicles`. Verified present in the smoke run's state.

## 2. Track 2 — first results (SC-RLGMI line)

| Run | Opportunities | Packet PRR | R_timely 100 ms | Legacy P_viol |
| :--- | ---: | ---: | ---: | ---: |
| SC-RLGMI (Option A) | 4 724 | 99.79 % | 99.79 % | 1.25 % |
| Candidate C | 4 599 | 99.91 % | 99.91 % | 0.50 % |
| Candidate D | 4 779 | 99.71 % | 99.71 % | 1.75 % |

All observed safety delays are below 50 ms, so the residual failures are
**losses, not late deliveries**.

The B1–B5 rows cannot be produced yet: see F1.

## 3. Findings that constrain the plan

### F1 — The frozen B1–B5 numbers are not reproducible from the repository

The comparison table cites `B1 = 39.20, B2 = 34.10, B3 = 36.50, B4 = 28.30,
B5 = 39.91`. No artifact under `sparc_performance_evaluation/results` contains
those values, and the only development-seed evaluations on 73300–73303 belong to
`scql_training`, `sc_rlgmi_training`, `Candidate_C_Surgical`, and
`Candidate_D_RiskCalibrated`. Track 2 cannot compare against B1–B5 until the
original campaign directory is located or the frozen baselines are re-run
unchanged on the dev seeds.

### F2 — The background safety floor is seed-dependent, and the training seeds are unusable for calibration

Zero-computation runs:

| Seed | Zero-offload violating windows | Window PRR |
| :--- | ---: | ---: |
| 73300 (dev) | 0 / 100 | 1.000 |
| 73301 (dev) | 0 / 100 | 1.000 |
| 73200 (train) | **20 / 100** | 0.938 |
| 73201 (train) | ≤ 2 / 100 (in guarded runs) | 0.995 |

On seed 73200 the 2 % criterion is violated by the radio environment alone,
before any computation is admitted. Consequence:

* Phase 4 ("compare on training scenarios") cannot calibrate a guard threshold:
  a 20 % floor swamps any effect of the guard. All three Stage B candidates score
  27–28 % on 73200 regardless of their strictness (C1 28 %, C2 28 %, C3 27 %).
* Any Track 1 claim must be stated **relative to the zero-offload floor of the
  same seed** (Δg), not as an absolute 0 %.

**Resolution (corrected-contract run).** The floor is a **small-sample,
phase-aligned window artifact**, not a radio-capacity event. On the payload
episode the low-PRR windows occur at epochs 7, 12, 17, 22, 27, … (every 5 epochs
= 100 ms, the beacon period), with values `12/14 = 0.857`, `14/16 = 0.875`,
`15/17 = 0.882`, while the cumulative PRR is **0.985** (24 errors / 1 451
attempts = 1.65 %). The errors are real path-loss/interference events at
140–410 m, but a ≈14-attempt window with 2 errors necessarily falls below
0.95. A zero-offload run on seed 73200 shows the same floor. Fix the metric
(longer window, minimum samples, EWMA/cumulative PRR) and report `Δg`; do not
read the 20 % as a capacity violation.

### F3 — The legacy metric is denominator-fragile; the packet-level result is strong

Across the four dev seeds SC-RLGMI loses **10 safety packets out of 4 724
attempts** (99.79 % packet reliability). Those 10 losses fall in **5 windows**
with ~14 packets each, producing a 1.25 % window-violation rate.

> Superseded in detail by `docs/dual_track/DUAL_TRACK_REPORT.md` §3; the numbers
> above are the authoritative ones.

The same underlying performance is therefore "1.25 % of windows fail" or
"99.79 % of packets arrive", depending on the metric. This is precisely the
distinction Track 2 exists to make explicit, and it favours the method.

### F4 — `V_done` credits a task after a single received packet (frozen contract)

`wilab_closed_loop_write_result.m` marks a task delivered if **any** positive
receive entry exists for its source:

```matlab
if any(stationManagement.pckReceived(:,ix) > 0)
    result.delivered_task_ids(end+1) = round(s);
```

`task_layer.apply_radio_outcomes` then sets the task complete (CPU finish within
deadline), and `completed_value()` sums its full value. A 79 kB / 80-packet task
is credited after 1 of 80 packets arrives.

This is the frozen completion contract, so it affects every method equally and is
not altered here. But it has two consequences that must be disclosed:

1. `V_done` is **not payload-faithful**; it is "admitted tasks with at least one
   decoded packet and a feasible CPU finish".
2. It creates an incentive to inject large bursts and collect the value
   immediately while the queue keeps interfering for several epochs — exactly
   the Stage A mechanism. Any "pacing" guard would exploit this accounting and
   produce a misleading "zero violations at unchanged value".

**Controlled result (corrected-contract run).** Same seed (73300), same frozen
SC-RLGMI checkpoint, same configuration; only the accounting differs:

| | Legacy | Payload |
| :--- | ---: | ---: |
| `V_done` | **38.82** | **0.00** |
| Tasks completed | 15 / 36 | 0 / 36 |
| Packets injected | 656 | 2 694 |
| Queue-overflow drops | 621 (95 %) | 2 585 (96 %) |
| ≈ Computation transmissions | ≈35 | ≈109 |
| Payload units credited | not tracked | 6 |

The 15 legacy-credited tasks required **656** units in total (21–80 each), but
at most ≈35 computation packets were transmitted (≈5.3 % of the required
units). A single packet cannot represent any of these payloads. The archived
44.96 is the dev-seed **mean**; 38.82 is the seed-73300 value. Both are
legacy-accounting numbers.

### F5 — The literal Track 1 inequality admits nothing at this operating point

Measured service capacity from the audits (net drain per 100 ms epoch,
`Q_{t-1} + injected_{t-1} + vehicles - Q_t`):

| Seed | median drain | mean drain |
| :--- | ---: | ---: |
| 73300 | 17 | 21.7 |
| 73301 | 16 | 18.8 |
| 73302 | 17 | 23.0 |
| 73303 | 16 | 21.8 |

The beacon load alone is ~15 packets/epoch (15 vehicles × 1 beacon / 100 ms).
With `S = 0.8·15 = 12` and `C ≈ 15` the headroom is 3 packets/epoch; with
`H = 2` the envelope is 6 packets. Every task in the workload has 29–109 packets,
so **no task is admissible**. The smoke run confirms it:

```
seed 73200: V_done = 0.00, P_viol = 20.00 % (20/100)  ← floor only
governor: packets_max = 4–6, admitted_packets = 0, reason = backlog_envelope
```

The rule as written is arithmetically correct but cannot be satisfied by atomic
task payloads. Options are in §4.

**Update (corrected-contract run).** The guard is not the binding constraint.
Two lower-level interfaces dominate:

* **Queue overflow (F8).** WiLab's single-packet `pckBuffer` drops one packet per
  TTI whenever the queue holds more than one packet. On the payload episode,
  2 585 / 2 694 injected packets (96 %) were dropped **before transmission**;
  pending depth reached 548. The traffic never reaches the scheduler, so this is
  a single-packet-queue artifact, not radio capacity.
* **Helper geometry (F9).** The legacy rule credits **any** receiver. The
  corrected rule requires the assigned helper, and the Python contact model
  (`r_comm = 250 m`, positions `U(100, 900)`) is independent of the WiLab trace
  (vehicles 66.7 m apart). Chosen helpers are often 400–600 m away, where the
  true PRR is 0.00–0.20. In the payload run, 12 of 14 tasks were assigned to
  helpers at 400–600 m and received **zero** units; the 2 tasks at 333 m
  received 6 units.

Consequence: redesigning the backlog guard on "physical radio capacity" before
fixing the queue and geometry would optimize the wrong layer.

### F6 — Distance-band PRR must not include WiLab's `blocked` column

`mainCV2XttiEnds.m:143` increments the blocked counter by *all neighbours* of a
half-duplex-blocked receiver, not by the single missed pair. On dev seed 73300
that column is 5 188 against 2 643 correct receptions, which is neighbour-count
inflation, not a 31 % reliability. The attempt-based PRR
(`NcorrectlyTxBeaconsTOT / (… + NerrorsTOT)`) is the defensible headline metric,
and the bridge export's `correct / (correct + errors)` is the correct
distance-banded analogue. Phase 1 §6.2 is corrected accordingly.

### F7 — The control clock and the task clock disagree (open)

The control cadence is **20 ms per epoch** (confirmed by 300 beacon
transmissions over 100 epochs = 15 vehicles × 10 Hz × 2 s). `task_layer` computes
deadlines with **10 ms per epoch** (`now_ms = 10 * current_epoch`). Deadlines
are therefore 2× more lenient in wall-clock time than the task model declares.
This affects legacy and payload modes equally; it was not changed. Decide the
task-layer clock explicitly before any final timing claim.

### F8 — The single-packet queue destroys multi-packet payloads (new)

`bufferOverflowLTE.m` drops the head packet whenever `pckBuffer > 1` at a TTI
end; its own comment says multi-packet queues are "not expected". A 21–80 packet
task therefore loses one packet per TTI, while the vehicle transmits only when it
wins a Mode-2 reservation (≈0.2 packets/TTI). Measured on the payload episode:

```
epoch  injected  scheduled  pending  overflow  credited
   10       249         39       95       151         0
   20       713         82      214       483         3
   40      1280        173       35      1198         6
   60      2355        255      548      1738         6
   90      2694        409        0      2585         6
```

96 % of injected packets never reach the transmitter. This is a queue-model
artifact, not a radio-capacity limit, and the 99.79 % beacon PRR does not cover
this traffic.

### F9 — Re-injection re-sends the whole payload (new)

Under payload accounting, incomplete tasks stay candidates and are re-proposed;
each re-proposal injects the **full** payload, not only the missing units. At
seed 73300 this produced 61 assignment events and 2 694 injected packets, versus
15 events and 656 packets in the legacy run. The feedback loop amplifies F8.
Retransmission must send only missing units, and injection must be paced at the
radio's service rate; neither is currently modelled.

## 4. Decisions required

**D1 — Delivery accounting (F4).** Keep the frozen "first packet completes the
task" contract, or introduce a payload-faithful completion (all packets or a
stated fraction) and re-run the comparison? The latter changes the reward, which
the plan currently freezes, so it needs explicit sign-off.

**D2 — Track 1 actuation (F5).** With atomic payloads the safety-reserved
capacity admits nothing. Choose one:
  (a) accept the negative result: at density 15, zero violations with this
      workload requires ~zero offloading;
  (b) allow **injection pacing** so a task's packets are spread across future
      windows (implements the per-window inequality literally, but interacts
      with F4 and must be reported as a shaper, not an admission rule);
  (c) allow a bounded excursion: admit a task when
      `Q_c + P(M) ≤ H·(Ĉ − S)` with `H` set from the Stage A spillover (1–3
      epochs), which rejects only bursts that would monopolise a window.

**D3 — Baseline re-run (F1).** Recover the original B1–B5 campaign, or re-run the
frozen baselines unchanged on 73300–73303 so Track 2 can report them?

**D4 — Background floor (F2).** For Track 1, evaluate `Δg = g(SC-RLGMI) −
g(∅)` per seed and state the floor alongside the absolute rate, or restrict the
safety claim to seeds whose floor is 0 %?

**Decision update (corrected-contract run).** D1 is implemented and demonstrated;
the remaining decisions are sequenced by the new evidence:

* **D1 (accounting): resolved** in favour of the payload-faithful rule for all
  new results. `accounting="legacy"` is kept only to reproduce archived values.
* **D5 (queue/injection, new):** fix the single-packet queue (F8) and full
  re-injection (F9) before any policy comparison. Options: paced injection, a
  dedicated computation queue, or partial retransmission. Each changes the
  simulator/workload and needs sign-off.
* **D6 (helper geometry, new):** align the Python kinematics with the WiLab
  trace (or select helpers from trace positions) before re-running. Without it,
  the assigned helper is often 400–600 m away (F9) and payload completion is
  impossible regardless of policy.
* **D2 (Track 1 actuation): defer the guard redesign.** The payload run shows
  the guard is not the binding constraint; F8/F9 are. Revisit only after D5/D6.
* **D3 (baselines):** the campaign is located (`evaluation/benchmark_panel.py`);
  a one-seed payload spot-check is 0.00 for B1–B5. The full multi-seed payload
  campaign is deferred until D5/D6.
* **D4 (floor): resolved to the metric fix** (longer window / minimum samples /
  `Δg`); the 20 % is not a capacity violation.

## 5. Not yet done

* Phase 4 training-seed comparison — blocked by D5/D6 (queue and geometry).
* Phase 5 dev evaluation of a Track 1 configuration — deferred until D5/D6.
* Phase 6 freeze of the safety contract — blocked by D4 (metric fix).
* Phase 7 sealed-test evaluation — **not touched**.
* Full multi-seed payload re-run of B1–B5 and SC-RLGMI — blocked by D5/D6.
