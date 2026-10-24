# Corrected task-completion contract: results and consequences

Status: **dev-seed evidence, uncommitted work tree**. Sealed seeds `73400–73407`
were not touched. This document supersedes the completion-accounting parts of
`DUAL_TRACK_REPORT.md`; the legacy tables there remain valid **as legacy
accounting**, not as payload-faithful utility.

---

## Executive summary

1. The frozen rule credits a task when **one** packet from its source is received
   by **any** vehicle. Under the corrected rule — every required payload unit
   reaches the **assigned helper** and computation finishes by the deadline — the
   same seed, policy, and radio configuration yield **`V_done = 0.00`** instead
   of **`38.82`** (SC-RLGMI, seed 73300, identical episode configuration). The
   original B1–B5 panel likewise yields **0.00 for every method** at seed 73100.
2. The corrected run is **not** a radio-capacity result. WiLab's single-packet
   vehicle queue discards **2 585 of 2 694** injected computation packets (96 %)
   before they are ever transmitted, because it is designed for one packet per
   vehicle and is not a multi-packet payload queue. Of the ≈109 computation
   packets that do transmit, only **6** reach the assigned helper.
3. The assigned helper is frequently **physically out of range** in the WiLab
   trace (true PRR 0.00 at 470 m) even though the Python contact model believes
   it is within 250 m, because the application-layer kinematics are generated
   independently of the trace geometry. The legacy "any receiver" rule hid this.

**Consequences.** The reported utilities (SC-RLGMI mean 44.96; B1–B5
39.20 / 34.10 / 36.50 / 28.30 / 39.91) are **legacy-accounting results, not
payload-faithful utility**. No policy ranking or safety claim can be sustained
under the corrected contract until the queue/injection and helper-geometry
interfaces are fixed. The 20 % safety floor is a small-sample window artifact,
not a capacity event.

---

## 1. The corrected rule

For task `k` with source `s(k)`, assigned helper `h(k)`, required payload units
`n_k`, and deadline `d_k`:

```
z_k = 1{ all n_k units of k reached h(k) at least once
         AND h(k)'s computation finished by d_k }
```

Operationally:

* each injected packet carries an identity tag `(task_id, unit_index)`;
* tags are held in a per-source FIFO in injection order;
* the head tag is credited only when that source's final replica is decoded
  **and** `pckReceived(h(k), s(k)) > 0`;
* a tag is popped without credit when the simulator drops or overflows the
  corresponding queued packet (explained via `reconcile`);
* duplicates and re-injections deduplicate per `(task, unit)`, so retransmission
  can recover a missing unit but cannot inflate credit;
* if an incomplete task is re-assigned to a different helper, the unit bitmap is
  reset and the payload must reach the new helper; units received by the old
  helper do not count toward the new one;
* a task is complete only when every unit has been credited **and** the helper's
  CPU finish time is `≤ d_k`; assigned-but-incomplete tasks are re-checked in a
  second pass for late completions.

`accounting="legacy"` is retained only to reproduce archived first-packet
results. Payload accounting is the default.

---

## 2. Implementation and verification

| Artifact | Path |
| :--- | :--- |
| Packet identity / dedup / completion core | `experiments/wilab_closed_loop/wilab_closed_loop_packet_accounting.m` |
| MATLAB capture / reconcile / register / mark-final wiring | `mainCV2XttiStarts.m`, `mainV2X.m`, `wilab_closed_loop_write_result.m` |
| Python payload contract | `experiments/wilab_closed_loop/task_layer.py` |
| Accounting threading / audit fields | `train_wilab_closed_loop.py` |
| Invariant checker | `scripts/verify_payload_accounting.py` |
| Python unit tests (11) | `tests/test_payload_accounting.py` |
| MATLAB unit tests | `tests/test_packet_accounting.m` |

Checks that hold on the integration episode:

* `packet_tag_deficits = 0` — every injected tag was eventually explained.
* `packet_capture_missing = 0` — every `mark_final` found a head tag.
* No task credited without `payload_received >= payload_required`, and no task
  credited above `n_k`.
* No epoch announces a completion that is absent from the final task state.

The near-zero utility is therefore a property of the simulator/queue and the
helper geometry, not of the accounting code.

---

## 3. Controlled comparison

Same seed (73300), same frozen SC-RLGMI checkpoint, same density (15), same
task seed (186100), same 100 epochs, same radio configuration; **only the
accounting mode differs**.

| | Legacy accounting | Payload accounting |
| :--- | ---: | ---: |
| `V_done` | **38.82** | **0.00** |
| Tasks completed | 15 / 36 | 0 / 36 |
| Tasks ever assigned | 15 | 14 |
| Assignment events | 15 | 61 |
| Packets injected | 656 | 2 694 |
| Packets dropped by queue overflow | 621 (95 %) | 2 585 (96 %) |
| Scheduled vehicle-attempts | 335 | 409 |
| ≈ Computation transmissions (335/409 − ≈300 beacons) | ≈35 | ≈109 |
| Payload units credited | not tracked (legacy) | 6 |
| Mean cumulative PRR | 0.9945 | 0.9851 |
| Windowed `P_viol` | 4 % | 12 % |

The two runs use the same `SCRLGMIPolicy` object, checkpoint, and
`select_matching` path. The legacy value reproduces the archived seed-73300
episode exactly (`episode_summary.json` = `38.823551375581474`), confirming that
the legacy path is unchanged and that the difference is the accounting rule.

The archived headline **44.96 is the mean over the dev seeds**; `38.82` is the
seed-73300 value. It is the same policy line, not a different method.

---

## 4. Finding 1 — first-packet credit (F4, confirmed)

The 15 tasks credited by the legacy run required, in total, **656 payload
units** (21–80 units each; median 38). Only **≈35 computation packets were even
transmitted** in the same 2 s window, i.e. at most **5.3 % of the required
units**, and 621 of the 656 injected packets were dropped before transmission.

Yet the legacy rule credited **15 / 15 assigned tasks**. Under the corrected
rule, **0 / 15** are payload-complete (only 2 tasks received any unit at all,
3 units each).

The per-task requirements of the credited tasks were:

```
29, 54, 21, 38, 31, 35, 36, 50, 24, 28, 80, 43, 60, 52, 75
```

A single decoded packet cannot represent any of these payloads. This is the
first-packet inflation described in F4; the controlled run now quantifies it as
**38.82 → 0.00**.

---

## 5. Finding 2 — any-receiver credit and the helper-geometry mismatch (F5 updated)

The legacy rule also does not check the assigned helper:

```matlab
if any(stationManagement.pckReceived(:,ix) > 0)   % any receiver
    result.delivered_task_ids(end+1) = round(s);
```

`task_layer` then applies the delivery to every assignment of that source. A
task is therefore credited if the packet reaches *any* neighbour, even when the
chosen helper receives nothing.

That is not a corner case. In the payload run, 11 of the 14 assigned tasks were
sent to helper 8 or 9, which sit 400–600 m from their sources. The
distance-banded true PRR (`correct / (correct + errors)`, excluding WiLab's
inflated `blocked` column) is:

| Distance | True PRR |
| ---: | ---: |
| 70 m | 1.000 |
| 140 m | 0.965 |
| 210 m | 0.959 |
| 270 m | 0.947 |
| 340 m | 0.589 |
| 410 m | 0.201 |
| 470 m | 0.000 |

Assigned pairs and credited units:

| Source → helper | Trace distance | True PRR | Units credited |
| :--- | ---: | ---: | ---: |
| 0 → 5 | 333 m | 0.589 | 6 |
| 1 → 8 | 467 m | 0.000 | 0 |
| 2 → 8 | 400 m | ≈0.20 | 0 |
| 1 → 9 | 533 m | 0.000 | 0 |
| 1 → 6 | 333 m | 0.589 | 0 |
| 0 → 9 | 600 m | 0.000 | 0 |

**Why the policy picks out-of-range helpers.** `task_layer.py` samples helper
and source positions independently from `U(100, 900)` m and evaluates contact
with `r_comm = 250 m`. The WiLab trace places the 15 vehicles 66.7 m apart on a
single lane. The two geometries are unrelated. A pair that is 333 m apart in
the trace can be 200 m apart in the Python model, so the policy believes it is
in contact while the radio sees a 0.59 PRR; at 467 m the radio sees 0.00.

The legacy "any receiver" rule made this invisible: the adjacent vehicle at
66.7 m (PRR 1.0) received the packet, so the task was credited regardless of
the helper.

**This is an interface defect, not a policy defect.** Fixing the accounting
without aligning the geometries cannot produce a meaningful payload-faithful
comparison, because the selected helper is often physically unreachable.

---

## 6. Finding 3 — the queue-overflow artifact (F8, new)

WiLab models one packet per vehicle: `pckBuffer` is a counter, and
`bufferOverflowLTE.m` drops the head packet whenever `pckBuffer > 1` at a TTI
end (its own comment says the multi-packet case is "not expected"). Injecting a
21–80 packet task therefore triggers one overflow drop **per TTI**, while the
vehicle can transmit only when it wins a Mode-2 reservation (≈0.2 packets per
TTI).

Measured on the payload episode:

```
epoch  injected  scheduled  pending  overflow  credited
   10       249         39       95       151         0
   20       713         82      214       483         3
   40      1280        173       35      1198         6
   60      2355        255      548      1738         6
   90      2694        409        0      2585         6
```

* 2 694 packets injected; 2 585 lost to overflow (96 %).
* Pending depth reaches 548 packets, versus ~94 in the legacy run.
* Only ≈109 computation packets are transmitted.
* Only 6 units reach the assigned helper, because the helper is often out of
  range (Finding 2).

The drop is **not** a radio-capacity limit: the packets never reach the
scheduler. It is a mismatch between an atomic multi-packet payload and a
single-packet queue. The `99.79 %` beacon PRR says nothing about this traffic,
because it measures beacons.

---

## 7. Finding 4 — the re-injection feedback loop (F9, new)

Under payload accounting, incomplete tasks remain candidates and are
re-proposed. Each re-proposal re-injects the **full** payload, not only the
missing units:

* legacy: 15 assignment events, 656 packets injected;
* payload: 61 assignment events, 2 694 packets injected (4.1×).

The re-injection both reflects the corrected objective (a task that is not
complete may be retried) and amplifies the queue overflow. It is a genuine
design issue: retransmission must send only the missing units, and injection
must be paced at the radio's service rate. Neither is currently modelled.

---

## 8. Finding 5 — the 20 % floor is a metric artifact (F2 resolved)

The floor reproduces under payload accounting with the same phase: low-PRR
windows at epochs **7, 12, 17, 22, 27, …** (every 5 epochs = 100 ms, the beacon
generation period). The window values are `12/14 = 0.857`, `14/16 = 0.875`,
`15/17 = 0.882`, … while the cumulative PRR is **0.985** and the total error
count is **24 / 1 451 attempts = 1.65 %**.

The errors themselves occur at epochs 6, 11, 16, 21, … (2 errors per batch) and
are real path-loss/interference events at 140–410 m. The violation is produced
by the **≈14-attempt window** and a one-epoch offset between the success and
error counters: 2 errors in a 14-attempt window is `0.857 < 0.95`, even though
the underlying rate is 98.5 %. A zero-offload run on seed 73200 shows the same
20 % floor, so it is a property of the radio + metric, not of offloading.

**Resolution:** use a longer aggregation window (e.g. ≥100 attempts or ≥500 ms)
or an EWMA/cumulative PRR with a minimum-sample rule, and report the seed's
zero-offload floor `Δg`. Do not read the 20 % as a capacity violation.

---

## 9. Finding 6 — the control clock (F7, open)

The control cadence is **20 ms per epoch**, confirmed by 300 beacon
transmissions over 100 epochs (15 vehicles × 10 Hz × 2 s). The task layer
computes deadlines with **10 ms per epoch** (`now_ms = 10 * current_epoch`).
Deadlines are therefore **2× more lenient in wall-clock time** than the task
model declares. This affects legacy and payload modes equally and was not
changed. It should be decided explicitly before any final timing claim.

---

## 10. Reassessment of RL superiority and safety

**Superiority.** Under the corrected contract, SC-RLGMI at seed 73300 is 0.00,
and the original B1–B5 panel at seed 73100 (density 35, TPS 4) is also 0.00 for
every method. The baselines did admit traffic — 12–13 tasks and 1 268–1 668
packets each — but 95–96 % of those packets were dropped by the queue and no
payload completed:

| Method | `V_done` | Assigned | Injected | Overflow | Scheduled |
| :--- | ---: | ---: | ---: | ---: | ---: |
| B1 greedy value-density | 0.00 | 13 | 1 268 | 1 207 | 755 |
| B2 CBR-threshold cap | 0.00 | 13 | 1 268 | 1 207 | 755 |
| B3 myopic dual | 0.00 | 13 | 1 268 | 1 207 | 755 |
| B4 CE-MILP | 0.00 | 12 | 1 668 | 1 594 | 768 |
| B5 PPO-Lagrangian | 0.00 | 13 | 1 268 | 1 207 | 757 |

The legacy ranking (SC-RLGMI 44.96 vs B1–B5 28.30–39.91) is a ranking of
first-packet/any-receiver credit, not of delivered payloads. **The RL
superiority claim is not supported.** It cannot be recovered by re-ranking the
same runs; the accounting change removes the quantity the ranking was based on.

**Safety.** The `P_viol` numbers (1.25–12 %) are windowed-PRR small-sample
artifacts (Finding 5); the cumulative PRR is 98.5–99.5 %. The standards metrics
remain valid **as beacon/radio metrics**. They do not license a claim that
computation payloads are delivered.

**Backlog guard.** Track 1's literal inequality admits nothing at this
operating point, but the payload run shows the guard is not the binding
constraint: the queue and the helper geometry are. Redesigning the guard on
"physical radio capacity" before fixing those two interfaces would optimize the
wrong layer.

---

## 11. What remains valid

* The packet-identity/dedup/completion implementation and its tests.
* The standards/beacon metrics (PRR, R_timely, CBR, p95) as **radio** metrics.
* The negative Track 1 result: with atomic payloads and the current capacity,
  no task is admissible under the literal inequality.
* The archived legacy values, **provided they are labelled legacy**.

---

## 12. Options and recommendation

| Option | What changes | What it fixes | Cost / risk |
| :--- | :--- | :--- | :--- |
| **A. Report as-is** | nothing | honest statement that the corrected contract yields 0; no ranking | no model change; claim is negative |
| **B. Pace injection** | inject one unit per reservation (or a bounded rate), not the whole burst | queue overflow; makes the payload physically carryable | changes the radio workload; needs sign-off |
| **C. Dedicated computation queue** | separate FIFO that does not trigger `bufferOverflowLTE` | queue artifact while keeping beacons intact | simulator change; needs sign-off |
| **D. Partial retransmission** | re-inject only missing units | re-injection feedback loop | task-layer/API change; needs sign-off |
| **E. Align helper geometry** | map Python kinematics to the WiLab trace, or select helpers from trace positions | out-of-range helper (0 % PRR) | interface fix; changes candidate sets and therefore results |
| **F. Metric fix** | longer PRR window / minimum samples / `Δg` | the 20 % artifact | measurement only; safe |
| **G. Change the completion contract** | require a fraction of payload, or single-packet tasks | makes the objective carryable | changes reward/workload; needs explicit sign-off |

**Recommendation.** Do **B + E + F**, then re-run the frozen B1–B5 and
SC-RLGMI campaigns under the corrected contract. Option A alone is defensible
but leaves the system unable to score any policy. Do **not** tune the backlog
guard, do **not** change the checkpoint, and do **not** open the sealed seeds
until the interfaces are fixed and the freeze is re-declared.

---

## 13. Reproduction

```bash
# Payload and legacy control (SC-RLGMI, seed 73300, identical config)
python scripts/run_payload_control.py --seed 73300 --eid 98002 --accounting payload
python scripts/run_payload_control.py --seed 73300 --eid 98003 --accounting legacy

# Invariants
python scripts/verify_payload_accounting.py \
  --episode-dir sparc_performance_evaluation/results/raw/payload_control/payload/val_s73300/episode_98002

# Unit tests
pytest -q tests/test_payload_accounting.py
```

The evidence quoted in this document is archived at
`sparc_performance_evaluation/results/raw/backlog_guard/payload_plain/val_s73300/episode_97002/episode_97002`
and `.../payload_legacy/val_s73300/episode_97003/episode_97003`; the script
reproduces the same comparison in a clean directory.

B1–B5 spot-check at the original configuration (density 35, TPS 4, seed 73100,
sim 12 s). All methods admitted traffic and all collapsed to zero:

| Method | `V_done` | `P_viol` | Mean PRR | p95 delay |
| :--- | ---: | ---: | ---: | ---: |
| B1 greedy value-density | 0.00 | 30 % | 0.9661 | 34.7 ms |
| B2 CBR-threshold cap | 0.00 | 30 % | 0.9661 | 34.7 ms |
| B3 myopic dual | 0.00 | 30 % | 0.9661 | 34.7 ms |
| B4 CE-MILP | 0.00 | 23 % | 0.9732 | 36.3 ms |
| B5 PPO-Lagrangian | 0.00 | 29 % | 0.9674 | 34.3 ms |

The `P_viol` values are the windowed-PRR artifact of Finding 5; the cumulative
PRRs are 0.966–0.973. The result confirms the collapse is policy-independent;
it is not a payload-faithful ranking.

---

## 14. Archived legacy values (do not present as payload-faithful)

| Method | Legacy value | Source |
| :--- | ---: | :--- |
| SC-RLGMI (mean, dev seeds) | 44.96 | archived dual-track tables |
| SC-RLGMI (seed 73300) | 38.82 | `sc_rlgmi_training/eval/val_s73300/episode_93001` |
| B1 / B2 / B3 / B4 / B5 | 39.20 / 34.10 / 36.50 / 28.30 / 39.91 | frozen baseline panel |

All of these were computed with first-packet, any-receiver credit. They are
retained for provenance and for reproducing the legacy path
(`accounting="legacy"`), not as delivered-payload utility.

Under the corrected contract the corresponding spot-checks are **0.00** for
SC-RLGMI at seed 73300 and **0.00** for every B1–B5 method at seed 73100. The
full multi-seed payload campaign is deferred until the queue/injection and
helper-geometry interfaces are fixed (Finding 3 and Finding 2).
