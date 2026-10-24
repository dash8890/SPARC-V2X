# Phase 1 — Evidence freeze and application mapping

Status: **complete for the SC-RLGMI line; one provenance gap in the baseline line (see §6).**
This document fixes what is frozen, what the simulator physically represents, and which
standards quantities the safety evaluation will be mapped onto. It does not change any
baseline, workload, reward, or simulator behaviour.

---

## 1. Frozen artifacts

| Artifact | Path | SHA-256 |
| :--- | :--- | :--- |
| SC-RLGMI advantage checkpoint | `checkpoints/sc_rlgmi_best.pt` | `230ea2a8…92cdaf7` |
| Mobility trace (dev/train density 15) | `traces/mode2_external_validation/synthetic_highway_rho_15_20s.txt` | `88cf8d66…71bdb0f0` |
| Simulator config template | `vendor/WiLabV2Xsim/ConfigFiles/test_trace.cfg` | (template, `rho` overridden per run) |
| Causal guard (currently active) | `controllers/rl/causal_safety_guard.py` | frozen for Track 1 comparison |
| Advantage network architecture | `controllers/rl/sc_rlgmi/sc_rlgmi_core.py` | frozen |

Runtime overrides applied by the evaluation driver (`train_sc_rlgmi.py::run_single_ep` →
`train_wilab_closed_loop.run_episode` → `run_wilab_closed_loop_episode.m`):

```
Technology      5G-V2X            Raw              150
MCS_NR          11                SCS_NR           15
sizeSubchannel  10 PRB            powerThreshold   -126 dBm
beaconSizeBytes 300 B             generation      100 ms
allocationPeriod 20 ms            simulationTime   12 s
controllerEpochs 100 (100 ms each) seed            per-seed
density         15 veh            FixedPdensity    false
```

The run-local `test_trace_runtime.cfg` is written per episode with `[rho] = 15`; the shared
template default (`rho = 35.5`) is never used.

## 2. What the simulator physically represents

| Quantity | Value | Evidence |
| :--- | :--- | :--- |
| Radio access | NR-V2X Mode 2 (sensing-based autonomous selection) | `MainOut.xls` `C-V2X allocation algorithm = Winner+`, `duplex = HD` |
| Duplexing | Half duplex | `MainOut.xls` `Phy (BW,MCS/Mode,Duplex) = 10.0,11,HD` |
| Bandwidth | 10 MHz, SCS 15 kHz | `MainOut.xls` |
| Safety traffic | Periodic awareness message, **300 B every 100 ms per vehicle** | `App (AllocationPeriod,PacketSize) = 0.020,300`; `Phy GenInterval = 0.100` |
| Safety max range | `Raw = 150 m`, distance accounting to `RawMax = 483 m` | `MainOut.xls` `Phy (MaxRange2Sigma = 483)` |
| Computation traffic | Payload-faithful packet stream, **1000 B per packet**, injected into the source vehicle's Mode-2 queue | `computation_packet_bytes = 1000` in `bridge_audit.json` |
| Network size | 15 vehicles, 12 s per episode, 100 control epochs | run config |

The safety workload is therefore a **10 Hz cooperative-awareness service with 300 B payloads**,
not a pre-crash or platooning service. That distinction determines which standard applies
(§4) and is the reason a universal "2 % of windows may fail" ceiling is not a
standards-derived quantity.

## 3. Metrics (implemented in `scripts/evaluate_standards_safety.py`)

Packet-level, standards-aligned:

```
R_timely(D_max) = Σ_i  # safety packets decoded with latency ≤ D_max in window i
                  ─────────────────────────────────────────────────────────────
                  Σ_i  # decode opportunities in window i        (rx_i + err_i)
```

with `D_max ∈ {20, 50, 100} ms` reported. Also emitted:

* cumulative packet PRR (`rx / (rx + err)`),
* per-window PRR and window p95 delay (1 ms bins, differenced from cumulative histograms),
* PRR versus distance in bands 0–50 / 50–150 / 150–300 / 300–500 m,
* longest run of consecutive loss windows (explicit window-level proxy for packet
  inter-reception interval; WiLab does not export per-packet receive timestamps),
* the legacy metric unchanged: `P_window = Pr[PRR_100ms < 0.95 ∨ P95 > 50 ms]`.

## 4. Standards mapping

The mapping below is the one supplied by the standards review; the numeric targets are
treated as **external inputs** to the evaluation, not re-derived here (the normative PDFs
could not be re-fetched in this environment — see §6).

| Source | Requirement | Maps to |
| :--- | :--- | :--- |
| 3GPP TS 22.185 | Basic V2X: 100 ms V2V latency; pre-crash sensing 20 ms | `R_timely(100 ms)` for the 10 Hz awareness workload |
| 3GPP TS 22.186 | Enhanced V2X / cooperative collision avoidance: 10 ms, 99.99 % reliability | Reference point for why a generic 2 % window ceiling is not a standard |
| 3GPP TR 37.885 | PRR and distance-dependent evaluation methodology | PRR-vs-distance bands |
| NR-V2X Mode 2 system-level literature | Report PRR, inter-reception interval, throughput distributions | PRR distribution, loss-run statistics |

Consequences for the claim structure:

1. The **primary** safety statement will be packet-level reliability and latency for the
   represented 10 Hz awareness service, with the evaluation range stated explicitly.
2. The legacy window metric and its 2 % ceiling are retained as a **secondary, historical**
   experimental metric for comparability with the frozen B1–B5 table.
3. No 2 % (or 0 %) ceiling is re-interpreted as standards-mandated.

## 5. What Track 1 is allowed to change

| Component | Decision |
| :--- | :--- |
| B1–B5 implementations | frozen, unmodified |
| Workload, mobility, radio configuration | frozen |
| Completion-only reward | unchanged |
| Advantage network `θ_RL` | frozen for the guard comparison |
| Beam search | unchanged |
| Guard | extended with backlog awareness (new variant, previous variant preserved) |

## 6. Provenance gaps found during the freeze

These block parts of Track 2 and must be resolved before the final manuscript.

1. **The frozen B1–B5 numbers are not present as artifacts in this repository.**
   The comparison table cites `B1 = 39.20, B2 = 34.10, B3 = 36.50, B4 = 28.30, B5 = 39.91`
   with `P_viol = 0/0/0/48.5/25 %`, but no result directory under
   `sparc_performance_evaluation/results` contains those values, and the only development
   evaluations on seeds 73300–73303 belong to `scql_training`, `sc_rlgmi_training`,
   `Candidate_C_Surgical`, and `Candidate_D_RiskCalibrated`. The numbers are therefore not
   yet reproducible from the frozen evidence. Options: recover the original campaign
   directory, or re-run the frozen baselines unchanged on the dev seeds.
2. **Distance-band PRR must not include WiLab's `blocked` column.** `mainCV2XttiEnds.m:143`
   increments the blocked counter by *all neighbours* of a half-duplex-blocked receiver, not by
   the single missed pair, so the column is neighbour-multiplicity inflated (dev seed 73300:
   5 188 blocked vs 2 643 correct). The bridge export's `correct / (correct + errors)` is the
   defensible distance-banded analogue of the attempt-based headline PRR; the `blocked` column
   is reported for completeness only and never used as a reliability denominator.
3. **Audit epochs do not record the simulator seed.** Seeds are recovered from the
   `val_s<seed>` directory convention; new runs record the seed explicitly.
4. **No standards PDF could be re-fetched from this environment**, so the numeric targets in
   §4 are inherited from the supplied review. They must be pinned to a specific spec version
   before publication.

## 7. Immediate baseline measurement (SC-RLGMI line only)

`sparc_performance_evaluation/results/aggregates/standards_safety/STANDARDS_SAFETY_REPORT.md`
already reports, for the existing dev runs:

| Run | Opportunities | Packet PRR | R_timely 100 ms | Legacy P_viol |
| :--- | ---: | ---: | ---: | ---: |
| SC-RLGMI (Option A) | 4724 | 99.79 % | 99.79 % | 1.25 % |
| Candidate C (surgical throttle) | 4599 | 99.91 % | 99.91 % | 0.50 % |
| Candidate D (risk-calibrated) | 4779 | 99.71 % | 99.71 % | 1.75 % |

All observed safety-packet delays are below 50 ms, so `R_timely(50 ms) = R_timely(100 ms)`:
the residual failures are **losses, not late deliveries**.
