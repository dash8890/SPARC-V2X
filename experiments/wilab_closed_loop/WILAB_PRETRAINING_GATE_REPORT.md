# WiLab closed-loop pre-training gate

Audit performed against the isolated MATLAB/Python bridge after the reward-path fix.

## Checks that pass

- MATLAB/WiLab starts and terminates normally.
- Every emitted state has exactly one action and one result.
- Smoke trace: 100 state files and 100 result files; epoch IDs match exactly.
- Application packets are delivered: 99/100 result intervals contain at least one delivered source.
- Realized task reward is nonzero: smoke completed value = 52.8149.
- The Python-to-WiLab action path changes application packet admission.

## Checks that do not pass yet

- `safe_budget` is fixed at 4 for every epoch; the estimator-derived safe action set is not connected to the bridge.
- Initial CBR samples are unavailable in WiLab (`-1` sentinel); valid samples occur only after CBR sensing starts. The bridge now filters the sentinel and reports `cbr_valid`.
- P95 delay is now extracted from WiLab's packet-delay histogram, but it is unavailable until delay observations exist; the bridge reports `delay_valid`.
- The bridge does not snapshot and replay the same WiLab pre-decision state for sampled-versus-anchor counterfactuals.

The shield-wiring smoke additionally showed 0/100 epochs with both CBR and delay
measurements valid. With the deterministic startup-withholding rule, this
correctly admitted no computation and produced zero reward. This is a genuine
initialization deadlock: the current isolated WiLab configuration has no valid
safety measurement before application computation is admitted.

## Consequence

The bridge is mechanically functional and produces real task rewards when the
legacy fixed budget is used, but it has not passed the scientific pre-training
gate for shielded WiLab learning. The existing 100- and 150-episode runs are
preserved as bridge integration runs; they must not be presented as validated
shielded WiLab training results until the safe-action mask, a non-deadlocking
measurement initialization, and paired counterfactual replay are connected.

## Warm-up smoke follow-up

The one-second safety-only warm-up was added and the controller horizon then
ran only after validity became true. In the current 10-second trace, validity
appeared at approximately 9.9 s, leaving 10 controller epochs. All 10 had
valid CBR/PRR/P95 inputs, but WiLab P95 delay was 97 ms, above the frozen
50 ms shield limit. The safe set therefore contained only the no-computation
fallback and the effective admitted computation count was zero. This resolves
the startup deadlock but does not pass the gate: the remaining issue is the
radio-metric/threshold compatibility and the trace duration, not fabricated
startup values.

## 10-ms service audit

An isolated application-parameter override now permits the manuscript-equivalent
`allocationPeriod=0.01 s` and `generationInterval=0.01 s`; canonical WiLab
files were not changed. The 20 s safety-only run produced 100 matched states
and results, first valid at 1.000 s, with online P95 10 ms and packet-delay-CDF
P95/P99 11/11 ms. CBR ranged 0.0662--0.9786 (mean 0.176986), and the shield
exposed `{0,1}` with `K_t=2` at every epoch; the smoke completed value was
46.985869.

Aggregate WiLab PRR from `MainOut.xls` was 0.801674, below the frozen 0.95
requirement. The 100-ms service mismatch is fixed, but the zero-computation
WiLab baseline still fails the safety gate. No WiLab RL training was started.

## Extended-trace safety-only audit

The original 10 s traces were extended to 20 s without changing the originals.
A 100-epoch safety-only smoke now starts at 1.000 s with valid CBR, PRR, and
P95 inputs at every epoch; 100 states and 100 results match exactly. The
final pending result is flushed before the MATLAB event loop exits.

The corrected extractor reads the latest valid CBR history sample instead of
the terminal `-1` sentinel. WiLab safety-only P95 is 99 ms for all 100 epochs,
with CBR mean 0.023432 (range 0--0.212), PRR 1.0, and safe action set `{0}`.
This is genuine frozen-shield infeasibility, not a warm-up artifact.

The runner explicitly configures `allocationPeriod=0.1 s` and
`generationInterval=0.1 s`; WiLab validates allocation periods only in the
0.1--1 s range. Its delay histogram is generation-to-reception packet delay
(`countPacketDelay.m`) at 1 ms resolution. The approximately 99 ms P95 is
consistent with the configured 100 ms safety-packet service. No WiLab RL
training was launched.
