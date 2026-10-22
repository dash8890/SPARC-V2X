# WiLab safety-only audit after trace extension

## Run

- Trace: `traces/mode2_external_validation/synthetic_highway_rho_50_20s.txt`
- Trace duration: 20.0 s; the original 10 s trace is retained unchanged.
- Controller horizon: 100 epochs after safety-only warm-up.
- Output: `results/additional_validation_20261007/wilab_safety_audit_20s_v4/`.

## Result

The 20 s trace completes without the earlier splice error. The bridge emits
100 states and 100 matching results. The first controller state occurs at
`t=1.000 s`; all controller epochs have valid CBR, PRR, and delay fields.
The final-result flush was fixed so epoch 99 also has a corresponding result.

| quantity | result |
|---|---:|
| valid controller states | 100/100 |
| matching results | 100/100 |
| P95 delay | 99 ms for all epochs |
| CBR | mean 0.023432; min 0; max 0.212 |
| PRR | 1.0 in emitted states |
| safe action set | `{0}` for every epoch |
| effective computation budget | 0 for every epoch |
| completed computation value | 0 |

The no-computation outcome is produced by the actual shield inputs, not the
old constant-budget path.

## Service semantics

The runner passes `allocationPeriod=0.1` s and `generationInterval=0.1` s.
WiLab documents that `generationInterval` defaults to `allocationPeriod`, and
its validation accepts allocation periods only in the `0.1:0.1:1` s range.
This run therefore generates one safety packet per vehicle per 100 ms, rather
than the manuscript's 10 ms service.

The delay extractor is packet generation-to-reception delay: WiLab's
`countPacketDelay.m` computes `subframeNow - subframeLastPacket`, and the
bridge converts its histogram using `delayResolution=1 ms`. The observed
99 ms P95 is consistent with the configured 100 ms generation period.

## Gate decision

Warm-up and bridge validity: **PASS**. Frozen shield feasibility: **FAIL**.
The shield requires P95 delay <= 50 ms, while safety-only WiLab produces 99 ms
before computation traffic is admitted. The safe set is therefore `{0}`.
No WiLab-trained SPARC experiment was started from this run.

The remaining blocker is configuration equivalence: implement the intended
10 ms safety service in WiLab without changing the shield, or explicitly
recalibrate a WiLab-native safety operating point.
