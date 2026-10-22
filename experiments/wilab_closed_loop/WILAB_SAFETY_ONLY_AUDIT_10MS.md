# WiLab 10-ms safety-service audit

## Configuration

- 20 s extended trace; original 10 s traces unchanged.
- Isolated WiLab application override permits `allocationPeriod=0.01` s.
- `generationInterval=0.01` s.
- Payload: 1000 B; 5G-V2X, MCS 11, SCS 15, same resource settings.
- No computation assignments were admitted during the safety-only warm-up.
- Output: `results/additional_validation_20261007/wilab_safety_audit_10ms/`.

## Results

| metric | result |
|---|---:|
| controller states/results | 100 / 100 |
| first valid controller epoch | 1.000 s |
| online PRR field | 1.000 |
| aggregate WiLab PRR (`MainOut.xls`) | 0.801674 |
| online P95 delay | 10 ms |
| packet-delay CDF P95 | 11 ms |
| packet-delay CDF P99 | 11 ms |
| CBR mean | 0.176986 |
| CBR range | 0.0662--0.9786 |
| safe action set | `{0,1}` at all 100 epochs |
| implemented `K_t=|A_safe|` | 2 |
| epochs with admitted computation | 100 |
| completed value | 46.985869 |

The online state PRR is per-interval bookkeeping. Aggregate WiLab PRR is the
end-to-end safety-service metric used for the frozen safety gate.

## Gate decision

The 10-ms service fixes the previous 100-ms service mismatch and produces a
nonzero shield action set. Delay passes (`P95 <= 50 ms`), but aggregate safety
PRR fails (`0.801674 < 0.95`). The zero-computation WiLab baseline therefore
still does not satisfy the frozen safety criterion, and no WiLab RL training
was started.
