# PB-CRL v3.1 — C2 / WiLab map (foundations)

**Status:** instrumented in code; not yet calibrated on a multi-seed WiLab panel.  
**Module:** `controllers/rl/pb_crl_c2.py`

## Definitions

| Symbol | Meaning in WiLab |
|--------|------------------|
| Valid window | `bridge_audit` epoch with `cbr_valid` and `delay_valid` |
| Violation \(z_t=1\) | valid window with PRR < 0.95 or p95 delay > 50 ms (same as suite) |
| \(Z\) | \(\sum z_t\) over valid windows in the episode |
| \(P_{\mathrm{viol}}\) | \(Z / n_{\mathrm{valid}}\) (panel rate constraint) |
| Pathwise C2 | \(Z \le K\) on an episode; cell-feasible if CP UB on \(P(Z>K) \le \delta\) |

## Defaults (to calibrate)

| Knob | Default | Note |
|------|---------|------|
| \(K\) | 2 | For ~100-epoch / 12 s runs; also report \(K_{\alpha}=\lfloor\alpha n_{\mathrm{valid}}\rfloor\) |
| \(\alpha\) | 0.02 | Suite rate (panel) |
| \(\delta\) | 0.05 | Cell feasibility UB threshold (tune on val) |

## Code API

- `episode_c2_from_audit(audit, K=…)` → `Z`, `C2_ok`, `P_viol`, …
- `c2_cell_summary(rows, K, delta)` → CP upper bound, `feasible_cell`

## Next calibration step

Re-score existing panel / B5′ rollouts with `episode_c2_from_audit` and choose \(K\) so that B5′ feasibility rate under C2 is non-degenerate (neither always fail nor always pass). Document chosen \(K\) here before Phase fine-tune.
