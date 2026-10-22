# G9 root cause — why CBR/PRR premise failed (and the fix)

**Date:** 2026-10-19

## What G9 showed (first pass)

Causal EMPTY vs HEAVY @ epoch 0 (tps4): mean ΔCBR ≈ 0.001; **ep1 CBR often identical** for EMPTY and HEAVY even when ~170 kB was offered. Half the seeds had `bytes_heavy=0`.

## Why (code + measurement)

1. **`dcc_active` was hard-coded `false`** — CBR sensed, DCC closed-loop path off.
2. **CBR sensing interval = 0.1 s** vs epoch = 0.01 s — averaging epochs 1–20 is mostly pre-sensing / safety-dominated.
3. **Zero-byte HEAVY dilution** — LVM epoch-0 capture returned empty packs on 4/8 seeds.
4. **Mode-2 duty cycle** — computation deepens `pckBuffer`; CBR only moves if backlog raises TX occupancy vs CAM-only. Short window + DCC-off hid that.

## Fix (applied — not a new stop gate)

| Change | Where |
|--------|--------|
| DCC **default ON** (`WILAB_DCC_ACTIVE=0` to disable) | `run_wilab_closed_loop_episode.m` |
| HEAVY = greedy envelope fill from epoch-0 candidates | `run_g9_endogenous_load_audit.py` |
| CBR/PRR window = epochs **10–100**; log Δpending / Δsched | same |
| Drop zero-byte pairs from premise mean | same |

Re-audit seeds `72600–72607` → update `SPARC_G9_ENDOGENOUS_LOAD_AUDIT.md`.

## Decision after re-audit (DONE)

**FAIL.** DCC ON + envelope HEAVY + CBR@10–100: mean ΔCBR ≈ −0.0003, mean ΔPRR ≈ −0.002 (not < −0.01). Premise unsupported.

**Close** radio-endogenous EC-SPO on this WiLab stack. Do not paper CBR feedback that the sim does not show.
