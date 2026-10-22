# G9 — Endogenous load audit (DCC on, envelope HEAVY, CBR@10–100)

**Seeds:** `72600–72607` (tps4)
**DCC:** default ON (`WILAB_DCC_ACTIVE=0` to disable)
**HEAVY:** greedy envelope fill (no zero-byte dilution)
**Premise supported:** `False`
**mean ΔCBR (heavy−empty, epochs 10–100):** -0.00027301587301588326
**mean ΔPRR:** -0.0019904605517156726
**mean Δpending (1–20):** -3.525000000000002
**mean Δscheduled_attempts (1–20):** 1.612499999999983
**frac CBR up:** 0.5
**frac PRR down:** 1.0

| seed | bytes_heavy | ΔCBR | ΔPRR | Δpending | Δsched | ΔV |
|---:|---:|---:|---:|---:|---:|---:|
| 72600 | 147045 | -0.0023 | -0.0024 | -62.6 | -0.3 | +0.00 |
| 72601 | 148795 | +0.0004 | -0.0034 | +26.7 | +3.3 | +0.00 |
| 72603 | 128300 | +0.0010 | -0.0021 | +23.8 | +4.0 | +0.00 |
| 72607 | 87802 | -0.0002 | -0.0000 | -2.0 | -0.6 | +0.00 |
