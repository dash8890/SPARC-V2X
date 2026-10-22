# Packing-pressure LVM gap gate (fresh seeds)

**Overall:** `FAIL`
**Problem:** density=35, tasks_per_source=4
**Seeds:** `72300`… (`n=8` complete)

## Bars (predeclared)

- mean(best−LVM) ≥ 1.0: **-2.0538174363833135** → `FAIL` (with frac)
- frac(best≥LVM) ≥ 0.75: **0.375**
- max mean P_viol ≤ 0.02: **0.09** → `FAIL`

## Means

- LVM: 51.27181597928735
- Direct: 48.63011409390528
- POPL: 43.321119664799625
- mean(POPL−LVM): -7.950696314487714
- mean(Direct−LVM): -2.6417018853820706
- mean(best−LVM): -2.0538174363833135

## Per seed

| seed | LVM | Direct | POPL | ΔPOPL | ΔDirect | Δbest |
|---:|---:|---:|---:|---:|---:|---:|
| 72300 | 60.18 | 52.93 | 43.17 | -17.01 | -7.25 | -7.25 |
| 72301 | 61.23 | 64.45 | 54.64 | -6.60 | +3.22 | +3.22 |
| 72302 | 23.99 | 21.72 | 17.48 | -6.51 | -2.28 | -2.28 |
| 72303 | 67.11 | 61.61 | 53.65 | -13.46 | -5.50 | -5.50 |
| 72304 | 56.00 | 43.23 | 47.93 | -8.06 | -12.77 | -8.06 |
| 72305 | 31.72 | 38.18 | 31.72 | +0.00 | +6.47 | +6.47 |
| 72306 | 51.93 | 48.91 | 44.02 | -7.92 | -3.02 | -3.02 |
| 72307 | 58.00 | 58.00 | 53.97 | -4.04 | +0.00 | +0.00 |

No training. Locked val unused for fitting. Final-test sealed.
Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\packing_pressure_gate`
