# RHO-SPARC — Surrogate Objective Audit

**Correction-gate verdict:** FAIL (retained)
**PPO training:** NO-GO
**Scope:** Existing traces/calculations only — no new replay, no training.

Stage 1 and JIPI remain closed. Direct remains the validated SPARC reference. Final-test sealed.

## 1. Horizon compatibility

| Quantity | What it measures | Typical scale |
|---|---|---:|
| Predicted E[V] | One epoch's pack \(\sum v_i\widehat p_i\) | ~4–8 |
| Realized \(V_{\mathrm{done}}\) | Full-episode completed value | ~35–59 |

**Magnitudes are not comparable.** Ranking vs episode \(V\) is only interpretable as an intervention test (force pack at epoch 0 → episode outcome), not as calibration of absolute scores.

Compatible check — sign of \(\Delta E\) (RHO−Direct) vs \(\Delta V\) (RHO−Direct) under the same force:

- \(\Delta E\): [5.522, 2.265, 4.754, 0.289]
- \(\Delta V\): [-1.58, -3.547, -2.647, 0.0]
- Sign agreement: **0/4** (0.00)

Predicted gains for RHO do **not** align in sign with realized episode losses.

## 2. LVM mean reconciliation (45.66 vs locked 46.82)

| Protocol | Mean \(V\) |
|---|---:|
| Locked matched LVM (full policy every epoch) | **46.8173** |
| Ranking replay: LVM pack @ epoch 0, then Direct | **45.6621** |

Locked 46.82 is the full-episode LVM policy (every epoch). Ranking headline 45.66 forces only the Stage-0 LVM epoch-0 pack, then continues with Direct. Different execution condition — not a corruption of the locked matched reference.

| Seed | Locked full LVM | Force LVM@0 + Direct | Gap |
|---:|---:|---:|---:|
| 82001 | 38.2612 | 38.2612 | +0.0000 |
| 82002 | 50.8360 | 49.9532 | +0.8828 |
| 82011 | 58.6936 | 58.6936 | +0.0000 |
| 82012 | 39.4785 | 35.7404 | +3.7381 |

## 3. Why the surrogate misranks packs

| Method | mean n | mean bytes | mean \(\widehat p\) |
|---|---:|---:|---:|
| Direct | 3.50 | 149237 | 0.555 |
| LVM | 4.25 | 152236 | 0.509 |
| RHO | 2.00 | 73121 | 0.968 |

Surrogate rewards high mean p̂ on small packs: RHO assigns fewer tasks/bytes than Direct/LVM, inflating E[V] while reducing epoch-0 admission coverage. Lost Direct tasks' nominal value is not charged unless they appear as pack-internal queue infeasibility.

### Per-seed coverage vs Direct

| Seed | RHO n | Direct n | Lost task IDs (nominal value) | Gained | \(\Delta E\) | \(\Delta V\) |
|---:|---:|---:|---|---|---:|---:|
| 82001 | 2 | 3 | [0, 18] (3.64) | [12] (4.86) | +5.52 | -1.58 |
| 82002 | 2 | 3 | [9, 15] (3.73) | [18] (4.32) | +2.26 | -3.55 |
| 82011 | 2 | 4 | [3, 12, 15] (4.73) | [0] (4.92) | +4.75 | -2.65 |
| 82012 | 2 | 4 | [15, 18] (5.04) | [] (0.00) | +0.29 | +0.00 |

### Mechanism classes

1. **Admission / coverage:** RHO's high-\(\widehat p\) small packs omit tasks Direct completes later; surrogate does not credit multi-epoch coverage.
2. **Helper-queue interactions:** Pack-internal projection can look safer with fewer concurrent assignments even when that underfills the envelope.
3. **Downstream deadlines:** Epoch-0 force changes later Direct decisions; episode \(V\) includes those effects, while \(E\) only scores the forced pack.

## 4. Decision

- Current \(\widehat p\) / pack objective: **rejected for PPO training**.
- RHO joint-search formulation: retained as an **unvalidated research candidate** (latency OK).
- Next work, if any, must redesign or replace the surrogate so that compatible-horizon rankings agree with realized intervention outcomes — still without PPO until that gate passes.

No architecture expansion toward RL training, manuscript promotion, or final-test access.
