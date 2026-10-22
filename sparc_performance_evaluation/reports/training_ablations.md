# Training and ablations log

**Status:** NOT_STARTED (historical SPARC train/val exists; Direct PPO / A3–A6 / multi-seed TBD)

| Config | Train budget | Checkpoint rule | Val seeds | Status |
|---|---|---|---|---|
| A0 SPARC | 96 ep / 24 updates (historical) | best ΔVA on val | 82001/02/11/12 | HISTORICAL |
| A1 safety gates off | frozen A0 ckpt | n/a | | NOT_STARTED |
| A2 ≡ B1 | n/a | n/a | | REUSE |
| A3 no VA-relative | matched | best ΔVA on val | | NOT_STARTED |
| A4 ≡ B3 | matched | ordinary best on val | | NOT_STARTED |
| A5 residual-only logits | matched | best ΔVA on val | | NOT_STARTED |
| A6 random residual init | matched | best ΔVA on val | | NOT_STARTED |
