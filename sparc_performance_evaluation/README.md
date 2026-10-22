# SPARC-V2X performance evaluation package

**Status: experimental specification.** This package defines the revised Performance Evaluation protocol for the definitive deadline-feasible, byte-governed VA-residual PPO (SPARC-V2X). It does **not** claim that the full campaign (B3–B5, ablations, final-test) has been executed.

## Start here

1. [`protocol/evaluation_protocol.md`](protocol/evaluation_protocol.md) — full protocol (Figs. 3–11, Tables II–IV, gates, stats).
2. [`protocol/controller_contracts.md`](protocol/controller_contracts.md) — fairness / governor / baseline contracts.
3. [`protocol/parameter_manifest.yaml`](protocol/parameter_manifest.yaml) — Table II sources; `UNVERIFIED`/`TBD` left explicit.
4. [`protocol/seed_registry.yaml`](protocol/seed_registry.yaml) — train/validation frozen; final-test sealed empty.
5. [`protocol/experiment_matrix.csv`](protocol/experiment_matrix.csv) — E0–E9 status tracker.
6. [`protocol/protocol_hashes.json`](protocol/protocol_hashes.json) — implementation fingerprints + envelope audit.

## Envelope audit (pre-freeze)

\(B_{\max}\) under `horizon_envelope_only` is a **per-epoch** offered-byte budget (not a cumulative episode budget). See `protocol_hashes.json` and contracts.

## Historical evidence already on disk (not final-test)

- Training: `results/wilab_residual_ppo_96ep_va_relative/`
- Validation bake-off (same val seeds): `results/wilab_bakeoff_residual_va_mf_best/`
- Checkpoint: `checkpoints/wilab_residual_ppo_96ep_va_relative.pt`

These seeds are **consumed**. Final claims require untouched `final_test` seeds after Stages 1–3.

## Execution order

Stage 1 correctness (G0–G9) → Stage 2 validation feasibility → Stage 3 train/ablate → Stage 4 sealed final-test → Stage 5 Figs 3–11 / Tables II–IV.
