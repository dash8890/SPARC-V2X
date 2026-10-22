# Main claim evidence audit

**Central question:** Does definitive SPARC-V2X yield a statistically supported increase in deadline-valid completed value over the strongest safety-feasible competitor on held-out Mode-2 final-test scenarios?

**Status:** FINAL-TEST NOMINAL (5 seeds) COMPLETE for VA/MF/SPARC — full six-method + lock blockers still open

## Seed partition

See `protocol/SEED_PARTITION.md`: 1 train · 4 validation · **5 final-test** · reuse calibration.

## Validation-only evidence (consumed seeds; not final)

Source: `tables/table_iii_comparison/table_iii_validation.*` (n=4 validation pairs).

## Final-test nominal

- Seal: `protocol/final_test_seeds.json` (**n=5**, sim `93000`–`93004`)
- Rows: `results/raw/final_test_nominal_d35/rows.json` (VA / MF / residual)
- Aggregates: run `scripts/postprocess_final_test.py`

## Missing for lock

B3–B5 on the same 5 seeds; ablations; sensitivity sweeps; simultaneous five-contrast inference.
