# RHO-SPARC — Surrogate Development Branch Closure

**Verdict:** `CLOSE_RHO_SURROGATE_BRANCH`

Replacement-aware \(\Delta J\) **does** expose opportunity costs that additive \(C_{\mathrm{excl}}\) missed (4/4 Direct packs with \(C_{\mathrm{excl}}=0\) still have nonzero replacement \(\Delta J\); 3/4 have a positive substitute). Aligning with realized force-pack \(V\) remains inadequate for a training gate (\(J^*\): sign 3/4, top-1 50%, Spearman 0.50). That is not a license to stack further penalties or open Phase B / PPO.

## What was tried

1. Interim pack expected-value objective — FAIL (sparse high-\(\widehat p\) packs).
2. Search latency fix — PASS (≤10 ms).
3. Coverage-aware \(J\) with \(C_{\mathrm{excl}}\) — Phase A PARTIAL (2/4 sign).
4. Replacement-aware \(\Delta J(e)\) on existing traces — no credible decision-quality signal.

## Status

| Item | Status |
|---|---|
| RHO surrogate / PPO path | **Closed** (unvalidated; do not train) |
| RHO joint-search *idea* | Closed with surrogate stack; not continued |
| Next design hypothesis | **POPL-SPARC** (paired realized outcomes) — see `POPL_SPARC_PAIRED_ROLLOUT_FEASIBILITY_SPEC.md` (spec only; no training) |
| Direct PPO | Validated SPARC reference |
| Stage 1 / JIPI | Closed |
| Final-test | Sealed |
| Manuscript promotion of RHO | Prohibited |

No further corrective-penalty iterations on this surrogate stack.
