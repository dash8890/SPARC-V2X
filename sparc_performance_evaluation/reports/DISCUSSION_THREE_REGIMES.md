# Discussion — three operating regimes

Companion to Stage-5 evidence (`STAGE5_NO_RESIDUAL_ANALYSIS.md`). Core experimental figures end at **Fig.9**.

## Fig.8 — Candidate-set sensitivity

\(R_{\mathrm{value}}\) vs \(k\in\{8,16,32,64\}\) for SPARC-V2X and VAS.
Peak at \(k=32\) (SPARC \(R=0.6078\)). No runtime/latency claims (not logged).

## Fig.9 — Density-50 adaptation (final experimental figure)

| variant | \(R_{\mathrm{value}}\) | \(P_{\mathrm{viol}}\) |
|---|---:|---:|
| SPARC-N | 0.3482 | 0.40 |
| SPARC-S50 | 0.3765 | 0.40 |
| VLRO | 0.4409 | 0.40 |

Specialization recovers some value; it does not remove the high-density limit.

## Regimes

1. **Low-to-moderate density (≈20–35).** SPARC-V2X strongest among practically comparable schedulers; also exceeds VLRO.
2. **Transition (≈40–45).** SPARC-V2X and VLRO become close; physical environment increasingly binds.
3. **High density (≥50).** VLRO and LVM extract more raw computation value; SPARC-V2X generally preserves better safety-message PRR/P95. Fig.9 specialization does not erase the feasibility boundary.
