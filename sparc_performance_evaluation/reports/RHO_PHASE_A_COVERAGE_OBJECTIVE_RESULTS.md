# RHO-SPARC — Phase A Coverage-Aware Objective Results

**Verdict:** `PHASE_A_PARTIAL_IMPROVEMENT`
**PPO training:** NO-GO
**Scope:** Re-score existing force-pack traces only — no new replay.

## 1. Paired Δ-sign agreement (RHO − Direct)

| Surrogate | Sign agreement with realized ΔV |
|---|---:|
| Interim \(E\) / score | **0/4** |
| Revised \(J\) | **2/4** |

| Seed | ΔE | ΔJ | ΔV | E agree | J agree |
|---:|---:|---:|---:|:---:|:---:|
| 82001 | +5.522 | +1.567 | -1.580 | N | N |
| 82002 | +2.265 | -7.954 | -3.547 | N | Y |
| 82011 | +4.754 | -0.498 | -2.647 | N | Y |
| 82012 | +0.289 | -3.997 | +0.000 | N | N |

## 2. Ranking vs realized outcomes

| Metric | Interim E | Revised J |
|---|---:|---:|
| Top-1 agreement | 0.00 | 0.25 |
| Mean Spearman vs V | 0.250 | 0.125 |

| Seed | rank E | rank J | rank V |
|---:|---|---|---|
| 82001 | RHO>LVM>Direct | RHO>LVM>Direct | Direct>LVM>RHO |
| 82002 | LVM>RHO>Direct | LVM>Direct>RHO | Direct>LVM>RHO |
| 82011 | LVM>RHO>Direct | LVM>Direct>RHO | Direct>LVM>RHO |
| 82012 | RHO>Direct>LVM | Direct>LVM>RHO | Direct>LVM>RHO |

## 3. Term separation (double-counting check)

- \(C_{\mathrm{excl}}\): uncovered eligible tasks × \(v\cdot\widehat p^{alt}\) only (no bytes).
- \(C_B\): included-pack bytes only.
- \(C_{\mathrm{load}}\): helper-load term + pending-risk (queued tasks harmed).

Mean Spearman(n, \(C_{\mathrm{excl}}\)) across methods within seed: **-0.500** (expected negative if coverage works).

| Seed | Method | n | V_in | C_excl | C_B | C_load | J | V_realized |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 82001 | Direct | 3 | 2.910 | 0.000 | 0.980 | 0.169 | 1.761 | 39.841 |
| 82001 | LVM | 4 | 3.951 | 0.000 | 0.968 | 0.370 | 2.612 | 38.261 |
| 82001 | RHO | 2 | 8.432 | 4.514 | 0.526 | 0.063 | 3.328 | 38.261 |
| 82002 | Direct | 3 | 6.615 | 0.000 | 0.798 | 0.252 | 5.564 | 49.953 |
| 82002 | LVM | 4 | 12.167 | 0.000 | 0.972 | 0.358 | 10.838 | 49.953 |
| 82002 | RHO | 2 | 8.880 | 10.850 | 0.355 | 0.065 | -2.390 | 46.407 |
| 82011 | Direct | 4 | 2.900 | 0.000 | 0.977 | 0.239 | 1.684 | 58.694 |
| 82011 | LVM | 5 | 8.095 | 0.000 | 0.884 | 0.356 | 6.855 | 58.694 |
| 82011 | RHO | 2 | 7.654 | 5.989 | 0.428 | 0.051 | 1.185 | 56.047 |
| 82012 | Direct | 4 | 6.292 | 0.000 | 0.655 | 0.356 | 5.280 | 35.740 |
| 82012 | LVM | 4 | 6.141 | 0.000 | 0.655 | 0.356 | 5.129 | 35.740 |
| 82012 | RHO | 2 | 6.581 | 4.796 | 0.363 | 0.140 | 1.283 | 35.740 |

## 4. Pack-size confounding

Fraction of seeds where \(J\)-winner has largest n: **0.75**

Member marginal \(\Delta J\) (assignment vs pack without it) — fraction positive:

| Seed | Direct | LVM | RHO |
|---:|---:|---:|---:|
| 82001 | 1.00 | 1.00 | 1.00 |
| 82002 | 1.00 | 1.00 | 1.00 |
| 82011 | 1.00 | 1.00 | 1.00 |
| 82012 | 1.00 | 1.00 | 1.00 |

## 5. Decision

- Phase A verdict: `PHASE_A_PARTIAL_IMPROVEMENT`
- Phase B (new replay): **not authorized** by this result alone unless PASS.
- PPO training: **NO-GO**
- Direct remains SPARC reference; Stage 1 / JIPI closed; RHO unvalidated.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\rho_phase_a_coverage_objective`
