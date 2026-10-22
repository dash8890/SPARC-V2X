# RHO-SPARC — Replacement-Aware ΔJ Assessment

**Verdict:** `CLOSE_RHO_SURROGATE_BRANCH`
**Credible decision-quality signal:** `False`
**PPO / Phase B / new replay:** NO-GO

Replacement-aware \(\Delta J\) exposes costs that additive \(C_{\mathrm{excl}}\) missed (4/4 Direct packs), and \(J^*\) improves alignment modestly (sign 3/4, top-1 50%, Spearman 0.50). That is still below a credible training/decision-quality bar. Close the surrogate-based RHO branch rather than stacking more penalties.

## Setup

On each existing force-pack state (Direct / LVM / RHO packs at epoch 0):

- Base score \(J(A)\) (coverage-aware).
- Enumerate feasible single **rebind** / **substitute** edits \(e\).
- \(\Delta J(e)=J(A(e))-J(A)\); \(J^\*(A)=\max\{J(A),\max_e J(A(e))\}\).

No new simulator replay. No PPO. Not a new algorithm.

## 1. Does replacement see costs that \(C_{\mathrm{excl}}\) missed?

| Check (Direct packs) | Count |
|---|---:|
| Seeds with \(C_{\mathrm{excl}}=0\) (envelope-blocked adds) | 4/4 |
| Among those, nonzero replacement \(\Delta J\) exists | 4/4 |
| Among those, some **positive** substitute \(\Delta J\) | 3/4 |

## 2. Alignment with realized force-pack outcomes

| Metric | \(J\) | \(J^*\) (best single edit) |
|---|---:|---:|
| Sign agree (RHO−Direct) | 2/4 | 3/4 |
| Top-1 vs realized V | 0.25 | 0.50 |
| Mean Spearman | 0.125 | 0.500 |

| Seed | rank J | rank J* | rank V | best edit type (Direct) | best ΔJ (Direct) |
|---:|---|---|---|---|---:|
| 82001 | RHO>LVM>Direct | Direct>LVM>RHO | Direct>LVM>RHO | SUBSTITUTE | +5.864 |
| 82002 | LVM>Direct>RHO | LVM>Direct>RHO | Direct>LVM>RHO | SUBSTITUTE | +3.617 |
| 82011 | LVM>Direct>RHO | LVM>Direct>RHO | Direct>LVM>RHO | REBIND | +2.766 |
| 82012 | Direct>LVM>RHO | Direct>LVM>RHO | Direct>LVM>RHO | REBIND | +2.736 |

### Per-method edit activity

| Seed | Method | n_edits | #+ΔJ | #−ΔJ | #+sub | best ΔJ |
|---:|---|---:|---:|---:|---:|---:|
| 82001 | Direct | 5 | 5 | 0 | 2 | +5.864 |
| 82001 | LVM | 3 | 3 | 0 | 0 | +4.797 |
| 82001 | RHO | 4 | 0 | 4 | 0 | -0.200 |
| 82002 | Direct | 5 | 5 | 0 | 2 | +3.617 |
| 82002 | LVM | 3 | 3 | 0 | 0 | +0.998 |
| 82002 | RHO | 4 | 0 | 4 | 0 | -1.604 |
| 82011 | Direct | 4 | 4 | 0 | 1 | +2.766 |
| 82011 | LVM | 3 | 3 | 0 | 0 | +2.355 |
| 82011 | RHO | 4 | 0 | 4 | 0 | -1.052 |
| 82012 | Direct | 3 | 3 | 0 | 0 | +2.736 |
| 82012 | LVM | 3 | 3 | 0 | 0 | +2.887 |
| 82012 | RHO | 4 | 0 | 4 | 0 | -2.970 |

## 3. Decision

- **`CLOSE_RHO_SURROGATE_BRANCH`**
- Do **not** stack further corrective penalties or open Phase B / PPO on this surrogate stack.
- Direct PPO remains the SPARC reference.
- Stage 1 and JIPI remain closed.
- Final-test sealed; no manuscript promotion.

Artifacts: `C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0\sparc_performance_evaluation\results\raw\rho_replacement_aware_delta_j`

Related: `RHO_PHASE_A_COVERAGE_OBJECTIVE_RESULTS.md`, `RHO_CORRECTION_GATE_FAIL.md`.
