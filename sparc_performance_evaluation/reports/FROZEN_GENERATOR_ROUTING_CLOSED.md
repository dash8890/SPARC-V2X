# Frozen-Generator Routing — CLOSED

**Date:** 2026-10-19  
**Decision:** Stop. No more Direct/POPL/LVM routers, pressure gates, or selector retunes on frozen checkpoints.

## Verdict

Routing among frozen generators **cannot** deliver a flat-out win on this panel. Continuing that loop is waste.

| Attempt | Result |
|---|---|
| RAO logistic selector | G4 FAIL (CLOSED) |
| MDP W-advantage (Phase 2d) | FAIL → CA-PPO CLOSED |
| Pressure-gate v1 (Direct↔POPL) | FAIL — ≥ best on **1/7** conditions; mean Δ vs Direct/LVM/POPL all negative |
| Pressure-gate v2 (Direct/POPL/LVM) | **Aborted** — same mechanism class; not authorized to burn another full panel |

## What the audits already established

**Success regimes (POPL):** `workload_tps4` (all 4 seeds); partial `density_35` / `deadline_1.5`. Mechanism = pack admission + co-scheduling companions, not helper-only flips.

**Failure regimes:** `deadline_0.5`, `workload_tps2`, `helper_homog`, `helper_heterog`. LVM or Direct wins; POPL omits valuable early tasks or mis-binds helpers.

**Headroom:** Direct∪POPL oracle ≈ +1.9% vs Direct; ~69% disagreeing packs have ΔG=0. Not enough for panel-wide dominance by switching frozen packs.

## What is NOT next

- Another threshold / third-policy / “smarter gate” on the same four val seeds  
- More MATLAB panels to re-prove selector failure  
- Reopening RAO or Phase-3 W-training without a new program lock

## What would be a real next program (only if you start one)

A **new** training protocol with a fresh train/dev split (not 82001/02/11/12 fitting), predeclared success = beat best matched baseline on the robustness panel with allowed ties/defensible losses — not another frozen-pack router.

Until you explicitly authorize that new program, **agent stops**.

SPARC reference remains **Direct PPO**. POPL-P5 stays a frozen experimental candidate.
