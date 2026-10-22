# Baseline nomenclature (Stage-5 / manuscript)

**Effective:** 2026-10-16. Display names for papers and evidence reports.
Internal eval/method keys (`value_aware`, `minfinish`, `residual`+`--logit-mode direct`, `lagrangian_*`, `itora_inspired`) stay unchanged for artifact compatibility.

| Code / legacy label | Full name | Acronym | Principle |
|---|---|---|---|
| `value_aware` / VA / B1 | Value-Aware Scheduling | **VAS** | ranks/selects primarily by task value |
| `minfinish` / MF / B2 | Latency-Aware Scheduling | **LAS** | favors lower predicted completion latency |
| Direct PPO / B3 (`residual` + direct logits) | SPARC-V2X (proposed) | **SPARC-V2X** | learned sequential task–helper scheduling (PPO) |
| B4 Lagrangian | Lagrangian Value Maximization | **LVM** | learned value-oriented policy with dual/soft constraints |
| `itora_inspired` / B5 | Value-Latency-Resource Offloading | **VLRO** | non-learning; score \(v/(F\cdot b)\) |

## Why VLRO (not VLAO / not ITORA)

Frozen B5 rule (`train_wilab_closed_loop.py` / `b5_itora_mapping_audit.md`):

\[
s = \frac{v_i}{\widehat{F}^{\mathrm{rel}}_{i,h}\cdot (b_i/10^4)}
\]

Terms are **value**, **predicted finish latency**, and **payload (Mode-2 resource)**. Energy/interference are not in the score. Naming after the ITORA paper would over-claim reproduction; **VLRO** names what this implementation does. Prior value-/delay-aware vehicular offloading (including ITORA) may be cited as related work, not as “this baseline is ITORA.”

## Table III (principle view)

| Method | Principle | Learning | Constraint handling |
|---|---|---:|---|
| **VAS** | task-value-aware scheduling | No | common safety-constrained admission |
| **LAS** | predicted-latency-aware scheduling | No | common safety-constrained admission |
| **SPARC-V2X** | learned sequential task–helper scheduling | PPO | common safety-constrained admission |
| **LVM** | Lagrangian value maximization | PPO | dual/soft constraint handling |
| **VLRO** | joint value–latency–resource offloading | No | unconstrained / external admission (byte cap; no SPARC governor) |

## Usage rules

- In prose and tables for humans: prefer **VAS / LAS / SPARC-V2X / LVM / VLRO**.
- Proposed method: always **SPARC-V2X** (not bare “Direct PPO”) unless discussing the training recipe.
- Excluded residual-family RL remains “SPARC residual RL (excluded)” — not SPARC-V2X.
- Artifact folders and `--methods` flags keep legacy code keys.
