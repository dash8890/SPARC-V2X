# B5 method-mapping audit — VLRO (Value-Latency-Resource Offloading)

**Display name:** **VLRO** — Value-Latency-Resource Offloading (code key: `itora_inspired`).
**Not** an exact ITORA reproduction. See `baseline_nomenclature.md`. No authoritative ITORA source PDF is checked into this worktree; the adaptation freezes a literature-typical iterative task-offloading structure onto the shared WiLab task/helper state.

## Mapping

| Original ITORA-style element | Mode-2 / TaskLayer adaptation | Rationale |
|---|---|---|
| Task utility / value | `task_value` \(v_i\) | Shared task value already in `TaskLayer` |
| Resource / delay cost | Predicted finish \(\widehat F\) via shared deadline-feasible finish model | Same \(\widehat F\) as all controllers |
| Communication / payload cost | `payload_bytes` in ranking denominator | Mode-2 offers consume sidelink bytes; no infrastructure RB allocator |
| Infrastructure radio allocation | **Dropped** — autonomous Mode-2 bridge only | Protocol: translate unsupported infrastructure APIs |
| Iterative selection under capacity | Greedy categorical/`argmax` under unique task/source + **fixed byte cap** \(B_{\max}=175\) kB | Validation-selected byte cap (same magnitude as SPARC envelope) **without** SPARC PRR/P95 governor |
| Learning / online value update | **Not used** in this frozen baseline | Keeps B5 a non-learned literature-informed comparator; SPARC retains learned residual |

## Frozen decision rule

For deadline-feasible candidates \((i,h)\):

\[
s^{\mathrm{VLRO}}_{i,h}
=
\frac{v_i}{\widehat F^{\mathrm{rel}}_{i,h}\cdot (b_i/10^4)}
\]

Select greedily (deterministic eval) under unique task, unique source, and byte budget. No SPARC-V2X safety governor; admission is byte-cap only.

## Disclosure

Cite as **VLRO** (Value-Latency-Resource Offloading), Mode-2 adaptation; this audit is provenance. Related value-/delay-aware vehicular offloading (including ITORA) may be referenced as prior art — do **not** claim bit-faithful reproduction of ITORA.
