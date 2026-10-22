# Controller contracts (SPARC-V2X performance evaluation)

**Status:** specification. Implementations must match this contract and the frozen closed-loop code under `experiments/wilab_closed_loop/` and `controllers/rl/`.

## Shared radio / task contract

- Native WiLabV2Xsim NR-V2X Mode 2 only; no analytic radio shortcut in any baseline.
- Application decisions are (task, helper) offers only — never physical subchannels.
- Shared `TaskLayer` candidate generation: released, unfinished, not queued, deadline-feasible under
  \(\widehat F_{i,h,t}=\max\{(t+1)\Delta,\tau_h\}+c_i/\mu_h\) with \(\Delta\) matching the implemented epoch step (10 ms application clock in `task_layer.py`).
- Unique task and unique source per epoch for all B1–B3 and SPARC/ablations that use the common governor path.
- Value earned only after radio delivery **and** finish within absolute deadline and horizon.

## Envelope and governor (`horizon_envelope_only`)

| Rule | Implementation fact |
|---|---|
| Byte budget scope | **Per epoch.** `sample_action(..., forced_byte_budget=B_max)` resets remaining bytes each epoch; governor checks each proposal against `max_bytes`. Not a cumulative episode budget. |
| `horizon_envelope_only=True` | Skips next-window risk-certificate gate; still requires valid CBR/delay measurements, PRR ≥ floor, P95 ≤ ceiling, and byte bound. |
| Blocked safety gate | Empty executed set (no computation). |
| Malformed / over-budget proposal | Deterministic deadline-feasible VA greedy fallback under the same envelope and unique task/source. |
| Invalid vs violation | Log missing/invalid measurements separately from PRR-floor and P95-ceiling failures. |

Defaults in `SafetyGovernor`: `prr_floor=0.95`, `p95_delay_ceiling_ms=50.0`, `max_bytes=175000` (definitive run).

## Method contracts

### A0 / SPARC-V2X

- Logits: \(\ell=\alpha(s^{\mathrm{VA}}+\beta r_\theta(x))\), residual MLP Linear–Tanh–Linear, separate critic.
- Residual last layer zero-initialized at train start.
- PPO with VA-relative advantage shift; checkpoint = best mean \(\Delta V_{\mathrm{done}}\) vs corrected VA on **validation** seeds only.
- Out of scope for SPARC: counterfactual cloned-state training, residual regression loss, MF deployment gate, cardinality shield, \(\kappa\) risk prediction, variable-delay discard model.

### B1 Corrected VA + governor

- Greedy descending \(s^{\mathrm{VA}}\); same masks and governor. No learning.

### B2 Min-Finish + governor

- Greedy ascending \(\widehat F\); same masks and governor. No learning.

### B3 Direct PPO + governor

- Logits \(\alpha q_\psi(x)\) without additive VA; matched architecture, masks, GAE/PPO schedule, governor; **no** VA-relative shift; ordinary init.

### B4 Lagrangian PPO (unshielded)

- No SPARC PRR/P95 governor / VA fallback.
- Explicit STOP / empty action; physical interface capacity disclosed (not \(B_{\max}\) safety envelope).
- Cost critic on union violation indicator; \(\widehat A^r-\lambda\widehat A^c\). Not a compliance guarantee.

### B5 ITORA-inspired

- Requires method-mapping audit before coding. Label **ITORA-inspired** unless faithful reproduction is demonstrated.
- Common deadline / uniqueness accounting; Mode-2 translation of unsupported infrastructure radio APIs.
- Admission: validation-selected fixed byte cap **or** adapted original mechanism — freeze before final test.

## Ablation reuse

- A2 ≡ B1 (reuse results).
- A4 ≡ B3 (reuse results).
- A1 = frozen SPARC actor with safety gates removed (byte/physical limit retained; label “without safety gates”).
- A3/A5/A6 require separate training as specified in the protocol.

## Excluded logs / metrics

Do not require: `counterfactual_log`, `gate_log`, `shield_prediction_coefficients`, `discarded_assignments`, or discarded-assignment fraction as a definitive-model KPI.
