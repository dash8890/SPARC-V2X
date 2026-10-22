# WiLab execution-time safety governor

The residual policy is no longer the final authority over computation
admission.  At every controller epoch it proposes task/helper assignments;
`controllers/rl/safety_governor.py` filters that proposal before the action is
written to the MATLAB bridge.

The governor enforces three conditions independently of the learner:

* CBR and delay measurements must be valid;
* the measured PRR must be at least 0.95 and P95 safety delay at most 50 ms;
* the executed computation payload must not exceed the configured hard byte
  envelope (175,000 bytes by default).

For positive computation, it also requires a frozen calibration artifact with
an upper confidence bound for the **next safety-window violation probability**.
The bound must satisfy `upper_violation_risk <= 0.05`.  The governor performs
no interpolation or extrapolation.  If the artifact is missing or does not
cover the proposed byte load, it fails safe to zero computation.

If radio measurements are invalid or outside the safety thresholds, the
governor executes no computation.  If a proposal is malformed or exceeds the
byte envelope, it replaces it with the deterministic Value-Aware ordering,
truncated to the same envelope.  Thus the policy cannot bypass the governor by
changing its scores or sampling a different action.

Each bridge audit records both `proposed_assignments` and
`selected_assignments`, together with `governor_intervened`,
`governor_reason`, and `governor_max_bytes`.  These fields distinguish learner
behavior from the executed safe action and quantify how often the governor had
to intervene.

The governor is enabled for ordinary training/evaluation.  It is disabled only
for explicit calibration/debug modes (`--safety-only`, forced budgets,
`--all-eligible`, `--forced-byte-budget`, or
`--disable-safety-governor`).  Existing calibration outputs are therefore
unchanged, while new policy runs use the independent execution filter.

This is the enforceable runtime policy: uncalibrated positive computation is
not admitted.  The calibration artifact still represents an empirical bound,
not a mathematical proof of WiLab reliability; its coverage must be reported
from held-out next-window data before positive budgets are enabled.
