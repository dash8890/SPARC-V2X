## Helper-Selection Generalization Audit — Decision

Verdict: The helper-selection opportunity is real, but the proposed `tight_slack_helper_rebind` rule is not yet justified.

The audit reveals an important asymmetry:

| Diagnostic                                     | Result          |
| ---------------------------------------------- | --------------- |
| Total Direct assignments                       | 70              |
| Assignments meeting the trigger                | 16 (22.9%)      |
| Failed assignments potentially recoverable     | 1               |
| Successful assignments exposed to modification | 15              |
| Confirmed successful local replay              | 1               |
| Generalized benefit                            | Not established |

The key concern is that 93.75% of the trigger firings occur on tasks Direct already completes.

A universal tight-slack rebinding rule would therefore modify substantially more successful assignments than failed ones. Even if the alternative helper has greater predicted slack, there is no guarantee that changing those successful assignments improves episode-level completed value.

### My recommendation

Authorize a bounded, training-free safety-and-value replay of the 15 completed-class events before considering any implementation.

This differs slightly from the proposed protocol: I would prioritize testing potential harm rather than attempting to reproduce the already-confirmed benefit.

The experiment should:

1. Replay each of the 15 successful-assignment events individually from its identical pre-decision state.
2. Change only the selected task's helper; retain all other assignments and matched exogenous randomness.
3. Measure net episode-level \\(V\_{\mathrm{done}}\\), preserved/lost completions, safety violations, and downstream effects.
4. Report all 15 outcomes, including negative and zero effects.
5. Keep the confirmed seed 82002/task 19 recovery as a separate positive-control result.

### Proposed decision criterion

- If the rule damages existing successful assignments: Reject the universal rebinding trigger. A more selective rule would require independent causal justification.
- If successful assignments remain successful without material losses: The rule passes this limited harm screen, but still needs independent validation of actual improvement.
- If some assignments benefit and others suffer: Identify observable differences between them before proposing a targeted implementation.

One further methodological correction: with only one miss-class event, the proposed “majority of misses improve” criterion cannot establish generalization. One successful recovery remains one successful recovery, regardless of its percentage.

### Authorization boundary

I recommend approving the 15-event replay, but this message does not authorize implementation or training. The existing Direct checkpoint, 175 kB envelope, governor, validation seeds, and final-test seal must remain unchanged.

The next objective is not to demonstrate another isolated gain. It is to determine whether a general helper-rebinding rule can avoid destroying value that Direct PPO already captures.