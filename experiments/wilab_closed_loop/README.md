# Isolated Python–WiLabV2Xsim closed-loop adapter

This directory contains the bridge for coupling the final residual policy to an isolated copy of the WiLab event loop. It does not modify `vendor/WiLabV2Xsim`, existing checkpoints, or existing results.

## Protocol

The MATLAB side calls `wilab_closed_loop_callback` at each control boundary and writes:

```text
state_<episode>_<epoch>.json
```

The JSON state contains time, density, CBR, PRR, P95 delay, active-vehicle count, radio load, and the application candidates supplied by the wrapper. Python reads the state, selects task/helper assignments with the frozen checkpoint, and writes:

```text
action_<episode>_<epoch>.json
```

The action contains the admission budget and selected task/helper pairs. MATLAB acknowledges the action before continuing the event loop.

The bridge uses a file protocol because MATLAB Engine for Python is not installed in this environment. The protocol is deterministic and supports a later MATLAB Engine replacement without changing the observation/action schema.

## Important scope

WiLabV2Xsim supplies the radio state and packet-level Mode-2 evolution. The Python wrapper supplies the application task/helper state because the vendored WiLab model has no native task/helper scheduler. The adapter therefore keeps application semantics explicit rather than silently treating WiLab beacons as task assignments.
