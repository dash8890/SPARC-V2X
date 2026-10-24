"""Execution-time safety governor for the WiLab residual scheduler.

The governor is deliberately outside the learner.  It can remove or replace a
proposed assignment, but the policy cannot bypass it or enlarge its budget.
"""
from __future__ import annotations
import json
from pathlib import Path


class SafetyGovernor:
    def __init__(self, max_bytes: int = 175_000, prr_floor: float = .95,
                 p95_delay_ceiling_ms: float = 50.0, risk_calibration=None,
                 epsilon: float = .05, horizon_envelope_only: bool = False,
                 cbr_ceiling: float | None = 0.60):
        self.max_bytes = int(max_bytes)
        self.prr_floor = float(prr_floor)
        self.p95_delay_ceiling_ms = float(p95_delay_ceiling_ms)
        self.epsilon = float(epsilon)
        self.horizon_envelope_only = bool(horizon_envelope_only)
        self.cbr_ceiling = None if cbr_ceiling is None else float(cbr_ceiling)
        self.risk_calibration = None
        if risk_calibration:
            p = Path(risk_calibration)
            if p.exists():
                self.risk_calibration = json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _valid(state):
        return bool(state.get("cbr_valid", False) and state.get("delay_valid", False))

    def _long_window_below_floor(self, state):
        """True when the long-window or cumulative PRR is below the floor.

        ``window_prr_long`` is only meaningful after the long-window has
        collected enough attempts; ``cumulative_prr`` is always meaningful
        once at least one packet has been received.  Missing measurements are
        not treated as a violation here, but the short-window and validity
        checks above remain fail-safe.
        """
        if bool(state.get("window_prr_long_valid", False)):
            prr_long = state.get("window_prr_long")
            if prr_long is not None and float(prr_long) < self.prr_floor:
                return True
        cum = state.get("cumulative_prr")
        if cum is not None:
            try:
                cum = float(cum)
            except (TypeError, ValueError):
                cum = None
            if cum is not None and cum > 0.0 and cum < self.prr_floor:
                return True
        return False

    def admission_allowed(self, state):
        """Return whether any positive computation action may be sampled."""
        if not self._valid(state):
            return False
        prr = state.get("prr")
        p95 = state.get("p95_delay_ms")
        if prr is not None and float(prr) < self.prr_floor:
            return False
        if p95 is not None and float(p95) > self.p95_delay_ceiling_ms:
            return False
        if self.cbr_ceiling is not None:
            cbr = state.get("cbr")
            if cbr is not None and float(cbr) > self.cbr_ceiling:
                return False
        # The acceptance criterion is the long-window / cumulative PRR, not
        # only the short window.  A short window can look healthy while the
        # cumulative service is already below the floor.
        if self._long_window_below_floor(state):
            return False
        return True

    def _anchor(self, candidates):
        """Deterministic Value-Aware fallback under the byte envelope.

        Skips deadline-infeasible candidates when that flag is present, and
        keeps at most one task per source so the governor matches the closed-
        loop VA policy used in training/eval.
        """
        selected, used_tasks, used_sources, used_bytes = [], set(), set(), 0
        for c in sorted(candidates, key=lambda x: float(x.get("analytic_score", 0.0)), reverse=True):
            if c.get("deadline_feasible") is False:
                continue
            tid = c.get("task_id")
            sid = c.get("source_id")
            size = int(c.get("payload_bytes", 0))
            if tid in used_tasks or sid in used_sources or used_bytes + size > self.max_bytes:
                continue
            used_tasks.add(tid); used_sources.add(sid); used_bytes += size
            selected.append({"task_id": tid, "helper_id": c.get("helper_id"),
                             "source_id": sid,
                             "payload_bytes": size,
                             "packet_count": int(c.get("packet_count", 0))})
        return selected

    def apply(self, proposed, candidates, state):
        """Return (executed_assignments, intervention, reason).

        Invalid/unsafe radio measurements force zero computation.  A valid
        proposal is independently checked for unique tasks and the hard byte
        envelope.  Over-budget proposals fall back to the deterministic
        Value-Aware ordering under the same envelope.
        """
        if not self._valid(state):
            return [], bool(proposed), "invalid_radio_state"
        prr = state.get("prr")
        p95 = state.get("p95_delay_ms")
        if prr is not None and float(prr) < self.prr_floor:
            return [], bool(proposed), "prr_below_floor"
        if p95 is not None and float(p95) > self.p95_delay_ceiling_ms:
            return [], bool(proposed), "p95_above_ceiling"
        if self.cbr_ceiling is not None:
            cbr = state.get("cbr")
            if cbr is not None and float(cbr) > self.cbr_ceiling:
                return [], bool(proposed), "cbr_above_ceiling"
        if self._long_window_below_floor(state):
            return [], bool(proposed), "prr_long_below_floor"

        proposed_bytes = sum(int(a.get("payload_bytes", 0)) for a in proposed)
        # A positive action is admissible only if a calibration artifact gives
        # a conservative upper bound on the *next safety-window* violation
        # probability.  Missing calibration is fail-safe, never permissive.
        if proposed_bytes > 0 and not self.horizon_envelope_only:
            upper_risk = self._upper_risk(state, proposed_bytes)
            if upper_risk is None:
                return [], True, "future_risk_uncertified"
            if upper_risk > self.epsilon:
                return [], True, "future_risk_above_epsilon"

        by_key = {(c.get("task_id"), c.get("helper_id")): c for c in candidates}
        executed, used_tasks, used_bytes = [], set(), 0
        malformed = False
        for a in proposed:
            key = (a.get("task_id"), a.get("helper_id"))
            c = by_key.get(key)
            if c is None or a.get("task_id") in used_tasks:
                malformed = True; continue
            size = int(c.get("payload_bytes", a.get("payload_bytes", 0)))
            if used_bytes + size > self.max_bytes:
                malformed = True; continue
            used_tasks.add(a.get("task_id")); used_bytes += size
            executed.append(dict(a, payload_bytes=size,
                                 packet_count=int(c.get("packet_count", a.get("packet_count", 0)))))
        if malformed:
            return self._anchor(candidates), True, "over_budget_or_invalid_assignment"
        return executed, False, "none"

    def _upper_risk(self, state, proposed_bytes):
        """Read a conservative risk bound from a frozen calibration artifact.

        The artifact must contain rows with ``max_bytes`` and
        ``upper_violation_risk``.  The smallest row covering the proposal is
        used.  No interpolation or extrapolation is performed.
        """
        if not self.risk_calibration:
            return None
        rows = self.risk_calibration.get("certified_rows", [])
        eligible = [r for r in rows
                    if int(r.get("max_bytes", -1)) >= int(proposed_bytes)
                    and float(r.get("upper_violation_risk", 1.0)) >= 0.0]
        if not eligible:
            return None
        row = min(eligible, key=lambda r: int(r["max_bytes"]))
        return float(row["upper_violation_risk"])
