"""Backlog-aware safety-reserved admission governor (Track 1).

Extends the execution-time governor with an explicit account of radio resources
already committed by computation packets that are still queued in the Mode-2
buffers.  The rule is the one specified for Track 1:

    L_backlog(t:t+H) + L(M, t:t+H)  <=  C_hat(t:t+H) - S(t:t+H)

evaluated in *packets per future window*, not as an aggregate byte budget, so a
single burst cannot monopolise one future safety window while still fitting a
multi-epoch budget on average.

Causal estimators (no future information):

  Q_t        total Mode-2 packets queued at the controller boundary, measured by
             WiLab *before* this epoch's admission is injected.
  Q^c_t      compute-attributable backlog = max(0, Q_t - A_t), where A_t is the
             number of active vehicles, each of which generates at most one
             periodic safety packet per 100 ms epoch.
  C_hat_t    EWMA of the *measured* net drain per epoch:
                 drained_{t-1} = Q_{t-1} + injected_{t-1} + A_{t-1} - Q_t
             i.e. what the channel actually served last epoch, not a modelled
             capacity.
  S_t        safety reservation = reserve_frac * A_t packets per window.

A proposal is admitted only up to

    packets_max = max(0, H * max(0, C_hat_t - S_t) - Q^c_t)

packets.  The proposal is truncated by analytic value density (the RL policy's
own ranking), never replaced wholesale, so the learned matching survives where
it fits.

Initialisation is deliberately conservative: before the first drain measurement
the headroom is two packets per window.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from controllers.rl.safety_governor import SafetyGovernor


class BacklogAwareGovernor(SafetyGovernor):
    def __init__(
        self,
        max_bytes: int = 175_000,
        horizon: int = 2,
        reserve_frac: float = 1.0,
        ewma_alpha: float = 0.3,
        init_headroom_packets: float = 2.0,
        prr_floor: float = 0.95,
        p95_delay_ceiling_ms: float = 50.0,
    ):
        super().__init__(
            max_bytes=max_bytes,
            prr_floor=prr_floor,
            p95_delay_ceiling_ms=p95_delay_ceiling_ms,
            horizon_envelope_only=True,
        )
        self.static_max_bytes = int(max_bytes)
        self.horizon = int(horizon)
        self.reserve_frac = float(reserve_frac)
        self.ewma_alpha = float(ewma_alpha)
        self.init_headroom_packets = float(init_headroom_packets)

        self.capacity_ewma: Optional[float] = None
        self.prev_pending: Optional[float] = None
        self.prev_active: Optional[float] = None
        self.prev_injected_packets: float = 0.0
        self.log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ state
    def _observables(self, state: Dict[str, Any]) -> Tuple[float, float]:
        pending = float(state.get("pending_packets") or 0.0)
        active = float(state.get("active_vehicles") or state.get("density") or 0.0)
        return pending, active

    def _update_capacity(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pending, active = self._observables(state)
        drained = None
        if self.prev_pending is not None and self.prev_active is not None:
            drained = (
                self.prev_pending
                + self.prev_injected_packets
                + self.prev_active
                - pending
            )
            drained = max(0.0, drained)
            if self.capacity_ewma is None:
                self.capacity_ewma = drained
            else:
                a = self.ewma_alpha
                self.capacity_ewma = (1.0 - a) * self.capacity_ewma + a * drained

        backlog_compute = max(0.0, pending - active)
        reserve = self.reserve_frac * active
        if self.capacity_ewma is None:
            headroom = self.init_headroom_packets
        else:
            headroom = max(0.0, self.capacity_ewma - reserve)
        packets_max = max(
            0.0, self.horizon * headroom - backlog_compute
        )
        diag = {
            "pending_packets": pending,
            "active_vehicles": active,
            "drained_last_epoch": drained,
            "capacity_ewma": self.capacity_ewma,
            "reserve_packets": reserve,
            "backlog_compute_packets": backlog_compute,
            "headroom_packets_per_epoch": headroom,
            "packets_max": packets_max,
        }
        self.prev_pending = pending
        self.prev_active = active
        return diag

    # -------------------------------------------------------------- execution
    def _truncate_to_packets(
        self, proposed: List[Dict[str, Any]], candidate_by_key: Dict[Any, Dict[str, Any]],
        packets_max: float,
    ) -> List[Dict[str, Any]]:
        if packets_max <= 0:
            return []
        ranked = sorted(
            proposed,
            key=lambda a: float(
                (candidate_by_key.get((a.get("task_id"), a.get("helper_id"))) or {}).get(
                    "analytic_score", 0.0
                )
            ),
            reverse=True,
        )
        kept: List[Dict[str, Any]] = []
        used = 0.0
        for a in ranked:
            c = candidate_by_key.get((a.get("task_id"), a.get("helper_id")))
            pk = float(
                (c or {}).get("packet_count", a.get("packet_count", 0)) or 0
            )
            if used + pk > packets_max:
                continue
            kept.append(a)
            used += pk
        # Preserve the policy's original ordering in the executed pack.
        order = {id(a): i for i, a in enumerate(proposed)}
        kept.sort(key=lambda a: order.get(id(a), 0))
        return kept

    def apply(self, proposed, candidates, state):
        diag = self._update_capacity(state)

        # Fail-safe checks are inherited unchanged.
        if not self._valid(state):
            self._commit(state, diag, 0, "invalid_radio_state")
            return [], bool(proposed), "invalid_radio_state"
        prr = state.get("prr")
        p95 = state.get("p95_delay_ms")
        if prr is not None and float(prr) < self.prr_floor:
            self._commit(state, diag, 0, "prr_below_floor")
            return [], bool(proposed), "prr_below_floor"
        if p95 is not None and float(p95) > self.p95_delay_ceiling_ms:
            self._commit(state, diag, 0, "p95_above_ceiling")
            return [], bool(proposed), "p95_above_ceiling"

        by_key = {(c.get("task_id"), c.get("helper_id")): c for c in candidates}
        proposed_bytes = sum(int(a.get("payload_bytes", 0)) for a in proposed)
        if proposed_bytes > self.static_max_bytes:
            executed, intervened, reason = self._fallback_envelope(proposed, by_key)
            return self._finish(state, diag, executed, intervened, reason)

        packets_max = diag["packets_max"]
        executed = self._truncate_to_packets(proposed, by_key, packets_max)
        intervened = len(executed) != len(proposed)
        reason = "backlog_envelope" if intervened else "none"
        return self._finish(state, diag, executed, intervened, reason)

    def _fallback_envelope(self, proposed, by_key) -> Tuple[List[Dict[str, Any]], bool, str]:
        """Static byte envelope path, mirroring the base governor's contract."""
        executed: List[Dict[str, Any]] = []
        used = 0
        for a in proposed:
            c = by_key.get((a.get("task_id"), a.get("helper_id")))
            size = int((c or a).get("payload_bytes", 0) or 0)
            if used + size > self.static_max_bytes:
                continue
            executed.append(a)
            used += size
        return executed, True, "static_envelope"

    def _finish(self, state, diag, executed, intervened, reason):
        packets = sum(int(a.get("packet_count", 0) or 0) for a in executed)
        self._commit(state, diag, packets, reason)
        return executed, intervened, reason

    def _commit(self, state, diag, packets, reason):
        self.prev_injected_packets = float(packets)
        self.log.append({**diag, "admitted_packets": float(packets), "reason": reason})

    # ----------------------------------------------------------------- report
    def summary(self) -> Dict[str, Any]:
        if not self.log:
            return {"epochs": 0}
        allowed = [e for e in self.log if e["packets_max"] > 0]
        return {
            "epochs": len(self.log),
            "epochs_with_headroom": len(allowed),
            "mean_capacity_ewma": _mean(e["capacity_ewma"] for e in self.log),
            "mean_headroom_packets": _mean(e["headroom_packets_per_epoch"] for e in self.log),
            "mean_packets_max": _mean(e["packets_max"] for e in self.log),
            "mean_admitted_packets": _mean(e["admitted_packets"] for e in self.log),
            "epochs_throttled": sum(1 for e in self.log if e["reason"] == "backlog_envelope"),
            "horizon": self.horizon,
            "reserve_frac": self.reserve_frac,
        }


def _mean(xs) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None
