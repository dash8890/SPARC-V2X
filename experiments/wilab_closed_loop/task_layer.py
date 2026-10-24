"""Application task/helper layer with trace-aligned kinematics and persistent helper queues.

WiLabV2Xsim does not model application tasks, helper CPU queues, or vehicular
contact windows.  This module keeps those semantics explicit on the Python
side, using the *same* mobility trace that WiLabV2Xsim loads.  A helper is only
eligible when the trace puts it inside the application contact radius; the
scheduler therefore cannot treat a physically unreachable helper as reachable.

Payload accounting is payload-faithful by default: a task earns value only
after every payload unit reaches the assigned helper and the helper finishes
computation before the task's absolute deadline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:  # package import (pytest: experiments.wilab_closed_loop.task_layer)
    from .vehicle_trace import VehicleTrace, vehicle_id_map
except ImportError:  # direct script import (sys.path includes the experiment dir)
    from vehicle_trace import VehicleTrace, vehicle_id_map


@dataclass
class VehicleKinematics:
    veh_id: int
    is_source: bool
    x_0: float          # longitudinal position at the snapshot time (meters)
    y: float            # lateral coordinate (meters)
    velocity: float     # longitudinal speed at the snapshot time (m/s)

    def x(self, t_sec: float) -> float:
        return self.x_0 + self.velocity * t_sec


def _normalize_task_radio(payload) -> Dict[int, dict]:
    """Normalize the MATLAB ``task_radio`` field to ``{task_id: record}``.

    MATLAB ``jsonencode`` emits a scalar object for a one-element struct array,
    an array for several elements, and ``[]`` for none.
    """
    if payload is None:
        return {}
    if isinstance(payload, dict):
        payload = [payload] if "task_id" in payload else list(payload.values())
    if not isinstance(payload, (list, tuple)):
        payload = [payload]
    out: Dict[int, dict] = {}
    for rec in payload:
        if isinstance(rec, dict) and "task_id" in rec:
            out[int(rec["task_id"])] = rec
    return out


def compute_contact_duration(
    v_src: VehicleKinematics,
    v_hlp: VehicleKinematics,
    t_sec: float,
    r_comm: float = 250.0,
) -> float:
    """Exact multi-lane Euclidean forward contact window at a snapshot.

    ``v_src`` and ``v_hlp`` are snapshots taken at ``t_sec``.  The function
    evaluates the remaining contact time from that instant.  Pass ``t_sec=0``
    when the snapshot already carries the current position/speed.
    """
    delta_y = v_src.y - v_hlp.y
    if abs(delta_y) > r_comm:
        return 0.0

    x_max = math.sqrt(r_comm ** 2 - delta_y ** 2)
    delta_x = v_src.x(t_sec) - v_hlp.x(t_sec)
    delta_v = v_src.velocity - v_hlp.velocity

    if abs(delta_v) < 1e-4:
        if -x_max <= delta_x <= x_max:
            return 999.0  # Same speed and currently in range
        return 0.0

    t1 = (-x_max - delta_x) / delta_v
    t2 = (x_max - delta_x) / delta_v
    t_min = min(t1, t2)
    t_max = max(t1, t2)

    if t_min <= 0.0 <= t_max:
        return max(0.0, t_max)
    if t_max < 0.0:
        return 0.0  # Contact already passed
    return 0.0      # Approaching but not yet within range


@dataclass
class Task:
    task_id: int
    source_id: int
    payload_bytes: int
    cycles: float
    deadline_ms: float
    value: float
    done: bool = False
    release_epoch: int = 0
    queued: bool = False
    completion_time_ms: float | None = None
    assigned_helper_id: int | None = None
    finish_estimate_ms: float | None = None
    assign_epoch: int | None = None
    radio_delivered: bool = False
    # Payload-faithful accounting (see wilab_closed_loop_packet_accounting.m).
    payload_required: int = 0
    payload_received: int = 0
    radio_complete_epoch: int | None = None
    radio_complete_ms: float | None = None
    legacy_radio_delivered: bool = False
    # Persistent transmission/execution lifecycle.  An admitted task is not
    # re-proposed while its payload is being paced through the sidelink queue.
    in_flight: bool = False
    cancelled: bool = False
    cancel_reason: str | None = None
    admitted_epoch: int | None = None
    source_wilab_id: int | None = None
    helper_wilab_id: int | None = None
    radio_ms_est: float | None = None


@dataclass
class Helper:
    helper_id: int
    cpu_cycles_s: float
    queue_cycles: float = 0.0
    ready_time_ms: float = 0.0
    kinematics: VehicleKinematics | None = None
    veh_id: int | None = None


class TaskLayer:
    def __init__(
        self,
        density: int,
        seed: int,
        tasks_per_source: int = 3,
        computation_packet_bytes: int = 300,
        spread_arrivals: bool = False,
        controller_epochs: int = 100,
        deadline_multiplier: float = 1.0,
        helper_cpu_lo: float = 1e9,
        helper_cpu_hi: float = 4e9,
        cluster_share: float | None = None,
        arrival_mode: str = "legacy",
        r_comm: float = 250.0,
        trace_path: str | Path | None = None,
        epoch_ms: float = 20.0,
        packets_per_epoch: int = 1,
        tb_bytes: int | None = None,
        segment_bytes: int = 2000,
        reservation_ms: float = 5.0,
        payload_bytes_lo: int = 10_000,
        payload_bytes_hi: int = 80_000,
        deadline_ms_lo: float = 100.0,
        deadline_ms_hi: float = 500.0,
        cycles_lo: float = 5e6,
        cycles_hi: float = 5e7,
    ):
        self.rng = random.Random(seed)
        # ``computation_packet_bytes`` is retained as the action/metadata name
        # for the PHY transport-block payload.  ``tb_bytes`` is the explicit
        # corrected-model name; when both are present ``tb_bytes`` wins.
        self.tb_bytes = int(tb_bytes if tb_bytes is not None else computation_packet_bytes)
        if self.tb_bytes <= 0:
            raise ValueError("tb_bytes must be positive")
        self.computation_packet_bytes = self.tb_bytes
        # Application-layer segmentation is independent of the PHY TB size:
        # a 2 kB segment is fragmented into ceil(2000 / tb_bytes) TBs.
        self.segment_bytes = int(segment_bytes)
        if self.segment_bytes <= 0:
            raise ValueError("segment_bytes must be positive")
        self.reservation_ms = float(reservation_ms)
        if self.reservation_ms <= 0.0:
            raise ValueError("reservation_ms must be positive")
        self.payload_bytes_lo = int(min(payload_bytes_lo, payload_bytes_hi))
        self.payload_bytes_hi = int(max(payload_bytes_lo, payload_bytes_hi))
        self.deadline_ms_lo = float(min(deadline_ms_lo, deadline_ms_hi))
        self.deadline_ms_hi = float(max(deadline_ms_lo, deadline_ms_hi))
        self.cycles_lo = float(min(cycles_lo, cycles_hi))
        self.cycles_hi = float(max(cycles_lo, cycles_hi))
        self.arrival_mode = str(arrival_mode)
        self.r_comm = float(r_comm)
        self.controller_epochs = int(controller_epochs)
        self.current_epoch = 0
        self.epoch_ms = float(epoch_ms)
        if self.epoch_ms <= 0.0:
            raise ValueError("epoch_ms must be positive")
        self.packets_per_epoch = max(1, int(packets_per_epoch))
        # One PSSCH transport block per semi-persistent reservation.  The
        # control clock (epoch_ms) and the reservation interval are separate:
        # a 20 ms control epoch contains several 5 ms reservations.
        self.packets_per_reservation = 1
        self.reservations_per_epoch = max(
            1, int(round(self.epoch_ms / self.reservation_ms))
        )

        lo = float(helper_cpu_lo)
        hi = float(helper_cpu_hi)
        if hi < lo:
            lo, hi = hi, lo

        if cluster_share is not None:
            n_sources = max(1, min(density, int(round(density * float(cluster_share)))))
        else:
            n_sources = max(1, density // 5)
        n_helpers = max(1, min(12, density - n_sources))
        self.n_sources = int(n_sources)
        self.n_helpers = int(n_helpers)
        self.trace: VehicleTrace | None = None
        self.vehicle_map: Dict[str, Dict[int, int]] = vehicle_id_map(
            density, self.n_sources, self.n_helpers
        )

        if trace_path is not None:
            trace_file = Path(trace_path)
            if trace_file.is_file():
                self.trace = VehicleTrace.load(trace_file)
                if len(self.trace.vehicle_ids) < self.n_sources + self.n_helpers:
                    raise ValueError(
                        f"trace {trace_file} has {len(self.trace.vehicle_ids)} vehicles; "
                        f"need {self.n_sources + self.n_helpers}"
                    )

        # Helpers
        self.helpers: List[Helper] = []
        for i in range(self.n_helpers):
            veh_id = self.vehicle_map["helpers"][i]
            kin = self._trace_snapshot(veh_id, is_source=False, t_sec=0.0)
            if kin is None:
                v_speed = min(25.0, max(15.0, self.rng.gauss(20.0, 2.5)))
                lane = self.rng.choice([4.0, 8.0, 12.0])
                x0 = self.rng.uniform(100.0, 900.0)
                kin = VehicleKinematics(veh_id, is_source=False, x_0=x0, y=lane, velocity=v_speed)
            self.helpers.append(
                Helper(i, self.rng.uniform(lo, hi), kinematics=kin, veh_id=veh_id)
            )

        # Sources
        sources = list(range(self.n_sources))
        self.source_kinematics: Dict[int, VehicleKinematics] = {}
        self.source_veh_ids: Dict[int, int] = {}
        for s in sources:
            veh_id = self.vehicle_map["sources"][s]
            self.source_veh_ids[s] = veh_id
            kin = self._trace_snapshot(veh_id, is_source=True, t_sec=0.0)
            if kin is None:
                v_speed = min(25.0, max(15.0, self.rng.gauss(20.0, 2.5)))
                lane = self.rng.choice([4.0, 8.0, 12.0])
                x0 = self.rng.uniform(100.0, 900.0)
                kin = VehicleKinematics(veh_id, is_source=True, x_0=x0, y=lane, velocity=v_speed)
            self.source_kinematics[s] = kin

        self.tasks: List[Task] = []
        self.spread_arrivals = bool(
            spread_arrivals or (self.arrival_mode in ("streaming", "mobility_streaming"))
        )
        deadline_multiplier = max(0.05, float(deadline_multiplier))

        if self.arrival_mode in ("streaming", "mobility_streaming"):
            for source in sources:
                cur_ep = self.rng.randint(0, 4)
                for task_index in range(tasks_per_source):
                    release_epoch = min(self.controller_epochs - 1, cur_ep)
                    self.tasks.append(Task(
                        len(self.tasks), source,
                        self.rng.randint(self.payload_bytes_lo, self.payload_bytes_hi),
                        self.rng.uniform(self.cycles_lo, self.cycles_hi),
                        self.rng.uniform(self.deadline_ms_lo, self.deadline_ms_hi)
                        * deadline_multiplier,
                        self.rng.uniform(1.0, 5.0),
                        release_epoch=release_epoch,
                        source_wilab_id=self.vehicle_map["sources"][source],
                    ))
                    # Calibrated Poisson streaming with bursts
                    if self.rng.random() < 0.35:
                        inter = self.rng.randint(1, 3)
                    else:
                        inter = max(3, int(round(self.rng.expovariate(1.0 / 6.0))))
                    cur_ep += inter
        else:
            for source in sources:
                for task_index in range(tasks_per_source):
                    release_epoch = (
                        task_index * self.controller_epochs // max(1, tasks_per_source)
                    ) if self.spread_arrivals else 0
                    self.tasks.append(Task(
                        len(self.tasks), source,
                        self.rng.randint(self.payload_bytes_lo, self.payload_bytes_hi),
                        self.rng.uniform(self.cycles_lo, self.cycles_hi),
                        self.rng.uniform(self.deadline_ms_lo, self.deadline_ms_hi)
                        * deadline_multiplier,
                        self.rng.uniform(1.0, 5.0),
                        release_epoch=release_epoch,
                        source_wilab_id=self.vehicle_map["sources"][source],
                    ))

    # ------------------------------------------------------------------
    # Clock and geometry helpers
    # ------------------------------------------------------------------
    def release_ms(self, task: Task) -> float:
        return self.epoch_ms * float(task.release_epoch)

    def absolute_deadline_ms(self, task: Task) -> float:
        return self.release_ms(task) + float(task.deadline_ms)

    def packet_count(self, task: Task) -> int:
        """Number of PHY transport blocks carrying the task payload."""
        return max(
            1,
            (int(task.payload_bytes) + self.tb_bytes - 1) // self.tb_bytes,
        )

    def segment_count(self, task: Task) -> int:
        """Number of 2 kB application segments in the task payload."""
        return max(
            1,
            (int(task.payload_bytes) + self.segment_bytes - 1)
            // self.segment_bytes,
        )

    def radio_ms_est(self, task: Task) -> float:
        packets = self.packet_count(task)
        reservations = max(
            1, math.ceil(packets / self.packets_per_reservation)
        )
        # One reservation for admission plus one reservation per TB.  This is
        # the physical reservation clock, not the 20 ms control clock.
        return self.reservation_ms * (1 + reservations)

    def _trace_snapshot(self, veh_id: int, is_source: bool, t_sec: float) -> VehicleKinematics | None:
        if self.trace is None:
            return None
        row = self.trace.sample(int(veh_id), float(t_sec))
        return VehicleKinematics(
            int(veh_id), is_source=is_source,
            x_0=float(row.x_m), y=float(row.y_m), velocity=float(row.speed_m_s),
        )

    def source_kinematics_at(self, source_id: int, t_sec: float) -> VehicleKinematics | None:
        if self.trace is not None:
            veh_id = self.source_veh_ids.get(int(source_id))
            if veh_id is None:
                return None
            return self._trace_snapshot(veh_id, is_source=True, t_sec=t_sec)
        return self.source_kinematics.get(int(source_id))

    def helper_kinematics_at(self, helper: Helper, t_sec: float) -> VehicleKinematics | None:
        if self.trace is not None and helper.veh_id is not None:
            return self._trace_snapshot(helper.veh_id, is_source=False, t_sec=t_sec)
        return helper.kinematics

    def pair_distance_m(self, source_id: int, helper_id: int, t_sec: float) -> float:
        src = self.source_kinematics_at(source_id, t_sec)
        helper = next((h for h in self.helpers if h.helper_id == int(helper_id)), None)
        if src is None or helper is None:
            return float("inf")
        hlp = self.helper_kinematics_at(helper, t_sec)
        if hlp is None:
            return float("inf")
        return math.hypot(src.x(0.0) - hlp.x(0.0), src.y - hlp.y)

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int):
        """Advance application time, drain helper queues, and finalize/cancel tasks."""
        self.current_epoch = int(epoch)
        now_ms = self.epoch_ms * self.current_epoch

        # Finalize tasks that completed computation.
        for task in self.tasks:
            if task.queued and not task.done and task.completion_time_ms is not None:
                absolute_deadline_ms = self.absolute_deadline_ms(task)
                if task.completion_time_ms <= now_ms and task.completion_time_ms <= absolute_deadline_ms:
                    task.done = True
                    task.queued = False
                    task.in_flight = False

        # Deadline-aware cancellation.  An in-flight task whose absolute
        # deadline has passed can no longer earn value; its remaining units
        # are dropped by the MATLAB pacing queue at the same deadline.
        for task in self.tasks:
            if task.done or task.cancelled:
                continue
            if now_ms > self.absolute_deadline_ms(task) + 1e-9:
                task.cancelled = True
                task.in_flight = False
                task.queued = False
                task.cancel_reason = task.cancel_reason or "deadline"

        # Drain helper backlog
        for helper in self.helpers:
            helper.queue_cycles = max(
                0.0, (helper.ready_time_ms - now_ms) * helper.cpu_cycles_s / 1000.0
            )

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------
    def candidates(self, cbr: float = 0.0, time_fraction: float = 0.0):
        out = []
        now_ms = self.epoch_ms * self.current_epoch
        now_sec = now_ms / 1000.0

        for task in self.tasks:
            if task.done or task.queued or task.in_flight or task.cancelled:
                continue
            if self.spread_arrivals:
                if task.release_epoch > self.current_epoch:
                    continue
                if now_ms > self.absolute_deadline_ms(task) + 1e-9:
                    continue

            src_kin = self.source_kinematics_at(task.source_id, now_sec)
            packet_count = self.packet_count(task)
            radio_ms = self.radio_ms_est(task)

            for helper in self.helpers:
                compute_ms = 1000.0 * task.cycles / helper.cpu_cycles_s
                interval_ms = self.epoch_ms
                # Helper queueing is part of the physical deadline clock:
                # CPU execution can only start when the payload has arrived
                # *and* the helper is free.
                helper_ready_ms = max(0.0, helper.ready_time_ms - now_ms)

                if self.spread_arrivals:
                    release_ms = self.release_ms(task)
                    absolute_deadline_ms = self.absolute_deadline_ms(task)
                    elapsed_ms = max(0.0, now_ms - release_ms)
                    # Payload transmission is paced through the Mode-2 queue,
                    # then one turnaround epoch elapses before computation.
                    finish_from_release_ms = (
                        elapsed_ms
                        + max(radio_ms, helper_ready_ms)
                        + compute_ms
                    )
                    finish_absolute_ms = release_ms + finish_from_release_ms
                    finish_ms = finish_from_release_ms
                    rem_dl_ms = max(1.0, absolute_deadline_ms - now_ms)

                    hlp_kin = self.helper_kinematics_at(helper, now_sec)
                    t_contact = 999.0
                    if src_kin is not None and hlp_kin is not None:
                        t_contact = compute_contact_duration(src_kin, hlp_kin, 0.0, self.r_comm)

                    finish_rel_sec = max(0.0, finish_absolute_ms - now_ms) / 1000.0
                    contact_feasible = bool(finish_rel_sec <= t_contact + 1e-5)
                    deadline_feasible = bool(finish_absolute_ms <= absolute_deadline_ms + 1e-9)
                    if not deadline_feasible or not contact_feasible:
                        continue
                else:
                    finish_ms = max(radio_ms, helper_ready_ms) + compute_ms
                    rem_dl_ms = task.deadline_ms
                    t_contact = 999.0
                    contact_feasible = True
                    deadline_feasible = bool(finish_ms <= task.deadline_ms + 1e-9)
                    if not deadline_feasible:
                        continue

                norm_val = task.value / 5.0
                lat_pressure = min(1.0, max(0.0, finish_ms / rem_dl_ms))
                occ_cost = min(
                    1.0,
                    (helper.queue_cycles + task.cycles)
                    / (helper.cpu_cycles_s * max(0.01, t_contact)),
                )
                rad_footprint = min(
                    1.0, (task.payload_bytes / 80_000.0) * (1.0 + float(cbr))
                )
                distance_m = self.pair_distance_m(task.source_id, helper.helper_id, now_sec)

                out.append({
                    "task_id": task.task_id,
                    "source_id": task.source_id,
                    "helper_id": helper.helper_id,
                    "source_wilab_id": int(self.vehicle_map["sources"][task.source_id]),
                    "helper_wilab_id": int(self.vehicle_map["helpers"][helper.helper_id]),
                    "payload_bytes": task.payload_bytes,
                    "packet_count": packet_count,
                    "radio_ms_est": float(radio_ms),
                    "analytic_score": task.value / max(finish_ms, 1e-6),
                    "finish_ms": finish_ms,
                    "deadline_ms": task.deadline_ms,
                    "task_value": task.value,
                    "deadline_feasible": True,
                    "contact_feasible": contact_feasible,
                    "t_contact_sec": t_contact,
                    "_dist": float(distance_m),
                    # Structured priority features
                    "norm_value": norm_val,
                    "lat_pressure": lat_pressure,
                    "occ_cost": occ_cost,
                    "rad_footprint": rad_footprint,
                    "V_kj": norm_val,
                    "L_kj": lat_pressure,
                    "O_kj": occ_cost,
                    "R_kj": rad_footprint,
                    # Legacy 10-feature vector for backwards compatibility
                    "features": [
                        task.payload_bytes / 80_000.0,
                        task.cycles / 5e7,
                        task.deadline_ms / 500.0,
                        task.value / 5.0,
                        helper.cpu_cycles_s / 4e9,
                        helper.queue_cycles / 5e7,
                        0.0,
                        cbr,
                        self.current_epoch / max(1.0, float(self.controller_epochs))
                        if self.spread_arrivals else time_fraction,
                        1.0,
                    ],
                })
        return out

    # ------------------------------------------------------------------
    # Radio outcome application
    # ------------------------------------------------------------------
    def apply_radio_outcomes(
        self,
        assignments,
        delivered_task_ids,
        interval_ms: float | None = None,
        task_radio=None,
        accounting: str = "payload",
    ):
        """Apply the radio result to task state.

        ``accounting='payload'`` requires the full task payload to be received
        by the assigned helper before computation can be credited.  The radio
        record carries the number of distinct payload units received and the
        control epoch at which the last one arrived (see
        ``wilab_closed_loop_packet_accounting.m``).

        ``accounting='legacy'`` reproduces the archived first-packet rule.  It
        exists only to re-run archived configurations; new experiments must
        use the default payload rule.
        """
        interval_ms = self.epoch_ms if interval_ms is None else float(interval_ms)
        delivered = set(int(x) for x in (delivered_task_ids or []))
        radio = _normalize_task_radio(task_radio)
        now_ms = self.epoch_ms * self.current_epoch
        now_sec = now_ms / 1000.0
        payload_mode = str(accounting).lower() != "legacy"

        for assignment in assignments:
            task = next((t for t in self.tasks if t.task_id == int(assignment["task_id"])), None)
            helper = next((h for h in self.helpers if h.helper_id == int(assignment["helper_id"])), None)
            if task is None or helper is None or task.done or task.cancelled:
                continue

            task.assigned_helper_id = int(helper.helper_id)
            task.helper_wilab_id = int(helper.veh_id) if helper.veh_id is not None else int(helper.helper_id) + 1
            task.assign_epoch = int(self.current_epoch)
            task.admitted_epoch = task.admitted_epoch if task.admitted_epoch is not None else int(self.current_epoch)
            task.in_flight = True
            task.radio_ms_est = self.radio_ms_est(task)
            absolute_deadline_ms = self.absolute_deadline_ms(task)

            if payload_mode:
                task.legacy_radio_delivered = task.legacy_radio_delivered or (task.task_id in delivered)
                self._apply_payload_record(
                    task, helper, radio.get(int(task.task_id)),
                    interval_ms, now_ms, now_sec,
                    fallback_required=int(assignment.get("packet_count", 0) or 0),
                )
                continue

            # ---- archived first-packet accounting (legacy reproduction) ----
            if task.task_id in delivered:
                task.radio_delivered = True
                task.legacy_radio_delivered = True
                if self.spread_arrivals:
                    start_ms = max(now_ms + interval_ms, helper.ready_time_ms)
                    compute_ms = 1000.0 * task.cycles / helper.cpu_cycles_s
                    finish_absolute_ms = start_ms + compute_ms

                    src_kin = self.source_kinematics_at(task.source_id, now_sec)
                    hlp_kin = self.helper_kinematics_at(helper, now_sec)
                    t_contact = 999.0
                    if src_kin is not None and hlp_kin is not None:
                        t_contact = compute_contact_duration(src_kin, hlp_kin, 0.0, self.r_comm)

                    finish_rel_sec = max(0.0, finish_absolute_ms - now_ms) / 1000.0
                    contact_ok = (finish_rel_sec <= t_contact + 1e-5)
                    deadline_ok = (finish_absolute_ms <= absolute_deadline_ms)

                    task.finish_estimate_ms = finish_absolute_ms - now_ms
                    if deadline_ok and contact_ok:
                        task.queued = True
                        task.completion_time_ms = finish_absolute_ms
                        helper.ready_time_ms = finish_absolute_ms
                        helper.queue_cycles = max(
                            0.0,
                            (helper.ready_time_ms - (now_ms + interval_ms))
                            * helper.cpu_cycles_s / 1000.0,
                        )
                        if finish_absolute_ms <= now_ms + interval_ms:
                            task.done = True
                            task.queued = False
                            task.in_flight = False
                else:
                    finish_ms = interval_ms + 1000.0 * task.cycles / helper.cpu_cycles_s
                    task.finish_estimate_ms = finish_ms
                    task.completion_time_ms = now_ms + finish_ms
                    if finish_ms <= task.deadline_ms:
                        task.done = True
                        task.in_flight = False
                        helper.queue_cycles = max(0.0, helper.queue_cycles - task.cycles)
            else:
                task.radio_delivered = False
                if not self.spread_arrivals:
                    helper.queue_cycles += task.cycles
                    task.finish_estimate_ms = (
                        interval_ms + 1000.0 * task.cycles / helper.cpu_cycles_s
                    )

        if payload_mode:
            # A payload may span several control epochs.  Keep crediting units
            # and finalize the task even on epochs where the policy does not
            # re-propose it, as long as its lifecycle is still active.
            seen = {int(a["task_id"]) for a in assignments}
            for task in self.tasks:
                if task.done or task.cancelled or task.task_id in seen:
                    continue
                if not task.in_flight or task.assigned_helper_id is None:
                    continue
                rec = radio.get(int(task.task_id))
                if rec is None:
                    continue
                helper = next(
                    (h for h in self.helpers if h.helper_id == task.assigned_helper_id), None
                )
                if helper is None:
                    continue
                self._apply_payload_record(
                    task, helper, rec, interval_ms, now_ms, now_sec,
                    fallback_required=int(task.payload_required or 0),
                )

    def _apply_payload_record(
        self,
        task: Task,
        helper: Helper,
        rec: dict | None,
        interval_ms: float,
        now_ms: float,
        now_sec: float,
        fallback_required: int = 0,
    ):
        """Credit a task only once its complete payload has reached the helper."""
        required = int(rec.get("required_packets", 0) or 0) if rec else int(fallback_required or 0)
        received = int(rec.get("received_packets", 0) or 0) if rec else 0
        if required > 0:
            task.payload_required = required
            task.payload_received = min(received, required)
        payload_ok = bool(
            rec is not None
            and required > 0
            and received >= required
            and rec.get("complete")
        )
        if not payload_ok:
            # Incomplete payload: the helper cannot start computation.  The
            # task remains in flight; its outstanding units are retransmitted
            # by the pacing queue, not re-injected as a new full payload.
            task.radio_delivered = False
            task.finish_estimate_ms = None
            return

        task.radio_delivered = True
        complete_epoch = rec.get("complete_epoch")
        complete_time_s = rec.get("complete_time_s")
        if complete_time_s is not None:
            try:
                complete_time_s = float(complete_time_s)
            except (TypeError, ValueError):
                complete_time_s = None
            if complete_time_s is not None and not math.isfinite(complete_time_s):
                complete_time_s = None
        if complete_time_s is not None:
            # Physical arrival time (seconds) from the WiLab accounting hook.
            # Do not quantize it to the 20 ms controller epoch: the payload is
            # paced on the 5 ms reservation clock and the deadline is physical.
            task.radio_complete_ms = 1000.0 * complete_time_s
            task.radio_complete_epoch = (
                int(complete_epoch) if complete_epoch is not None
                else int(self.current_epoch)
            )
        elif complete_epoch is not None:
            task.radio_complete_epoch = int(complete_epoch)
            task.radio_complete_ms = self.epoch_ms * float(complete_epoch)
        elif task.radio_complete_ms is None:
            task.radio_complete_epoch = int(self.current_epoch)
            task.radio_complete_ms = now_ms
        arrival_ms = float(task.radio_complete_ms)
        absolute_deadline_ms = self.absolute_deadline_ms(task)
        compute_ms = 1000.0 * task.cycles / helper.cpu_cycles_s

        if self.spread_arrivals:
            start_ms = max(arrival_ms, helper.ready_time_ms)
            finish_absolute_ms = start_ms + compute_ms
            src_kin = self.source_kinematics_at(task.source_id, now_sec)
            hlp_kin = self.helper_kinematics_at(helper, now_sec)
            t_contact = 999.0
            if src_kin is not None and hlp_kin is not None:
                t_contact = compute_contact_duration(src_kin, hlp_kin, 0.0, self.r_comm)
            finish_rel_sec = max(0.0, finish_absolute_ms - now_ms) / 1000.0
            contact_ok = (finish_rel_sec <= t_contact + 1e-5)
            deadline_ok = (finish_absolute_ms <= absolute_deadline_ms + 1e-9)
            task.finish_estimate_ms = finish_absolute_ms - arrival_ms
            if deadline_ok and contact_ok:
                task.queued = True
                task.completion_time_ms = finish_absolute_ms
                helper.ready_time_ms = finish_absolute_ms
                helper.queue_cycles = max(
                    0.0,
                    (helper.ready_time_ms - arrival_ms)
                    * helper.cpu_cycles_s / 1000.0,
                )
                if finish_absolute_ms <= now_ms:
                    task.done = True
                    task.queued = False
                    task.in_flight = False
            else:
                task.cancelled = True
                task.in_flight = False
                task.queued = False
                task.cancel_reason = "deadline_or_contact"
        else:
            completion_absolute_ms = max(arrival_ms, helper.ready_time_ms) + compute_ms
            task.finish_estimate_ms = completion_absolute_ms - arrival_ms
            task.completion_time_ms = completion_absolute_ms
            if completion_absolute_ms <= absolute_deadline_ms + 1e-9:
                task.done = True
                task.in_flight = False
                helper.queue_cycles = max(0.0, helper.queue_cycles - task.cycles)
            else:
                task.cancelled = True
                task.in_flight = False
                task.cancel_reason = "deadline"

    def completed_value(self):
        return sum(t.value for t in self.tasks if t.done)

    def state(self):
        return {"tasks": [asdict(t) for t in self.tasks], "helpers": [asdict(h) for h in self.helpers]}
