"""Application task/helper layer with continuous kinematics and persistent helper queues.

WiLabV2Xsim does not model application tasks, helper CPU queues, or vehicular contact windows.
This module keeps those semantics explicit on the Python side, maintaining causal Mode-2
sidelink execution semantics.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class VehicleKinematics:
    veh_id: int
    is_source: bool
    x_0: float          # longitudinal position at t=0 (meters)
    y: float            # lateral lane coordinate (meters)
    velocity: float     # constant longitudinal speed (m/s)

    def x(self, t_sec: float) -> float:
        return self.x_0 + self.velocity * t_sec


def compute_contact_duration(
    v_src: VehicleKinematics,
    v_hlp: VehicleKinematics,
    t_sec: float,
    r_comm: float = 250.0,
) -> float:
    """Exact multi-lane Euclidean forward contact window (Definition 1 in specification)."""
    delta_y = v_src.y - v_hlp.y
    if abs(delta_y) > r_comm:
        return 0.0

    x_max = math.sqrt(r_comm ** 2 - delta_y ** 2)
    delta_x = v_src.x(t_sec) - v_hlp.x(t_sec)
    delta_v = v_src.velocity - v_hlp.velocity

    if abs(delta_v) < 1e-4:
        if -x_max <= delta_x <= x_max:
            return 999.0  # Traveling at identical speeds in range
        return 0.0

    t1 = (-x_max - delta_x) / delta_v
    t2 = (x_max - delta_x) / delta_v
    t_min = min(t1, t2)
    t_max = max(t1, t2)

    # Vehicles are currently in contact if t_min <= 0 <= t_max
    if t_min <= 0.0 <= t_max:
        return max(0.0, t_max)
    elif t_max < 0.0:
        return 0.0  # Contact already passed
    else:
        # Approaching but not yet within range
        return 0.0


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


@dataclass
class Helper:
    helper_id: int
    cpu_cycles_s: float
    queue_cycles: float = 0.0
    ready_time_ms: float = 0.0
    kinematics: VehicleKinematics | None = None


class TaskLayer:
    def __init__(
        self,
        density: int,
        seed: int,
        tasks_per_source: int = 3,
        computation_packet_bytes: int = 1000,
        spread_arrivals: bool = False,
        controller_epochs: int = 100,
        deadline_multiplier: float = 1.0,
        helper_cpu_lo: float = 2e8,
        helper_cpu_hi: float = 1e9,
        cluster_share: float | None = None,
        arrival_mode: str = "legacy",
        r_comm: float = 250.0,
    ):
        self.rng = random.Random(seed)
        self.computation_packet_bytes = int(computation_packet_bytes)
        self.arrival_mode = str(arrival_mode)
        self.r_comm = float(r_comm)
        self.controller_epochs = int(controller_epochs)
        self.current_epoch = 0

        lo = float(helper_cpu_lo)
        hi = float(helper_cpu_hi)
        if hi < lo:
            lo, hi = hi, lo

        # Helpers
        n_helpers = min(12, density)
        self.helpers = []
        for i in range(n_helpers):
            v_speed = min(25.0, max(15.0, self.rng.gauss(20.0, 2.5)))
            lane = self.rng.choice([4.0, 8.0, 12.0])
            x0 = self.rng.uniform(100.0, 900.0)
            kin = VehicleKinematics(i, is_source=False, x_0=x0, y=lane, velocity=v_speed)
            self.helpers.append(Helper(i, self.rng.uniform(lo, hi), kinematics=kin))

        # Sources
        if cluster_share is not None:
            n_sources = max(1, min(density, int(round(density * float(cluster_share)))))
        else:
            n_sources = max(1, density // 5)
        sources = list(range(n_sources))
        self.source_kinematics = {}
        for s in sources:
            v_speed = min(25.0, max(15.0, self.rng.gauss(20.0, 2.5)))
            lane = self.rng.choice([4.0, 8.0, 12.0])
            x0 = self.rng.uniform(100.0, 900.0)
            self.source_kinematics[s] = VehicleKinematics(s, is_source=True, x_0=x0, y=lane, velocity=v_speed)

        self.tasks = []
        self.spread_arrivals = bool(spread_arrivals or (self.arrival_mode in ("streaming", "mobility_streaming")))
        deadline_multiplier = max(0.05, float(deadline_multiplier))

        if self.arrival_mode in ("streaming", "mobility_streaming"):
            for source in sources:
                cur_ep = self.rng.randint(0, 4)
                for task_index in range(tasks_per_source):
                    release_epoch = min(self.controller_epochs - 1, cur_ep)
                    self.tasks.append(Task(
                        len(self.tasks), source,
                        self.rng.randint(10_000, 80_000),
                        self.rng.uniform(1e7, 1e8),
                        self.rng.uniform(30.0, 150.0) * deadline_multiplier,
                        self.rng.uniform(1.0, 5.0),
                        release_epoch=release_epoch,
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
                    release_epoch = (task_index * self.controller_epochs // max(1, tasks_per_source)) if self.spread_arrivals else 0
                    self.tasks.append(Task(
                        len(self.tasks), source,
                        self.rng.randint(10_000, 80_000),
                        self.rng.uniform(1e7, 1e8),
                        self.rng.uniform(30.0, 150.0) * deadline_multiplier,
                        self.rng.uniform(1.0, 5.0),
                        release_epoch=release_epoch,
                    ))

    def set_epoch(self, epoch: int):
        """Advance application time, drain helper queues, and finalize completed tasks."""
        self.current_epoch = int(epoch)
        now_ms = 10.0 * self.current_epoch

        if not self.spread_arrivals:
            return

        # Finalize tasks that completed computation
        for task in self.tasks:
            if task.queued and not task.done and task.completion_time_ms is not None:
                absolute_deadline_ms = 10.0 * task.release_epoch + task.deadline_ms
                if task.completion_time_ms <= now_ms and task.completion_time_ms <= absolute_deadline_ms:
                    task.done = True
                    task.queued = False

        # Drain helper backlog
        for helper in self.helpers:
            helper.queue_cycles = max(0.0, (helper.ready_time_ms - now_ms) * helper.cpu_cycles_s / 1000.0)

    def candidates(self, cbr: float = 0.0, time_fraction: float = 0.0):
        out = []
        now_ms = 10.0 * self.current_epoch
        now_sec = now_ms / 1000.0

        for task in self.tasks:
            if task.done or task.queued:
                continue
            if self.spread_arrivals:
                if task.release_epoch > self.current_epoch:
                    continue
                if now_ms > 10.0 * task.release_epoch + task.deadline_ms:
                    continue

            src_kin = self.source_kinematics.get(task.source_id)

            for helper in self.helpers:
                compute_ms = 1000.0 * task.cycles / helper.cpu_cycles_s
                interval_ms = 10.0

                if self.spread_arrivals:
                    start_ms = max(now_ms + interval_ms, helper.ready_time_ms)
                    finish_absolute_ms = start_ms + compute_ms
                    finish_ms = finish_absolute_ms - now_ms
                    absolute_deadline_ms = 10.0 * task.release_epoch + task.deadline_ms

                    # Contact feasibility check
                    t_contact = 999.0
                    if src_kin is not None and helper.kinematics is not None:
                        t_contact = compute_contact_duration(src_kin, helper.kinematics, now_sec, self.r_comm)

                    contact_feasible = bool(finish_ms / 1000.0 <= t_contact + 1e-5)
                    deadline_feasible = bool(finish_absolute_ms <= absolute_deadline_ms)

                    if not deadline_feasible or not contact_feasible:
                        continue
                else:
                    finish_ms = interval_ms + compute_ms
                    t_contact = 999.0
                    contact_feasible = True
                    deadline_feasible = bool(finish_ms <= task.deadline_ms)
                    if not deadline_feasible:
                        continue

                # Normalized candidate features (Section 3.2 in specification)
                rem_dl_ms = max(1.0, (10.0 * task.release_epoch + task.deadline_ms) - now_ms) if self.spread_arrivals else task.deadline_ms
                norm_val = task.value / 5.0
                lat_pressure = min(1.0, max(0.0, finish_ms / rem_dl_ms))
                occ_cost = min(1.0, (helper.queue_cycles + task.cycles) / (helper.cpu_cycles_s * max(0.01, t_contact)))
                rad_footprint = min(1.0, (task.payload_bytes / 175_000.0) * (1.0 + float(cbr)))

                out.append({
                    "task_id": task.task_id,
                    "source_id": task.source_id,
                    "helper_id": helper.helper_id,
                    "payload_bytes": task.payload_bytes,
                    "packet_count": (task.payload_bytes + self.computation_packet_bytes - 1) // self.computation_packet_bytes,
                    "analytic_score": task.value / max(finish_ms, 1e-6),
                    "finish_ms": finish_ms,
                    "deadline_ms": task.deadline_ms,
                    "task_value": task.value,
                    "deadline_feasible": True,
                    "contact_feasible": contact_feasible,
                    "t_contact_sec": t_contact,
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
                        task.payload_bytes / 1e6,
                        task.cycles / 1e8,
                        task.deadline_ms / 1000.0,
                        task.value / 5.0,
                        helper.cpu_cycles_s / 1e9,
                        helper.queue_cycles / 1e8,
                        0.0,
                        cbr,
                        self.current_epoch / max(1.0, float(self.controller_epochs)) if self.spread_arrivals else time_fraction,
                        1.0,
                    ],
                })
        return out

    def apply_radio_outcomes(self, assignments, delivered_task_ids, interval_ms: float = 10.0):
        delivered = set(delivered_task_ids)
        now_ms = 10.0 * self.current_epoch
        now_sec = now_ms / 1000.0

        for assignment in assignments:
            task = next((t for t in self.tasks if t.task_id == assignment["task_id"]), None)
            helper = next((h for h in self.helpers if h.helper_id == assignment["helper_id"]), None)
            if task is None or helper is None or task.done:
                continue

            task.assigned_helper_id = int(helper.helper_id)
            task.assign_epoch = int(self.current_epoch)

            if task.task_id in delivered:
                task.radio_delivered = True
                if self.spread_arrivals:
                    start_ms = max(now_ms + interval_ms, helper.ready_time_ms)
                    compute_ms = 1000.0 * task.cycles / helper.cpu_cycles_s
                    finish_absolute_ms = start_ms + compute_ms
                    absolute_deadline_ms = 10.0 * task.release_epoch + task.deadline_ms

                    src_kin = self.source_kinematics.get(task.source_id)
                    t_contact = 999.0
                    if src_kin is not None and helper.kinematics is not None:
                        t_contact = compute_contact_duration(src_kin, helper.kinematics, now_sec, self.r_comm)

                    finish_rel_sec = (finish_absolute_ms - now_ms) / 1000.0
                    contact_ok = (finish_rel_sec <= t_contact + 1e-5)
                    deadline_ok = (finish_absolute_ms <= absolute_deadline_ms)

                    task.finish_estimate_ms = finish_absolute_ms - now_ms
                    if deadline_ok and contact_ok:
                        task.queued = True
                        task.completion_time_ms = finish_absolute_ms
                        helper.ready_time_ms = finish_absolute_ms
                        helper.queue_cycles = max(0.0, (helper.ready_time_ms - (now_ms + interval_ms)) * helper.cpu_cycles_s / 1000.0)
                        if finish_absolute_ms <= now_ms + interval_ms:
                            task.done = True
                            task.queued = False
                else:
                    finish_ms = interval_ms + 1000.0 * task.cycles / helper.cpu_cycles_s
                    task.finish_estimate_ms = finish_ms
                    task.completion_time_ms = 10.0 * self.current_epoch + finish_ms
                    if finish_ms <= task.deadline_ms:
                        task.done = True
                        helper.queue_cycles = max(0.0, helper.queue_cycles - task.cycles)
            else:
                task.radio_delivered = False
                if not self.spread_arrivals:
                    helper.queue_cycles += task.cycles
                    task.finish_estimate_ms = interval_ms + 1000.0 * task.cycles / helper.cpu_cycles_s

    def completed_value(self):
        return sum(t.value for t in self.tasks if t.done)

    def state(self):
        return {"tasks": [asdict(t) for t in self.tasks], "helpers": [asdict(h) for h in self.helpers]}
