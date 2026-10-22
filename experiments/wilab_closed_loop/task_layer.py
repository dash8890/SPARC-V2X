"""Application task/helper layer for the isolated WiLab bridge.

WiLabV2Xsim does not model application tasks or helper CPUs. This module keeps
those semantics explicit on the Python side and exposes only the selected
task/helper packets to the MATLAB bridge.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import random


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


class TaskLayer:
    def __init__(self, density: int, seed: int, tasks_per_source: int = 3,
                 computation_packet_bytes: int = 1000,
                 spread_arrivals: bool = False,
                 controller_epochs: int = 100,
                 deadline_multiplier: float = 1.0,
                 helper_cpu_lo: float = 2e8,
                 helper_cpu_hi: float = 1e9,
                 cluster_share: float | None = None):
        self.rng = random.Random(seed)
        self.computation_packet_bytes = int(computation_packet_bytes)
        lo = float(helper_cpu_lo)
        hi = float(helper_cpu_hi)
        if hi < lo:
            lo, hi = hi, lo
        self.helpers = [Helper(i, self.rng.uniform(lo, hi)) for i in range(min(12, density))]
        if cluster_share is not None:
            n_sources = max(1, min(density, int(round(density * float(cluster_share)))))
        else:
            n_sources = max(1, density // 5)
        sources = list(range(n_sources))
        self.tasks = []
        self.spread_arrivals = bool(spread_arrivals)
        self.controller_epochs = int(controller_epochs)
        self.current_epoch = 0
        deadline_multiplier = max(0.05, float(deadline_multiplier))
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
        """Advance application time and finalize queued computation."""
        self.current_epoch = int(epoch)
        if not self.spread_arrivals:
            return
        now_ms = 10.0 * self.current_epoch
        for task in self.tasks:
            if task.queued and not task.done and task.completion_time_ms is not None:
                absolute_deadline_ms = 10.0 * task.release_epoch + task.deadline_ms
                if task.completion_time_ms <= now_ms and task.completion_time_ms <= absolute_deadline_ms:
                    task.done = True
                    task.queued = False
        for helper in self.helpers:
            helper.queue_cycles = max(0.0, (helper.ready_time_ms - now_ms) * helper.cpu_cycles_s / 1000.0)

    def candidates(self, cbr: float, time_fraction: float):
        out = []
        for task in self.tasks:
            if task.done or task.queued:
                continue
            if self.spread_arrivals:
                if task.release_epoch > self.current_epoch:
                    continue
                if 10.0 * self.current_epoch > 10.0 * task.release_epoch + task.deadline_ms:
                    continue
            for helper in self.helpers:
                now_ms = 10.0 * self.current_epoch
                compute_ms = 1000.0 * task.cycles / helper.cpu_cycles_s
                # Match apply_radio_outcomes: work starts after this controller
                # interval (or when the helper is free), not at "now".
                interval_ms = 10.0
                if self.spread_arrivals:
                    start_ms = max(now_ms + interval_ms, helper.ready_time_ms)
                    finish_absolute_ms = start_ms + compute_ms
                    finish_ms = finish_absolute_ms - now_ms
                    absolute_deadline_ms = 10.0 * task.release_epoch + task.deadline_ms
                    # Value-Aware may only rank helpers that can meet the deadline.
                    # Scoring value/finish on infeasible pairs made WiLab VA prefer
                    # fast-but-late assignments (~80% top picks missed the deadline).
                    if finish_absolute_ms > absolute_deadline_ms:
                        continue
                else:
                    finish_ms = interval_ms + compute_ms
                    if finish_ms > task.deadline_ms:
                        continue
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

    def apply_radio_outcomes(self, assignments, delivered_task_ids, interval_ms: float):
        delivered = set(delivered_task_ids)
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
                    now_ms = 10.0 * self.current_epoch
                    start_ms = max(now_ms + interval_ms, helper.ready_time_ms)
                    finish_absolute_ms = start_ms + 1000.0 * task.cycles / helper.cpu_cycles_s
                    absolute_deadline_ms = 10.0 * task.release_epoch + task.deadline_ms
                    task.finish_estimate_ms = finish_absolute_ms - now_ms
                    if finish_absolute_ms <= absolute_deadline_ms:
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
