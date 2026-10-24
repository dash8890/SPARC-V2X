"""Acceptance tests for the corrected payload-faithful contract.

These tests are the gate before any retraining:

1. geometry agreement between the Python task layer and the WiLab trace,
2. disjoint source/helper WiLab IDs,
3. one physical clock (20 ms) shared by controller, radio, and task layer,
4. payload conservation and completion only after all units,
5. deadline enforcement (a complete payload that arrives late earns nothing),
6. safety metrics reported as short-window, long-window, cumulative, and
   zero-offload-relative quantities.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.wilab_closed_loop.metrics_logger import safety_from_audit_list
from experiments.wilab_closed_loop.task_layer import Helper, Task, TaskLayer
from experiments.wilab_closed_loop.vehicle_trace import (
    VehicleTrace,
    resolve_trace_path,
    vehicle_id_map,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Geometry agreement
# ---------------------------------------------------------------------------
def test_geometry_matches_shared_wilab_trace():
    trace_path = resolve_trace_path(ROOT, 15)
    assert trace_path.is_file(), trace_path
    trace = VehicleTrace.load(trace_path)
    layer = TaskLayer(
        density=15, seed=1, tasks_per_source=1, controller_epochs=10,
        trace_path=trace_path, epoch_ms=20.0,
    )
    assert layer.n_sources == 3
    assert layer.n_helpers == 12
    assert layer.vehicle_map["sources"] == {0: 1, 1: 2, 2: 3}
    assert layer.vehicle_map["helpers"] == {j: 4 + j for j in range(12)}
    for t_s in (0.0, 0.25, 1.0, 2.5):
        expected = trace.distance(1, 4, t_s)
        assert layer.pair_distance_m(0, 0, t_s) == pytest.approx(expected, abs=1e-6)


def test_source_and_helper_ids_are_disjoint():
    for density, n_src, n_hlp in ((15, 3, 12), (35, 7, 12)):
        mapping = vehicle_id_map(density, n_src, n_hlp)
        src = set(mapping["sources"].values())
        hlp = set(mapping["helpers"].values())
        assert src.isdisjoint(hlp)
        assert src == set(range(1, n_src + 1))
        assert hlp == set(range(n_src + 1, n_src + n_hlp + 1))
    with pytest.raises(ValueError):
        vehicle_id_map(10, 5, 6)


# ---------------------------------------------------------------------------
# 2. One physical clock
# ---------------------------------------------------------------------------
def test_physical_clock_is_20ms_and_drives_deadlines():
    layer = TaskLayer(
        density=15, seed=1, tasks_per_source=1, controller_epochs=100,
        epoch_ms=20.0, computation_packet_bytes=2000,
    )
    task = Task(
        task_id=0, source_id=0, payload_bytes=10_000, cycles=1e7,
        deadline_ms=150.0, value=1.0, release_epoch=3,
    )
    assert layer.epoch_ms == 20.0
    assert layer.release_ms(task) == pytest.approx(60.0)
    assert layer.absolute_deadline_ms(task) == pytest.approx(210.0)
    assert layer.packet_count(task) == 5
    # The 20 ms control clock is separate from the 5 ms physical reservation
    # clock: one admission reservation plus one reservation per TB.
    assert layer.radio_ms_est(task) == pytest.approx(30.0)

    layer.tasks = [task]
    layer.helpers = [Helper(helper_id=0, cpu_cycles_s=1e9)]
    layer.current_epoch = 3
    candidates = layer.candidates()
    assert candidates, "the 10 kB / 1e7-cycle / 1e9-Hz task must be admissible"
    best = min(candidates, key=lambda c: c["finish_ms"])
    assert best["finish_ms"] == pytest.approx(40.0)
    assert best["deadline_ms"] == pytest.approx(150.0)


def test_candidate_radio_estimate_grows_with_payload():
    layer = TaskLayer(
        density=15, seed=1, tasks_per_source=1, controller_epochs=10,
        epoch_ms=20.0, computation_packet_bytes=2000,
    )
    small = Task(task_id=0, source_id=0, payload_bytes=10_000, cycles=1e7,
                 deadline_ms=150.0, value=1.0)
    large = Task(task_id=1, source_id=0, payload_bytes=20_000, cycles=1e7,
                 deadline_ms=150.0, value=1.0)
    assert layer.packet_count(small) == 5
    assert layer.packet_count(large) == 10
    assert layer.radio_ms_est(large) > layer.radio_ms_est(small)


# ---------------------------------------------------------------------------
# 3. Payload conservation and completion
# ---------------------------------------------------------------------------
def _layer_with_task(*, payload_bytes: int = 10_000, deadline_ms: float = 150.0,
                     release_epoch: int = 0) -> TaskLayer:
    layer = TaskLayer(
        density=15, seed=1, tasks_per_source=1, controller_epochs=100,
        epoch_ms=20.0, computation_packet_bytes=2000, spread_arrivals=True,
    )
    layer.tasks = [Task(
        task_id=0, source_id=0, payload_bytes=payload_bytes, cycles=1e7,
        deadline_ms=deadline_ms, value=2.0, release_epoch=release_epoch,
    )]
    layer.helpers = [Helper(helper_id=0, cpu_cycles_s=1e9)]
    layer.current_epoch = release_epoch + 1
    return layer


def _assignment(packet_count: int, payload_bytes: int = 10_000) -> dict:
    return {
        "task_id": 0, "source_id": 0, "helper_id": 0,
        "packet_count": packet_count, "payload_bytes": payload_bytes,
    }


def _radio(received: int, required: int, complete: bool,
           epoch: int = 1) -> dict:
    return {
        "task_id": 0, "source_id": 1, "helper_id": 1,
        "required_packets": required, "received_packets": received,
        "complete": complete,
        "complete_epoch": epoch if complete else None,
    }


def test_partial_payload_is_not_credited():
    layer = _layer_with_task()
    layer.apply_radio_outcomes(
        [_assignment(5)], [], 20.0,
        task_radio=[_radio(4, 5, complete=False)],
    )
    task = layer.tasks[0]
    assert task.payload_required == 5
    assert task.payload_received == 4
    assert task.done is False
    assert layer.completed_value() == 0.0


def test_all_units_before_deadline_completes_exactly_once():
    layer = _layer_with_task()
    layer.apply_radio_outcomes(
        [_assignment(5)], [], 20.0,
        task_radio=[_radio(5, 5, complete=True, epoch=1)],
    )
    task = layer.tasks[0]
    # The payload is complete; computation starts immediately at the physical
    # arrival time (no artificial 20 ms control turnaround).
    assert task.payload_received == 5
    assert task.queued is True
    assert task.completion_time_ms == pytest.approx(30.0)
    assert task.done is False
    layer.set_epoch(3)                       # now = 60 ms
    assert task.done is True
    assert layer.completed_value() == pytest.approx(2.0)
    # A later duplicate report must not add value or inflate the unit count.
    layer.apply_radio_outcomes(
        [_assignment(5)], [], 20.0,
        task_radio=[_radio(9, 5, complete=True, epoch=3)],
    )
    assert layer.tasks[0].payload_received == 5
    assert layer.completed_value() == pytest.approx(2.0)


def test_complete_payload_after_deadline_earns_nothing():
    # Released at epoch 0, deadline 50 ms; the last unit arrives at epoch 5
    # (100 ms) and computation would finish at 120 ms.
    layer = _layer_with_task(deadline_ms=50.0)
    layer.apply_radio_outcomes(
        [_assignment(5)], [], 20.0,
        task_radio=[_radio(5, 5, complete=True, epoch=5)],
    )
    task = layer.tasks[0]
    assert task.radio_delivered is True
    assert task.done is False
    assert layer.completed_value() == 0.0


# ---------------------------------------------------------------------------
# 4. Safety metric separation and zero-offload floor
# ---------------------------------------------------------------------------
def _audit(prr: float, long_prr: float, cum_prr: float, long_valid: bool = True) -> list[dict]:
    return [
        {
            "cbr_valid": True,
            "delay_valid": True,
            "prr": prr,
            "p95_delay_ms": 10.0,
            "window_prr": prr,
            "window_prr_long": long_prr,
            "window_prr_long_valid": long_valid,
            "safety_prr_cumulative": cum_prr,
            "safety_p95_delay_cumulative_ms": 10.0,
        }
        for _ in range(3)
    ]


def test_safety_metrics_are_reported_separately():
    audit = [
        _audit(0.0, 0.99, 0.99)[0],
        _audit(1.0, 0.98, 0.98)[0],
        _audit(1.0, 0.97, 0.98)[0],
    ]
    safety = safety_from_audit_list(audit)
    # The short window is a small-sample quantity and can read 0.0.
    assert safety["P_viol"] == pytest.approx(1.0 / 3.0)
    # The long window and cumulative PRR are the claim quantities.
    assert safety["P_viol_long"] == pytest.approx(0.0)
    assert safety["P_prr_fail_long"] == pytest.approx(0.0)
    assert safety["mean_prr_long"] == pytest.approx((0.99 + 0.98 + 0.97) / 3.0)
    assert safety["P_viol_cum"] == pytest.approx(0.0)
    assert safety["compliant"] is True


def test_zero_offload_floor_is_subtracted():
    audit = _audit(1.0, 0.98, 0.98)
    safety = safety_from_audit_list(
        audit,
        zero_offload={"P_viol": 0.5, "P_viol_long": 0.4, "PRR_cum": 0.90},
    )
    assert safety["Delta_P_viol"] == pytest.approx(-0.5)
    assert safety["Delta_P_viol_long"] == pytest.approx(-0.4)
    assert safety["Delta_PRR_cum"] == pytest.approx(0.08)
    assert safety["zero_offload_P_viol"] == 0.5
    assert safety["zero_offload_PRR_cum"] == 0.90


def test_invalid_long_window_falls_back_to_cumulative_claim():
    audit = _audit(0.0, 0.98, 0.98, long_valid=False)
    safety = safety_from_audit_list(audit)
    assert safety["P_viol_long"] is None
    assert safety["P_viol_cum"] == pytest.approx(0.0)
    assert safety["compliant"] is True
