"""Packet-level tests for payload-faithful task completion.

The legacy bridge credited a task as soon as *any* packet from its source was
received.  These tests pin the corrected contract:

    z_k = 1{ all required payload units received by the helper
             AND helper computation finished by the task deadline }

They also pin the normalisation of the MATLAB ``task_radio`` field and the
legacy reproduction mode used only to re-run archived configurations.
"""
from __future__ import annotations

import pytest

from experiments.wilab_closed_loop.task_layer import (
    Helper,
    Task,
    TaskLayer,
    _normalize_task_radio,
)


def make_layer(*, spread: bool = False, release_epoch: int = 0,
               deadline_ms: float = 100.0, cycles: float = 1e7) -> TaskLayer:
    # 20 ms is the physical controller/Mode-2 reservation epoch used by the
    # frozen evaluation configuration.  The unit tests pin that clock instead
    # of the legacy 10 ms placeholder.
    layer = TaskLayer(density=15, seed=1, tasks_per_source=1,
                      controller_epochs=100, spread_arrivals=spread,
                      epoch_ms=20.0)
    layer.tasks = [Task(
        task_id=0, source_id=0, payload_bytes=3000, cycles=cycles,
        deadline_ms=deadline_ms, value=1.0, release_epoch=release_epoch,
    )]
    layer.helpers = [Helper(helper_id=0, cpu_cycles_s=1e9)]
    layer.current_epoch = 1
    return layer


def assignment(packet_count: int = 3) -> dict:
    return {
        "task_id": 0, "helper_id": 0, "source_id": 0,
        "packet_count": packet_count, "payload_bytes": 3000,
    }


def radio(*, received: int, required: int = 3, complete: bool,
          complete_epoch: int | None = 1) -> dict:
    return {
        "task_id": 0,
        "source_id": 1,
        "helper_id": 1,
        "required_packets": required,
        "received_packets": received,
        "complete": complete,
        "complete_epoch": complete_epoch if complete else None,
    }


# ---------------------------------------------------------------------------
# Normalisation of the MATLAB struct-array boundary
# ---------------------------------------------------------------------------
def test_normalize_task_radio_handles_scalar_array_and_empty():
    rec = radio(received=1, complete=False)
    assert _normalize_task_radio(None) == {}
    assert _normalize_task_radio([]) == {}
    assert _normalize_task_radio(rec) == {0: rec}
    assert _normalize_task_radio([rec]) == {0: rec}


# ---------------------------------------------------------------------------
# Non-spread (legacy arrival mode) payload contract
# ---------------------------------------------------------------------------
def test_partial_payload_never_completes():
    layer = make_layer()
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=2, complete=False)],
    )
    task = layer.tasks[0]
    assert task.payload_required == 3
    assert task.payload_received == 2
    assert task.radio_delivered is False
    assert task.done is False
    assert layer.completed_value() == 0.0


def test_first_packet_is_not_enough_even_if_legacy_flag_fires():
    layer = make_layer()
    # Legacy delivery flag says "something arrived", but only 1 of 3 units did.
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=1, complete=False)],
    )
    task = layer.tasks[0]
    assert task.legacy_radio_delivered is True
    assert task.done is False


def test_complete_payload_before_deadline_completes():
    layer = make_layer()
    # Physical clock: payload arrival at epoch 1 (20 ms) + 10 ms compute.
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=3, complete=True, complete_epoch=1)],
    )
    task = layer.tasks[0]
    assert task.radio_delivered is True
    assert task.done is True
    assert task.radio_complete_epoch == 1
    assert task.completion_time_ms == pytest.approx(30.0)
    assert layer.completed_value() == pytest.approx(1.0)


def test_complete_payload_after_deadline_does_not_complete():
    # Released at epoch 0 with a 50 ms deadline, but the last unit only
    # arrives at epoch 10 (100 ms).  Completion is 120 ms, past the deadline.
    layer = make_layer(deadline_ms=50.0)
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=3, complete=True, complete_epoch=10)],
    )
    task = layer.tasks[0]
    assert task.radio_delivered is True
    assert task.done is False


def test_missing_radio_record_does_not_complete():
    layer = make_layer()
    layer.apply_radio_outcomes([assignment()], [0], 10.0, task_radio=None)
    task = layer.tasks[0]
    assert task.done is False
    assert task.radio_delivered is False


def test_received_units_are_capped_at_required():
    layer = make_layer()
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=9, required=3, complete=True)],
    )
    task = layer.tasks[0]
    assert task.payload_received == 3
    assert task.done is True


def test_payload_completes_even_without_a_new_assignment():
    layer = make_layer()
    # Epoch 1: assigned, one unit arrives.
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=1, complete=False)],
    )
    assert layer.tasks[0].done is False
    # Epoch 2: policy does not re-propose the task, but the remaining units
    # arrive.  The task must still be finalized.
    layer.current_epoch = 2
    layer.apply_radio_outcomes(
        [], [0], 10.0,
        task_radio=[radio(received=3, complete=True, complete_epoch=2)],
    )
    task = layer.tasks[0]
    assert task.done is True
    assert task.radio_complete_epoch == 2


# ---------------------------------------------------------------------------
# Streaming/spread arrival mode
# ---------------------------------------------------------------------------
def test_spread_mode_requires_full_payload():
    layer = make_layer(spread=True)
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=2, complete=False)],
    )
    assert layer.tasks[0].done is False
    assert layer.tasks[0].queued is False

    layer.current_epoch = 2
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=3, complete=True, complete_epoch=2)],
    )
    task = layer.tasks[0]
    assert task.radio_delivered is True
    assert task.queued is True
    assert task.done is False  # computation still running
    layer.set_epoch(10)
    assert task.done is True


# ---------------------------------------------------------------------------
# Archived legacy reproduction
# ---------------------------------------------------------------------------
def test_legacy_mode_still_credits_first_packet():
    layer = make_layer()
    layer.apply_radio_outcomes(
        [assignment()], [0], 10.0,
        task_radio=[radio(received=1, complete=False)],
        accounting="legacy",
    )
    task = layer.tasks[0]
    assert task.done is True
    assert task.radio_delivered is True
    assert layer.completed_value() == pytest.approx(1.0)


def test_legacy_mode_ignores_payload_record():
    layer = make_layer()
    layer.apply_radio_outcomes(
        [assignment()], [], 10.0,
        task_radio=[radio(received=3, complete=True)],
        accounting="legacy",
    )
    assert layer.tasks[0].done is False
