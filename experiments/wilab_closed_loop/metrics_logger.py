"""Comprehensive performance-metric export for SPARC-V2X evaluation.

Writes per-episode machine-readable traces:
  - episode_summary.csv / .json
  - epoch_trace.csv
  - task_trace.csv
  - decision_trace.csv

Also provides campaign-level helpers and training_history export.
Designed to capture every field that can serve as a performance metric
(computation, safety, offloading, deadline, scheduling, helper, admission,
radio, complexity). Missing physical quantities are left blank rather than
invented.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _f(x: Any, default: float | None = None) -> float | None:
    if x is None:
        return default
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _i(x: Any, default: int | None = None) -> int | None:
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _sum_payload(assignments: list | None) -> int:
    if not assignments:
        return 0
    return int(sum(int(a.get("payload_bytes") or 0) for a in assignments))


def _sum_packets(assignments: list | None) -> int:
    if not assignments:
        return 0
    return int(sum(int(a.get("packet_count") or 0) for a in assignments))


def enrich_audit_epoch(
    epoch_rec: dict,
    *,
    delivered_task_ids: list[int] | None = None,
    gate_blocked: bool = False,
) -> dict:
    """Add derived offloading / admission scalars onto one audit epoch dict."""
    proposed = epoch_rec.get("proposed_assignments") or []
    selected = epoch_rec.get("selected_assignments") or []
    proposed_bytes = _sum_payload(proposed)
    executed_bytes = _sum_payload(selected)
    proposed_packets = _sum_packets(proposed)
    executed_packets = _sum_packets(selected)
    delivered = set(int(t) for t in (delivered_task_ids or []))
    delivered_bytes = int(
        sum(int(a.get("payload_bytes") or 0) for a in selected if int(a.get("task_id", -1)) in delivered)
    )
    delivered_packets = int(
        sum(int(a.get("packet_count") or 0) for a in selected if int(a.get("task_id", -1)) in delivered)
    )
    wasted_bytes = max(0, executed_bytes - delivered_bytes)
    suppressed_bytes = max(0, proposed_bytes - executed_bytes)
    reason = epoch_rec.get("governor_reason") or "none"
    epoch_rec.update(
        {
            "proposed_bytes": proposed_bytes,
            "executed_bytes": executed_bytes,
            "delivered_bytes": delivered_bytes,
            "wasted_bytes": wasted_bytes,
            "suppressed_bytes": suppressed_bytes,
            "proposed_packets": proposed_packets,
            "executed_packets": executed_packets,
            "delivered_packets": delivered_packets,
            "n_proposed": len(proposed),
            "n_executed": len(selected),
            "n_delivered_tasks": len(delivered),
            "admission_gate_blocked": bool(gate_blocked),
            "admission_fallback": reason == "over_budget_or_invalid_assignment",
            "admission_rejected_proposal": bool(proposed) and not selected and reason not in ("none", "disabled"),
        }
    )
    return epoch_rec


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Stable union of keys, first-row order then extras sorted
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in keys:
                v = r.get(k)
                if isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, separators=(",", ":"))
                else:
                    out[k] = v
            w.writerow(out)


def build_epoch_rows(audit: list[dict], meta: dict) -> list[dict]:
    rows = []
    for e in audit:
        row = {
            **{k: meta.get(k) for k in (
                "method", "sim_seed", "task_seed", "density", "episode_id",
                "envelope_bytes", "policy_mode", "checkpoint",
            )},
            "epoch": e.get("epoch"),
            "prr": e.get("prr"),
            "p95_delay_ms": e.get("p95_delay_ms"),
            "cbr": e.get("cbr"),
            "cbr_valid": e.get("cbr_valid"),
            "delay_valid": e.get("delay_valid"),
            "safety_window_valid": bool(e.get("cbr_valid") and e.get("delay_valid")),
            "K_t": e.get("K_t"),
            "n_candidates": e.get("n_candidates"),
            "num_selected": e.get("num_selected"),
            "n_proposed": e.get("n_proposed"),
            "n_executed": e.get("n_executed"),
            "n_delivered_tasks": e.get("n_delivered_tasks"),
            "proposed_bytes": e.get("proposed_bytes"),
            "executed_bytes": e.get("executed_bytes", e.get("computation_bytes_offered")),
            "delivered_bytes": e.get("delivered_bytes"),
            "wasted_bytes": e.get("wasted_bytes"),
            "suppressed_bytes": e.get("suppressed_bytes"),
            "proposed_packets": e.get("proposed_packets"),
            "executed_packets": e.get("executed_packets", e.get("computation_packets_offered")),
            "delivered_packets": e.get("delivered_packets"),
            "computation_packets_rx_delta": e.get("computation_packets_rx_delta"),
            "computation_packet_bytes": e.get("computation_packet_bytes"),
            "wilab_enqueued_packets": e.get("wilab_enqueued_packets"),
            "wilab_enqueued_bytes": e.get("wilab_enqueued_bytes"),
            "wilab_scheduled_attempts": e.get("wilab_scheduled_attempts"),
            "wilab_pending_packets": e.get("wilab_pending_packets"),
            "window_prr": e.get("window_prr"),
            "window_prr_long": e.get("window_prr_long"),
            "window_prr_long_valid": e.get("window_prr_long_valid"),
            "window_attempts": e.get("window_attempts"),
            "window_duration_ms": e.get("window_duration_ms"),
            "safety_prr_cumulative": e.get("safety_prr_cumulative"),
            "safety_p95_delay_cumulative_ms": e.get("safety_p95_delay_cumulative_ms"),
            "safety_rx_cumulative": e.get("safety_rx_cumulative"),
            "safety_error_cumulative": e.get("safety_error_cumulative"),
            "packet_paced": e.get("packet_paced"),
            "packet_retransmitted": e.get("packet_retransmitted"),
            "packet_cancelled": e.get("packet_cancelled"),
            "packet_dropped": e.get("packet_dropped"),
            "packet_pending": e.get("packet_pending"),
            "packet_overflow_losses": e.get("packet_overflow_losses"),
            "packet_tag_deficits": e.get("packet_tag_deficits"),
            "packet_capture_missing": e.get("packet_capture_missing"),
            "packet_units_credited": e.get("packet_units_credited"),
            "packet_units_duplicate": e.get("packet_units_duplicate"),
            "payload_complete_task_ids": e.get("payload_complete_task_ids"),
            "governor_intervened": e.get("governor_intervened"),
            "governor_reason": e.get("governor_reason"),
            "governor_max_bytes": e.get("governor_max_bytes"),
            "admission_gate_blocked": e.get("admission_gate_blocked"),
            "admission_fallback": e.get("admission_fallback"),
            "admission_rejected_proposal": e.get("admission_rejected_proposal"),
            "inference_ms": e.get("inference_ms"),
            "candidate_gen_ms": e.get("candidate_gen_ms"),
            "score_select_ms": e.get("score_select_ms"),
            "governor_ms": e.get("governor_ms"),
            "reward": e.get("reward"),
            "helper_payload_bytes_json": e.get("helper_payload_bytes"),
            "helper_packet_counts_json": e.get("helper_packet_counts"),
            "delivered_source_ids_json": e.get("delivered_source_ids"),
            "selected_assignments_json": e.get("selected_assignments"),
            "proposed_assignments_json": e.get("proposed_assignments"),
        }
        rows.append(row)
    return rows


def build_task_rows(layer, audit: list[dict], meta: dict) -> list[dict]:
    """Task-level outcomes from TaskLayer + assignment history on tasks."""
    # Map task_id -> first/last assignment epochs from audit
    first_assign: dict[int, dict] = {}
    last_assign: dict[int, dict] = {}
    delivered_epochs: dict[int, int] = {}
    for e in audit:
        epoch = int(e.get("epoch") or 0)
        delivered_sources = {int(x) for x in (e.get("delivered_source_ids") or [])}
        for a in e.get("selected_assignments") or []:
            tid = int(a.get("task_id"))
            rec = {
                "epoch": epoch,
                "helper_id": a.get("helper_id"),
                "source_id": a.get("source_id"),
                "payload_bytes": a.get("payload_bytes"),
                "packet_count": a.get("packet_count"),
                "radio_delivered": int(a.get("source_id", -1)) in delivered_sources,
            }
            first_assign.setdefault(tid, rec)
            last_assign[tid] = rec
            if rec["radio_delivered"]:
                delivered_epochs.setdefault(tid, epoch)

    rows = []
    helpers = {h.helper_id: h for h in layer.helpers}
    epoch_ms = float(meta.get("epoch_ms", getattr(layer, "epoch_ms", 20.0)) or 20.0)
    for t in layer.tasks:
        asg = last_assign.get(t.task_id) or first_assign.get(t.task_id)
        hid = getattr(t, "assigned_helper_id", None)
        if hid is None and asg:
            hid = asg.get("helper_id")
        helper = helpers.get(int(hid)) if hid is not None else None
        release_ms = epoch_ms * float(getattr(t, "release_epoch", 0) or 0)
        deadline_abs_ms = release_ms + float(t.deadline_ms)
        completion_ms = getattr(t, "completion_time_ms", None)
        finish_est_ms = getattr(t, "finish_estimate_ms", None)
        slack_ms = None
        if completion_ms is not None:
            slack_ms = deadline_abs_ms - float(completion_ms)
        elif finish_est_ms is not None:
            slack_ms = float(t.deadline_ms) - float(finish_est_ms)
        status = "completed" if t.done else ("queued" if t.queued else "incomplete")
        if not t.done and asg and not asg.get("radio_delivered"):
            status = "radio_loss"
        elif not t.done and asg and slack_ms is not None and slack_ms < 0:
            status = "deadline_miss"
        rows.append({
            **{k: meta.get(k) for k in (
                "method", "sim_seed", "task_seed", "density", "episode_id",
                "envelope_bytes", "policy_mode",
            )},
            "task_id": t.task_id,
            "source_id": t.source_id,
            "payload_bytes": t.payload_bytes,
            "cycles": t.cycles,
            "deadline_ms": t.deadline_ms,
            "task_value": t.value,
            "release_epoch": getattr(t, "release_epoch", 0),
            "release_ms": release_ms,
            "deadline_abs_ms": deadline_abs_ms,
            "done": bool(t.done),
            "queued": bool(t.queued),
            "status": status,
            "completion_time_ms": completion_ms,
            "finish_estimate_ms": finish_est_ms,
            "deadline_slack_ms": slack_ms,
            "deadline_miss": bool(slack_ms is not None and slack_ms < 0 and not t.done),
            "assigned_helper_id": hid,
            "helper_cpu_cycles_s": None if helper is None else helper.cpu_cycles_s,
            "first_assign_epoch": None if t.task_id not in first_assign else first_assign[t.task_id]["epoch"],
            "last_assign_epoch": None if t.task_id not in last_assign else last_assign[t.task_id]["epoch"],
            "radio_delivered_epoch": delivered_epochs.get(t.task_id),
            "radio_complete_epoch": getattr(t, "radio_complete_epoch", None),
            "packet_count": int(getattr(t, "payload_required", 0) or 0)
            or (None if asg is None else asg.get("packet_count")),
            "payload_required": int(getattr(t, "payload_required", 0) or 0),
            "payload_received": int(getattr(t, "payload_received", 0) or 0),
            "radio_ms_est": getattr(t, "radio_ms_est", None),
            "in_flight": bool(getattr(t, "in_flight", False)),
            "cancelled": bool(getattr(t, "cancelled", False)),
            "cancel_reason": getattr(t, "cancel_reason", None),
            "source_wilab_id": getattr(t, "source_wilab_id", None),
            "helper_wilab_id": getattr(t, "helper_wilab_id", None),
            "epoch_ms": epoch_ms,
        })
    return rows


def build_helper_rows(layer, audit: list[dict], meta: dict) -> list[dict]:
    assigned_counts: dict[int, int] = {h.helper_id: 0 for h in layer.helpers}
    assigned_bytes: dict[int, int] = {h.helper_id: 0 for h in layer.helpers}
    for e in audit:
        for a in e.get("selected_assignments") or []:
            hid = int(a.get("helper_id", -1))
            if hid in assigned_counts:
                assigned_counts[hid] += 1
                assigned_bytes[hid] += int(a.get("payload_bytes") or 0)
    rows = []
    for h in layer.helpers:
        rows.append({
            **{k: meta.get(k) for k in (
                "method", "sim_seed", "task_seed", "density", "episode_id",
            )},
            "helper_id": h.helper_id,
            "cpu_cycles_s": h.cpu_cycles_s,
            "final_queue_cycles": h.queue_cycles,
            "final_ready_time_ms": h.ready_time_ms,
            "assigned_task_count": assigned_counts.get(h.helper_id, 0),
            "assigned_payload_bytes": assigned_bytes.get(h.helper_id, 0),
            "utilization_proxy": (
                assigned_bytes.get(h.helper_id, 0) / max(h.cpu_cycles_s, 1e-9)
            ),
        })
    return rows


def build_decision_rows(trace: list[dict], audit: list[dict], meta: dict) -> list[dict]:
    """One row per (epoch, candidate) plus summary fields for selected set."""
    audit_by_epoch = {int(e.get("epoch") or i): e for i, e in enumerate(audit)}
    rows = []
    for i, tr in enumerate(trace):
        epoch = i
        ae = audit_by_epoch.get(epoch, {})
        tids = tr.get("candidate_task_ids") or []
        sids = tr.get("candidate_source_ids") or []
        payloads = tr.get("candidate_payload_bytes") or []
        scores = tr.get("candidate_analytic_scores") or []
        features = tr.get("candidate_features") or []
        helpers = tr.get("candidate_helper_ids") or []
        finishes = tr.get("candidate_finish_ms") or []
        deadlines = tr.get("candidate_deadline_ms") or []
        values = tr.get("candidate_task_values") or []
        cycles = tr.get("candidate_cycles") or []
        selected = set(int(x) for x in (tr.get("selected_candidate_indices") or []))
        n = max(len(tids), len(scores), len(features))
        for j in range(n):
            feat = features[j] if j < len(features) else None
            rows.append({
                **{k: meta.get(k) for k in (
                    "method", "sim_seed", "task_seed", "density", "episode_id", "policy_mode",
                )},
                "epoch": epoch,
                "candidate_index": j,
                "selected": j in selected,
                "task_id": tids[j] if j < len(tids) else None,
                "source_id": sids[j] if j < len(sids) else None,
                "helper_id": helpers[j] if j < len(helpers) else None,
                "payload_bytes": payloads[j] if j < len(payloads) else None,
                "analytic_score": scores[j] if j < len(scores) else None,
                "task_value": values[j] if j < len(values) else None,
                "finish_ms": finishes[j] if j < len(finishes) else None,
                "deadline_ms": deadlines[j] if j < len(deadlines) else None,
                "cycles": cycles[j] if j < len(cycles) else None,
                "feat_payload_norm": None if not feat else feat[0],
                "feat_cycles_norm": None if not feat else feat[1],
                "feat_deadline_norm": None if not feat else feat[2],
                "feat_value_norm": None if not feat else feat[3],
                "feat_helper_cpu_norm": None if not feat else feat[4],
                "feat_helper_queue_norm": None if not feat else feat[5],
                "feat_cbr": None if not feat or len(feat) < 8 else feat[7],
                "n_candidates": n,
                "executed_bytes": tr.get("bytes"),
                "residual_changed_anchor": tr.get("residual_changed_anchor"),
                "governor_intervened": tr.get("governor_intervened", ae.get("governor_intervened")),
                "governor_reason": tr.get("governor_reason", ae.get("governor_reason")),
                "proposed_bytes": ae.get("proposed_bytes"),
                "suppressed_bytes": ae.get("suppressed_bytes"),
                "inference_ms": ae.get("inference_ms"),
                "reward": tr.get("reward"),
            })
    return rows


def safety_from_audit_list(
    audit: list[dict],
    prr_floor: float = 0.95,
    p95_ceiling: float = 50.0,
    zero_offload: dict | None = None,
) -> dict:
    """Separate short-window, long-window, and cumulative safety metrics.

    The controller's per-epoch window is a small-sample quantity (a handful of
    beacon attempts); a single lost beacon can make it read 0.0.  The safety
    *claim* therefore uses the long window (when the simulator emits it) and the
    cumulative PRR, and is reported relative to the seed's zero-offload floor.
    """
    n = len(audit)
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    n_valid = len(valid)
    zg = zero_offload or {}
    if n_valid == 0:
        last = audit[-1] if audit else {}
        return {
            "n_epochs": n, "n_valid": 0, "invalid_frac": 1.0,
            "P_viol": None, "P_prr_fail": None, "P_p95_fail": None,
            "mean_prr_valid": None, "mean_p95_valid": None,
            "P_viol_long": None, "P_prr_fail_long": None,
            "mean_prr_long": None, "n_valid_long": 0,
            "P_viol_cum": None, "PRR_cum": last.get("safety_prr_cumulative"),
            "P95_cum": last.get("safety_p95_delay_cumulative_ms"),
            "zero_offload_P_viol": zg.get("P_viol"),
            "zero_offload_PRR_cum": zg.get("PRR_cum"),
            "Delta_P_viol": None,
            "Delta_P_viol_long": None,
            "Delta_PRR_cum": None,
            "compliant": False,
        }
    prr_fail = sum(1 for e in valid if e.get("prr") is not None and float(e["prr"]) < prr_floor)
    p95_fail = sum(1 for e in valid if e.get("p95_delay_ms") is not None and float(e["p95_delay_ms"]) > p95_ceiling)
    viol = sum(
        1 for e in valid
        if (e.get("prr") is not None and float(e["prr"]) < prr_floor)
        or (e.get("p95_delay_ms") is not None and float(e["p95_delay_ms"]) > p95_ceiling)
    )
    long_valid = [
        e for e in valid
        if e.get("window_prr_long") is not None and e.get("window_prr_long_valid")
    ]
    n_long = len(long_valid)
    if n_long:
        long_prr_fail = sum(
            1 for e in long_valid if float(e["window_prr_long"]) < prr_floor
        )
        long_p95_fail = sum(
            1 for e in long_valid
            if e.get("p95_delay_ms") is not None and float(e["p95_delay_ms"]) > p95_ceiling
        )
        long_viol = sum(
            1 for e in long_valid
            if float(e["window_prr_long"]) < prr_floor
            or (e.get("p95_delay_ms") is not None and float(e["p95_delay_ms"]) > p95_ceiling)
        )
        mean_prr_long = sum(float(e["window_prr_long"]) for e in long_valid) / n_long
        p_viol_long = long_viol / n_long
        p_prr_long = long_prr_fail / n_long
        p95_long = long_p95_fail / n_long
    else:
        mean_prr_long = None
        p_viol_long = None
        p_prr_long = None
        p95_long = None
    last = audit[-1]
    prr_cum = last.get("safety_prr_cumulative")
    p_viol_cum = None
    if prr_cum is not None:
        p_viol_cum = 0.0 if float(prr_cum) >= prr_floor else 1.0
    short_p_viol = viol / n_valid
    delta_p_viol = (
        None if zg.get("P_viol") is None else short_p_viol - float(zg["P_viol"])
    )
    delta_p_viol_long = (
        None
        if zg.get("P_viol_long") is None or p_viol_long is None
        else p_viol_long - float(zg["P_viol_long"])
    )
    delta_prr_cum = (
        None
        if zg.get("PRR_cum") is None or prr_cum is None
        else float(prr_cum) - float(zg["PRR_cum"])
    )
    return {
        "n_epochs": n,
        "n_valid": n_valid,
        "invalid_frac": (n - n_valid) / max(n, 1),
        "P_viol": short_p_viol,
        "P_prr_fail": prr_fail / n_valid,
        "P_p95_fail": p95_fail / n_valid,
        "mean_prr_valid": sum(float(e.get("prr") or 0) for e in valid) / n_valid,
        "mean_p95_valid": sum(float(e.get("p95_delay_ms") or 0) for e in valid) / n_valid,
        "P_viol_long": p_viol_long,
        "P_prr_fail_long": p_prr_long,
        "P_p95_fail_long": p95_long,
        "mean_prr_long": mean_prr_long,
        "n_valid_long": n_long,
        "P_viol_cum": p_viol_cum,
        "PRR_cum": prr_cum,
        "P95_cum": last.get("safety_p95_delay_cumulative_ms"),
        "zero_offload_P_viol": zg.get("P_viol"),
        "zero_offload_PRR_cum": zg.get("PRR_cum"),
        "Delta_P_viol": delta_p_viol,
        "Delta_P_viol_long": delta_p_viol_long,
        "Delta_PRR_cum": delta_prr_cum,
        # The claim requires both the long-window and cumulative floors.  A
        # zero-offload-relative margin is reported separately in the summary.
        "compliant": bool(
            (p_viol_long is None or p_viol_long <= 0.02)
            and (prr_cum is None or float(prr_cum) >= prr_floor)
        ),
    }


def build_episode_summary(
    *,
    audit: list[dict],
    layer,
    meta: dict,
    total_return: float,
    wall_seconds: float | None = None,
    safety: dict | None = None,
) -> dict:
    offered_value = sum(float(t.value) for t in layer.tasks)
    completed_value = float(layer.completed_value())
    n_tasks = len(layer.tasks)
    n_done = sum(1 for t in layer.tasks if t.done)
    n_miss = sum(1 for t in layer.tasks if not t.done)
    proposed_bytes = sum(int(e.get("proposed_bytes") or 0) for e in audit)
    executed_bytes = sum(int(e.get("executed_bytes") or e.get("computation_bytes_offered") or 0) for e in audit)
    delivered_bytes = sum(int(e.get("delivered_bytes") or 0) for e in audit)
    # The audit credits delivered bytes only on epochs where the completed
    # task is re-selected.  Payload-faithful delivery is a property of the
    # task, not of the epoch: a task whose full payload arrived is delivered
    # even when it is no longer in the admission action (it is in flight).
    task_delivered_bytes = sum(
        int(t.payload_bytes)
        for t in layer.tasks
        if t.done
        and int(getattr(t, "payload_required", 0) or 0) > 0
        and int(getattr(t, "payload_received", 0) or 0)
        >= int(getattr(t, "payload_required", 0) or 0)
    )
    delivered_bytes = max(delivered_bytes, task_delivered_bytes)
    wasted_bytes = sum(int(e.get("wasted_bytes") or 0) for e in audit)
    wasted_bytes = max(wasted_bytes, executed_bytes - delivered_bytes)
    suppressed_bytes = sum(int(e.get("suppressed_bytes") or 0) for e in audit)
    gate_blocks = sum(
        1 for e in audit
        if (e.get("governor_reason") or "none") not in ("none", "disabled")
        or e.get("admission_gate_blocked")
    )
    interventions = sum(1 for e in audit if e.get("governor_intervened"))
    fallbacks = sum(1 for e in audit if e.get("admission_fallback"))
    rejected = sum(1 for e in audit if e.get("admission_rejected_proposal"))
    infer = [_f(e.get("inference_ms")) for e in audit if e.get("inference_ms") is not None]
    infer = [x for x in infer if x is not None]
    n_cand = [_i(e.get("n_candidates")) for e in audit if e.get("n_candidates") is not None]
    n_cand = [x for x in n_cand if x is not None]
    value_per_executed_MB = (
        completed_value / (executed_bytes / 1e6) if executed_bytes > 0 else None
    )
    value_per_delivered_MB = (
        completed_value / (delivered_bytes / 1e6) if delivered_bytes > 0 else None
    )
    safety = safety or safety_from_audit_list(audit)
    row = {
        **meta,
        "return": total_return,
        "V_done": completed_value,
        "V_off": offered_value,
        "R_value": completed_value / max(offered_value, 1e-9),
        "n_tasks": n_tasks,
        "n_deadline_valid_completed": n_done,
        "n_incomplete_tasks": n_miss,
        "task_completion_rate": n_done / max(n_tasks, 1),
        "deadline_miss_rate": n_miss / max(n_tasks, 1),
        "proposed_bytes_sum": proposed_bytes,
        "executed_bytes_sum": executed_bytes,
        "delivered_bytes_sum": delivered_bytes,
        "wasted_bytes_sum": wasted_bytes,
        "suppressed_bytes_sum": suppressed_bytes,
        "value_per_executed_MB": value_per_executed_MB,
        "value_per_delivered_MB": value_per_delivered_MB,
        "governor_intervention_epochs": interventions,
        "governor_gate_block_epochs": gate_blocks,
        "governor_fallback_epochs": fallbacks,
        "governor_rejected_proposal_epochs": rejected,
        "governor_intervention_rate": interventions / max(len(audit), 1),
        "n_epochs": len(audit),
        "inference_ms_mean": (sum(infer) / len(infer)) if infer else None,
        "inference_ms_p50": sorted(infer)[len(infer) // 2] if infer else None,
        "inference_ms_p95": sorted(infer)[int(0.95 * (len(infer) - 1))] if infer else None,
        "mean_n_candidates": (sum(n_cand) / len(n_cand)) if n_cand else None,
        "wall_seconds": wall_seconds,
        "safety_P_viol": safety.get("P_viol"),
        "safety_P_prr_fail": safety.get("P_prr_fail"),
        "safety_P_p95_fail": safety.get("P_p95_fail"),
        "safety_mean_prr_valid": safety.get("mean_prr_valid"),
        "safety_mean_p95_valid": safety.get("mean_p95_valid"),
        "safety_n_valid": safety.get("n_valid"),
        "safety_n_epochs": safety.get("n_epochs"),
        "safety_invalid_frac": safety.get("invalid_frac"),
        "safety_PRR_cum": safety.get("PRR_cum"),
        "safety_P95_cum": safety.get("P95_cum"),
        "safety_P_viol_long": safety.get("P_viol_long"),
        "safety_P_prr_fail_long": safety.get("P_prr_fail_long"),
        "safety_mean_prr_long": safety.get("mean_prr_long"),
        "safety_n_valid_long": safety.get("n_valid_long"),
        "safety_P_viol_cum": safety.get("P_viol_cum"),
        "zero_offload_P_viol": safety.get("zero_offload_P_viol"),
        "zero_offload_PRR_cum": safety.get("zero_offload_PRR_cum"),
        "Delta_P_viol": safety.get("Delta_P_viol"),
        "Delta_P_viol_long": safety.get("Delta_P_viol_long"),
        "Delta_PRR_cum": safety.get("Delta_PRR_cum"),
        "safety_compliant": safety.get("compliant"),
    }
    return row


def write_episode_metrics(
    out_dir: Path,
    *,
    audit: list[dict],
    trace: list[dict],
    layer,
    meta: dict,
    total_return: float,
    wall_seconds: float | None = None,
    safety: dict | None = None,
) -> dict:
    """Write all per-episode metric files into out_dir. Returns episode_summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_rows = build_epoch_rows(audit, meta)
    task_rows = build_task_rows(layer, audit, meta)
    helper_rows = build_helper_rows(layer, audit, meta)
    decision_rows = build_decision_rows(trace, audit, meta)
    summary = build_episode_summary(
        audit=audit,
        layer=layer,
        meta=meta,
        total_return=total_return,
        wall_seconds=wall_seconds,
        safety=safety,
    )

    _write_csv(out_dir / "epoch_trace.csv", epoch_rows)
    _write_csv(out_dir / "task_trace.csv", task_rows)
    _write_csv(out_dir / "helper_trace.csv", helper_rows)
    _write_csv(out_dir / "decision_trace.csv", decision_rows)
    _write_csv(out_dir / "episode_summary.csv", [summary])
    (out_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Logging-only full candidate dump for Stage 0 identical-candidate shadow scoring.
    # Does not alter scheduling, admission, rewards, or simulator behavior.
    cand_epochs = []
    for tr in trace:
        feats = tr.get("candidate_features") or []
        n = len(feats) if feats else len(tr.get("candidate_task_ids") or [])
        epoch_cands = []
        for j in range(n):
            feat = feats[j] if j < len(feats) else []
            epoch_cands.append(
                {
                    "task_id": (tr.get("candidate_task_ids") or [None] * n)[j],
                    "source_id": (tr.get("candidate_source_ids") or [None] * n)[j],
                    "helper_id": (tr.get("candidate_helper_ids") or [None] * n)[j],
                    "payload_bytes": (tr.get("candidate_payload_bytes") or [None] * n)[j],
                    "analytic_score": (tr.get("candidate_analytic_scores") or [None] * n)[j],
                    "task_value": (tr.get("candidate_task_values") or [None] * n)[j],
                    "finish_ms": (tr.get("candidate_finish_ms") or [None] * n)[j],
                    "deadline_ms": (tr.get("candidate_deadline_ms") or [None] * n)[j],
                    "features": feat,
                    "feat_cbr": feat[7] if len(feat) > 7 else None,
                    "feat_deadline_norm": feat[2] if len(feat) > 2 else None,
                    "feat_helper_queue_norm": feat[5] if len(feat) > 5 else None,
                    "selected": j in set(int(x) for x in (tr.get("selected_candidate_indices") or [])),
                    "governor_intervened": bool(tr.get("governor_intervened")),
                    "governor_reason": tr.get("governor_reason", "none"),
                }
            )
        cand_epochs.append(epoch_cands)
    (out_dir / "candidate_epochs.json").write_text(
        json.dumps(cand_epochs) + "\n", encoding="utf-8"
    )
    # Keep a compact metrics sidecar for quick loads
    (out_dir / "metrics_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    "episode_summary.csv",
                    "episode_summary.json",
                    "epoch_trace.csv",
                    "task_trace.csv",
                    "helper_trace.csv",
                    "decision_trace.csv",
                    "candidate_epochs.json",
                    "bridge_audit.json",
                ],
                "n_epochs": len(epoch_rows),
                "n_tasks": len(task_rows),
                "n_helpers": len(helper_rows),
                "n_decision_rows": len(decision_rows),
                "R_value": summary.get("R_value"),
                "P_viol": summary.get("safety_P_viol"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def append_campaign_episode_summary(campaign_root: Path, summary: dict) -> None:
    """Append one episode_summary row into campaign_root/episode_summary.csv."""
    campaign_root = Path(campaign_root)
    path = campaign_root / "episode_summary.csv"
    rows = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    # replace same episode_id+method if re-run
    key = (str(summary.get("episode_id")), str(summary.get("method")), str(summary.get("sim_seed")))
    rows = [
        r for r in rows
        if (str(r.get("episode_id")), str(r.get("method")), str(r.get("sim_seed"))) != key
    ]
    rows.append({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in summary.items()})
    _write_csv(path, rows)


def export_training_history_csv(metrics_json: Path, out_csv: Path | None = None) -> Path:
    """Convert training_metrics.json → training_history.csv."""
    metrics_json = Path(metrics_json)
    out_csv = out_csv or metrics_json.with_name("training_history.csv")
    rows = json.loads(metrics_json.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("episodes") or rows.get("history") or [rows]
    _write_csv(out_csv, list(rows))
    return out_csv
