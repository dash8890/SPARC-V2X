"""Verify payload-faithful completion accounting for one closed-loop episode.

The corrected contract is

    z_k = 1{ all n_k payload units reached the helper at least once
             AND the helper's computation finished by d_k }.

This script checks the structural invariants that must hold no matter how the
radio behaves:

  * a completed task has ``radio_delivered`` set,
  * a radio-delivered task has ``payload_received >= payload_required > 0``,
  * no task is credited with more units than it requires,
  * every task named in an epoch's ``payload_complete_task_ids`` is marked
    complete in the final task state,
  * the sum of credited units never exceeds the number of injected packets.

It also reports the queue-drop counters, because WiLab's shared
``pckBuffer`` can discard computation packets before they ever transmit.  A
large ``packet_overflow_losses`` with near-zero payload completion is a
simulator queue artifact, not a radio-capacity result.

Usage
-----
    python scripts/verify_payload_accounting.py \
        --episode-dir sparc_performance_evaluation/results/raw/.../episode_97001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_episode(ep_dir: Path):
    state_file = ep_dir / "task_layer_state.json"
    audit_file = ep_dir / "bridge_audit.json"
    summary_file = ep_dir / "episode_summary.json"
    if not state_file.is_file():
        raise SystemExit(f"missing {state_file}")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
    return state, audit, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ep_dir = Path(args.episode_dir)
    state, audit, summary = load_episode(ep_dir)
    tasks = state.get("tasks", [])

    violations: list[str] = []

    modes = sorted({str(e.get("accounting_mode")) for e in audit if "accounting_mode" in e})
    payload_tasks = [t for t in tasks if int(t.get("payload_required") or 0) > 0]
    payload_applicable = bool(payload_tasks) or any(m == "payload" for m in modes)
    done_tasks = [t for t in tasks if t.get("done")]
    complete_tasks = [t for t in tasks if t.get("radio_delivered")]
    credited_units = sum(int(t.get("payload_received") or 0) for t in tasks)

    for t in tasks:
        tid = t.get("task_id")
        req = int(t.get("payload_required") or 0)
        got = int(t.get("payload_received") or 0)
        if got > req:
            violations.append(f"task {tid}: received {got} > required {req}")
        if t.get("done") and not t.get("radio_delivered"):
            violations.append(f"task {tid}: done without radio delivery")
        if payload_applicable and t.get("radio_delivered") and not (req > 0 and got >= req):
            violations.append(
                f"task {tid}: radio_delivered with {got}/{req} payload units"
            )

    last_epoch = audit[-1] if audit else {}
    enqueued_units = int(last_epoch.get("wilab_enqueued_packets") or 0)
    if enqueued_units and credited_units > enqueued_units:
        violations.append(
            f"credited units {credited_units} exceed injected packets {enqueued_units}"
        )

    # Cross-check the epoch-level completion announcements against final state.
    known_complete = {int(t["task_id"]) for t in complete_tasks}
    announced = set()
    for e in audit:
        for tid in e.get("payload_complete_task_ids") or []:
            announced.add(int(tid))
    missing_announcements = sorted(announced - known_complete)
    if missing_announcements:
        violations.append(
            f"epoch announcements missing from final state: {missing_announcements}"
        )

    capture_missing = int(last_epoch.get("packet_capture_missing") or 0)
    overflow_losses = int(last_epoch.get("packet_overflow_losses") or 0)
    tag_deficits = int(last_epoch.get("packet_tag_deficits") or 0)

    report = {
        "episode_dir": str(ep_dir),
        "accounting_modes": modes,
        "payload_invariants_applicable": payload_applicable,
        "n_tasks": len(tasks),
        "n_payload_tracked": len(payload_tasks),
        "n_payload_complete": len(complete_tasks),
        "n_done": len(done_tasks),
        "n_done_without_payload": sum(1 for t in done_tasks if not t.get("radio_delivered")),
        "credited_units": credited_units,
        "enqueued_packets": enqueued_units,
        "packet_overflow_losses": overflow_losses,
        "packet_tag_deficits": tag_deficits,
        "packet_capture_missing": capture_missing,
        "V_done": summary.get("V_done"),
        "completion_rate": summary.get("task_completion_rate"),
        "violations": violations,
        "tasks": [
            {
                "task_id": t.get("task_id"),
                "release_epoch": t.get("release_epoch"),
                "deadline_ms": t.get("deadline_ms"),
                "assigned_helper_id": t.get("assigned_helper_id"),
                "payload_required": t.get("payload_required"),
                "payload_received": t.get("payload_received"),
                "radio_complete_epoch": t.get("radio_complete_epoch"),
                "radio_delivered": t.get("radio_delivered"),
                "done": t.get("done"),
                "completion_time_ms": t.get("completion_time_ms"),
            }
            for t in tasks
        ],
    }

    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)

    if violations:
        print(f"\nFAIL: {len(violations)} accounting invariant violation(s)")
        return 1
    if not payload_applicable:
        print("\nOK: legacy-accounting episode; payload invariants not applicable")
        return 0
    print("\nOK: payload-faithful accounting invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
