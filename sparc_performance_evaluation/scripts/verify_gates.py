"""Offline / unit correctness gates G0,G2-partial,G3,G5,G6,G7 for SPARC-V2X."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from controllers.rl.residual_ppo import _sequence_logp, build_ppo_steps
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor
from task_layer import TaskLayer

OUT = ROOT / "sparc_performance_evaluation" / "verification"
REPORT = ROOT / "sparc_performance_evaluation" / "reports" / "controller_verification.md"


def gate_g3_deadline_commit_parity():
    layer = TaskLayer(density=35, seed=424242, tasks_per_source=3,
                      spread_arrivals=True, controller_epochs=100)
    mismatches = []
    for epoch in range(0, 100, 7):
        layer.set_epoch(epoch)
        cands = layer.candidates(cbr=0.2, time_fraction=epoch / 100.0)
        for c in cands:
            assert c["deadline_feasible"] is True
            task = next(t for t in layer.tasks if t.task_id == c["task_id"])
            helper = next(h for h in layer.helpers if h.helper_id == c["helper_id"])
            now = 10.0 * epoch
            start = max(now + 10.0, helper.ready_time_ms)
            finish = start + 1000.0 * task.cycles / helper.cpu_cycles_s
            abs_d = 10.0 * task.release_epoch + task.deadline_ms
            if finish > abs_d + 1e-6:
                mismatches.append((epoch, c["task_id"], c["helper_id"], finish, abs_d))
            # finish_ms in candidate equals finish - now
            if abs((finish - now) - float(c["finish_ms"])) > 1e-4:
                mismatches.append(("finish_ms", epoch, c["task_id"], c["finish_ms"], finish - now))
    return {"gate": "G3", "pass": len(mismatches) == 0, "mismatches": mismatches[:5]}


def gate_g5_zero_init_matches_va():
    policy = ResidualTaskPolicy(logit_scale=100.0, residual_gain=1.0, logit_mode="va_residual")
    with torch.no_grad():
        policy.net[-1].weight.zero_()
        policy.net[-1].bias.zero_()
    layer = TaskLayer(density=35, seed=111, tasks_per_source=3,
                      spread_arrivals=True, controller_epochs=100)
    cands = []
    for epoch in range(0, 100, 5):
        layer.set_epoch(epoch)
        cands = layer.candidates(0.15, epoch / 100.0)
        if cands:
            break
    if not cands:
        return {"gate": "G5", "pass": False, "error": "no candidates"}
    x = torch.tensor([c["features"] for c in cands], dtype=torch.float32)
    analytic = torch.tensor([c["analytic_score"] for c in cands], dtype=torch.float32)
    logits = policy.score_logits(x, analytic)
    # argmax of logits == argmax of analytic; logits == scale * analytic
    ok_argmax = int(torch.argmax(logits)) == int(torch.argmax(analytic))
    ok_scale = torch.allclose(logits, 100.0 * analytic, atol=1e-5)
    return {
        "gate": "G5",
        "pass": bool(ok_argmax and ok_scale),
        "ok_argmax": ok_argmax,
        "ok_scale": ok_scale,
        "n_candidates": len(cands),
    }


def gate_g2_governor_envelope():
    gov = SafetyGovernor(max_bytes=50_000, horizon_envelope_only=True)
    cands = [
        {"task_id": 0, "helper_id": 0, "source_id": 0, "payload_bytes": 40_000,
         "analytic_score": 1.0, "deadline_feasible": True, "packet_count": 1},
        {"task_id": 1, "helper_id": 1, "source_id": 1, "payload_bytes": 40_000,
         "analytic_score": 0.5, "deadline_feasible": True, "packet_count": 1},
        {"task_id": 2, "helper_id": 0, "source_id": 2, "payload_bytes": 20_000,
         "analytic_score": 2.0, "deadline_feasible": False, "packet_count": 1},
    ]
    state_ok = {"cbr_valid": True, "delay_valid": True, "prr": 0.99, "p95_delay_ms": 20.0}
    # over-budget proposal -> VA fallback under envelope
    proposed = [
        {"task_id": 0, "helper_id": 0, "source_id": 0, "payload_bytes": 40_000},
        {"task_id": 1, "helper_id": 1, "source_id": 1, "payload_bytes": 40_000},
    ]
    executed, intervened, reason = gov.apply(proposed, cands, state_ok)
    bytes_ok = sum(a["payload_bytes"] for a in executed) <= 50_000
    no_infeas = all(True for a in executed)  # fallback skips deadline_feasible False
    # invalid radio blocks
    executed2, intervened2, reason2 = gov.apply(proposed, cands, {"cbr_valid": False, "delay_valid": True})
    # prr floor
    executed3, _, reason3 = gov.apply(proposed, cands, {**state_ok, "prr": 0.5})
    return {
        "gate": "G2",
        "pass": bool(intervened and bytes_ok and reason == "over_budget_or_invalid_assignment"
                     and executed2 == [] and reason2 == "invalid_radio_state"
                     and executed3 == [] and reason3 == "prr_below_floor"
                     and all(a["task_id"] != 2 for a in executed)),
        "fallback_n": len(executed),
        "fallback_reason": reason,
        "invalid_reason": reason2,
        "prr_reason": reason3,
    }


def gate_g6_sequence_logp_mask():
    policy = ResidualTaskPolicy(logit_mode="va_residual")
    with torch.no_grad():
        policy.net[-1].weight.zero_()
        policy.net[-1].bias.zero_()
    # two candidates, select first under byte mask
    item = {
        "candidate_features": [[0.1] * 10, [0.2] * 10],
        "candidate_analytic_scores": [0.05, 0.01],
        "candidate_task_ids": [0, 1],
        "candidate_source_ids": [0, 1],
        "candidate_payload_bytes": [10_000, 10_000],
        "selected_candidate_indices": [0],
        "normalized_reward": 0.1,
        "value": torch.tensor(0.0),
        "logp": torch.tensor(0.0),
        "observation": [0.0] * 10,
        "residual_changed_anchor": False,
    }
    evaluated = _sequence_logp(policy, item, envelope_bytes=175000, unique_sources=True)
    # selecting both same source should fail uniqueness
    bad = dict(item)
    bad["candidate_source_ids"] = [0, 0]
    bad["selected_candidate_indices"] = [0, 1]
    bad_eval = _sequence_logp(policy, bad, envelope_bytes=175000, unique_sources=True)
    return {
        "gate": "G6",
        "pass": evaluated is not None and bad_eval is None,
        "logp": None if evaluated is None else float(evaluated[0].detach()),
        "unique_source_rejected": bad_eval is None,
    }


def gate_g7_seed_disjointness():
    used_train_sim = set(range(72000, 72096))
    used_val_sim = {82001, 82002, 82011, 82012}
    used_train_task = set(range(182000, 182096))
    used_val_task = {182001, 182002, 182011, 182012}
    final_path = ROOT / "sparc_performance_evaluation" / "protocol" / "final_test_seeds.json"
    if not final_path.exists():
        return {"gate": "G7", "pass": False, "error": "final_test_seeds.json missing"}
    final = json.loads(final_path.read_text(encoding="utf-8"))
    sims = set(final["sim_seeds"])
    tasks = set(final["task_seeds"])
    overlap_sim = sims & (used_train_sim | used_val_sim)
    overlap_task = tasks & (used_train_task | used_val_task)
    return {
        "gate": "G7",
        "pass": len(overlap_sim) == 0 and len(overlap_task) == 0 and len(sims) == len(final["sim_seeds"]),
        "n_final": len(sims),
        "overlap_sim": sorted(overlap_sim),
        "overlap_task": sorted(overlap_task),
        "seal_status": final.get("status"),
    }


def gate_g0_task_reproducibility():
    a = TaskLayer(35, 90001, 3, spread_arrivals=True, controller_epochs=100)
    b = TaskLayer(35, 90001, 3, spread_arrivals=True, controller_epochs=100)
    same = [(t.payload_bytes, t.cycles, t.deadline_ms, t.value, t.release_epoch, t.source_id)
            for t in a.tasks] == [(t.payload_bytes, t.cycles, t.deadline_ms, t.value, t.release_epoch, t.source_id)
                                  for t in b.tasks]
    c = TaskLayer(35, 90002, 3, spread_arrivals=True, controller_epochs=100)
    diff = [(t.payload_bytes, t.cycles) for t in a.tasks] != [(t.payload_bytes, t.cycles) for t in c.tasks]
    return {"gate": "G0", "pass": bool(same and diff), "n_tasks": len(a.tasks)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [
        gate_g0_task_reproducibility(),
        gate_g2_governor_envelope(),
        gate_g3_deadline_commit_parity(),
        gate_g5_zero_init_matches_va(),
        gate_g6_sequence_logp_mask(),
        gate_g7_seed_disjointness(),
    ]
    # G1/G4/G8/G9 require WiLab / provenance — mark pending
    results.extend([
        {"gate": "G1", "pass": None, "status": "REQUIRES_WILAB_RUNTIME_AUDIT"},
        {"gate": "G4", "pass": None, "status": "REQUIRES_WILAB_DELIVERY_TRACE_AUDIT"},
        {"gate": "G8", "pass": None, "status": "REQUIRES_SAFETY_DENOMINATOR_AUDIT"},
        {"gate": "G9", "pass": None, "status": "REQUIRES_PROVENANCE_INDEX_AFTER_RUNS"},
    ])
    (OUT / "gate_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Controller verification (G0–G9)", "", "**Status:** auto-updated by `scripts/verify_gates.py`", "",
             "| Gate | Result | Notes |", "|---|---|---|"]
    for r in results:
        g = r["gate"]
        if r.get("pass") is True:
            res = "PASS"
        elif r.get("pass") is False:
            res = "FAIL"
        else:
            res = "PENDING"
        note = r.get("status") or json.dumps({k: v for k, v in r.items() if k not in ("gate", "pass", "status")})[:120]
        lines.append(f"| {g} | {res} | {note} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    hard = [r for r in results if r.get("pass") is False]
    if hard:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
