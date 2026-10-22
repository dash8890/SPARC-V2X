"""Python side of the isolated WiLab closed-loop protocol.

The client deliberately reuses the frozen final residual checkpoint. It does
not alter the canonical checkpoint or evaluation files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from controllers.rl.residual_task_policy import ResidualTaskPolicy
from wilab_closed_loop_protocol import atomic_json_write, wait_json
from task_layer import TaskLayer


def choose_action(state: dict, policy: ResidualTaskPolicy) -> dict:
    candidates = state.get("candidates", [])
    budget = int(state.get("safe_budget", 0))
    if not candidates or budget <= 0:
        return {"budget": 0, "assignments": []}

    x = torch.tensor([c["features"] for c in candidates], dtype=torch.float32)
    with torch.no_grad():
        residual = policy.residual(x)
    scores = torch.tensor([float(c["analytic_score"]) for c in candidates]) + residual
    order = torch.argsort(scores, descending=True).tolist()
    used = set()
    assignments = []
    for i in order:
        task_id = candidates[i]["task_id"]
        if task_id in used:
            continue
        used.add(task_id)
        assignments.append({"task_id": task_id, "helper_id": candidates[i]["helper_id"]})
        if len(assignments) >= budget:
            break
    source_by_task = {c["task_id"]: c.get("source_id", c["task_id"]) for c in candidates}
    return {
        "budget": budget,
        "assignments": assignments,
        "allowed_sources": [source_by_task[a["task_id"]] for a in assignments],
    }


def serve(bridge_dir: Path, checkpoint: Path, timeout_s: float) -> None:
    policy = ResidualTaskPolicy()
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu")["policy"])
    policy.eval()
    layers = {}
    for request in sorted(bridge_dir.glob("state_*.json")):
        action_path = bridge_dir / request.name.replace("state_", "action_", 1)
        if action_path.exists():
            continue
        state = wait_json(request, timeout_s)
        episode = int(state.get("episode", 0))
        layer = layers.setdefault(episode, TaskLayer(int(state.get("density", 50)), 90000 + episode))
        state["candidates"] = layer.candidates(float(state.get("cbr", 0.0)), float(state.get("time_s", 0.0)))
        action = choose_action(state, policy)
        action["episode"] = state.get("episode")
        action["epoch"] = state.get("epoch")
        atomic_json_write(action_path, action)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/residual_task_policy_cf.pt"))
    ap.add_argument("--timeout-s", type=float, default=300.0)
    args = ap.parse_args()
    serve(args.bridge_dir, args.checkpoint, args.timeout_s)
