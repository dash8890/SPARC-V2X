"""Corrected-VA / MF / residual@init bake-off on WiLab closed-loop seeds.

Uses the deadline-feasible candidate set and the same envelope/governor settings
as residual PPO training. Reports completed value and R_value per method.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from train_wilab_closed_loop import ROOT, run_episode
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor


def final_safety(output_root: Path, episode: int):
    audit_path = output_root / f"episode_{episode:04d}" / "bridge_audit.json"
    if not audit_path.exists():
        return None, None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    last = audit[-1] if audit else {}
    return last.get("safety_prr_cumulative"), last.get("safety_p95_delay_cumulative_ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=int, default=35)
    ap.add_argument("--controller-epochs", type=int, default=100)
    ap.add_argument("--tasks-per-source", type=int, default=3)
    ap.add_argument("--envelope-bytes", type=int, default=175000)
    ap.add_argument("--safety-payload-bytes", type=int, default=300)
    ap.add_argument("--simulation-time", type=float, default=12.0)
    ap.add_argument("--logit-scale", type=float, default=100.0)
    ap.add_argument("--residual-gain", type=float, default=1.0)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="optional residual policy checkpoint; zero-init if omitted")
    ap.add_argument("--methods", nargs="+", default=["value_aware", "minfinish", "residual"])
    ap.add_argument("--sim-seeds", type=int, nargs="+", default=[82001, 82002, 82003, 82004, 82005])
    ap.add_argument("--task-seeds", type=int, nargs="+", default=[182001, 182002, 182003, 182004, 182005])
    ap.add_argument("--output-root", default="results/wilab_bakeoff_va_mf_residual_corrected")
    args = ap.parse_args()
    if len(args.sim_seeds) != len(args.task_seeds):
        raise ValueError("sim-seeds and task-seeds must match in length")

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    runs = root / "runs"
    runs.mkdir()

    policy = ResidualTaskPolicy(logit_scale=args.logit_scale, residual_gain=args.residual_gain)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(ckpt["policy"])
        init_note = f"loaded {args.checkpoint}"
    else:
        with torch.no_grad():
            policy.net[-1].weight.zero_()
            policy.net[-1].bias.zero_()
        init_note = "residual is zero-init (equals corrected VA under argmax)"
    policy.eval()
    rows = []
    episode_id = 0
    for scenario, (sim_seed, task_seed) in enumerate(zip(args.sim_seeds, args.task_seeds)):
        for method in args.methods:
            episode_id += 1
            bridge = Path(tempfile.gettempdir()) / f"wilab_bakeoff_{os.getpid()}_{episode_id}"
            if bridge.exists():
                shutil.rmtree(bridge)
            os.environ["WILAB_OUTPUT_ROOT"] = str(runs)
            governor = SafetyGovernor(args.envelope_bytes, horizon_envelope_only=True)
            total, _, completed, trace = run_episode(
                ROOT, bridge, episode_id, args.density, policy, 900.0,
                False, False, 100.0, 20.0, sim_seed, args.controller_epochs,
                None, args.tasks_per_source, False, args.envelope_bytes,
                args.safety_payload_bytes, args.simulation_time, method, None,
                True, "off", "", True, task_seed, governor,
                args.envelope_bytes, True, True,
            )
            offered = float(trace[0]["offered_value"]) if trace else 0.0
            prr, p95 = final_safety(runs, episode_id)
            row = {
                "scenario": scenario,
                "method": method,
                "sim_seed": sim_seed,
                "task_seed": task_seed,
                "return": float(total),
                "completed_value": float(completed),
                "offered_value": offered,
                "R_value": float(completed) / max(offered, 1e-9),
                "PRR": prr,
                "P95_ms": p95,
                "selected_epochs": sum(bool(x.get("selected_candidate_indices")) for x in trace),
                "deviation_fraction": sum(bool(x.get("residual_changed_anchor")) for x in trace) / max(1, len(trace)),
                "governor_interventions": sum(bool(x.get("governor_intervened")) for x in trace),
                "note": init_note,
            }
            rows.append(row)
            (root / "bakeoff_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)

    # paired summaries
    from collections import defaultdict
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[(r["sim_seed"], r["task_seed"])][r["method"]] = r
    methods = list(args.methods)
    summary = {"n_seeds": len(by_seed), "methods": {}}
    for m in methods:
        vals = [by_seed[s][m]["R_value"] for s in by_seed if m in by_seed[s]]
        comps = [by_seed[s][m]["completed_value"] for s in by_seed if m in by_seed[s]]
        summary["methods"][m] = {
            "mean_R": sum(vals) / len(vals),
            "mean_completed": sum(comps) / len(comps),
        }
    if "value_aware" in methods:
        deltas = {}
        for m in methods:
            if m == "value_aware":
                continue
            dR, dC, better = [], [], 0
            for s, d in by_seed.items():
                if "value_aware" not in d or m not in d:
                    continue
                dr = d[m]["R_value"] - d["value_aware"]["R_value"]
                dc = d[m]["completed_value"] - d["value_aware"]["completed_value"]
                dR.append(dr); dC.append(dc)
                better += int(dr > 1e-12)
            deltas[m] = {
                "mean_delta_R_vs_VA": sum(dR) / len(dR),
                "mean_delta_completed_vs_VA": sum(dC) / len(dC),
                "frac_better_R_than_VA": better / len(dR),
            }
        summary["paired_vs_value_aware"] = deltas
    if "minfinish" in methods and "value_aware" in methods:
        mf_vs_va = []
        for s, d in by_seed.items():
            mf_vs_va.append(d["minfinish"]["R_value"] - d["value_aware"]["R_value"])
        summary["VA_vs_MF"] = {
            "mean_VA_minus_MF_R": -sum(mf_vs_va) / len(mf_vs_va),
            "frac_VA_beats_MF": sum(1 for x in mf_vs_va if x < -1e-12) / len(mf_vs_va),
            "frac_MF_beats_VA": sum(1 for x in mf_vs_va if x > 1e-12) / len(mf_vs_va),
            "frac_tie": sum(1 for x in mf_vs_va if abs(x) <= 1e-12) / len(mf_vs_va),
        }
    if "residual" in methods and "minfinish" in methods:
        dR, dC, better = [], [], 0
        for s, d in by_seed.items():
            if "residual" not in d or "minfinish" not in d:
                continue
            dr = d["residual"]["R_value"] - d["minfinish"]["R_value"]
            dc = d["residual"]["completed_value"] - d["minfinish"]["completed_value"]
            dR.append(dr); dC.append(dc)
            better += int(dr > 1e-12)
        summary["paired_vs_minfinish"] = {
            "residual": {
                "mean_delta_R_vs_MF": sum(dR) / len(dR),
                "mean_delta_completed_vs_MF": sum(dC) / len(dC),
                "frac_better_R_than_MF": better / len(dR),
            }
        }

    cfg = vars(args).copy()
    cfg["residual_init"] = init_note
    cfg["candidate_policy"] = "deadline-feasible only; finish aligned with apply_radio_outcomes"
    (root / "bakeoff_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (root / "bakeoff_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "summary": summary, "output_root": str(root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
