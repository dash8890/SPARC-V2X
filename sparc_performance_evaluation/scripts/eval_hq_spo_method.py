"""Eval HQ-SPO gate vs LVM / Direct on fresh tps4 seeds (post premise PASS)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.lvm_rpi.hq_spo_gate import HQSPOGatedPolicy
from sparc_performance_evaluation.scripts.run_paired_eval import safety_from_audit

REPORTS = ROOT / "sparc_performance_evaluation" / "reports"
LVM_CKPT = ROOT / "results" / "wilab_lagrangian_ppo_96ep" / "checkpoint_best_delta_va.pt"
DIRECT_CKPT = ROOT / "results" / "wilab_direct_ppo_96ep" / "checkpoint_best_delta_va.pt"
ENVELOPE = 175000
# Fresh after v1 FAIL (72900–07, mean Δ vs LVM ≈ −1.27)
SEEDS = list(range(72910, 72918))
RAW = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "hq_spo_method_eval_v2"


def load_pol(ckpt):
    pol = ResidualTaskPolicy(logit_mode="direct", logit_scale=100.0)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu")["policy"], strict=False)
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    return pol


def run_one(policy, *, episode_id, sim_seed, task_seed, runs_dir):
    bridge = Path(tempfile.gettempdir()) / f"hqm_{os.getpid()}_{episode_id}"
    if bridge.exists():
        shutil.rmtree(bridge)
    os.environ["WILAB_OUTPUT_ROOT"] = str(runs_dir)
    gov = SafetyGovernor(ENVELOPE, horizon_envelope_only=True)
    total, _, completed, _ = run_episode(
        WILAB_ROOT, bridge, episode_id, 35, policy, 900.0,
        False, False, 100.0, 20.0, sim_seed, 100,
        None, 4, False, ENVELOPE,
        300, 12.0, "residual", None,
        True, "off", "", True, task_seed, gov,
        ENVELOPE, True, True, 1.0, None, None, None, 2e8, 1e9,
    )
    ep = runs_dir / f"episode_{episode_id:04d}"
    safety = safety_from_audit(ep / "bridge_audit.json")
    choices = {"LVM": 0, "Direct": 0, "Spread": 0}
    try:
        audit = json.loads((ep / "bridge_audit.json").read_text(encoding="utf-8"))
        for e in audit:
            g = e.get("hq_spo_gate") or {}
            ch = g.get("choice")
            if ch in choices:
                choices[ch] += 1
    except Exception:
        pass
    return {
        "V_done": float(completed),
        "P_viol": safety.get("P_viol"),
        "choices": choices,
    }


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    runs = RAW / "runs"
    runs.mkdir(exist_ok=True)
    lvm = load_pol(LVM_CKPT)
    direct = load_pol(DIRECT_CKPT)
    hq = HQSPOGatedPolicy(lvm=lvm, direct=direct, envelope=ENVELOPE)
    methods = {"LVM": lvm, "Direct": direct, "HQSPO": hq}
    rows = []
    eid = 1
    t0 = time.time()
    for sim_seed in SEEDS:
        task_seed = 184910 + (sim_seed - 72910)
        row = {"sim_seed": sim_seed}
        for name, pol in methods.items():
            print(json.dumps({"event": "run", "method": name, "sim_seed": sim_seed}), flush=True)
            out = run_one(pol, episode_id=eid, sim_seed=sim_seed, task_seed=task_seed, runs_dir=runs)
            eid += 1
            row[name] = out
            print(json.dumps({"event": "done", "method": name, "sim_seed": sim_seed, **out}, default=str), flush=True)
        rows.append(row)
        (RAW / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    per = []
    for r in rows:
        per.append({
            "sim_seed": r["sim_seed"],
            "V_LVM": r["LVM"]["V_done"],
            "V_Direct": r["Direct"]["V_done"],
            "V_HQSPO": r["HQSPO"]["V_done"],
            "d_vs_LVM": r["HQSPO"]["V_done"] - r["LVM"]["V_done"],
            "d_vs_best": r["HQSPO"]["V_done"] - max(r["LVM"]["V_done"], r["Direct"]["V_done"]),
            "P_viol": r["HQSPO"]["P_viol"],
            "choices": r["HQSPO"]["choices"],
        })

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    summary = {
        "n": len(per),
        "mean_V_LVM": mean([p["V_LVM"] for p in per]),
        "mean_V_Direct": mean([p["V_Direct"] for p in per]),
        "mean_V_HQSPO": mean([p["V_HQSPO"] for p in per]),
        "mean_d_vs_LVM": mean([p["d_vs_LVM"] for p in per]),
        "mean_d_vs_best": mean([p["d_vs_best"] for p in per]),
        "frac_beat_LVM": mean([p["d_vs_LVM"] > 0 for p in per]),
        "frac_beat_best": mean([p["d_vs_best"] > 0 for p in per]),
        "flat_win_vs_best": bool(mean([p["d_vs_best"] for p in per]) is not None and mean([p["d_vs_best"] for p in per]) > 0),
        "per_seed": per,
        "elapsed_seconds": time.time() - t0,
    }
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# HQ-SPO method eval v2 (conservative gate) vs LVM / Direct",
        "",
        f"**Seeds:** `{SEEDS[0]}–{SEEDS[-1]}`",
        f"**Prior v1:** FAIL mean Δ vs LVM ≈ −1.27 on 72900–07",
        f"**mean Δ vs LVM:** {summary['mean_d_vs_LVM']}",
        f"**mean Δ vs best(LVM,Direct):** {summary['mean_d_vs_best']}",
        f"**flat win vs best:** `{summary['flat_win_vs_best']}`",
        f"**frac beat LVM:** {summary['frac_beat_LVM']}",
        "",
        f"- mean V LVM: {summary['mean_V_LVM']}",
        f"- mean V Direct: {summary['mean_V_Direct']}",
        f"- mean V HQSPO: {summary['mean_V_HQSPO']}",
        "",
        "| seed | LVM | Direct | HQSPO | Δ vs LVM | Δ vs best | choices |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for p in per:
        lines.append(
            f"| {p['sim_seed']} | {p['V_LVM']:.2f} | {p['V_Direct']:.2f} | {p['V_HQSPO']:.2f} | "
            f"{p['d_vs_LVM']:+.2f} | {p['d_vs_best']:+.2f} | {p['choices']} |"
        )
    (REPORTS / "SPARC_HQ_SPO_METHOD_EVAL_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Keep pointer on main eval report
    (REPORTS / "SPARC_HQ_SPO_METHOD_EVAL.md").write_text(
        "\n".join([
            "# HQ-SPO method eval — status",
            "",
            "v1 FAIL: `SPARC_HQ_SPO_METHOD_EVAL.md` history / raw `hq_spo_method_eval`",
            "v2 (conservative): see `SPARC_HQ_SPO_METHOD_EVAL_V2.md`",
            "",
        ]) + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
