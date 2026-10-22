import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

ROOT = Path("C:/Users/dash/.codex/worktrees/b2cd/5GProject3.0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WILAB_ROOT, run_episode
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor

RESULTS_BASE = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step1_pick_cell"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

B_MAX = 175000
K_BUDGET = 2
GIT_COMMIT = "8b96b3af2c84b3ef86b693223deaa67fe41da0db"


def run_single_probe(cell_name, density, cluster_share, tps, method, sim_seed, task_seed, ep_id, runs_dir):
    ep_dir = runs_dir / f"ep_{cell_name}_{method}_{sim_seed}"
    audit_file = ep_dir / "bridge_audit.json"
    forced_budget = 0 if method == "No-Offload" else B_MAX
    pol = ResidualTaskPolicy(logit_mode="direct", logit_scale=100.0)
    pol.eval()
    gov = SafetyGovernor(B_MAX, horizon_envelope_only=True)

    wall = 0.0
    if not audit_file.is_file():
        ep_dir.mkdir(parents=True, exist_ok=True)
        bridge = Path(tempfile.gettempdir()) / f"p1_{os.getpid()}_{cell_name}_{method}_{sim_seed}"
        if bridge.exists():
            shutil.rmtree(bridge, ignore_errors=True)
        os.environ["WILAB_OUTPUT_ROOT"] = str(runs_dir)
        os.environ["WILAB_WARMUP_SECONDS"] = "0.0"

        t0 = time.perf_counter()
        run_episode(
            WILAB_ROOT, bridge, ep_id, density, pol, 300.0,
            False, False, 100.0, 20.0, sim_seed, 50,
            None, tps, True, forced_budget,
            300, 2.5, "greedy_vb", None,
            True, "off", "", True, task_seed, gov,
            B_MAX, True, True, 1.0, None, None, None, 2e8, 1e9,
            cluster_share,
        )
        wall = time.perf_counter() - t0

        # Move generated folder if needed
        default_out = runs_dir / f"episode_{ep_id:04d}"
        if default_out.is_dir() and default_out != ep_dir:
            for item in default_out.glob("*"):
                shutil.move(str(item), str(ep_dir / item.name))
            shutil.rmtree(default_out, ignore_errors=True)

    data = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    valid = [e for e in data if e.get("cbr_valid") and e.get("delay_valid")]
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    cbr = [float(e.get("cbr", 0.0)) for e in valid]
    delay = [float(e.get("p95_delay_ms", 0.0)) for e in valid]
    resel = [float(e.get("resel_t", 0.0)) for e in valid]
    bytes_offered = [int(e.get("computation_bytes_offered", 0)) for e in valid]

    # Violation definition: existing window violation (PRR < 0.95 or delay > 50 ms)
    viols = [1 if (p < 0.95 or d > 50.0) else 0 for p, d in zip(prr, delay)]
    Z = sum(viols)
    exceeded = (Z > K_BUDGET)

    # Delivered throughput / utility
    delivered_val = sum(float(e.get("reward", 0.0)) for e in valid)
    delivered_tasks = sum(int(e.get("n_delivered_tasks", 0)) for e in valid)
    total_tasks_offered = sum(int(e.get("n_executed", 0)) for e in valid)
    ttcr = (delivered_tasks / max(total_tasks_offered, 1)) if total_tasks_offered > 0 else 0.0

    return {
        "cell": cell_name,
        "method": method,
        "sim_seed": sim_seed,
        "task_seed": task_seed,
        "ep_id": ep_id,
        "wall_s": wall,
        "n_valid": len(valid),
        "Z": Z,
        "exceeded": exceeded,
        "mean_prr": float(np.mean(prr)) if prr else 1.0,
        "min_prr": float(np.min(prr)) if prr else 1.0,
        "mean_cbr": float(np.mean(cbr)) if cbr else 0.0,
        "mean_delay_ms": float(np.mean(delay)) if delay else 0.0,
        "mean_resel": float(np.mean(resel)) if resel else 0.0,
        "mean_bytes": float(np.mean(bytes_offered)) if bytes_offered else 0.0,
        "U": delivered_val,
        "TTCR": ttcr,
    }


def evaluate_cell(cell_name, density, cluster_share, tps, sim_seeds, ep_offset):
    runs_dir = RESULTS_BASE / cell_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    task_seeds = [s + 100000 for s in sim_seeds]

    tasks = []
    ep_id = ep_offset
    for s, t in zip(sim_seeds, task_seeds):
        for m in ["No-Offload", "B1"]:
            ep_id += 1
            tasks.append((cell_name, density, cluster_share, tps, m, s, t, ep_id, runs_dir))

    print(f"\n{'='*80}")
    print(f"Evaluating Cell: {cell_name} (N={density}, cs={cluster_share:.2f}, tps={tps})")
    print(f"Seeds: {sim_seeds[0]} - {sim_seeds[-1]} ({len(sim_seeds)} seeds x 2 arms = {len(tasks)} episodes)")
    print(f"{'='*80}")

    t0 = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_single_probe, *args): args for args in tasks}
        completed = 0
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            results.append(res)
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                el = time.perf_counter() - t0
                print(f"  [{cell_name}] Completed {completed:2d}/{len(tasks)} episodes in {el/60:.2f}m...", flush=True)

    no_rows = sorted([r for r in results if r["method"] == "No-Offload"], key=lambda r: r["sim_seed"])
    b1_rows = sorted([r for r in results if r["method"] == "B1"], key=lambda r: r["sim_seed"])

    no_exceeded = sum(1 for r in no_rows if r["exceeded"])
    b1_exceeded = sum(1 for r in b1_rows if r["exceeded"])

    # Clean seeds: seeds where No-Offload has Z <= K (Z <= 2)
    clean_seeds = [r["sim_seed"] for r in no_rows if not r["exceeded"]]
    b1_clean_rows = [r for r in b1_rows if r["sim_seed"] in clean_seeds]
    b1_clean_exceeded = sum(1 for r in b1_clean_rows if r["exceeded"])

    mean_cbr_no = float(np.mean([r["mean_cbr"] for r in no_rows]))
    mean_cbr_b1 = float(np.mean([r["mean_cbr"] for r in b1_rows]))
    cbr_ratio = (mean_cbr_b1 / max(mean_cbr_no, 1e-6)) - 1.0

    mean_bytes_b1 = float(np.mean([r["mean_bytes"] for r in b1_rows]))
    mean_prr_no = float(np.mean([r["mean_prr"] for r in no_rows]))
    mean_prr_b1 = float(np.mean([r["mean_prr"] for r in b1_rows]))

    print(f"\n--- Results for {cell_name} ---")
    print(f"  No-Offload Exceedances (Z > 2): {no_exceeded}/20 ({no_exceeded/20*100:.1f}%) | Mean PRR: {mean_prr_no:.4f} | Mean CBR: {mean_cbr_no:.4f}")
    print(f"  B1 Exceedances (Z > 2):         {b1_exceeded}/20 ({b1_exceeded/20*100:.1f}%) | Mean PRR: {mean_prr_b1:.4f} | Mean CBR: {mean_cbr_b1:.4f}")
    print(f"  CBR Increase under B1:          {cbr_ratio*100:+.1f}% (Required: >= +20%)")
    print(f"  B1 Mean Bytes Offered:          {mean_bytes_b1:.0f} B (Required: > 0)")
    print(f"  Clean Seeds (No-Offload Z <= 2): {len(clean_seeds)}/20")
    print(f"  B1 Attributable Exceedances:     {b1_clean_exceeded}/{len(clean_seeds)} ({b1_clean_exceeded/max(len(clean_seeds),1)*100:.1f}%)")

    # Pass criteria:
    # 1. No-Offload has Z > 2 on <= 2 of 20 seeds
    # 2. B1 has Z > 2 on >= 6 of 20 seeds
    # 3. Mean CBR is at least 20% higher under B1, and computation_bytes_offered > 0
    cond1 = (no_exceeded <= 2)
    cond2 = (b1_exceeded >= 6)
    cond3 = (cbr_ratio >= 0.20 and mean_bytes_b1 > 0)
    passed = cond1 and cond2 and cond3

    print(f"\n  Condition 1 (No-Offload Z > 2 <= 2/20):    {'PASS' if cond1 else 'FAIL'} ({no_exceeded}/20)")
    print(f"  Condition 2 (B1 Z > 2 >= 6/20):            {'PASS' if cond2 else 'FAIL'} ({b1_exceeded}/20)")
    print(f"  Condition 3 (CBR rise >= +20% & Bytes > 0): {'PASS' if cond3 else 'FAIL'} (+{cbr_ratio*100:.1f}%, {mean_bytes_b1:.0f} B)")
    print(f"  >>> CELL VERDICT: {'PASSED (CALIBRATED CELL)' if passed else 'FAILED'} <<<\n")

    # Save summary
    summary = {
        "cell": cell_name,
        "density": density,
        "cluster_share": cluster_share,
        "tps": tps,
        "git_commit": GIT_COMMIT,
        "passed": passed,
        "cond1": cond1,
        "cond2": cond2,
        "cond3": cond3,
        "no_exceeded": no_exceeded,
        "b1_exceeded": b1_exceeded,
        "clean_seeds_count": len(clean_seeds),
        "b1_clean_exceeded": b1_clean_exceeded,
        "mean_cbr_no": mean_cbr_no,
        "mean_cbr_b1": mean_cbr_b1,
        "cbr_ratio": cbr_ratio,
        "mean_bytes_b1": mean_bytes_b1,
        "mean_prr_no": mean_prr_no,
        "mean_prr_b1": mean_prr_b1,
        "results": results,
    }
    (runs_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return passed, summary


def main():
    sim_seeds = [97000 + i for i in range(20)]
    print(f"Starting Step 1 Cell Selection with Git Commit {GIT_COMMIT}")
    print(f"Seeds: {sim_seeds[0]} - {sim_seeds[-1]}")

    # Primary Cell: N=30, cluster_share=0.70, tps=5
    passed, summary = evaluate_cell("N30_cs0.70_tps5", 30, 0.70, 5, sim_seeds, 10000)
    if passed:
        print("Primary cell N30_cs0.70_tps5 PASSED! Proceeding to Step 2.")
        return

    # Attempt 1: cluster share 0.80
    print("\nPrimary cell did not meet all criteria. Attempt 1: cluster share 0.80...")
    passed, summary = evaluate_cell("N30_cs0.80_tps5", 30, 0.80, 5, sim_seeds, 10100)
    if passed:
        print("Attempt 1 (N30_cs0.80_tps5) PASSED! Proceeding to Step 2.")
        return

    # Attempt 2: tasks per source 6
    print("\nAttempt 1 did not meet all criteria. Attempt 2: tasks per source 6...")
    passed, summary = evaluate_cell("N30_cs0.70_tps6", 30, 0.70, 6, sim_seeds, 10200)
    if passed:
        print("Attempt 2 (N30_cs0.70_tps6) PASSED! Proceeding to Step 2.")
        return

    # Attempt 3: N = 32
    print("\nAttempt 2 did not meet all criteria. Attempt 3: N = 32...")
    passed, summary = evaluate_cell("N32_cs0.70_tps5", 32, 0.70, 5, sim_seeds, 10300)
    if passed:
        print("Attempt 3 (N32_cs0.70_tps5) PASSED! Proceeding to Step 2.")
        return

    print("\nAll 3 attempts failed. Coupling is too weak in this configuration.")


if __name__ == "__main__":
    main()
