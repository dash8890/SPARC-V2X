from pathlib import Path
import pandas as pd
import numpy as np

runs_dir = Path("sparc_performance_evaluation/results/raw/step6_baseline_comparison/B4_ce_milp/runs")
print("=" * 80)
print("AUDIT: B4 SCHEDULING OPPORTUNITY ANALYSIS (DENSITY 35)")
print("=" * 80)

for ep in sorted(runs_dir.glob("episode_*")):
    csv_file = ep / "task_trace.csv"
    if not csv_file.exists():
        continue
    df = pd.read_csv(csv_file)
    total_val = df["task_value"].sum()
    done_df = df[df["done"] == True]
    done_val = done_df["task_value"].sum()
    incomp = df[df["done"] == False]
    
    # Check max helper CPU across helpers
    # Helper CPU is typically up to 1e9 cycles/s (1 GHz)
    max_cpu = 1e9
    
    feasible_rej = []
    infeasible_rej = []
    for idx, row in incomp.iterrows():
        min_exec_ms = (row["cycles"] / max_cpu) * 1000.0
        # If pure exec time alone exceeds deadline, it is strictly physically infeasible
        if min_exec_ms > row["deadline_ms"]:
            infeasible_rej.append((row["task_id"], row["task_value"], min_exec_ms, row["deadline_ms"], "exec_exceeds_deadline"))
        else:
            feasible_rej.append((row["task_id"], row["task_value"], min_exec_ms, row["deadline_ms"]))
            
    print(f"\n{ep.name}:")
    print(f"  Total Offered Tasks: {len(df)} | Total Value: {total_val:.2f}")
    print(f"  B4 Completed Tasks:  {len(done_df)} | Completed Value: {done_val:.2f} ({done_val/total_val*100:.1f}%)")
    print(f"  B4 Rejected Tasks:   {len(incomp)} | Rejected Value: {incomp['task_value'].sum():.2f}")
    print(f"  - Physically Impossible (min_exec > deadline): {len(infeasible_rej)} tasks (Value: {sum(x[1] for x in infeasible_rej):.2f})")
    print(f"  - Potentially Feasible (min_exec <= deadline):  {len(feasible_rej)} tasks (Value: {sum(x[1] for x in feasible_rej):.2f})")
    if feasible_rej:
        for fid, fval, fexec, fdl in feasible_rej:
            print(f"      Task {fid}: val={fval:.2f}, min_exec={fexec:.1f}ms, deadline={fdl:.1f}ms, slack={fdl-fexec:.1f}ms")
