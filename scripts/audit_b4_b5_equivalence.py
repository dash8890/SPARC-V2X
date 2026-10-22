"""Verify B4 and B5' comparator equivalence by comparing actual decisions epoch-by-epoch."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base_s6 = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step6_baseline_comparison"

print("=" * 90)
print("B4 vs B5' COMPARATOR IMPLEMENTATION VERIFICATION")
print("=" * 90)

seeds = [73300, 73301, 73302, 73303]
b4_eids = {73300: 80001, 73301: 80004, 73302: 80007, 73303: 80010}
b5_eids = {73300: 80002, 73301: 80005, 73302: 80008, 73303: 80011}

for s in seeds:
    b4_d = list(base_s6.rglob(f"episode_{b4_eids[s]:04d}"))[0]
    b5_d = list(base_s6.rglob(f"episode_{b5_eids[s]:04d}"))[0]

    b4_audit = json.loads((b4_d / "bridge_audit.json").read_text(encoding="utf-8"))
    b5_audit = json.loads((b5_d / "bridge_audit.json").read_text(encoding="utf-8"))

    mismatches = 0
    total_epochs = len(b4_audit)

    for ep in range(total_epochs):
        b4_sel = b4_audit[ep].get("selected_assignments") or b4_audit[ep].get("assignments") or []
        b5_sel = b5_audit[ep].get("selected_assignments") or b5_audit[ep].get("assignments") or []

        b4_pairs = sorted([(int(a["task_id"]), int(a["helper_id"])) for a in b4_sel if "task_id" in a])
        b5_pairs = sorted([(int(a["task_id"]), int(a["helper_id"])) for a in b5_sel if "task_id" in a])

        if b4_pairs != b5_pairs:
            mismatches += 1

    print(f"Seed {s}: {total_epochs} epochs compared.")
    print(f"  Exact (task_id, helper_id) matching epochs: {total_epochs - mismatches} / {total_epochs}")
    print(f"  Mismatch epochs: {mismatches}")
    assert mismatches == 0, f"B4 and B5' differed on seed {s}!"

print("\nConclusion: B4 (ce_milp) and B5' (b5_prime) make 100% identical task-helper decisions at every epoch.")
print("Explanation:")
print("  - In B4, ce_byte_cap(cbr, resel, w) under CBR~0.09 evaluates to b_eff = 175,000 (full budget), then packs with lam=0.")
print("  - In B5', the preserved checkpoint policy outputs beta~1.0 (175,000) and lam~0, then packs with lam=0.")
print("  - Therefore, both algorithms reduce to identical greedy knapsack packing with identical tie-breaking.")
