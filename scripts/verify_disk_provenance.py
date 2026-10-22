from pathlib import Path
import json
import numpy as np

print("=" * 80)
print("PROVENANCE VERIFICATION ACROSS DISK FILES")
print("=" * 80)

# 1. Density 12 Audit Check
d12_dir = Path("sparc_performance_evaluation/results/raw/authoritative_density12_audit/runs")
print("\n--- Density 12 Directory Inspection ---")
for ep in sorted(d12_dir.glob("episode_*")):
    audit_file = ep / "bridge_audit.json"
    if audit_file.exists():
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        cbr = [float(e.get("cbr", 0.0)) for e in valid]
        viols = sum(1 for p in prr if p < 0.95)
        print(f"  {ep.name}: viols={viols}/{len(valid)} ({viols/max(1,len(valid)):.4f}) | Mean PRR={np.mean(prr):.4f} | Mean CBR={np.mean(cbr):.4f}")

# Check summary file if exists
s12 = Path("sparc_performance_evaluation/results/raw/authoritative_density12_audit/authoritative_density12_summary.json")
if s12.exists():
    print("  Summary JSON contents:")
    for row in json.loads(s12.read_text(encoding="utf-8")):
        print(f"    Seed {row['seed']}: p_viol={row['p_viol']}, prr={row['mean_prr']}, cbr={row['mean_cbr']}")

# 2. Density 35 Zero-Offload Check
d35_zero_dir = Path("sparc_performance_evaluation/results/raw/zero_offload_audit/runs")
print("\n--- Density 35 Zero-Offload Directory Inspection ---")
if d35_zero_dir.exists():
    for ep in sorted(d35_zero_dir.glob("episode_*")):
        audit_file = ep / "bridge_audit.json"
        if audit_file.exists():
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
            valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
            prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
            cbr = [float(e.get("cbr", 0.0)) for e in valid]
            viols = sum(1 for p in prr if p < 0.95)
            bytes_off = sum(int(e.get("computation_bytes_offered", 0)) for e in audit)
            print(f"  {ep.name}: bytes={bytes_off} | viols={viols}/{len(valid)} ({viols/max(1,len(valid)):.4f}) | Mean PRR={np.mean(prr):.4f} | Mean CBR={np.mean(cbr):.4f}")
else:
    print("  zero_offload_audit/runs directory not found!")
