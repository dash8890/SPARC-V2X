"""Audit script to reconcile Step 5 and Step 6 episode results directly from disk."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base_s6 = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step6_baseline_comparison"
base_s5 = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "step5_matched_comparison"

print("=" * 90)
print("RECONCILIATION AUDIT: LOCATING ACTUAL EPISODE OUTPUTS")
print("=" * 90)

print("\n--- STEP 6 EPISODES ---")
eids_s6 = {
    80001: ("B4_ce_milp", 73300),
    80002: ("B5_prime", 73300),
    80003: ("PB_CRL_repaired", 73300),
    80004: ("B4_ce_milp", 73301),
    80005: ("B5_prime", 73301),
    80006: ("PB_CRL_repaired", 73301),
    80007: ("B4_ce_milp", 73302),
    80008: ("B5_prime", 73302),
    80009: ("PB_CRL_repaired", 73302),
    80010: ("B4_ce_milp", 73303),
    80011: ("B5_prime", 73303),
    80012: ("PB_CRL_repaired", 73303),
}

for eid, (m, s) in eids_s6.items():
    matches = list(base_s6.rglob(f"episode_{eid:04d}"))
    if not matches:
        print(f"Eid {eid} ({m}, seed {s}): NOT FOUND!")
        continue
    d = matches[0]
    audit_file = d / "bridge_audit.json"
    if not audit_file.is_file():
        print(f"Eid {eid} ({m}, seed {s}) in {d}: audit file missing!")
        continue
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    cbr = [float(e.get("cbr", 0.0)) for e in valid]
    viols = sum(1 for p in prr if p < 0.95)
    mean_prr = sum(prr) / len(prr) if prr else 1.0
    mean_cbr = sum(cbr) / len(cbr) if cbr else 0.0

    st_file = d / "task_layer_state.json"
    comp_val = 0.0
    if st_file.is_file():
        st = json.loads(st_file.read_text(encoding="utf-8"))
        comp_val = float(st.get("completed_value", 0.0))

    bytes_offered = sum(int(e.get("computation_bytes_offered", 0)) for e in audit)
    rel_path = d.relative_to(base_s6)
    print(f"Eid {eid} | {m:<16} | Seed {s} | V_done={comp_val:6.2f} | PRR={mean_prr:.4f} | CBR={mean_cbr:.4f} | Viols={viols:2d}/100 | Bytes={bytes_offered:6d} | Path={rel_path}")

print("\n--- STEP 5 EPISODES ---")
for cond in ["zero_offload", "pb_crl"]:
    for i, s in enumerate([73300, 73301, 73302, 73303]):
        eid = (70000 + i) if cond == "zero_offload" else (71000 + i)
        d = base_s5 / cond / "runs" / f"episode_{eid:04d}"
        audit_file = d / "bridge_audit.json"
        if not audit_file.is_file():
            print(f"Step 5 {cond} seed {s} (eid {eid}): audit missing at {d}")
            continue
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
        prr = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
        cbr = [float(e.get("cbr", 0.0)) for e in valid]
        viols = sum(1 for p in prr if p < 0.95)
        mean_prr = sum(prr) / len(prr) if prr else 1.0
        mean_cbr = sum(cbr) / len(cbr) if cbr else 0.0

        st_file = d / "task_layer_state.json"
        comp_val = 0.0
        if st_file.is_file():
            st = json.loads(st_file.read_text(encoding="utf-8"))
            comp_val = float(st.get("completed_value", 0.0))

        bytes_offered = sum(int(e.get("computation_bytes_offered", 0)) for e in audit)
        print(f"Eid {eid} | {cond:<12} | Seed {s} | V_done={comp_val:6.2f} | PRR={mean_prr:.4f} | CBR={mean_cbr:.4f} | Viols={viols:2d}/100 | Bytes={bytes_offered:6d}")
