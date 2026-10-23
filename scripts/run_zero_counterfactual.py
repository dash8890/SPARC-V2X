"""Counterfactual Replay: Zero Computation on Seeds 73300 and 73301."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor


class ZeroComputationPolicy:
    is_scql_policy = True
    def _sequential_match(self, candidates, **kwargs):
        return [], {"selected_mode": "zero_computation", "n_pack": 0, "bytes": 0}


def run_zero_counterfactual(seed: int, eid: int, out_dir: Path):
    t_seed = 186100 + (seed - 73300) if seed >= 73300 else 191000 + (seed - 73200)
    bridge = Path(tempfile.gettempdir()) / f"bridge_zero_cf_{seed}_{eid}"
    if bridge.exists():
        shutil.rmtree(bridge, ignore_errors=True)
    gov = SafetyGovernor(175000, horizon_envelope_only=True)
    policy = ZeroComputationPolicy()

    comp_val = run_episode(
        WROOT,
        bridge,
        eid,
        15,
        policy,
        900.0,
        False,
        False,
        100.0,
        20.0,
        seed,
        100,
        None,
        12,
        False,
        175000,
        300,
        12.0,
        "scql",
        None,
        True,
        "off",
        "",
        True,
        t_seed,
        gov,
        175000,
        True,
        True,
        1.0,
        None,
        helper_swap=None,
        force_pack=None,
        helper_cpu_lo=2e8,
        helper_cpu_hi=1e9,
        output_root=out_dir,
        arrival_mode="streaming",
    )[2]

    audit_file = out_dir / f"episode_{eid:04d}" / "bridge_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.is_file() else []
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    prrs = [float(e.get("window_prr", e.get("prr", 1.0))) for e in valid]
    viols = sum(1 for p in prrs if p < 0.95)
    pv = viols / len(valid) if valid else 1.0
    return comp_val, pv, audit


if __name__ == "__main__":
    out_base = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "cf_zero_audit"
    for s, eid in [(73300, 95000), (73301, 95001)]:
        print(f"\nRunning Zero Computation Counterfactual on Seed {s} (eid {eid})...")
        c_val, pv, audit = run_zero_counterfactual(s, eid, out_base / f"zero_s{s}")
        print(f"Seed {s} Zero CF Result: V_done = {c_val:.2f} | P_viol = {pv*100:.2f}% | Total Valid Windows = {len(audit)}")
