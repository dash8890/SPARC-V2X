"""Verify exact deterministic replay integrity for WiLab closed loop."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from train_wilab_closed_loop import ROOT as WROOT, run_episode
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.causal_safety_guard import CausalGuardedPolicy


def test_wilab_replay_exact_determinism():
    policy = CausalGuardedPolicy(r_thresh=0.20)
    policy.eval()

    seed = 73301
    t_seed = 186101
    results = []
    temp_dir = Path(tempfile.gettempdir()) / "test_replay_det"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    for rep in (1, 2):
        bridge = temp_dir / f"bridge_{rep}"
        out = temp_dir / f"out_{rep}"
        gov = SafetyGovernor(175000, horizon_envelope_only=True)
        _, _, comp, _ = run_episode(
            WROOT, bridge, 9990 + rep, 15, policy, 900.0,
            False, False, 100.0, 20.0, seed, 100,
            None, 12, False, 175000,
            300, 12.0, "structured_priority", None,
            True, "off", "", True, t_seed, gov,
            175000, True, True, 1.0, None, None, None, 2e8, 1e9,
            output_root=out, arrival_mode="streaming",
        )
        audit = json.loads((out / f"episode_{9990+rep:04d}" / "bridge_audit.json").read_text())
        results.append((comp, audit))

    c1, a1 = results[0]
    c2, a2 = results[1]
    assert c1 == pytest.approx(c2, abs=1e-6), f"Completed value diverged: {c1} vs {c2}"
    assert len(a1) == len(a2), f"Audit length diverged: {len(a1)} vs {len(a2)}"
    
    # Check that decisions and PRR match epoch by epoch
    for ep_idx, (e1, e2) in enumerate(zip(a1, a2)):
        assert e1.get("computation_bytes_offered") == e2.get("computation_bytes_offered"), (
            f"Epoch {ep_idx} bytes diverged: {e1.get('computation_bytes_offered')} vs {e2.get('computation_bytes_offered')}"
        )
        assert e1.get("delivered_source_ids") == e2.get("delivered_source_ids"), (
            f"Epoch {ep_idx} delivered sources diverged"
        )
        assert e1.get("prr") == pytest.approx(e2.get("prr"), abs=1e-6), (
            f"Epoch {ep_idx} PRR diverged"
        )


if __name__ == "__main__":
    test_wilab_replay_exact_determinism()
    print("Exact replay determinism test passed!")
