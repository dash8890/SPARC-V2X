"""Phase B: Counterfactual Opportunity & Policy Representability Analysis.

Analyzes the counterfactual replay dataset to determine:
1. Are there safety-compatible decisions that yield higher downstream utility?
2. Are these beneficial assignments representable by the 5D priority-weight policy (w in R_+^4, beta in [0, 1])?
3. Enforces the strict Phase B gate:
   If no beneficial representable actions exist, STOP and report.
   If beneficial representable actions exist, generate training targets for Phase C.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from controllers.rl.structured_priority_policy import score_and_match
from controllers.rl.causal_safety_guard import filter_candidates_causal_guard
from experiments.wilab_closed_loop.task_layer import TaskLayer


def is_assignment_representable(
    candidates: List[Dict[str, Any]],
    target_assignments: List[Tuple[int, int]],
    b_max: int = 175000,
    search_resolution: int = 7,
) -> Tuple[bool, Optional[List[float]], Optional[float]]:
    """Determines whether target_assignments is reproducible by w in [0, 5]^4 and beta in [0, 1].

    Uses grid search over priority weights w = (w_v, w_q, w_o, w_r) and beta.
    Returns (is_representable, best_w, best_beta).
    """
    if not target_assignments:
        return True, [1.0, 1.0, 1.0, 1.0], 0.0

    target_set = set(target_assignments)

    # Grid values for priority weights in [0, 5]
    wv_vals = [1.0, 2.5, 4.0, 5.0]
    wq_vals = [0.0, 0.5, 1.0, 2.0]
    wo_vals = [0.0, 0.5, 1.0, 2.0]
    wr_vals = [0.0, 0.5, 1.0]
    beta_vals = [0.85, 0.95, 1.0]

    for w_v in wv_vals:
        for w_q in wq_vals:
            for w_o in wo_vals:
                for w_r in wr_vals:
                    for beta in beta_vals:
                        pack, _ = score_and_match(
                            candidates,
                            weights=[w_v, w_q, w_o, w_r],
                            beta_throttle=beta,
                            b_max=b_max,
                            unique_sources=True,
                            unique_helpers=True,
                        )
                        pack_set = set((a["task_id"], a["helper_id"]) for a in pack)
                        if pack_set == target_set:
                            return True, [w_v, w_q, w_o, w_r], beta

    return False, None, None


def analyze_opportunities(
    dataset_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = data.get("records", [])

    print(f"\n=======================================================")
    print(f"PHASE B: COUNTERFACTUAL OPPORTUNITY & REPRESENTABILITY ANALYSIS")
    print(f"Total Evaluated Replay Decisions: {len(records)}")
    print(f"=======================================================\n")

    beneficial_records = [r for r in records if r["is_beneficial"]]
    print(f"Beneficial Decisions Found (Delta V > 0, P_viol <= 2%): {len(beneficial_records)}")

    representable_count = 0
    analyzed_records = []

    for r in records:
        seed = r["seed"]
        epoch = r["epoch"]
        target_assn = [tuple(x) for x in r["assn_b4"]]

        # Retrieve candidate pool from the matched decision replay
        observed_cands = r.get("candidates")
        if not observed_cands:
            # Combine packs from B4 and SP (unique by (task_id, helper_id))
            seen = set()
            observed_cands = []
            for c in (r.get("b4_pack", []) + r.get("sp_pack", [])):
                key = (c["task_id"], c["helper_id"])
                if key not in seen:
                    seen.add(key)
                    observed_cands.append(c)

        eff_b = max([sum(c.get("payload_bytes", 0) for c in r.get("b4_pack", [])), 55000, 175000])
        is_rep, best_w, best_b = is_assignment_representable(observed_cands, target_assn, b_max=eff_b)

        r_copy = dict(r)
        r_copy["is_representable"] = is_rep
        r_copy["best_w"] = best_w
        r_copy["best_beta"] = best_b

        if r["is_beneficial"] and is_rep:
            representable_count += 1
            print(f"  [Actionable Opportunity] Seed {seed} Epoch {epoch:02d}: Delta V = {r['delta_v']:+.2f} | Representable with w={best_w}, beta={best_b}")
        elif r["is_beneficial"] and not is_rep:
            print(f"  [Non-Representable Opportunity] Seed {seed} Epoch {epoch:02d}: Delta V = {r['delta_v']:+.2f} | Target assignment cannot be formed by priority weights")

        analyzed_records.append(r_copy)

    pass_gate = bool(representable_count > 0)

    summary = {
        "total_evaluated": len(records),
        "total_beneficial": len(beneficial_records),
        "total_beneficial_representable": representable_count,
        "gate_passed": pass_gate,
        "actionable_opportunities": [r for r in analyzed_records if r["is_beneficial"] and r["is_representable"]],
    }

    out_file = output_dir / "counterfactual_opportunities_analysis.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=======================================================")
    print(f"PHASE B GATE RESULT: {'PASS (Authorize Phase C Learning)' if pass_gate else 'FAIL (STOP - No Beneficial Representable Actions)'}")
    print(f"Actionable Beneficial & Representable Opportunities: {representable_count}")
    print(f"Saved analysis to {out_file}")
    print(f"=======================================================\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze counterfactual opportunities")
    parser.add_argument("--dataset", type=str, default="sparc_performance_evaluation/results/raw/counterfactual_replay/counterfactual_replay_dataset.json", help="Dataset path")
    parser.add_argument("--out-dir", type=str, default="sparc_performance_evaluation/results/raw/counterfactual_replay", help="Output directory")
    args = parser.parse_args()

    analyze_opportunities(ROOT / args.dataset, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
