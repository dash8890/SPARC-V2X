"""Minimum Control-Authority Diagnostics (D1–D5).

Executes the formal diagnostic suite recommended in the Model Repair & Improvement Audit:
  D1 — Budget sensitivity: Evaluate packer across beta in [0, 0.25, 0.5, 0.75, 1.0]
  D2 — Price sensitivity: Sweep shadow price lambda across [0, 50]
  D4 — Observability audit: Verify observation features respond to task mix, helper queues, radio history
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from task_layer import TaskLayer
from controllers.rl.pb_crl_packer import price_budget_pack
from controllers.rl.pb_crl_b5_prime import state_vec_from_wilab


def run_d1_d2():
    print("=" * 80)
    print("DIAGNOSTIC D1 & D2: PACKER SENSITIVITY TO (beta, lambda)")
    print("=" * 80)
    
    # Test on multiple epochs / seeds
    for seed in [186100, 186101]:
        layer = TaskLayer(35, seed, tasks_per_source=4, controller_epochs=100)
        cands = layer.candidates(0.09, 0.0)
        task_map = {t.task_id: t for t in layer.tasks}
        
        print(f"\n--- Task Seed {seed}: {len(cands)} Candidates Generated at Epoch 0 ---")
        
        # D1: Budget sensitivity with lambda = 0
        print("\n[D1: Budget Sensitivity Sweep (lambda = 0.0)]")
        print(f"{'beta':<8} | {'Tasks':<6} | {'Offered Bytes':<14} | {'Delivered Value':<16} | {'Packed Task IDs'}")
        print("-" * 75)
        for beta in [0.0, 0.25, 0.50, 0.75, 1.0]:
            pack = price_budget_pack(cands, lam=0.0, beta_exec=beta, b_max=175000)
            b_sum = sum(int(a["payload_bytes"]) for a in pack)
            v_sum = sum(float(task_map[a["task_id"]].value) for a in pack)
            tids = [a["task_id"] for a in pack]
            print(f"{beta:<8.2f} | {len(pack):<6} | {b_sum:<14} | {v_sum:<16.2f} | {tids}")
            
        # D2: Price sensitivity with beta = 1.0
        print("\n[D2: Price Sensitivity Sweep (beta = 1.0)]")
        print(f"{'lambda':<8} | {'Tasks':<6} | {'Offered Bytes':<14} | {'Delivered Value':<16} | {'Packed Task IDs'}")
        print("-" * 75)
        for lam in [0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 20.0, 35.0, 50.0]:
            pack = price_budget_pack(cands, lam=lam, beta_exec=1.0, b_max=175000)
            b_sum = sum(int(a["payload_bytes"]) for a in pack)
            v_sum = sum(float(task_map[a["task_id"]].value) for a in pack)
            tids = [a["task_id"] for a in pack]
            print(f"{lam:<8.1f} | {len(pack):<6} | {b_sum:<14} | {v_sum:<16.2f} | {tids}")


def run_d4():
    print("\n" + "=" * 80)
    print("DIAGNOSTIC D4: OBSERVABILITY & FEATURE SENSITIVITY AUDIT")
    print("=" * 80)
    
    layer = TaskLayer(35, 186100, tasks_per_source=4, controller_epochs=100)
    fake_state = {"cbr": 0.09, "resel_t": 0.35, "reselection_rate": 0.35}
    
    # Check what state_vec_from_wilab actually produces
    svec = state_vec_from_wilab(fake_state, layer, dim=16)
    print(f"\nCurrent `state_vec_from_wilab` output (dim={len(svec)}):")
    print(f"  {svec}")
    non_zeros = sum(1 for x in svec if abs(x) > 1e-6)
    print(f"  Active non-zero features: {non_zeros} / {len(svec)}")
    print(f"  Dead zero-padded slots:   {len(svec) - non_zeros} / {len(svec)}")
    
    # Audit candidates and helpers
    cands = layer.candidates(0.09, 0.0)
    print(f"\nTask Layer Rich Information Available but Ignored:")
    print(f"  Candidate tasks available: {len(cands)}")
    if cands:
        print(f"  Sample candidate features: {cands[0].get('features')}")
    print(f"  Helper nodes available: {len(layer.helpers)}")
    if layer.helpers:
        h0 = layer.helpers[0]
        print(f"  Sample helper state: CPU={h0.cpu_cycles_s/1e9:.2f} GHz, Queue={h0.queue_cycles/1e9:.2f} Gcyc, Ready={h0.ready_time_ms:.1f} ms")


if __name__ == "__main__":
    run_d1_d2()
    run_d4()
