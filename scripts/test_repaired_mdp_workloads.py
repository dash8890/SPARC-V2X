"""Simulation test of repaired TaskLayer with persistent helper queues and deadline tracking.

Tests whether sequential helper-queue persistence and buffer oversubscription
expose a reachable value gap where PB-CRL's (lambda, beta) control interface can outperform B4.
"""
from __future__ import annotations

import sys
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from controllers.rl.pb_crl_packer import price_budget_pack
from controllers.rl.hq_spo_baselines import build_baseline_pack


class RepairedTaskLayer:
    """Rigorous MDP TaskLayer with faithful helper queues and deadline expiration."""

    def __init__(self, density: int, seed: int, n_tasks_per_source: int = 8,
                 epoch_interval_ms: float = 10.0, arrival_mode: str = "burst_and_stream",
                 controller_epochs: int = 100):
        self.rng = random.Random(seed)
        self.epoch_interval_ms = float(epoch_interval_ms)
        self.controller_epochs = int(controller_epochs)
        self.current_epoch = 0
        
        # Helpers
        self.n_helpers = min(12, density)
        self.helpers = []
        for i in range(self.n_helpers):
            cpu = self.rng.uniform(2e8, 1e9)
            self.helpers.append({
                "helper_id": i,
                "cpu_cycles_s": cpu,
                "ready_time_ms": 0.0,
                "queue_cycles": 0.0,
            })
            
        # Sources
        n_sources = max(1, density // 5)
        self.tasks = []
        
        tid = 0
        for s in range(n_sources):
            for k in range(n_tasks_per_source):
                if arrival_mode == "burst":
                    rel_epoch = 0
                elif arrival_mode == "burst_and_stream":
                    # Wave 0 (burst) at epoch 0, then waves every 10 epochs
                    rel_epoch = 0 if k < 2 else (k - 1) * 10
                elif arrival_mode == "poisson":
                    rel_epoch = int(self.rng.expovariate(1.0 / 15.0) * k)
                    rel_epoch = min(rel_epoch, controller_epochs - 1)
                else:
                    rel_epoch = (k * controller_epochs) // n_tasks_per_source

                b = self.rng.randint(15_000, 75_000)
                cyc = self.rng.uniform(2e7, 1e8)
                dl = self.rng.uniform(40.0, 150.0)
                val = self.rng.uniform(1.0, 5.0)
                self.tasks.append({
                    "task_id": tid,
                    "source_id": s,
                    "payload_bytes": b,
                    "cycles": cyc,
                    "deadline_ms": dl,
                    "value": val,
                    "release_epoch": rel_epoch,
                    "release_time_ms": rel_epoch * self.epoch_interval_ms,
                    "absolute_deadline_ms": rel_epoch * self.epoch_interval_ms + dl,
                    "done": False,
                    "queued": False,
                    "assigned_helper": None,
                    "finish_ms": None,
                })
                tid += 1

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        now_ms = self.current_epoch * self.epoch_interval_ms
        
        # 1. Complete finished tasks
        for t in self.tasks:
            if t["queued"] and not t["done"]:
                if t["finish_ms"] is not None and t["finish_ms"] <= now_ms:
                    if t["finish_ms"] <= t["absolute_deadline_ms"]:
                        t["done"] = True
                    t["queued"] = False

        # 2. Update helper queues
        for h in self.helpers:
            rem_ms = max(0.0, h["ready_time_ms"] - now_ms)
            h["queue_cycles"] = rem_ms * h["cpu_cycles_s"] / 1000.0

    def candidates(self, cbr: float = 0.04):
        now_ms = self.current_epoch * self.epoch_interval_ms
        cands = []
        for t in self.tasks:
            if t["done"] or t["queued"]:
                continue
            # Must be released
            if t["release_epoch"] > self.current_epoch:
                continue
            # Must not be expired
            if now_ms >= t["absolute_deadline_ms"]:
                continue
                
            for h in self.helpers:
                compute_ms = 1000.0 * t["cycles"] / h["cpu_cycles_s"]
                start_ms = max(now_ms + self.epoch_interval_ms, h["ready_time_ms"])
                finish_ms = start_ms + compute_ms
                
                # Prune pairs that miss deadline
                if finish_ms > t["absolute_deadline_ms"]:
                    continue
                    
                cands.append({
                    "task_id": t["task_id"],
                    "source_id": t["source_id"],
                    "helper_id": h["helper_id"],
                    "payload_bytes": t["payload_bytes"],
                    "packet_count": (t["payload_bytes"] + 999) // 1000,
                    "finish_ms": finish_ms - now_ms,
                    "deadline_ms": t["deadline_ms"],
                    "task_value": t["value"],
                    "deadline_feasible": True,
                    "features": [
                        t["payload_bytes"] / 1e6,
                        t["cycles"] / 1e8,
                        t["deadline_ms"] / 1000.0,
                        t["value"] / 5.0,
                        h["cpu_cycles_s"] / 1e9,
                        h["queue_cycles"] / 1e8,
                        0.0,
                        cbr,
                        self.current_epoch / 100.0,
                        1.0,
                    ],
                })
        return cands

    def apply_assignments(self, assignments):
        now_ms = self.current_epoch * self.epoch_interval_ms
        for a in assignments:
            t = next(x for x in self.tasks if x["task_id"] == a["task_id"])
            h = next(x for x in self.helpers if x["helper_id"] == a["helper_id"])
            
            compute_ms = 1000.0 * t["cycles"] / h["cpu_cycles_s"]
            start_ms = max(now_ms + self.epoch_interval_ms, h["ready_time_ms"])
            finish_ms = start_ms + compute_ms
            
            if finish_ms <= t["absolute_deadline_ms"]:
                t["queued"] = True
                t["finish_ms"] = finish_ms
                t["assigned_helper"] = h["helper_id"]
                h["ready_time_ms"] = finish_ms
                h["queue_cycles"] = max(0.0, (h["ready_time_ms"] - (now_ms + self.epoch_interval_ms)) * h["cpu_cycles_s"] / 1000.0)

    def completed_value(self):
        return sum(t["value"] for t in self.tasks if t["done"])


def test_workload(arrival_mode: str, n_tasks: int):
    print(f"\n=======================================================")
    print(f"EVALUATING WORKLOAD: {arrival_mode.upper()} (Tasks/Source={n_tasks})")
    print(f"=======================================================")

    seeds = [73300, 73301, 73302, 73303]
    b_max = 175000
    
    lam_grid = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    beta_grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    b4_vals = []
    max_pbcrl_vals = []
    
    for s in seeds:
        t_seed = 186100 + (s - 73300)
        
        # 1. B4 Run (lam=0, beta=1.0)
        l_b4 = RepairedTaskLayer(15, t_seed, n_tasks_per_source=n_tasks, arrival_mode=arrival_mode)
        for ep in range(100):
            l_b4.set_epoch(ep)
            cands = l_b4.candidates()
            if cands:
                pack = price_budget_pack(cands, lam=0.0, beta_exec=1.0, b_max=b_max, unique_sources=True)
                l_b4.apply_assignments(pack)
        l_b4.set_epoch(100)
        v_b4 = l_b4.completed_value()
        b4_vals.append(v_b4)
        
        # 2. Grid Search over PB-CRL action space (lam, beta)
        best_v = 0.0
        best_cfg = None
        for b in beta_grid:
            for l in lam_grid:
                l_pb = RepairedTaskLayer(15, t_seed, n_tasks_per_source=n_tasks, arrival_mode=arrival_mode)
                for ep in range(100):
                    l_pb.set_epoch(ep)
                    cands = l_pb.candidates()
                    if cands:
                        pack = price_budget_pack(cands, lam=l, beta_exec=b, b_max=b_max, unique_sources=True)
                        l_pb.apply_assignments(pack)
                l_pb.set_epoch(100)
                v = l_pb.completed_value()
                if v > best_v:
                    best_v = v
                    best_cfg = (l, b)
                    
        max_pbcrl_vals.append(best_v)
        gap = best_v - v_b4
        print(f"Seed {s}: B4 V_done={v_b4:6.2f} | Max Reachable V={best_v:6.2f} ({best_cfg}) | Gap = {gap:+6.2f}")
        
    print(f"MEAN: B4 = {sum(b4_vals)/len(b4_vals):.2f} | Max Reachable PB-CRL = {sum(max_pbcrl_vals)/len(max_pbcrl_vals):.2f} | Mean Gap = {sum(max_pbcrl_vals)/len(max_pbcrl_vals) - sum(b4_vals)/len(b4_vals):+.2f}")


if __name__ == "__main__":
    test_workload("burst_and_stream", 8)
    test_workload("poisson", 10)
    test_workload("burst", 8)
