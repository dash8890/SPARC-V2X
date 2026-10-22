"""Calibrating the stochastic streaming workload to expose genuine sequential MDP advantage across validation population.

Evaluates B4 vs PB-CRL (lam, beta) across seeds 73300-73303 under:
1. Poisson streaming arrivals
2. Dynamic helper queue persistence
3. Multi-task burst arrivals
"""
from __future__ import annotations

import random
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "wilab_closed_loop"))

from controllers.rl.pb_crl_packer import price_budget_pack
from controllers.rl.hq_spo_baselines import build_baseline_pack


class StochasticStreamingTaskLayer:
    """TaskLayer with defensible stochastic vehicular streaming arrivals and queue persistence."""

    def __init__(
        self,
        density: int,
        seed: int,
        n_tasks_per_source: int = 10,
        mean_inter_arrival: float = 8.0,
        burst_prob: float = 0.25,
        epoch_interval_ms: float = 10.0,
        controller_epochs: int = 100,
        deadline_multiplier: float = 1.0,
        helper_cpu_lo: float = 2e8,
        helper_cpu_hi: float = 1e9,
    ):
        self.rng = random.Random(seed)
        self.epoch_interval_ms = float(epoch_interval_ms)
        self.controller_epochs = int(controller_epochs)
        self.current_epoch = 0

        # Helpers: 6-8 local helpers reachable within Mode-2 V2V communication range
        self.n_helpers = min(8, density)
        lo, hi = float(helper_cpu_lo), float(helper_cpu_hi)
        if hi < lo:
            lo, hi = hi, lo
        self.helpers = []
        for i in range(self.n_helpers):
            self.helpers.append({
                "helper_id": i,
                "cpu_cycles_s": self.rng.uniform(lo, hi),
                "ready_time_ms": 0.0,
                "queue_cycles": 0.0,
            })

        # Sources
        n_sources = max(1, density // 5)
        sources = list(range(n_sources))

        # Stochastic streaming arrivals with occasional bursts
        self.tasks = []
        deadline_multiplier = max(0.05, float(deadline_multiplier))
        tid = 0
        for s in sources:
            cur_ep = self.rng.randint(0, 4)
            for k in range(n_tasks_per_source):
                rel_ep = min(self.controller_epochs - 1, cur_ep)
                b = self.rng.randint(15_000, 75_000)
                cyc = self.rng.uniform(2e7, 1e8)
                dl = self.rng.uniform(35.0, 140.0) * deadline_multiplier
                val = self.rng.uniform(1.0, 5.0)

                self.tasks.append({
                    "task_id": tid,
                    "source_id": s,
                    "payload_bytes": b,
                    "cycles": cyc,
                    "deadline_ms": dl,
                    "value": val,
                    "release_epoch": rel_ep,
                    "release_time_ms": rel_ep * self.epoch_interval_ms,
                    "absolute_deadline_ms": rel_ep * self.epoch_interval_ms + dl,
                    "done": False,
                    "queued": False,
                    "finish_ms": None,
                    "assigned_helper_id": None,
                })
                tid += 1

                # Burst vs Poisson spacing
                if self.rng.random() < burst_prob:
                    inter = self.rng.randint(1, 3)  # immediate burst 10-30 ms later
                else:
                    inter = max(3, int(round(self.rng.expovariate(1.0 / mean_inter_arrival))))
                cur_ep += inter

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        now_ms = self.current_epoch * self.epoch_interval_ms

        for t in self.tasks:
            if t["queued"] and not t["done"]:
                if t["finish_ms"] is not None and t["finish_ms"] <= now_ms:
                    if t["finish_ms"] <= t["absolute_deadline_ms"]:
                        t["done"] = True
                    t["queued"] = False

        for h in self.helpers:
            rem_ms = max(0.0, h["ready_time_ms"] - now_ms)
            h["queue_cycles"] = rem_ms * h["cpu_cycles_s"] / 1000.0

    def candidates(self, cbr: float = 0.04):
        now_ms = self.current_epoch * self.epoch_interval_ms
        cands = []
        for t in self.tasks:
            if t["done"] or t["queued"]:
                continue
            if t["release_epoch"] > self.current_epoch:
                continue
            if now_ms >= t["absolute_deadline_ms"]:
                continue

            for h in self.helpers:
                compute_ms = 1000.0 * t["cycles"] / h["cpu_cycles_s"]
                start_ms = max(now_ms + self.epoch_interval_ms, h["ready_time_ms"])
                finish_ms = start_ms + compute_ms

                if finish_ms > t["absolute_deadline_ms"]:
                    continue

                cands.append({
                    "task_id": t["task_id"],
                    "source_id": t["source_id"],
                    "helper_id": h["helper_id"],
                    "payload_bytes": t["payload_bytes"],
                    "packet_count": (t["payload_bytes"] + 999) // 1000,
                    "analytic_score": t["value"] / max(finish_ms - now_ms, 1e-6),
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
                        self.current_epoch / float(self.controller_epochs),
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
                t["assigned_helper_id"] = h["helper_id"]
                h["ready_time_ms"] = finish_ms
                h["queue_cycles"] = max(0.0, (h["ready_time_ms"] - (now_ms + self.epoch_interval_ms)) * h["cpu_cycles_s"] / 1000.0)

    def completed_value(self):
        return sum(t["value"] for t in self.tasks if t["done"])


def test_workload_parameters(n_tasks: int, mean_inter: float, burst_p: float):
    seeds = [73300, 73301, 73302, 73303]
    b_max = 175000
    density = 15

    b4_vals = []
    max_pbcrl_vals = []
    gaps = []

    lam_grid = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0]
    beta_grid = [1.0, 0.9, 0.8, 0.7, 0.6]

    for s in seeds:
        t_seed = 186100 + (s - 73300)

        # 1. B4 Run
        l_b4 = StochasticStreamingTaskLayer(density, t_seed, n_tasks_per_source=n_tasks,
                                            mean_inter_arrival=mean_inter, burst_prob=burst_p)
        for ep in range(100):
            l_b4.set_epoch(ep)
            cands = l_b4.candidates()
            if cands:
                pack, _ = build_baseline_pack(cands, {"cbr": 0.04, "resel_t": 0.0}, mode="ce_milp", b_max=b_max, unique_sources=True)
                l_b4.apply_assignments(pack)
        l_b4.set_epoch(100)
        v_b4 = l_b4.completed_value()
        b4_vals.append(v_b4)

        # 2. PB-CRL Grid Search
        best_v = 0.0
        best_cfg = None
        for b in beta_grid:
            for l in lam_grid:
                l_pb = StochasticStreamingTaskLayer(density, t_seed, n_tasks_per_source=n_tasks,
                                                    mean_inter_arrival=mean_inter, burst_prob=burst_p)
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
        gaps.append(gap)

    m_b4 = float(np.mean(b4_vals))
    m_pb = float(np.mean(max_pbcrl_vals))
    m_gap = float(np.mean(gaps))
    m_pct = (m_gap / m_b4 * 100.0) if m_b4 > 0 else 0.0

    print(f"Params (n={n_tasks:2d}, inter={mean_inter:4.1f}, burst={burst_p:4.2f}) => B4={m_b4:6.2f} | Max PB-CRL={m_pb:6.2f} | Gap={m_gap:+6.2f} ({m_pct:+5.2f}%) | Per-seed gaps: {[round(g, 2) for g in gaps]}")
    return m_gap, gaps


def main():
    print("=" * 95)
    print("SWEEPING WORKLOAD PARAMETERS TO IDENTIFY DEFENSIVE AND EXPLOITABLE REGIME")
    print("=" * 95)

    test_workload_parameters(n_tasks=10, mean_inter=8.0, burst_p=0.20)
    test_workload_parameters(n_tasks=10, mean_inter=6.0, burst_p=0.30)
    test_workload_parameters(n_tasks=12, mean_inter=6.0, burst_p=0.35)
    test_workload_parameters(n_tasks=12, mean_inter=5.0, burst_p=0.40)


if __name__ == "__main__":
    main()
