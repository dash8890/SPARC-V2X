"""Calibrated Stochastic Streaming Workload Implementation & Validation.

Evaluates reachable value gap (PB-CRL vs B4) across the 4 validation seeds
[73300, 73301, 73302, 73303] at Density 15 under a rigorous stochastic streaming model.
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


class StreamingTaskLayer:
    """TaskLayer with calibrated Poisson streaming arrivals and persistent queues."""

    def __init__(
        self,
        density: int,
        seed: int,
        tasks_per_source: int = 8,
        mean_interval_epochs: float = 10.0,
        epoch_interval_ms: float = 10.0,
        controller_epochs: int = 100,
        deadline_multiplier: float = 1.0,
        helper_cpu_lo: float = 2e8,
        helper_cpu_hi: float = 1e9,
        cluster_share: float | None = None,
    ):
        self.rng = random.Random(seed)
        self.epoch_interval_ms = float(epoch_interval_ms)
        self.controller_epochs = int(controller_epochs)
        self.current_epoch = 0

        # Helper pool (helpers = min(12, density))
        lo, hi = float(helper_cpu_lo), float(helper_cpu_hi)
        if hi < lo:
            lo, hi = hi, lo
        n_helpers = min(12, density)
        self.helpers = []
        for i in range(n_helpers):
            self.helpers.append({
                "helper_id": i,
                "cpu_cycles_s": self.rng.uniform(lo, hi),
                "ready_time_ms": 0.0,
                "queue_cycles": 0.0,
            })

        # Sources
        if cluster_share is not None:
            n_sources = max(1, min(density, int(round(density * float(cluster_share)))))
        else:
            n_sources = max(1, density // 5)
        sources = list(range(n_sources))

        # Calibrated stochastic streaming arrivals
        self.tasks = []
        deadline_multiplier = max(0.05, float(deadline_multiplier))
        tid = 0
        for s in sources:
            cur_ep = self.rng.randint(0, 3)  # initial release epoch near start
            for k in range(tasks_per_source):
                rel_ep = min(self.controller_epochs - 1, cur_ep)
                b = self.rng.randint(10_000, 80_000)
                cyc = self.rng.uniform(1e7, 1e8)
                dl = self.rng.uniform(30.0, 150.0) * deadline_multiplier
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
                # Sample next arrival via exponential inter-arrival time
                inter = max(2, int(round(self.rng.expovariate(1.0 / mean_interval_epochs))))
                cur_ep += inter

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        now_ms = self.current_epoch * self.epoch_interval_ms

        # Finalize completed tasks
        for t in self.tasks:
            if t["queued"] and not t["done"]:
                if t["finish_ms"] is not None and t["finish_ms"] <= now_ms:
                    if t["finish_ms"] <= t["absolute_deadline_ms"]:
                        t["done"] = True
                    t["queued"] = False

        # Drain helper queues
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


def evaluate_population(density: int = 15, tasks_per_source: int = 8, mean_interval: float = 10.0):
    seeds = [73300, 73301, 73302, 73303]
    b_max = 175000

    print("=" * 85)
    print(f"REACHABLE VALUE ANALYSIS ON CALIBRATED STREAMING WORKLOAD (DENSITY {density})")
    print(f"Tasks per Source: {tasks_per_source} | Mean Arrival Interval: {mean_interval} epochs ({mean_interval*10:.0f} ms)")
    print("=" * 85)

    b4_results = []
    best_pbcrl_results = []
    gaps = []

    lam_grid = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    beta_grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    for s in seeds:
        t_seed = 186100 + (s - 73300)

        # 1. B4 baseline execution
        l_b4 = StreamingTaskLayer(density, t_seed, tasks_per_source=tasks_per_source, mean_interval_epochs=mean_interval)
        for ep in range(100):
            l_b4.set_epoch(ep)
            cands = l_b4.candidates()
            if cands:
                pack, _ = build_baseline_pack(cands, {"cbr": 0.04, "resel_t": 0.0}, mode="ce_milp", b_max=b_max, unique_sources=True)
                l_b4.apply_assignments(pack)
        l_b4.set_epoch(100)
        v_b4 = l_b4.completed_value()
        b4_results.append(v_b4)

        # 2. Exhaustive PB-CRL grid search over (lam, beta)
        best_v = 0.0
        best_cfg = None
        for b in beta_grid:
            for l in lam_grid:
                l_pb = StreamingTaskLayer(density, t_seed, tasks_per_source=tasks_per_source, mean_interval_epochs=mean_interval)
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

        best_pbcrl_results.append(best_v)
        gap = best_v - v_b4
        gaps.append(gap)
        pct = (gap / v_b4 * 100.0) if v_b4 > 0 else 0.0
        print(f"Seed {s}: B4 V_done = {v_b4:6.2f} | Max PB-CRL V = {best_v:6.2f} (action: lam={best_cfg[0]:4.1f}, beta={best_cfg[1]:4.2f}) | Gap = {gap:+6.2f} ({pct:+5.1f}%)")

    mean_b4 = float(np.mean(b4_results))
    mean_pb = float(np.mean(best_pbcrl_results))
    mean_gap = float(np.mean(gaps))
    mean_pct = (mean_gap / mean_b4 * 100.0) if mean_b4 > 0 else 0.0

    print("-" * 85)
    print(f"POPULATION MEAN: B4 = {mean_b4:6.2f} | Max Reachable PB-CRL = {mean_pb:6.2f} | Reachable Advantage = {mean_gap:+6.2f} ({mean_pct:+5.2f}%)")
    print("=" * 85)
    return mean_gap


if __name__ == "__main__":
    evaluate_population(density=15, tasks_per_source=8, mean_interval=10.0)
    evaluate_population(density=15, tasks_per_source=10, mean_interval=8.0)
