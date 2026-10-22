import copy
import numpy as np
import math
from env.rewards import Rewards

class RolloutController:
    """Predictive receding-horizon / rollout model-based controller."""
    def __init__(self, horizon: int = 3):
        self.horizon = horizon

    def _reward_upper_bound(self, env, uv_id):
        """Optimistic bound for the existing slot simulator, not a new policy.

        Ignore UL loss, queues and DL contention. Even then k computations must
        finish. No later candidate can beat that recovery time. This only prunes
        once a candidate actually attains the bound, preserving first-best ties.
        """
        state = env.jobs[uv_id]
        job = state["job"]
        k = env.coding_scheme.recovery_threshold(job)
        cycles = env.coding_scheme.subtask_cycles(job)
        helpers = env.world.get_feasible_helpers(env.time_s, uv_id, env.helpers)
        earliest_slots = []
        for hid in helpers:
            rate = env.helpers[hid].available_cpu_rate()
            if rate > 0:
                # Round downward at integer boundaries to remain optimistic.
                steps = math.ceil(math.nextafter(cycles / (rate * 0.001), -math.inf))
                earliest_slots.append(max(0, steps - 1))
        if len(earliest_slots) < k:
            return float("inf")
        earliest = env.time_s + sorted(earliest_slots)[k - 1] * 0.001
        return Rewards.calculate_job_reward(True, earliest - job.generation_time, job.deadline_ms)
        
    def select_action(self, k: int, feasible_helpers: int, current_env, uv_id: str) -> int | None:
        """
        Executes a forward tree search (rollout) over physically feasible actions
        to find the n that maximizes expected multi-agent reward.
        """
        if feasible_helpers < k:
            return None
            
        best_action = k
        best_reward = -float('inf')
        reward_bound = self._reward_upper_bound(current_env, uv_id)
        
        # Exact search is retained for small action spaces. For very large
        # helper sets, use a deterministic grid that preserves both bounds and
        # resolution near k while keeping the rollout tractable under the
        # endogenous shared-radio model.
        if feasible_helpers - k <= 64:
            candidates = list(range(k, feasible_helpers + 1))
        else:
            stride = (feasible_helpers - k) / 15
            candidates = sorted(set([k, feasible_helpers] +
                                    [round(k + i * stride) for i in range(16)]))

        for n in candidates:
            # In a true deployment, we deepcopy a lightweight forward model.
            # Here we simulate the trajectory using the true transition function.
            try:
                sim_env = copy.deepcopy(current_env)
            except TypeError:
                # Fallback if uncopyable (e.g., sockets/files in real envs)
                return k 
                
            sim_time = 0.0
            cumulative_reward = 0.0
            
            # Step the simulation forward over the receding horizon
            actions = {uv_id: n}
            for step in range(self.horizon):
                _, rewards, _, _ = sim_env.step(actions)
                cumulative_reward += rewards.get(uv_id, 0.0)
                if uv_id not in sim_env.jobs:
                    break  # This search injects no new jobs; no further UV reward exists.
                # Next steps assume baseline continuation (e.g. n=k)
                actions = {uv_id: k}
                
            if cumulative_reward > best_reward:
                best_reward = cumulative_reward
                best_action = n
            if best_reward >= reward_bound:
                break
                
        return best_action
