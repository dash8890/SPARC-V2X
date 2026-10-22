import copy
import numpy as np
from numbers import Integral
from env.rewards import Rewards
from radio.packets import Packet
from controllers.service_belief import ServiceHistory

class MultiAgentEnv:
    """The central Multi-Agent RL environment for Mode-2 coded worker admission."""
    def __init__(self, trace_loader, config, coding_scheme, mac_engine, world):
        self.trace_loader = trace_loader
        self.config = config
        self.coding_scheme = coding_scheme
        self.mac = mac_engine
        self.world = world
        self.time_s = 0.0
        self.step_dt = config.get("step_length_s", 0.1)
        
        self.jobs = {}             # active jobs
        self.helpers = {}          # helper states (CPUQueue)
        self.event_ledger = []     # deterministic verification ledger
        self.pending_ul = []       # packets waiting for UL
        self.pending_dl = []       # packets waiting for DL
        self.in_compute = []       # tasks in helper queues
        self._active_helpers = set()
        self.job_records = {}
        self.job_archive = {}
        self.episode_seed = None
        self._event_time = 0.0
        self.service_history = ServiceHistory(window=config.get("belief_window", 32))

    def __deepcopy__(self, memo):
        """Copy transition state for Rollout without copying audit history."""
        clone = copy.copy(self)
        memo[id(self)] = clone
        clone.config = self.config.copy()
        clone.jobs = copy.deepcopy(self.jobs, memo)
        clone.helpers = copy.deepcopy(self.helpers, memo)
        clone.pending_ul = copy.deepcopy(self.pending_ul, memo)
        clone.pending_dl = copy.deepcopy(self.pending_dl, memo)
        clone.in_compute = copy.deepcopy(self.in_compute, memo)
        clone._active_helpers = copy.deepcopy(self._active_helpers, memo)
        clone.job_records = copy.deepcopy(self.job_records, memo)
        clone.job_archive = copy.deepcopy(self.job_archive, memo)
        clone.event_ledger = []
        clone.mac = copy.deepcopy(self.mac, memo)
        clone.service_history = copy.deepcopy(self.service_history, memo)
        clone.trace_loader = self.trace_loader
        clone.world = self.world
        return clone

    def counterfactual_snapshot(self, helper_ids):
        """Copy causal dynamic state while sharing the immutable helper universe."""
        clone = copy.copy(self)
        clone.config = self.config.copy()
        clone.jobs = copy.deepcopy(self.jobs)
        clone.pending_ul = copy.deepcopy(self.pending_ul)
        clone.pending_dl = copy.deepcopy(self.pending_dl)
        clone.in_compute = copy.deepcopy(self.in_compute)
        clone._active_helpers = set(self._active_helpers)
        clone.job_records = copy.deepcopy(self.job_records)
        clone.job_archive = copy.deepcopy(self.job_archive)
        clone.event_ledger = []
        clone.mac = copy.deepcopy(self.mac)
        clone.service_history = copy.deepcopy(self.service_history)
        # Helper metadata and queues outside the causal action footprint remain
        # shared; reachable helpers receive private queue/service state.
        ids = set(helper_ids)
        clone.helpers = {hid: (copy.deepcopy(h) if hid in ids else h)
                         for hid, h in self.helpers.items()}
        return clone
        
    def reset(self, seed=None):
        self.time_s = 0.0
        self._event_time = 0.0
        self.episode_seed = seed
        if self.mac is not None:
            self.mac.reset(seed)
        self.job_records.clear()
        self.job_archive.clear()
        self.jobs.clear()
        # Keep helpers dictionary but clear their queues
        for h in self.helpers.values():
            h.cpu_queue.queued_cycles = 0.0
        self.event_ledger.clear()
        self.pending_ul.clear()
        self.pending_dl.clear()
        self.in_compute.clear()
        self._active_helpers.clear()
        self.service_history.reset()
        return self._get_observations()

    def add_job(self, job):
        if job.uv_id in self.jobs or job.job_id in self.job_records:
            raise ValueError("Overlapping vehicle job or duplicate job ID")
        self._event_time = self.time_s
        self.jobs[job.uv_id] = {"job": job, "state": "new",
                                "k": self.coding_scheme.recovery_threshold(job)}
        self._register_job(job)

    def _register_job(self, job):
        if job.job_id in self.job_records:
            return
        self.job_archive[job.job_id] = job
        self.job_records[job.job_id] = {
            "job_id": job.job_id, "uv_id": job.uv_id, "seed": self.episode_seed,
            "generation_time": job.generation_time, "d_in": job.d_in,
            "cycles": job.cycles, "d_out": job.d_out, "deadline_ms": job.deadline_ms,
            "k": self.coding_scheme.recovery_threshold(job), "H_F": None,
            "n": None, "workers_dispatched": 0, "outcome": "pending",
            "success": 0, "t_rec": None,
        }
        self._record_event(job.job_id, -1, "generation")
        
    def _record_event(self, job_id, subtask_id, event_type, details=""):
        self.event_ledger.append({
            "time": self._event_time,
            "job_id": job_id,
            "subtask_id": subtask_id,
            "event": event_type,
            "details": details
        })
        
    def step(self, actions):
        """
        Executes one epoch of:
        1. Worker admission (actions) & Helper ranking
        2. UL Packet generation -> Mode-2 MAC simulation
        3. CPU queue evolution
        4. DL Packet generation -> Mode-2 MAC
        5. Deadline accounting
        """
        # Epoch-level admission events use the current epoch boundary.
        self._event_time = self.time_s
        # 1. Process new admissions
        for uv_id, action in actions.items():
            if uv_id in self.jobs and self.jobs[uv_id].get("state") == "new":
                job = self.jobs[uv_id]["job"]
                n_workers = action
                feasible_helpers = self.world.get_feasible_helpers(self.time_s, uv_id, self.helpers)
                k = self.coding_scheme.recovery_threshold(job)
                if action is None:
                    if len(feasible_helpers) >= k:
                        raise AssertionError("Infeasibility signal on a feasible job")
                    self._register_job(job)
                    self.job_records[job.job_id].update(
                        H_F=len(feasible_helpers), outcome="infeasible", success=0)
                    self.jobs[uv_id]["state"] = "infeasible"
                    self._record_event(job.job_id, -1, "infeasible", "H_F < k; S=0")
                    continue
                # Validate before dispatch: slicing otherwise silently clips n to H_F.
                # Raise explicitly so the invariant also holds under python -O.
                if (isinstance(n_workers, (bool, np.bool_))
                        or not isinstance(n_workers, Integral)
                        or not k <= n_workers <= len(feasible_helpers)):
                    raise AssertionError(
                        f"Invalid worker admission: job_id={job.job_id}, uv={uv_id}, "
                        f"time={self.time_s}, n={n_workers}, k={k}, "
                        f"H_F={len(feasible_helpers)}; require k <= n <= H_F. "
                        "H_F < k requires a separate fallback, not an edge action."
                    )
                # Sort feasible helpers by estimated return time (mocked here as index sort for brevity)
                assigned_helpers = feasible_helpers[:n_workers]
                self._register_job(job)
                self.job_records[job.job_id].update(
                    H_F=len(feasible_helpers), n=int(n_workers),
                    workers_dispatched=len(assigned_helpers), outcome="dispatched")
                
                self.jobs[uv_id]["state"] = "dispatched"
                self.jobs[uv_id]["n_t_minus_1"] = int(n_workers)
                self.jobs[uv_id]["returns"] = []
                self.jobs[uv_id]["k"] = k
                
                for idx, h_id in enumerate(assigned_helpers):
                    self._record_event(job.job_id, idx, "worker dispatch", f"Assigned to {h_id}")
                    # Create UL packet
                    pkt = Packet(
                        source=uv_id, destination=h_id,
                        job_id=job.job_id, subtask_id=idx, direction="UL",
                        payload_size=self.coding_scheme.uplink_payload(job),
                        generation_time=self.time_s, deadline=job.absolute_deadline
                    )
                    self.pending_ul.append(pkt)
                    self._record_event(job.job_id, idx, "UL scheduling")
                    
        # 2-4. Simulate MAC slots within this epoch
        slots = int(self.step_dt / 0.001)  # 1ms slots
        for slot in range(slots):
            self.mac.advance_slot()
            current_time = self.time_s + (slot * 0.001)
            self._event_time = current_time
            
            # Attempt UL transmissions
            for pkt in list(self.pending_ul):
                if current_time > pkt.deadline:
                    self.pending_ul.remove(pkt)
                    self._record_event(pkt.job_id, pkt.subtask_id, "UL failure", "Deadline exceeded")
                    continue
                if self.mac.attempt_transmission(pkt, pkt.source, pkt.destination, None, current_time):
                    self.pending_ul.remove(pkt)
                    self._record_event(pkt.job_id, pkt.subtask_id, "UL delivery")
                    self._record_event(pkt.job_id, pkt.subtask_id, "helper queue start")
                    
                    job = self._get_job_by_id(pkt.job_id)
                    cycles = self.coding_scheme.subtask_cycles(job)
                    self.helpers[pkt.destination].cpu_queue.add_job(cycles)
                    self._active_helpers.add(pkt.destination)
                    
                    self.in_compute.append({
                        "job_id": pkt.job_id, "subtask_id": pkt.subtask_id,
                        "helper": pkt.destination, "cycles_rem": cycles, "uv": pkt.source
                    })
                    
            # Advance CPU queues
            for h_id in list(self._active_helpers):
                h = self.helpers[h_id]
                if h.cpu_queue.queued_cycles > 0:
                    h.advance_time(0.001)
                if (h.cpu_queue.queued_cycles <= 0
                        and not any(comp["helper"] == h_id for comp in self.in_compute)):
                    self._active_helpers.discard(h_id)
                
            # Check compute completions
            for comp in list(self.in_compute):
                h = self.helpers[comp["helper"]]
                # Simplified tracking: we assume FIFO per helper. If helper queue <= what was there, it's done.
                # Actually, we subtract cycles explicitly here for tracking
                comp["cycles_rem"] -= h.available_cpu_rate() * 0.001
                if comp["cycles_rem"] <= 0:
                    self.in_compute.remove(comp)
                    self._record_event(comp["job_id"], comp["subtask_id"], "compute completion")
                    self._record_event(comp["job_id"], comp["subtask_id"], "DL scheduling")
                    
                    job = self._get_job_by_id(comp["job_id"])
                    dl_pkt = Packet(
                        source=comp["helper"], destination=comp["uv"],
                        job_id=job.job_id, subtask_id=comp["subtask_id"], direction="DL",
                        payload_size=self.coding_scheme.return_payload(job),
                        generation_time=current_time, deadline=job.absolute_deadline
                    )
                    self.pending_dl.append(dl_pkt)
                    
            # Attempt DL transmissions
            for pkt in list(self.pending_dl):
                if current_time > pkt.deadline:
                    self.pending_dl.remove(pkt)
                    continue
                if self.mac.attempt_transmission(pkt, pkt.source, pkt.destination, None, current_time):
                    self.pending_dl.remove(pkt)
                    self._record_event(pkt.job_id, pkt.subtask_id, "DL return")
                    # Causal sample: it becomes visible only after this return.
                    self.service_history.observe(
                        pkt.source,
                        max(0.0, current_time - self._get_job_by_id(pkt.job_id).generation_time),
                        success=True,
                    )
                    
                    # Record return
                    job_state = self.jobs.get(pkt.destination)
                    if (job_state and job_state["state"] == "dispatched"
                            and job_state["job"].job_id == pkt.job_id):
                        job_state["returns"].append(current_time)
                        if len(job_state["returns"]) == job_state["k"]:
                            job_state["state"] = "recovered"
                            job_state["t_rec"] = current_time
                            self.job_records[pkt.job_id].update(
                                outcome="recovered", success=1, t_rec=current_time)
                            self._record_event(pkt.job_id, -1, "recovery/deadline", f"Recovered at {current_time}")

        # Epoch-boundary events use the epoch time, not the last radio slot.
        self._event_time = self.time_s
        # 5. Deadline accounting & Rewards
        self.time_s = round(self.time_s + self.step_dt, 10)
        rewards = {uv: 0.0 for uv in actions.keys()}
        dones = {uv: False for uv in actions.keys()}
        infos = {}
        
        for uv, state in list(self.jobs.items()):
            if state["state"] == "recovered":
                rewards[uv] = Rewards.calculate_job_reward(True, state["t_rec"] - state["job"].generation_time, state["job"].deadline_ms)
                del self.jobs[uv]
            elif (state["state"] == "infeasible"
                  or self.time_s + 1e-10 >= state["job"].absolute_deadline):
                rewards[uv] = Rewards.calculate_job_reward(False, float('inf'), state["job"].deadline_ms)
                if state["state"] != "infeasible":
                    self.job_records[state["job"].job_id].update(outcome="deadline_missed", success=0)
                    self._record_event(state["job"].job_id, -1, "recovery/deadline", "Deadline missed")
                del self.jobs[uv]
        
        return self._get_observations(), rewards, dones, infos
        
    def project_b_beliefs(self, uv_id, job):
        """Return causal per-helper service beliefs before the next action."""
        beliefs = {}
        feasible = self.world.get_feasible_helpers(self.time_s, uv_id, self.helpers)
        for hid in feasible:
            helper = self.helpers[hid]
            compute_s = self.coding_scheme.subtask_cycles(job) / max(helper.available_cpu_rate(), 1e-9)
            radio_budget = max(0.0, job.deadline_ms / 1000.0 - compute_s)
            beliefs[hid] = self.service_history.belief(hid, radio_budget)
        return beliefs

    def project_b_observation(self, uv_id, job):
        """Compact task/queue/belief observation for Project-B controllers."""
        import numpy as np
        beliefs = self.project_b_beliefs(uv_id, job)
        rows = []
        for hid, b in beliefs.items():
            helper = self.helpers[hid]
            rows.append([
                b["p_success"], b["q50"], b["q90"], b["q95"],
                b["uncertainty"], helper.cpu_queue.get_remaining_cycles() / 5e8,
                helper.available_cpu_rate() / 1e10,
            ])
        rows.sort(reverse=True)
        flat = [job.d_in / .8, job.cycles / 5e8, job.deadline_ms / 100.0,
                self.mac.get_cbr_history_mean() if self.mac else 0.0]
        flat.extend(v for row in rows[:8] for v in row)
        while len(flat) < 4 + 8 * 7:
            flat.append(0.0)
        return np.asarray(flat[:4 + 8 * 7], dtype=np.float32)

    def _get_job_by_id(self, job_id):
        for uv, data in self.jobs.items():
            if data["job"].job_id == job_id:
                return data["job"]
        return self.job_archive.get(job_id)
        
    def _get_observations(self):
        """Builds true decentralized observations for each active UV."""
        obs = {}
        for uv_id, data in self.jobs.items():
            job = data["job"]
            
            # Dimensionless, fixed-reference features. These scales are part of
            # the interface contract and do not depend on evaluation data.
            job_features = [job.d_in / 0.8, job.cycles / 5e8,
                            job.d_out / 64.0, job.deadline_ms / 100.0]
            
            # 2. Feasible helper summaries (Cap at 5 for fixed tensor dimension)
            feasible = self.world.get_feasible_helpers(self.time_s, uv_id, self.helpers)
            helper_summaries = []
            for h_id in feasible[:5]:
                h = self.helpers[h_id]
                helper_summaries.extend([h.available_cpu_rate() / 1e10,
                                         h.cpu_queue.get_remaining_cycles() / 5e8])
                
            # Pad missing helpers with zeros
            while len(helper_summaries) < 10:
                helper_summaries.extend([0.0, 0.0])
                
            # 3. CBR history
            cbr = self.mac.get_cbr_history_mean() if self.mac is not None else 0.0
            
            # 4. Previous action
            prev_a = data.get("n_t_minus_1", 0) / max(1, len(self.helpers))
            
            obs_vector = np.array(job_features + helper_summaries + [cbr, prev_a], dtype=np.float32)
            obs[uv_id] = obs_vector
            
        return obs
