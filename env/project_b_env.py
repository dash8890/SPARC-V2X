"""Native local/single-helper environment for the rebased Project-B paper."""
import numpy as np
from numbers import Integral
from env.multiagent_env import MultiAgentEnv
from env.rewards import Rewards
from radio.packets import Packet

class ProjectBEnv(MultiAgentEnv):
    """Project-B action semantics: ``local`` or one explicit helper ID.

    The legacy MultiAgentEnv remains unchanged. This adapter owns a separate
    transition path and therefore cannot silently reinterpret an action as a
    coded worker count.
    """
    def reset(self, seed=None):
        out = super().reset(seed)
        return self._get_observations()

    def add_job(self, job):
        if job.uv_id in self.jobs or job.job_id in self.job_records:
            raise ValueError("Overlapping vehicle job or duplicate job ID")
        self._event_time = self.time_s
        self.jobs[job.uv_id] = {"job": job, "state": "new", "returns": []}
        self._register_job(job)

    def feasible_actions(self, uv_id):
        helpers = list(self.world.get_feasible_helpers(self.time_s, uv_id, self.helpers))
        return ["local"] + helpers + [(h, "DS") for h in helpers] + [(h, "SPS") for h in helpers]

    def _record_terminal(self, uv, success, t_rec=None, outcome=None):
        st = self.jobs[uv]; job = st["job"]
        if st.get("terminal"):
            return
        st["terminal"] = True
        self.job_records[job.job_id].update(
            outcome=outcome or ("recovered" if success else "deadline_missed"),
            success=int(success), t_rec=t_rec)

    def _project_b_observation(self, uv_id, job):
        # Same causal interface as MultiAgentEnv, but whole-task CPU service.
        import numpy as np
        beliefs = {}
        for hid in self.world.get_feasible_helpers(self.time_s, uv_id, self.helpers):
            helper = self.helpers[hid]
            compute_s = job.cycles / max(helper.available_cpu_rate(), 1e-9)
            budget = max(0.0, job.deadline_ms / 1000.0 - compute_s)
            beliefs[hid] = self.service_history.belief(hid, budget)
        rows=[]
        for hid,b in beliefs.items():
            h=self.helpers[hid]
            rows.append([b["p_success"],b["q50"],b["q90"],b["q95"],b["uncertainty"],h.cpu_queue.get_remaining_cycles()/5e8,h.available_cpu_rate()/1e10])
        rows.sort(reverse=True)
        flat=[job.d_in/.8,job.cycles/5e8,job.deadline_ms/100.,self.mac.get_cbr_history_mean() if self.mac else 0.]
        flat.extend(v for row in rows[:8] for v in row)
        flat += [0.] * (4+8*7-len(flat))
        return np.asarray(flat[:4+8*7],dtype=np.float32)

    def _get_observations(self):
        return {uv:self._project_b_observation(uv,st["job"])
                for uv,st in self.jobs.items()}

    def step(self, actions):
        self._event_time=self.time_s
        # Admission is explicit and validated before any state mutation.
        for uv, action in actions.items():
            if uv not in self.jobs or self.jobs[uv]["state"] != "new":
                continue
            st=self.jobs[uv]; job=st["job"]; feasible=self.world.get_feasible_helpers(self.time_s,uv,self.helpers)
            if action == "local":
                st.update(state="local", local_cycles=float(job.cycles), action="local")
                self.job_records[job.job_id].update(H_F=len(feasible), outcome="dispatched", action="local")
                self._record_event(job.job_id,-1,"local dispatch")
            elif isinstance(action, str) and action in feasible:
                helper, mode = action, "DS"
                st.update(state="offloaded", helper=helper, action=(helper, mode), scheduling_mode=mode)
                self.job_records[job.job_id].update(H_F=len(feasible), outcome="dispatched", action=(helper, mode), n=1, workers_dispatched=1)
                pkt=Packet(source=uv,destination=helper,job_id=job.job_id,subtask_id=0,direction="UL",payload_size=self.coding_scheme.uplink_payload(job),generation_time=self.time_s,deadline=job.absolute_deadline,scheduling_mode=mode,flow_id=uv)
                self.pending_ul.append(pkt); self._record_event(job.job_id,0,"worker dispatch",f"{helper}/{mode}"); self._record_event(job.job_id,0,"UL scheduling")
            elif isinstance(action, (tuple, list)) and len(action)==2 and action[0] in feasible and action[1] in {"DS", "SPS"}:
                helper, mode = action[0], action[1]
                st.update(state="offloaded", helper=helper, action=(helper, mode), scheduling_mode=mode)
                self.job_records[job.job_id].update(H_F=len(feasible), outcome="dispatched", action=(helper, mode), n=1, workers_dispatched=1)
                pkt=Packet(source=uv,destination=helper,job_id=job.job_id,subtask_id=0,direction="UL",payload_size=self.coding_scheme.uplink_payload(job),generation_time=self.time_s,deadline=job.absolute_deadline,scheduling_mode=mode,flow_id=uv)
                self.pending_ul.append(pkt); self._record_event(job.job_id,0,"worker dispatch",f"{helper}/{mode}"); self._record_event(job.job_id,0,"UL scheduling")
            else:
                raise AssertionError(f"Invalid or masked Project-B action {action!r}; feasible={self.feasible_actions(uv)}")

        slots=int(self.step_dt/.001)
        for slot in range(slots):
            self.mac.advance_slot(); current=self.time_s+slot*.001; self._event_time=current
            for pkt in list(self.pending_ul):
                if current>pkt.deadline:
                    self.pending_ul.remove(pkt); self._record_event(pkt.job_id,pkt.subtask_id,"UL failure","deadline"); continue
                if self.mac.attempt_transmission(pkt,pkt.source,pkt.destination,None,current):
                    self.pending_ul.remove(pkt); self._record_event(pkt.job_id,pkt.subtask_id,"UL delivery")
                    job=self._get_job_by_id(pkt.job_id); self._record_event(pkt.job_id,0,"helper queue start")
                    self.helpers[pkt.destination].cpu_queue.add_job(job.cycles); self._active_helpers.add(pkt.destination)
                    self.in_compute.append({"job_id":pkt.job_id,"subtask_id":0,"helper":pkt.destination,"cycles_rem":float(job.cycles),"uv":pkt.source})
            for hid in list(self._active_helpers):
                h=self.helpers[hid]
                if h.cpu_queue.queued_cycles>0: h.advance_time(.001)
                if h.cpu_queue.queued_cycles<=0 and not any(x["helper"]==hid for x in self.in_compute): self._active_helpers.discard(hid)
            for comp in list(self.in_compute):
                h=self.helpers[comp["helper"]]; comp["cycles_rem"]-=h.available_cpu_rate()*.001
                if comp["cycles_rem"]<=0:
                    self.in_compute.remove(comp); job=self._get_job_by_id(comp["job_id"]); self._record_event(job.job_id,0,"compute completion")
                    self.pending_dl.append(Packet(source=comp["helper"],destination=comp["uv"],job_id=job.job_id,subtask_id=0,direction="DL",payload_size=self.coding_scheme.return_payload(job),generation_time=current,deadline=job.absolute_deadline,scheduling_mode=self.jobs.get(comp["uv"],{}).get("scheduling_mode","DS"),flow_id=comp["uv"]))
            for pkt in list(self.pending_dl):
                if current>pkt.deadline: self.pending_dl.remove(pkt); continue
                if self.mac.attempt_transmission(pkt,pkt.source,pkt.destination,None,current):
                    self.pending_dl.remove(pkt); self._record_event(pkt.job_id,0,"DL return"); job=self._get_job_by_id(pkt.job_id)
                    self.service_history.observe(pkt.source,max(0.,current-job.generation_time),True)
                    st=self.jobs.get(pkt.destination)
                    if st and st["job"].job_id==pkt.job_id: st["returns"].append(current); self._record_terminal(pkt.destination,True,current,"recovered")
            for uv,st in list(self.jobs.items()):
                if st["state"]=="local" and st["local_cycles"]>0:
                    st["local_cycles"]-=self.config.get("local_cpu_rate",10e9)*.001
                    if st["local_cycles"]<=0: self._record_terminal(uv,True,self.time_s,"recovered")
        self._event_time=self.time_s; self.time_s=round(self.time_s+self.step_dt,10)
        rewards={uv:0. for uv in actions}; dones={uv:False for uv in actions}; infos={}
        for uv,st in list(self.jobs.items()):
            job=st["job"]
            if st.get("terminal"):
                rec=self.job_records[job.job_id]; t=(rec["t_rec"]-job.generation_time) if rec["t_rec"] is not None else float("inf")
                rewards[uv]=Rewards.calculate_job_reward(bool(rec["success"]),t,job.deadline_ms); dones[uv]=True; del self.jobs[uv]
            elif self.time_s+1e-10>=job.absolute_deadline:
                self._record_terminal(uv,False,None,"deadline_missed"); rewards[uv]=Rewards.calculate_job_reward(False,float("inf"),job.deadline_ms); dones[uv]=True; del self.jobs[uv]
        return self._get_observations(),rewards,dones,infos
