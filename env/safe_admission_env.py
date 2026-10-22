"""Safety/computation coexistence environment over the live Mode-2 MAC."""
from __future__ import annotations
from collections import deque
from radio.packets import Packet

class SafetyAdmissionEnv:
    def __init__(self, mac, vehicles, horizon=100, step_ms=10, safety_per_step=2, comp_payload_mb=.05):
        self.mac=mac; self.vehicles=list(vehicles); self.horizon=horizon; self.step_ms=step_ms
        self.safety_per_step=safety_per_step; self.comp_payload_mb=comp_payload_mb; self.reset()
    def reset(self, seed=None):
        self.mac.reset(seed); self.t=0; self.pending=[]; self.safety_ok=deque(maxlen=20); self.safety_delay=deque(maxlen=2000); self.comp_done=0; self.safety_sent=0; self.safety_successes=0; self.safety_fail=0; self.safety_generated=0; self.comp_admitted=0; return self.obs()
    def obs(self):
        cbr=self.mac.get_cbr_history_mean(); ok=sum(self.safety_ok)/len(self.safety_ok) if self.safety_ok else 1.0; delay=sum(self.safety_delay)/len(self.safety_delay) if self.safety_delay else 0.;
        return [cbr,self.safety_per_step/10.,ok,delay/100.,len([p for p in self.pending if p.traffic_class=='computation'])/20.,self.comp_admitted/20.,self.safety_fail/max(1,self.safety_sent),self.t/self.horizon]
    def step(self, action):
        action=int(action); self.comp_admitted+=action
        now=self.t*self.step_ms/1000.;
        for i in range(self.safety_per_step):
            src=self.vehicles[(self.t*self.safety_per_step+i)%len(self.vehicles)]; dst=self.vehicles[(self.t*self.safety_per_step+i+1)%len(self.vehicles)]
            self.pending.append(Packet(src,dst,-100000-self.t*10-i,0,'UL',.002,now,now+.1,traffic_class='safety'))
            self.safety_generated += 1
        for i in range(action):
            src=self.vehicles[(self.t+i)%len(self.vehicles)]; dst=self.vehicles[(self.t+i+2)%len(self.vehicles)]
            self.pending.append(Packet(src,dst,100000+self.t*10+i,0,'UL',self.comp_payload_mb,now,now+.2,traffic_class='computation'))
        before_done=self.comp_done
        for _ in range(max(1,self.step_ms)):
            self.mac.advance_slot(); current=now+_/1000.;
            for p in list(self.pending):
                if current>p.deadline:
                    self.pending.remove(p)
                    if p.traffic_class=='safety': self.safety_fail+=1; self.safety_ok.append(0)
                    continue
                if self.mac.attempt_transmission(p,p.source,p.destination,None,current):
                    self.pending.remove(p)
                    if p.traffic_class=='safety': self.safety_sent+=1; self.safety_successes+=1; self.safety_ok.append(1); self.safety_delay.append((current-p.generation_time)*1000); 
                    else: self.comp_done+=1
        self.t+=1
        violation=1.0 if (self.safety_ok and self.safety_ok[-1]==0) else 0.0
        utility=(self.comp_done-before_done)/10.0
        reward=utility-2.0*violation
        done=self.t>=self.horizon
        return self.obs(),reward,done,{'safety_violation':violation,'cbr':self.mac.get_cbr_history_mean(),'comp_done':self.comp_done,'safety_prr':self.safety_successes/max(1,self.safety_generated),'safety_p95_ms':float(sorted(self.safety_delay)[max(0,int(.95*len(self.safety_delay))-1)]) if self.safety_delay else 0.0}
