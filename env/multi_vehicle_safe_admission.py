from collections import deque
from radio.packets import Packet

class MultiVehicleSafeAdmissionEnv:
    """Shared-sidelink multi-vehicle safety/computation admission environment."""
    def __init__(self, mac, density_level, horizon=50, comp_fraction=.4):
        self.mac=mac; self.n_active=int(density_level); self.horizon=horizon; self.comp_fraction=comp_fraction; self.vehicles=[str(i) for i in range(self.n_active)]; self.comp_sources=self.vehicles[:max(1,int(self.n_active*comp_fraction))]; self.reset()
    def reset(self,seed=None):
        self.mac.reset(seed); self.t=0; self.pending=[]; self.safe_ok=deque(maxlen=2000); self.safe_delay=deque(maxlen=2000); self.safe_gen=0; self.safe_success=0; self.safe_fail=0; self.comp_done=0; self.comp_admitted=0; return self.observations()
    def observations(self):
        cbr=self.mac.get_cbr_history_mean(); prr=self.safe_success/max(1,self.safe_gen); p95=sorted(self.safe_delay)[max(0,int(.95*len(self.safe_delay))-1)] if self.safe_delay else 0.; base=[cbr,self.n_active/100.,prr,p95/100.,len(self.pending)/max(1,self.n_active*4),0.0,0.0,self.t/self.horizon]
        return {v:base[:] for v in self.comp_sources}
    def step(self,actions):
        now=self.t*.01
        for i,v in enumerate(self.vehicles):
            dst=self.vehicles[(i+1)%self.n_active]; self.pending.append(Packet(v,dst,-100000-self.t*100-i,0,'UL',.002,now,now+.1,traffic_class='safety')); self.safe_gen+=1
        for i,v in enumerate(self.comp_sources):
            for j in range(int(actions.get(v,0))):
                dst=self.vehicles[(i+j+2)%self.n_active]; self.pending.append(Packet(v,dst,100000+self.t*100+i*10+j,0,'UL',.05,now,now+.2,traffic_class='computation')); self.comp_admitted+=1
        before=self.comp_done
        for slot in range(10):
            self.mac.advance_slot(); cur=now+slot*.001
            for p in list(self.pending):
                if cur>p.deadline:
                    self.pending.remove(p)
                    if p.traffic_class=='safety': self.safe_fail+=1; self.safe_ok.append(0)
                    continue
                if self.mac.attempt_transmission(p,p.source,p.destination,None,cur):
                    self.pending.remove(p)
                    if p.traffic_class=='safety': self.safe_success+=1; self.safe_ok.append(1); self.safe_delay.append((cur-p.generation_time)*1000)
                    else: self.comp_done+=1
        self.t+=1; prr=self.safe_success/max(1,self.safe_gen); p95=sorted(self.safe_delay)[max(0,int(.95*len(self.safe_delay))-1)] if self.safe_delay else 0.; violation=max(0.,.95-prr)+max(0.,p95-50.)/50.; reward=(self.comp_done-before)/10.-2.*violation; done=self.t>=self.horizon
        info={'safety_prr':prr,'safety_p95_ms':p95,'safety_violation':violation,'comp_done':self.comp_done,'cbr':self.mac.get_cbr_history_mean()}
        return self.observations(),reward,done,info
