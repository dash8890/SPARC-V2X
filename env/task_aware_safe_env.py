import random, math
from radio.packets import Packet
from controllers.rl.safe_admission import conservative_safe_mask

class TaskAwareSafeEnv:
    def __init__(self, mac, density, horizon=20, seed=0, shield_enabled=True, task_multiplier=1.0):
        self.mac=mac; self.n=int(density); self.horizon=horizon; self.shield_enabled=shield_enabled; self.task_multiplier=float(task_multiplier); self.rng=random.Random(seed); self.vehicles=[str(i) for i in range(self.n)]; self.sources=self.vehicles[:max(1,self.n//5)]; self.helpers={str(i):{'cpu':self.rng.uniform(2e8,1e9),'queue':0.} for i in range(min(12,self.n))}; self.reset(seed)
    def reset(self,seed=None):
        if seed is not None:
            self.rng.seed(seed)
            # Keep task generation and Mode-2 randomness paired and repeatable.
            self.mac.reset(seed + 1000003)
        self.t=0; self.pending=[]; self.safe_gen=self.safe_ok=self.comp_done=0; self.delays=[]; self.tasks=[]
        self.safety_windows=[]; self.cbr_samples=[]; self.comp_generated=0; self.comp_delivered=0
        self.tx_attempts=0; self.tx_successes=0; self.resource_bits=0.0
        for src in self.sources:
            for j in range(max(1,int(round(3*self.task_multiplier)))): self.tasks.append({'id':len(self.tasks),'source':src,'payload':self.rng.uniform(.01,.08),'cycles':self.rng.uniform(1e7,1e8),'deadline':self.rng.uniform(.03,.15),'value':self.rng.uniform(1.,5.),'done':False})
        return self.obs()
    def obs(self):
        cbr=self.mac.get_cbr_history_mean(); prr=self.safe_ok/max(1,self.safe_gen); p95=(sorted(self.delays)[max(0,int(.95*len(self.delays))-1)] if self.delays else 0.)
        return [cbr,self.n/100.,prr,p95/100.,sum(not t['done'] for t in self.tasks)/max(1,len(self.tasks)),self.t/self.horizon,0.,0.]
    def safe_budget(self):
        return int(torch_where(conservative_safe_mask(self.obs()))) if self.shield_enabled else min(5, len(self.helpers))
    def step(self, assignments):
        now=self.t*.01; k=self.safe_budget(); assignments=assignments[:k]; used=set();
        safe_gen_before=self.safe_gen; safe_ok_before=self.safe_ok; delay_before=len(self.delays)
        tx_attempts_before=self.mac.tx_attempts; tx_successes_before=self.mac.tx_successes
        for i,v in enumerate(self.vehicles):
            dst=self.vehicles[(i+1)%self.n]; self.pending.append(Packet(v,dst,-100000-i-self.t*1000,0,'UL',.002,now,now+.1,traffic_class='safety')); self.safe_gen+=1
        for tid,hid in assignments:
            if tid in used: continue
            task=next((x for x in self.tasks if x['id']==tid and not x['done']),None)
            if task is None or hid not in self.helpers: continue
            used.add(tid); self.comp_generated += 1
            self.pending.append(Packet(task['source'],hid,tid,0,'UL',task['payload'],now,now+task['deadline'],traffic_class='computation'))
        for slot in range(10):
            self.mac.advance_slot(); cur=now+slot*.001
            for p in list(self.pending):
                if cur>p.deadline: self.pending.remove(p); continue
                if self.mac.attempt_transmission(p,p.source,p.destination,None,cur):
                    self.pending.remove(p)
                    if p.traffic_class=='safety': self.safe_ok+=1; self.delays.append((cur-p.generation_time)*1000)
                    else:
                        self.comp_delivered += 1
                        task=next(x for x in self.tasks if x['id']==p.job_id); finish=(cur-p.generation_time)+task['cycles']/self.helpers[p.destination]['cpu'];
                        if finish<=task['deadline']: task['done']=True; self.comp_done+=1
        self.t+=1
        interval_gen=self.safe_gen-safe_gen_before; interval_ok=self.safe_ok-safe_ok_before
        interval_delays=self.delays[delay_before:]
        interval_prr=interval_ok/max(1,interval_gen)
        interval_p95=(sorted(interval_delays)[max(0,int(.95*len(interval_delays))-1)] if interval_delays else 0.)
        self.safety_windows.append({'prr':interval_prr,'p95_ms':interval_p95})
        self.cbr_samples.extend(list(self.mac.cbr_history)[-10:])
        self.tx_attempts=self.mac.tx_attempts; self.tx_successes=self.mac.tx_successes
        self.resource_bits += sum(float(getattr(p,'payload_size',0.0))*(8_000_000 if getattr(p,'direction','UL')=='UL' else 8_000) for p in self.pending)
        prr=self.safe_ok/max(1,self.safe_gen); p95=(sorted(self.delays)[max(0,int(.95*len(self.delays))-1)] if self.delays else 0.)
        return self.obs(),sum(t['value'] for t in self.tasks if t['done']),{'prr':prr,'p95_ms':p95,'completed':self.comp_done}
def torch_where(mask):
    return int(mask.sum().item())
