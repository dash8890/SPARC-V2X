"""Train the counterfactual residual policy under calibrated static caps."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from radio.mode2 import Mode2MAC
from env.task_aware_safe_env import TaskAwareSafeEnv
from controllers.rl.residual_task_policy import ResidualTaskPolicy
CAPS={10:3,25:4,50:4,80:4,100:3,125:4,150:0,175:2,200:0}
torch.set_num_threads(1)
def features(t,z,e): return [t['payload'],t['cycles']/1e8,t['deadline'],t['value']/5.,z['cpu']/1e9,z['queue']/1e8,e.n/100.,e.mac.get_cbr_history_mean(),e.t/e.horizon,1.]
def candidates(e): return [(t,h,features(t,z,e)) for t in e.tasks if not t['done'] for h,z in e.helpers.items()]
def choose(e,p,cap,stochastic):
 c=candidates(e); k=min(CAPS[e.n],cap); 
 if not c or k<=0:return [],[],c
 x=torch.tensor([z[2] for z in c]); fin=torch.tensor([z[0]['cycles']/e.helpers[z[1]]['cpu'] for z in c]); val=torch.tensor([z[0]['value'] for z in c]); logits=val/(fin+1e-6)+p.residual(x); rem=list(range(len(c))); used=set(); pairs=[]; lps=[]
 for _ in range(min(k,len(rem))):
  v=[i for i in rem if c[i][0]['id'] not in used]
  if not v: break
  if stochastic:
   d=torch.distributions.Categorical(logits=torch.stack([logits[i] for i in v])); q=int(d.sample()); lps.append(d.log_prob(torch.tensor(q)))
  else:q=int(torch.argmax(torch.stack([logits[i] for i in v])))
  ix=v[q]; t,h,_=c[ix]; pairs.append((t['id'],h)); used.add(t['id']); rem.remove(ix)
 return pairs,lps,c
p=ResidualTaskPolicy(); p.load_state_dict(torch.load('checkpoints/residual_task_policy_cf_cap.pt',map_location='cpu')['policy']); opt=torch.optim.Adam(p.parameters(),lr=1e-4); rows=[]
for ep in range(100):
 d=(10,25,50,80,100,125,150,175,200)[ep%9]; e=TaskAwareSafeEnv(Mode2MAC(slot_capacity_bits=1_500_000),d,seed=60000+ep,shield_enabled=False); data=[]; total=0.
 for _ in range(e.horizon):
  sampled,lps,c=choose(e,p,CAPS[d],True); anchor,_,_=choose(e,p,CAPS[d],False)
  if not c or not lps:e.step([]); continue
  ea,es=copy.deepcopy(e),copy.deepcopy(e); ba=sum(t['value'] for t in ea.tasks if t['done']); ea.step(anchor); ra=sum(t['value'] for t in ea.tasks if t['done'])-ba; bb=sum(t['value'] for t in es.tasks if t['done']); es.step(sampled); rb=sum(t['value'] for t in es.tasks if t['done'])-bb; e.step(sampled); total+=rb; data.append((lps,torch.tensor([z[2] for z in c]),rb-ra))
 if data:
  pl=[];cl=[]
  for lps,x,a in data:
   adv=torch.tensor(a); cl.append((p.residual(x).mean()-adv)**2); pl.extend([-lp*adv for lp in lps])
  loss=torch.stack(pl).mean()+.2*torch.stack(cl).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(p.parameters(),1); opt.step()
 rows.append({'episode':ep,'density':d,'return':total,'cf_mean':sum(z[2] for z in data)/max(1,len(data))})
torch.save({'policy':p.state_dict(),'algorithm':'counterfactual residual RL calibrated static cap','episodes':100,'caps':CAPS},'checkpoints/residual_task_policy_cf_cap.pt'); Path('cap_sparc_training_v2.json').write_text(json.dumps(rows,indent=2)); print('saved cap checkpoint')


