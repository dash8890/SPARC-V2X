import copy,json,torch
from pathlib import Path
from radio.mode2 import Mode2MAC
from env.task_aware_safe_env import TaskAwareSafeEnv
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safe_admission import conservative_safe_mask

torch.set_num_threads(1)
def f(t,z,e): return [t['payload'],t['cycles']/1e8,t['deadline'],t['value']/5.,z['cpu']/1e9,z['queue']/1e8,e.n/100.,e.mac.get_cbr_history_mean(),e.t/e.horizon,1.]
def cs(e): return [(t,h,f(t,z,e)) for t in e.tasks if not t['done'] for h,z in e.helpers.items()]
def choose(e,p,stochastic):
 c=cs(e); k=int(conservative_safe_mask(e.obs()).sum().item());
 if not c or k==0:return [],[],c
 x=torch.tensor([z[2] for z in c]); fin=torch.tensor([z[0]['cycles']/e.helpers[z[1]]['cpu'] for z in c]); val=torch.tensor([z[0]['value'] for z in c]); an=val/(fin+1e-6); logits=an+p.residual(x); rem=list(range(len(c))); used=set(); pairs=[]; lps=[]
 for _ in range(min(k,len(rem))):
  v=[i for i in rem if c[i][0]['id'] not in used]
  if not v:break
  if stochastic:
   d=torch.distributions.Categorical(logits=torch.stack([logits[i] for i in v])); q=int(d.sample()); lps.append(d.log_prob(torch.tensor(q)))
  else:q=int(torch.argmax(torch.stack([logits[i] for i in v])))
  ix=v[q]; t,h,_=c[ix]; pairs.append((t['id'],h)); used.add(t['id']); rem.remove(ix)
 return pairs,lps,c
p=ResidualTaskPolicy();p.load_state_dict(torch.load('checkpoints/residual_task_policy_v2.pt',map_location='cpu')['policy']);opt=torch.optim.Adam(p.parameters(),lr=1e-4);rows=[]
for ep in range(100):
 e=TaskAwareSafeEnv(Mode2MAC(slot_capacity_bits=1_500_000),[25,50,80,100,150,200][ep%6],seed=50000+ep); data=[];total=0
 for _ in range(e.horizon):
  sampled,lps,c=choose(e,p,True); anchor,_,_=choose(e,p,False)
  if not c or not lps:e.step([]);continue
  ea,es=copy.deepcopy(e),copy.deepcopy(e); ba=sum(t['value'] for t in ea.tasks if t['done']); ea.step(anchor); ra=sum(t['value'] for t in ea.tasks if t['done'])-ba; bb=sum(t['value'] for t in es.tasks if t['done']); es.step(sampled); rb=sum(t['value'] for t in es.tasks if t['done'])-bb; e.step(sampled); total+=rb; data.append((lps,torch.tensor([z[2] for z in c]),rb-ra))
 if data:
  pl=[];cl=[]
  for lps,x,a in data:
   adv=torch.tensor(a); q=p.residual(x).mean(); cl.append((q-adv)**2); pl.extend([-lp*adv for lp in lps])
  loss=torch.stack(pl).mean()+0.2*torch.stack(cl).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1);opt.step()
 rows.append({'episode':ep,'return':total,'cf_mean':sum(z[2] for z in data)/max(1,len(data))})
 if ep%10==0:print(rows[-1])
torch.save({'policy':p.state_dict(),'algorithm':'counterfactual residual RL','episodes':100},'checkpoints/residual_task_policy_cf.pt');Path('results/residual_task_cf_training.json').write_text(json.dumps(rows,indent=2))
