import json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from radio.mode2 import Mode2MAC
from env.task_aware_safe_env import TaskAwareSafeEnv
from controllers.rl.residual_task_policy import ResidualTaskPolicy
CAPS={10:3,25:4,50:4,80:4,100:3,125:4,150:0,175:2,200:0}
torch.set_num_threads(1)
def run(d,seed,policy):
 e=TaskAwareSafeEnv(Mode2MAC(slot_capacity_bits=1_500_000),d,seed=seed,shield_enabled=False); p=None
 if policy=='residual': p=ResidualTaskPolicy(); p.load_state_dict(torch.load('checkpoints/residual_task_policy_cf_cap.pt',map_location='cpu')['policy']); p.eval()
 total=0.; times=[]; admitted=[]; done=set()
 for _ in range(e.horizon):
  tic=time.perf_counter(); c=[(t,h,[t['payload'],t['cycles']/1e8,t['deadline'],t['value']/5.,z['cpu']/1e9,z['queue']/1e8,e.n/100.,e.mac.get_cbr_history_mean(),e.t/e.horizon,1.]) for t in e.tasks if not t['done'] for h,z in e.helpers.items()]; k=CAPS[d]
  if not c:
   times.append((time.perf_counter()-tic)*1000); admitted.append(0); e.step([]); continue
  if policy=='value': order=sorted(c,key=lambda x:x[0]['value']/max(x[0]['cycles']/e.helpers[x[1]]['cpu'],1e-6),reverse=True)
  else:
   with torch.no_grad():
    x=torch.tensor([z[2] for z in c]); fin=torch.tensor([z[0]['cycles']/e.helpers[z[1]]['cpu'] for z in c]); val=torch.tensor([z[0]['value'] for z in c]); score=val/(fin+1e-6)+p.residual(x); order=[c[i] for i in torch.argsort(score,descending=True).tolist()]
  pairs=[]; used=set()
  for t,h,_ in order:
   if t['id'] in used: continue
   pairs.append((t['id'],h)); used.add(t['id'])
   if len(pairs)>=k: break
  times.append((time.perf_counter()-tic)*1000); admitted.append(len(pairs)); before={t['id'] for t in e.tasks if t['done']}; e.step(pairs); after={t['id'] for t in e.tasks if t['done']}; total+=sum(t['value'] for t in e.tasks if t['id'] in after-before)
 offered=sum(t['value'] for t in e.tasks); viol=sum(w['prr']<.95 or w['p95_ms']>50 for w in e.safety_windows); ds=sorted(e.delays); cbr=list(e.cbr_samples)
 return {'density':d,'seed':seed,'policy':policy,'cap':CAPS[d],'R_value':total/max(offered,1e-9),'P_viol':viol/max(1,len(e.safety_windows)),'prr':e.safe_ok/max(1,e.safe_gen),'p95_ms':ds[max(0,int(.95*len(ds))-1)] if ds else 0.,'p99_ms':ds[max(0,int(.99*len(ds))-1)] if ds else 0.,'cbr_mean':statistics.mean(cbr) if cbr else 0.,'mean_admitted':statistics.mean(admitted),'decision_ms_mean':statistics.mean(times),'decision_ms_p95':sorted(times)[max(0,int(.95*len(times))-1)]}
if __name__=='__main__':
    rows=[run(d,s,p) for d in CAPS for s in range(16000,16010) for p in ('value','residual')]; Path('cap_sparc_final_eval.json').write_text(json.dumps(rows,indent=2)); print('wrote',len(rows))
    for d in CAPS:
     for p in ('value','residual'):
      x=[r for r in rows if r['density']==d and r['policy']==p]; print(d,p,round(statistics.mean(r['R_value'] for r in x),4),round(statistics.mean(r['P_viol'] for r in x),4),round(statistics.mean(r['prr'] for r in x),4),round(statistics.mean(r['mean_admitted'] for r in x),2))
    
    
    

