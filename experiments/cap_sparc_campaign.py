"""Isolated calibrated-static-cap experiment.

Calibration and evaluation are intentionally separate from the estimator-shield
results. The cap is chosen from calibration seeds only, then frozen.
"""
import json, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from radio.mode2 import Mode2MAC
from env.task_aware_safe_env import TaskAwareSafeEnv
from controllers.rl.residual_task_policy import ResidualTaskPolicy

torch.set_num_threads(1)
DENSITIES=(10,25,50,80,100,125,150,175,200)
CAL_SEEDS=tuple(range(10000,10005)); COMPAT_SEEDS=tuple(range(15000,15005)); EPSILON=.05

def assign(env,k):
 c=[]
 for t in env.tasks:
  if t['done']: continue
  for h,z in env.helpers.items():
   finish=t['cycles']/max(z['cpu'],1.); c.append((t['value']/max(finish,1e-6),t['id'],h))
 c.sort(reverse=True); out=[]; used=set()
 for _,tid,h in c:
  if tid in used: continue
  out.append((tid,h)); used.add(tid)
  if len(out)>=k: break
 return out

def run(density,seed,cap,policy='value_aware'):
 env=TaskAwareSafeEnv(Mode2MAC(slot_capacity_bits=1_500_000),density,seed=seed,shield_enabled=False)
 pol=None
 if policy=='residual_rl':
  pol=ResidualTaskPolicy(); pol.load_state_dict(torch.load('checkpoints/residual_task_policy_cf.pt',map_location='cpu')['policy']); pol.eval()
 total=0.; done=set(); times=[]; admitted=[]
 for _ in range(env.horizon):
  tic=time.perf_counter(); k=min(int(cap),len(env.helpers)); pairs=assign(env,k)
  if policy=='residual_rl' and pairs:
   # Preserve the final residual scorer while applying the common cap.
   cand=[(t,h,[t['payload'],t['cycles']/1e8,t['deadline'],t['value']/5.,env.helpers[h]['cpu']/1e9,env.helpers[h]['queue']/1e8,env.n/100.,env.mac.get_cbr_history_mean(),env.t/env.horizon,1.]) for t in env.tasks if not t['done'] for h in env.helpers]
   with torch.no_grad():
    x=torch.tensor([z[2] for z in cand]); fin=torch.tensor([z[0]['cycles']/env.helpers[z[1]]['cpu'] for z in cand]); val=torch.tensor([z[0]['value'] for z in cand]); scores=val/(fin+1e-6)+pol.residual(x); order=torch.argsort(scores,descending=True)
   pairs=[]; used=set()
   for ix in order.tolist():
    t,h,_=cand[ix]
    if t['id'] in used: continue
    pairs.append((t['id'],h)); used.add(t['id'])
    if len(pairs)>=k: break
  times.append((time.perf_counter()-tic)*1000); admitted.append(len(pairs)); before={t['id'] for t in env.tasks if t['done']}; env.step(pairs); after={t['id'] for t in env.tasks if t['done']}
  total += sum(t['value'] for t in env.tasks if t['id'] in after-before)
 offered=sum(t['value'] for t in env.tasks); viol=sum(w['prr']<.95 or w['p95_ms']>50 for w in env.safety_windows); ds=sorted(env.delays); cbr=list(env.cbr_samples)
 return {'density':density,'seed':seed,'cap':cap,'policy':policy,'R_value':total/max(offered,1e-9),'P_viol':viol/max(1,len(env.safety_windows)),'prr':env.safe_ok/max(1,env.safe_gen),'p95_ms':ds[max(0,int(.95*len(ds))-1)] if ds else 0.,'p99_ms':ds[max(0,int(.99*len(ds))-1)] if ds else 0.,'cbr_mean':statistics.mean(cbr) if cbr else 0.,'cbr_p95':sorted(cbr)[max(0,int(.95*len(cbr))-1)] if cbr else 0.,'mean_admitted':statistics.mean(admitted),'decision_ms_mean':statistics.mean(times),'decision_ms_p95':sorted(times)[max(0,int(.95*len(times))-1)]}

def calibrate(d):
 out=[]
 for cap in range(6):
  rows=[run(d,s,cap,'value_aware') for s in CAL_SEEDS]; pv=statistics.mean(x['P_viol'] for x in rows)
  if pv<=EPSILON: out.append((statistics.mean(x['R_value'] for x in rows),cap,pv))
 return max(out,key=lambda x:x[0]) if out else (0,0,None)

if __name__=='__main__':
 tuning={}; compat=[]
 for d in DENSITIES:
  score=calibrate(d); tuning[str(d)]={'cap':score[1],'cal_R':score[0],'cal_P_viol':score[2]}; print(d,tuning[str(d)])
  compat += [run(d,s,score[1],'residual_rl') for s in COMPAT_SEEDS]
 out={'epsilon':EPSILON,'calibration_seeds':CAL_SEEDS,'compatibility_seeds':COMPAT_SEEDS,'tuning':tuning,'compatibility':compat}
 Path('cap_sparc_calibration_compatibility.json').write_text(json.dumps(out,indent=2)); print('wrote',len(compat),'compatibility rows')
