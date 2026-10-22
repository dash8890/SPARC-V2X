import json, statistics, time, torch
from pathlib import Path
from radio.mode2 import Mode2MAC
from env.task_aware_safe_env import TaskAwareSafeEnv
from controllers.rl.safe_admission import conservative_safe_mask
from controllers.rl.residual_task_policy import ResidualTaskPolicy
torch.set_num_threads(1)

def run(method,density,seed,cap=None,horizon=20,task_multiplier=1.0):
    env=TaskAwareSafeEnv(Mode2MAC(slot_capacity_bits=1_500_000),density,horizon=horizon,seed=seed,shield_enabled=True,task_multiplier=task_multiplier)
    pol=None
    if method=='residual_rl':
        pol=ResidualTaskPolicy(); pol.load_state_dict(torch.load('checkpoints/residual_task_policy_cf.pt',map_location='cpu')['policy']); pol.eval()
    total=0.; done=set(); times=[]
    for _ in range(env.horizon):
        tic=time.perf_counter(); obs=env.obs(); k=int(conservative_safe_mask(obs).sum().item()); k=min(k,cap) if cap is not None else k; pending=[t for t in env.tasks if not t['done']]; cand=[(t,h,[t['payload'],t['cycles']/1e8,t['deadline'],t['value']/5.,env.helpers[h]['cpu']/1e9,env.helpers[h]['queue']/1e8,env.n/100.,env.mac.get_cbr_history_mean(),env.t/env.horizon,1.]) for t in pending for h in env.helpers]; pairs=[]
        if cand:
            with torch.no_grad():
                x=torch.tensor([c[2] for c in cand],dtype=torch.float32); finish=torch.tensor([c[0]['cycles']/env.helpers[c[1]]['cpu'] for c in cand]); value=torch.tensor([c[0]['value'] for c in cand]);
                if method=='value_aware': scores=value/(finish+1e-6)
                else:
                    analytic=value/(finish+1e-6); scores=analytic+pol.residual(x); mf=int(torch.argmin(finish)); top=int(torch.argmax(scores))
                    if float((scores[top]-analytic[mf]).detach())<=.05: scores=analytic
                order=torch.argsort(scores,descending=True)
            used=set()
            for ix in order.tolist():
                t,h,_=cand[ix]
                if t['id'] not in used: pairs.append((t['id'],h)); used.add(t['id'])
                if len(pairs)>=k: break
        times.append((time.perf_counter()-tic)*1000); env.step(pairs)
        for t in env.tasks:
            if t['done'] and t['id'] not in done: done.add(t['id']); total+=t['value']
    offered=sum(t['value'] for t in env.tasks); violations=sum(w['prr']<.95 or w['p95_ms']>50 for w in env.safety_windows); bits=env.mac.resource_bits_by_class.get('computation',0.)
    return {'method':method,'density':density,'seed':seed,'cap':cap,'R_value':total/max(offered,1e-9),'P_viol':violations/max(1,len(env.safety_windows)),'prr':env.safe_ok/max(1,env.safe_gen),'p95_ms':sorted(env.delays)[max(0,int(.95*len(env.delays))-1)] if env.delays else 0.,'p99_ms':sorted(env.delays)[max(0,int(.99*len(env.delays))-1)] if env.delays else 0.,'cbr_mean':statistics.mean(env.cbr_samples) if env.cbr_samples else 0.,'eta_value':total/max(bits,1.),'decision_ms_mean':statistics.mean(times),'decision_ms_p95':sorted(times)[max(0,int(.95*len(times))-1)]}

rows=[]
for d in (10,25,50,80,100,125,150,175,200):
    for s in range(10):
        for m in (('value_aware','residual_rl')): rows.append(run(m,d,16000+d+s))
for d in (100,125):
    for cap in (1,2,3,4):
        for s in range(10):
            for m in (('value_aware','residual_rl')): rows.append(run(m,d,18000+cap*100+d+s,cap))
Path('results/residual_rl_eval.json').write_text(json.dumps(rows,indent=2)); print(f'wrote {len(rows)} rows')




