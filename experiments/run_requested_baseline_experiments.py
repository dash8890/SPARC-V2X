"""Run the frozen computation-load and shield-instrumentation campaigns.

Outputs are deliberately separate from earlier result files.  The SPARC row
uses the 100-episode final residual checkpoint.
"""
import argparse, csv, json, random, statistics, time
from pathlib import Path
import torch
from radio.mode2 import Mode2MAC
from env.task_aware_safe_env import TaskAwareSafeEnv
from controllers.rl.safe_admission import conservative_safe_mask
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.task_pair_policy import TaskPairPolicy

torch.set_num_threads(1)
SEEDS = tuple(range(7000, 7010))
DENSITIES = (10, 25, 50, 80, 100, 125, 150, 175, 200)
ROBUST_DENSITIES = (50, 100, 150, 200)
METHODS = ("sparc", "value_aware", "minfinish", "task_rl", "edf", "random", "shield_greedy")

def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def percentile(xs, q):
    if not xs: return 0.0
    return sorted(xs)[max(0, min(len(xs)-1, int(q*len(xs))-1))]

def load_policy(kind):
    if kind == "sparc":
        p = ResidualTaskPolicy(); ck = "checkpoints/residual_task_policy_cf.pt"
    elif kind == "task_rl":
        p = TaskPairPolicy(); ck = "checkpoints/task_pair_policy.pt"
    else: return None
    p.load_state_dict(torch.load(ck, map_location="cpu")["policy"]); p.eval(); return p

def candidates(env):
    out=[]
    for t in env.tasks:
        if t["done"]: continue
        for h,z in env.helpers.items():
            out.append((t,h,[t["payload"],t["cycles"]/1e8,t["deadline"],t["value"]/5.,z["cpu"]/1e9,z["queue"]/1e8,env.n/100.,env.mac.get_cbr_history_mean(),env.t/env.horizon,1.]))
    return out

def select(env, method, policy, safe_k, rng):
    pending=[t for t in env.tasks if not t["done"]]; cand=candidates(env); limit=min(safe_k,len(cand))
    if limit <= 0 or not cand: return [], cand
    if method in ("sparc","task_rl"):
        x=torch.tensor([c[2] for c in cand],dtype=torch.float32)
        with torch.no_grad():
            if method == "sparc":
                fin=torch.tensor([c[0]["cycles"]/env.helpers[c[1]]["cpu"] for c in cand]); val=torch.tensor([c[0]["value"] for c in cand])
                analytic=val/(fin+1e-6)
                scores=analytic+policy.residual(x)
                mf=int(torch.argmin(fin)); top=int(torch.argmax(scores))
                if float((scores[top]-analytic[mf]).detach()) <= 0.05:
                    scores=analytic
            else: scores=policy.score(x)
        order=torch.argsort(scores, descending=True).tolist()
    elif method == "value_aware": order=sorted(range(len(cand)),key=lambda i:cand[i][0]["value"]/(cand[i][0]["cycles"]/env.helpers[cand[i][1]]["cpu"]+1e-6),reverse=True)
    elif method == "minfinish": order=sorted(range(len(cand)),key=lambda i:cand[i][0]["cycles"]/env.helpers[cand[i][1]]["cpu"])
    elif method == "edf": order=sorted(range(len(cand)),key=lambda i:cand[i][0]["deadline"])
    elif method == "random": order=list(range(len(cand))); rng.shuffle(order)
    else: order=list(range(len(cand)))
    pairs=[]; used=set()
    for ix in order:
        t,h,_=cand[ix]
        if t["id"] in used: continue
        pairs.append((t["id"],h)); used.add(t["id"])
        if len(pairs)>=limit: break
    return pairs,cand

def run(method,density,seed,load=1.0,epoch_log=False):
    env=TaskAwareSafeEnv(Mode2MAC(slot_capacity_bits=1_500_000),density,seed=seed,task_multiplier=load,shield_enabled=True)
    policy=load_policy(method); rng=random.Random(seed+17); done=set(); total=0.; offered=sum(t["value"] for t in env.tasks); times=[]; epochs=[]
    for epoch in range(env.horizon):
        tic=time.perf_counter(); obs=env.obs(); safe_mask=conservative_safe_mask(obs); safe_k=int(safe_mask.sum().item()); cand=candidates(env)
        pairs, _=select(env,method,policy,safe_k,rng); raw_pairs,_=select(env,method,policy,min(5,len(cand)),rng)
        intervened=int(len(raw_pairs)>len(pairs) or safe_k<5)
        before_done=set(done); env.step(pairs)
        for t in env.tasks:
            if t["done"] and t["id"] not in done: done.add(t["id"]); total+=t["value"]
        dt=(time.perf_counter()-tic)*1000; times.append(dt)
        if epoch_log:
            recent=env.safety_windows[-1] if env.safety_windows else {"prr":0.,"p95_ms":0.}
            epochs.append({"seed":seed,"density":density,"method":method,"epoch":epoch,"num_candidate_actions_before_shield":len(cand),"num_safe_actions_after_shield":safe_k,"shield_intervened":intervened,"raw_selected_action":len(raw_pairs),"executed_action":len(pairs),"num_admitted_computation_tasks":len(pairs),"computation_bits_admitted":sum(next((t["payload"]*8e6 for t in env.tasks if t["id"]==tid),0.) for tid,_ in pairs),"computation_bits_transmitted":env.mac.resource_bits_by_class.get("computation",0.),"completed_value_this_epoch":sum(t["value"] for t in env.tasks if t["done"] and t["id"] not in before_done),"completed_tasks_this_epoch":len(done)-len(before_done),"R_value_contribution":sum(t["value"] for t in env.tasks if t["done"] and t["id"] not in before_done)/max(offered,1e-9),"CBR":env.mac.get_cbr_history_mean(),"PRR_safe":recent["prr"],"P95_safety_delay_ms":recent["p95_ms"],"safety_violation_indicator":int(recent["prr"]<.95 or recent["p95_ms"]>50.)})
    violations=sum(w["prr"]<.95 or w["p95_ms"]>50 for w in env.safety_windows); bits=env.mac.resource_bits_by_class.get("computation",0.)
    row={"seed":seed,"density":density,"load_multiplier":load,"method":method,"R_value":total/max(offered,1e-9),"P_viol":violations/max(1,len(env.safety_windows)),"PRR_safe":env.safe_ok/max(1,env.safe_gen),"P95_safety_delay_ms":percentile(env.delays,.95),"P99_safety_delay_ms":percentile(env.delays,.99),"CBR_mean":statistics.mean(env.cbr_samples) if env.cbr_samples else 0.,"CBR_P95":percentile(env.cbr_samples,.95),"Comp_PDR":env.comp_delivered/max(1,env.comp_generated),"completed_value":total,"offered_value":offered,"completed_tasks":len(done),"offered_tasks":len(env.tasks),"computation_bits":bits,"admitted_computation_tasks":env.comp_generated,"decision_mean_ms":statistics.mean(times),"decision_P95_ms":percentile(times,.95)}
    return row,epochs

def bootstrap(rows, a, b, group):
    out=[]; rng=random.Random(918273)
    keys=sorted(set(tuple(r[k] for k in group) for r in rows))
    for key in keys:
        sub=[r for r in rows if tuple(r[k] for k in group)==key]; seeds=sorted(set(r["seed"] for r in sub)); by={s:{r["method"]:r["R_value"] for r in sub if r["seed"]==s} for s in seeds}; ds=[by[s].get(a,0)-by[s].get(b,0) for s in seeds]
        if not ds: continue
        vals=[]
        for _ in range(5000): vals.append(statistics.mean(ds[rng.randrange(len(ds))] for _ in ds))
        vals.sort(); out.append({**dict(zip(group,key)),"method_a":a,"method_b":b,"mean_delta":statistics.mean(ds),"ci_low":vals[125],"ci_high":vals[4874],"n_seeds":len(ds)})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("robustness","instrumentation")); args=ap.parse_args()
    if args.mode=="robustness":
        rows=[]
        for load in (.5,1.,1.5):
            for d in ROBUST_DENSITIES:
                for seed in SEEDS:
                    for m in METHODS: rows.append(run(m,d,seed,load,False)[0])
        csv_write("results/load_robustness_raw.csv",rows)
        summary=[]
        for key in sorted(set((r["density"],r["load_multiplier"],r["method"]) for r in rows)):
            sub=[r for r in rows if (r["density"],r["load_multiplier"],r["method"])==key]; summary.append({"density":key[0],"load_multiplier":key[1],"method":key[2],**{k:statistics.mean(r[k] for r in sub) for k in ("R_value","P_viol","PRR_safe","P95_safety_delay_ms","P99_safety_delay_ms","CBR_mean","CBR_P95","Comp_PDR","completed_value","offered_value","completed_tasks","offered_tasks","computation_bits","admitted_computation_tasks","decision_mean_ms","decision_P95_ms")}})
        csv_write("results/load_robustness_summary.csv",summary); diffs=[]
        for r in rows:
            base=next((x for x in rows if x["seed"]==r["seed"] and x["density"]==r["density"] and x["load_multiplier"]==r["load_multiplier"] and x["method"]=="sparc"),None)
            if base and r["method"]!="sparc": diffs.append({"seed":r["seed"],"density":r["density"],"load_multiplier":r["load_multiplier"],"method":r["method"],"delta_R_value":base["R_value"]-r["R_value"]})
        csv_write("results/load_robustness_paired_differences.csv",diffs); boot=bootstrap(rows,"sparc","value_aware",("density","load_multiplier"))+bootstrap(rows,"sparc","minfinish",("density","load_multiplier")); csv_write("results/load_robustness_bootstrap.csv",boot)
    else:
        raw=[]; per=[]
        for d in DENSITIES:
            for seed in SEEDS:
                for m in METHODS:
                    row,epochs=run(m,d,seed,1.,True); raw.extend(epochs); per.append(row)
        csv_write("results/resource_shield_epoch_raw.csv",raw); csv_write("results/resource_shield_per_seed.csv",per)
        summary=[]
        for key in sorted(set((r["density"],r["method"]) for r in per)):
            sub=[r for r in per if (r["density"],r["method"])==key]; summary.append({"density":key[0],"method":key[1],**{k:statistics.mean(r[k] for r in sub) for k in ("R_value","P_viol","PRR_safe","CBR_mean","CBR_P95","computation_bits","admitted_computation_tasks","Comp_PDR","decision_mean_ms","decision_P95_ms")}})
        csv_write("results/resource_shield_summary.csv",summary); csv_write("results/resource_shield_paired_differences.csv",[{"density":r["density"],"seed":r["seed"],"method":r["method"],"delta_R_value_vs_sparc":r["R_value"]-next(x["R_value"] for x in per if x["density"]==r["density"] and x["seed"]==r["seed"] and x["method"]=="sparc")} for r in per if r["method"]!="sparc"])

if __name__=="__main__": main()
