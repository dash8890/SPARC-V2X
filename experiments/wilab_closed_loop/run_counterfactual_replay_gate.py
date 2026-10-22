"""Deterministic prefix-replay gate for WiLab residual-vs-anchor branches."""
from __future__ import annotations
import argparse, json, os, shutil, tempfile
from pathlib import Path
import torch
from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.residual_task_policy import ResidualTaskPolicy

def load_policy(path):
    p=ResidualTaskPolicy(); p.load_state_dict(torch.load(path,map_location='cpu')['policy']); p.eval(); return p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=909); ap.add_argument('--branch-epoch',type=int,default=5); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--density',type=int,default=50); ap.add_argument('--output-root',default='results/wilab_counterfactual_gate'); ap.add_argument('--checkpoint',default='checkpoints/residual_task_policy_cf.pt'); args=ap.parse_args()
    policy=load_policy(args.checkpoint); base=Path(args.output_root); base.mkdir(parents=True,exist_ok=True)
    results=[]
    for mode in ('residual','anchor'):
        out=base/mode; bridge=Path(tempfile.gettempdir())/f'wilab_cf_{mode}_{args.seed}'
        if bridge.exists(): shutil.rmtree(bridge)
        bridge.mkdir(parents=True)
        os.environ['WILAB_OUTPUT_ROOT']=str(out)
        total,loss,completed=run_episode(ROOT,bridge,0,args.density,policy,600.0,False,False,100.0,20.0,args.seed,args.epochs,None,3,False,250000,300,12.0,mode,args.branch_epoch,True)
        results.append({'mode':mode,'return':total,'completed_value':completed,'output':str(out)})
    def read(mode): return json.loads((base/mode/'episode_0000'/'bridge_audit.json').read_text())
    a=read('residual'); b=read('anchor'); n=min(args.branch_epoch,len(a),len(b))
    fields=['prr','p95_delay_ms','cbr','safety_rx_cumulative','safety_error_cumulative','safety_prr_cumulative','safety_p95_delay_cumulative_ms','wilab_scheduled_attempts','wilab_pending_packets']
    identical_prefix=all(a[i].get(k)==b[i].get(k) for i in range(n) for k in fields)
    first_diff=next((i for i in range(min(len(a),len(b))) if a[i].get('computation_bytes_offered')!=b[i].get('computation_bytes_offered') or a[i].get('delivered_source_ids')!=b[i].get('delivered_source_ids')),None)
    report={'seed':args.seed,'branch_epoch':args.branch_epoch,'epochs':args.epochs,'same_seed':True,'identical_prefix':identical_prefix,'first_observed_branch_difference_epoch':first_diff,'branches':results,'safety_envelope_bytes':250000}
    (base/'counterfactual_replay_gate.json').write_text(json.dumps(report,indent=2)); (base/'counterfactual_replay_gate.md').write_text('# WiLab counterfactual prefix-replay gate\n\n'+json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
