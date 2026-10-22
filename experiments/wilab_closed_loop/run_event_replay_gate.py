"""Short reference-record/replay gate for WiLab stochastic draws."""
from __future__ import annotations
import argparse,json,os,shutil,tempfile
from pathlib import Path
import torch
from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.residual_task_policy import ResidualTaskPolicy

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=909); ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--output-root',default='results/wilab_event_replay_gate'); args=ap.parse_args()
    policy=ResidualTaskPolicy(); policy.load_state_dict(torch.load('checkpoints/residual_task_policy_cf.pt',map_location='cpu')['policy']); policy.eval()
    base=Path(args.output_root); base.mkdir(parents=True,exist_ok=True); event=(base/'events.mat').resolve()
    paths=[]
    for mode in ('record','replay'):
        out=base/mode; bridge=Path(tempfile.gettempdir())/f'wilab_event_{mode}_{args.seed}'
        if bridge.exists(): shutil.rmtree(bridge)
        bridge.mkdir(parents=True); os.environ['WILAB_OUTPUT_ROOT']=str(out)
        run_episode(ROOT,bridge,0,50,policy,600.0,False,False,100.0,20.0,args.seed,args.epochs,None,3,False,250000,300,12.0,'anchor',None,True,mode,str(event))
        paths.append(out/'episode_0000'/'bridge_audit.json')
    a=json.loads(paths[0].read_text()); b=json.loads(paths[1].read_text()); fields=['prr','p95_delay_ms','cbr','safety_rx_cumulative','safety_error_cumulative','safety_prr_cumulative','safety_p95_delay_cumulative_ms','wilab_scheduled_attempts','wilab_pending_packets','computation_bytes_offered','delivered_source_ids']; diffs=[]
    for i,(x,y) in enumerate(zip(a,b)):
        bad=[k for k in fields if x.get(k)!=y.get(k)]
        if bad: diffs.append({'epoch':i,'fields':bad})
    report={'seed':args.seed,'epochs':args.epochs,'event_file':str(event),'reference_replay_identical':not diffs,'differences':diffs}
    (base/'event_replay_gate.json').write_text(json.dumps(report,indent=2)); (base/'event_replay_gate.md').write_text('# WiLab event replay gate\n\n'+json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
