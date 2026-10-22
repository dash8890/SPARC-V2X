"""Locate first epoch of divergence for identical WiLab replays."""
from __future__ import annotations
import argparse,json,os,shutil,tempfile
from pathlib import Path
import torch
from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.residual_task_policy import ResidualTaskPolicy

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=909); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--density',type=int,default=50); ap.add_argument('--output-root',default='results/wilab_same_action_determinism'); ap.add_argument('--checkpoint',default='checkpoints/residual_task_policy_cf.pt'); args=ap.parse_args()
    p=ResidualTaskPolicy(); p.load_state_dict(torch.load(args.checkpoint,map_location='cpu')['policy']); p.eval()
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True)
    for rep in (1,2):
        out=root/f'replay_{rep}'; bridge=Path(tempfile.gettempdir())/f'wilab_same_action_{args.seed}_{rep}'
        if bridge.exists(): shutil.rmtree(bridge)
        bridge.mkdir(parents=True); os.environ['WILAB_OUTPUT_ROOT']=str(out)
        run_episode(ROOT,bridge,0,args.density,p,600.0,False,False,100.0,20.0,args.seed,args.epochs,None,3,False,250000,300,12.0,'anchor',None,True)
    a=json.loads((root/'replay_1/episode_0000/bridge_audit.json').read_text()); b=json.loads((root/'replay_2/episode_0000/bridge_audit.json').read_text())
    fields=['prr','p95_delay_ms','cbr','safety_rx_cumulative','safety_error_cumulative','safety_prr_cumulative','safety_p95_delay_cumulative_ms','wilab_scheduled_attempts','wilab_pending_packets','computation_bytes_offered','delivered_source_ids']
    first=None; diffs=[]
    for i,(x,y) in enumerate(zip(a,b)):
        bad=[k for k in fields if x.get(k)!=y.get(k)]
        if bad and first is None: first=i
        if bad: diffs.append({'epoch':i,'fields':bad})
    report={'seed':args.seed,'epochs':args.epochs,'same_action':'Value-Aware anchor','envelope_bytes':250000,'identical_trajectory':not diffs,'first_difference_epoch':first,'differences':diffs}
    (root/'same_action_determinism.json').write_text(json.dumps(report,indent=2)); (root/'same_action_determinism.md').write_text('# Same-action WiLab determinism\n\n'+json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
