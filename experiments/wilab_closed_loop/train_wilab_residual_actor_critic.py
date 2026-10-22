"""Ordinary WiLab rollout actor-critic for the anchored residual policy.

The 250-kB envelope is fixed; no counterfactual branches are used.
"""
from __future__ import annotations
import argparse,json,os,shutil,tempfile
from pathlib import Path
import torch
from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.residual_task_policy import ResidualTaskPolicy

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episodes',type=int,default=20,help='total target episode count'); ap.add_argument('--start-episode',type=int,default=0); ap.add_argument('--seed',type=int,default=909); ap.add_argument('--density',type=int,default=50); ap.add_argument('--controller-epochs',type=int,default=20); ap.add_argument('--output-root',default='results/wilab_residual_actor_critic_smoke'); ap.add_argument('--checkpoint',default='checkpoints/residual_task_policy_cf.pt'); args=ap.parse_args()
    p=ResidualTaskPolicy(); p.load_state_dict(torch.load(args.checkpoint,map_location='cpu')['policy'],strict=False); p.train(); opt=torch.optim.Adam(p.parameters(),lr=1e-4); out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True); rows=[]
    for ep in range(args.start_episode,args.episodes):
        bridge=Path(tempfile.gettempdir())/f'wilab_ac_{args.seed}_{ep}';
        if bridge.exists(): shutil.rmtree(bridge)
        os.environ['WILAB_OUTPUT_ROOT']=str(out)
        total,_,completed,tr=run_episode(ROOT,bridge,ep,args.density,p,600.0,False,False,100.0,20.0,args.seed,args.controller_epochs,None,3,False,250000,300,12.0,'residual',None,False,'off','',True)
        rewards=torch.tensor([x['reward'] for x in tr],dtype=torch.float32); values=torch.stack([x['value'] for x in tr]) if tr else torch.zeros(0)
        if len(tr):
            adv=torch.zeros_like(rewards); gae=torch.tensor(0.0); next_v=torch.tensor(0.0)
            for t in reversed(range(len(tr))):
                delta=rewards[t]+0.99*next_v-values[t].detach(); gae=delta+0.99*0.95*gae; adv[t]=gae; next_v=values[t].detach()
            returns=adv+values.detach(); norm_adv=(adv-adv.mean())/(adv.std()+1e-6); logps=torch.stack([x['logp'] for x in tr]); actor=-(logps*norm_adv.detach()).mean(); critic=0.5*((values-returns)**2).mean(); loss=actor+critic
            opt.zero_grad(); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(p.parameters(),1.0); opt.step()
        else: actor=critic=loss=torch.tensor(0.0); grad=torch.tensor(0.0)
        rows.append({'episode':ep,'return':float(total),'completed_value':float(completed),'actor_loss':float(actor.detach()),'critic_loss':float(critic.detach()),'gradient_norm':float(grad),'decisions':len(tr),'nonzero_reward_steps':int((rewards!=0).sum()) if len(tr) else 0,'residual_deviation_fraction':sum(x['residual_changed_anchor'] for x in tr)/len(tr) if tr else 0.0,'envelope_bytes':250000})
        print(rows[-1],flush=True)
    (out/'training_metrics.json').write_text(json.dumps(rows,indent=2)); torch.save({'policy':p.state_dict(),'algorithm':'WiLab anchored residual actor-critic','episodes':args.episodes,'start_episode':args.start_episode,'envelope_bytes':250000},out/'residual_actor_critic_smoke.pt')
if __name__=='__main__': main()
