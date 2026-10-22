"""WiLab residual actor-critic with rollout actor updates and replay critics."""
from __future__ import annotations
import argparse, json, os, random, shutil, tempfile
from pathlib import Path
import numpy as np
import torch
from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.safety_governor import SafetyGovernor
from controllers.rl.replay_residual_actor_critic import ReplayResidualActorCritic, TransitionReplay


def rollout_actor_loss(trace, gamma=0.99, gae_lambda=0.95):
    """Compute a normalized GAE policy loss and value-baseline loss."""
    if not trace:
        return None
    rewards = [float(x.get('reward', 0.0)) for x in trace]
    values = [x['value'].reshape(()) for x in trace]
    advantages = []
    gae = torch.tensor(0.0)
    next_value = torch.tensor(0.0)
    for reward, value in reversed(list(zip(rewards, values))):
        delta = torch.tensor(reward) + gamma * next_value.detach() - value.detach()
        gae = delta + gamma * gae_lambda * gae
        advantages.append(gae)
        next_value = value
    advantages = torch.stack(list(reversed(advantages)))
    returns = advantages + torch.stack(values).detach()
    norm_adv = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    policy_terms = []
    for item, advantage in zip(trace, norm_adv):
        logp = item['logp']
        if isinstance(logp, torch.Tensor) and logp.requires_grad:
            policy_terms.append(-logp * advantage.detach())
    if not policy_terms:
        return None
    actor_loss = torch.stack(policy_terms).mean()
    value_loss = 0.5 * ((torch.stack(values) - returns) ** 2).mean()
    return actor_loss, value_loss


def save_checkpoint(path, policy, actor_optimizer, critic_optimizer, replay, cfg, rows, episode):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy': policy.state_dict(),
        'actor_optimizer': actor_optimizer.state_dict(),
        'critic_optimizer': critic_optimizer.state_dict(),
        'replay': replay.state_dict(),
        'environment': cfg,
        'rows': rows,
        'episode': int(episode),
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
    }, path)


def add_n_step_rollout(replay, trace, policy, envelope_bytes, n_step=5, gamma=0.99):
    """Convert one WiLab rollout to n-step replay transitions."""
    for start, item in enumerate(trace):
        reward=0.0; steps=0
        for offset in range(n_step):
            index=start+offset
            if index >= len(trace):
                break
            reward += (gamma ** offset) * float(trace[index].get('reward',0.0))
            steps += 1
        next_index=start+steps
        terminal=next_index >= len(trace)
        next_item=trace[-1] if terminal else trace[next_index]
        obs=np.asarray(item.get('observation',[0.0]*10),dtype=np.float32)
        nxt=np.asarray(next_item.get('observation',obs.tolist()),dtype=np.float32)
        replay.add({'obs':obs,'action':policy.action_summary(item,envelope_bytes),
                    'reward':reward,'next_obs':nxt,'next_meta':{
                        'candidate_features':next_item.get('candidate_features',[]),
                        'candidate_analytic_scores':next_item.get('candidate_analytic_scores',[]),
                        'candidate_task_ids':next_item.get('candidate_task_ids',[]),
                        'candidate_payload_bytes':next_item.get('candidate_payload_bytes',[]),
                        'assignments':next_item.get('assignments',[]),
                        'bytes':next_item.get('bytes',0),
                        'selected_candidate_indices':next_item.get('selected_candidate_indices',[]),
                        'residual_changed_anchor':next_item.get('residual_changed_anchor',False)},
                    'discount':gamma**steps,'done':float(terminal),
                    'envelope_bytes':int(envelope_bytes)})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episodes',type=int,default=100); ap.add_argument('--batch-size',type=int,default=5)
    ap.add_argument('--controller-epochs',type=int,default=100); ap.add_argument('--density',type=int,default=35)
    ap.add_argument('--seed-base',type=int,default=70000); ap.add_argument('--task-seed-base',type=int,default=170000)
    ap.add_argument('--output-root',default='results/wilab_residual_actor_critic_replay_actor_100ep'); ap.add_argument('--checkpoint-out',default='checkpoints/wilab_residual_actor_critic_replay_actor_100ep.pt')
    ap.add_argument('--envelope-bytes',type=int,default=175000); ap.add_argument('--safety-payload-bytes',type=int,default=300); ap.add_argument('--simulation-time',type=float,default=12.0); ap.add_argument('--lr',type=float,default=1e-4)
    args=ap.parse_args(); random.seed(515151); np.random.seed(515151); torch.manual_seed(515151)
    out=Path(args.output_root); out.mkdir(parents=True,exist_ok=False)
    policy=ReplayResidualActorCritic(); policy.train()
    actor_optimizer=torch.optim.Adam(policy.actor.parameters(),lr=args.lr)
    critic_optimizer=torch.optim.Adam(list(policy.q1.parameters())+list(policy.q2.parameters()),lr=args.lr)
    replay=TransitionReplay()
    cfg=vars(args).copy(); cfg.update({'algorithm':'WiLab residual actor-critic: on-policy GAE actor/value update plus prioritized n-step Double-Q replay critics','initialization':'fresh; residual output zeroed; seed=515151','replay':'prioritized transition replay; n_step=5; persisted contents','actor_update':'5-rollout normalized GAE batch','counterfactual_replay':False})
    (out/'training_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8'); rows=[]; actor_batch=[]
    for ep in range(args.episodes):
        bridge=Path(tempfile.gettempdir())/f'wilab_replay_ac_{os.getpid()}_{ep}';
        if bridge.exists(): shutil.rmtree(bridge)
        os.environ['WILAB_OUTPUT_ROOT']=str(out); gov=SafetyGovernor(args.envelope_bytes,horizon_envelope_only=True)
        total,_,completed,trace=run_episode(ROOT,bridge,ep,args.density,policy,900.0,False,False,100.0,20.0,args.seed_base+ep,args.controller_epochs,None,3,False,None,args.safety_payload_bytes,args.simulation_time,'residual',None,False,'off','',True,args.task_seed_base+ep,gov,args.envelope_bytes)
        add_n_step_rollout(replay,trace,policy,args.envelope_bytes,n_step=5,gamma=0.99)
        actor_batch.append(trace)
        actor_metrics=None
        if len(actor_batch) >= args.batch_size:
            losses=[rollout_actor_loss(x) for x in actor_batch]
            losses=[x for x in losses if x is not None]
            if losses:
                actor_loss=torch.stack([x[0] for x in losses]).mean()
                value_loss=torch.stack([x[1] for x in losses]).mean()
                total_actor_loss=actor_loss + 0.5*value_loss
                actor_optimizer.zero_grad(set_to_none=True); total_actor_loss.backward()
                actor_grad=float(torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.0))
                actor_optimizer.step()
                actor_metrics={'actor_loss':float(actor_loss.detach()),'value_loss':float(value_loss.detach()),'actor_grad_norm':actor_grad}
            actor_batch=[]
        metrics=policy.replay_update(replay,critic_optimizer,batch_size=128) if len(replay)>=32 else None
        row={'episode':ep,'sim_seed':args.seed_base+ep,'task_seed':args.task_seed_base+ep,'return':float(total),'completed_value':float(completed),'decisions':len(trace),'nonzero_reward_steps':sum(float(x.get('reward',0))!=0 for x in trace),'replay_size':len(replay)}
        if metrics: row.update(metrics)
        if actor_metrics: row.update(actor_metrics)
        rows.append(row); (out/'training_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); print(json.dumps(row),flush=True)
        if (ep+1)%5==0: save_checkpoint(out/f'checkpoint_episode_{ep+1:04d}.pt',policy,actor_optimizer,critic_optimizer,replay,cfg,rows,ep+1)
    save_checkpoint(Path(args.checkpoint_out),policy,actor_optimizer,critic_optimizer,replay,cfg,rows,args.episodes)
    print(json.dumps({'status':'complete','episodes':args.episodes,'output_root':str(out),'checkpoint':args.checkpoint_out}),flush=True)
if __name__=='__main__': main()
