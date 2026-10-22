"""Isolated WiLab-trained residual actor-critic run.

This runner is deliberately separate from the historical Python-validation
training and from the earlier WiLab integration smoke outputs.  It uses real
WiLab rollouts for every update; the Python checkpoint is only initialization.
"""
from __future__ import annotations
import argparse, json, os, shutil, tempfile
from pathlib import Path
import torch

from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--start-episode', type=int, default=0)
    ap.add_argument('--seed', type=int, default=70000)
    ap.add_argument('--task-seed', type=int, default=170000)
    ap.add_argument('--density', type=int, default=35)
    ap.add_argument('--controller-epochs', type=int, default=20)
    ap.add_argument('--output-root', default='results/wilab_final_training_20261012')
    ap.add_argument('--checkpoint-init', default='checkpoints/residual_task_policy_cf.pt')
    ap.add_argument('--checkpoint-out', default='checkpoints/wilab_residual_actor_critic_final_50ep.pt')
    ap.add_argument('--envelope-bytes', type=int, default=175000)
    ap.add_argument('--safety-payload-bytes', type=int, default=300)
    ap.add_argument('--simulation-time', type=float, default=12.0)
    ap.add_argument('--generation-ms', type=float, default=100.0)
    ap.add_argument('--allocation-ms', type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    init = Path(args.checkpoint_init)
    policy = ResidualTaskPolicy()
    payload = torch.load(init, map_location='cpu')
    policy.load_state_dict(payload['policy'], strict=False)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    rows = []

    config = vars(args).copy()
    config.update({
        'algorithm': 'WiLab anchored residual actor-critic',
        'training_environment': 'WiLabV2Xsim closed loop',
        'initialization_only': str(init),
        'python_validation_artifacts_preserved': True,
        'training_seed_schedule': 'sim_seed=seed+episode; task_seed=task_seed+episode',
        'counterfactual_replay': False,
        'safety_governor': 'external horizon envelope, 175000 bytes',
    })
    (out / 'training_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    for ep in range(args.start_episode, args.episodes):
        sim_seed = args.seed + ep
        task_seed = args.task_seed + ep
        bridge = Path(tempfile.gettempdir()) / f'wilab_final_ac_{args.seed}_{ep}'
        if bridge.exists():
            shutil.rmtree(bridge)
        os.environ['WILAB_OUTPUT_ROOT'] = str(out)
        governor = SafetyGovernor(args.envelope_bytes, horizon_envelope_only=True)
        total, _, completed, trace = run_episode(
            ROOT, bridge, ep, args.density, policy, 900.0,
            False, False, args.generation_ms, args.allocation_ms,
            sim_seed, args.controller_epochs, None, 3, False, None,
            args.safety_payload_bytes, args.simulation_time, 'residual',
            None, False, 'off', '', True, task_seed, governor,
            args.envelope_bytes)

        rewards = torch.tensor([x['reward'] for x in trace], dtype=torch.float32)
        values = torch.stack([x['value'] for x in trace]) if trace else torch.zeros(0)
        if len(trace):
            adv = torch.zeros_like(rewards)
            gae = torch.tensor(0.0)
            next_v = torch.tensor(0.0)
            for t in reversed(range(len(trace))):
                delta = rewards[t] + 0.99 * next_v - values[t]
                gae = delta + 0.99 * 0.95 * gae
                adv[t] = gae
                next_v = values[t].detach()
            returns = adv.detach() + values
            norm_adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            logps = torch.stack([x['logp'] for x in trace])
            actor = -(logps * norm_adv.detach()).mean()
            critic = 0.5 * ((values - returns.detach()) ** 2).mean()
            loss = actor + critic
            optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        else:
            actor = critic = loss = torch.tensor(0.0)
            grad = torch.tensor(0.0)
        interventions = sum(bool(x.get('governor_intervened', False)) for x in trace)
        row = {
            'episode': ep, 'sim_seed': sim_seed, 'task_seed': task_seed,
            'return': float(total), 'completed_value': float(completed),
            'actor_loss': float(actor.detach()), 'critic_loss': float(critic.detach()),
            'total_loss': float(loss.detach()), 'gradient_norm': float(grad),
            'decisions': len(trace),
            'nonzero_reward_steps': int((rewards != 0).sum()) if len(trace) else 0,
            'residual_deviation_fraction': sum(x['residual_changed_anchor'] for x in trace) / len(trace) if trace else 0.0,
            'governor_interventions': interventions,
            'envelope_bytes': args.envelope_bytes,
            'density': args.density,
        }
        rows.append(row)
        (out / 'training_metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        torch.save({'policy': policy.state_dict(), 'algorithm': config['algorithm'],
                    'environment': config, 'episode': ep, 'episodes': args.episodes},
                   out / f'checkpoint_episode_{ep:04d}.pt')
        print(row, flush=True)

    final = Path(args.checkpoint_out)
    final.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'policy': policy.state_dict(), 'algorithm': config['algorithm'],
                'environment': config, 'episodes': args.episodes, 'rows': rows}, final)
    torch.save({'policy': policy.state_dict(), 'algorithm': config['algorithm'],
                'environment': config, 'episodes': args.episodes, 'rows': rows}, out / 'wilab_residual_actor_critic_final.pt')
    print(json.dumps({'status': 'complete', 'checkpoint': str(final), 'output_root': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
