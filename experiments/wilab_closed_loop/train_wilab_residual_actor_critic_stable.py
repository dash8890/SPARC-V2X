"""Stabilized WiLab residual actor-critic training.

This is an isolated training protocol. Existing checkpoints and results are
never overwritten. The run uses a fixed scenario pool, batched rollouts,
normalized GAE advantages, gradient clipping, and checkpoints containing the
optimizer plus RNG state so continuation is a real continuation.
"""
from __future__ import annotations
import argparse, json, os, random, shutil, tempfile
from pathlib import Path
import numpy as np
import torch

from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor


def terms(trace, gamma=0.99, lam=0.95):
    if not trace:
        z = torch.zeros(0)
        return z, z, z, z
    rewards = torch.tensor([float(x['reward']) for x in trace], dtype=torch.float32)
    values = torch.stack([x['value'] for x in trace]).float()
    adv = torch.zeros_like(rewards)
    gae = torch.tensor(0.0)
    next_v = torch.tensor(0.0)
    for t in reversed(range(len(trace))):
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_v = values[t].detach()
    return torch.stack([x['logp'] for x in trace]), values, adv.detach(), (adv + values.detach()).detach()


def save_checkpoint(path, policy, optimizer, rows, cfg, episode):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode': episode,
        'episodes': episode + 1,
        'rows': rows,
        'algorithm': 'WiLab anchored residual actor-critic, stabilized fixed-pool batched training',
        'environment': cfg,
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--density', type=int, default=35)
    ap.add_argument('--controller-epochs', type=int, default=20)
    ap.add_argument('--seed-pool', type=int, nargs='+', default=[909,1001,1111,1212,1313,1414,1515,1616])
    ap.add_argument('--task-seed-base', type=int, default=170000)
    ap.add_argument('--checkpoint-init', default='checkpoints/residual_task_policy_cf.pt')
    ap.add_argument('--output-root', default='results/wilab_residual_actor_critic_stable_200ep')
    ap.add_argument('--checkpoint-out', default='checkpoints/wilab_residual_actor_critic_stable_200ep.pt')
    ap.add_argument('--envelope-bytes', type=int, default=175000)
    ap.add_argument('--safety-payload-bytes', type=int, default=300)
    ap.add_argument('--simulation-time', type=float, default=12.0)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()
    out = Path(args.output_root)
    if out.exists() and (out / 'training_metrics.json').exists():
        raise RuntimeError(f'Output exists; choose a new output root: {out}')
    out.mkdir(parents=True, exist_ok=False)
    init = Path(args.checkpoint_init)
    payload = torch.load(init, map_location='cpu')
    policy = ResidualTaskPolicy(); policy.load_state_dict(payload['policy'], strict=False); policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    cfg = vars(args).copy()
    cfg.update({'algorithm': 'WiLab anchored residual actor-critic, stabilized fixed-pool batched training',
                'scenario_pool': args.seed_pool, 'fixed_pool': True, 'batch_update_rollouts': args.batch_size,
                'advantage': 'GAE gamma=0.99 lambda=0.95, batch-normalized',
                'gradient_clip_norm': 1.0, 'initialization': str(init),
                'optimizer_state_saved': True, 'counterfactual_replay': False})
    (out / 'training_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    rows, batch = [], []
    for ep in range(args.episodes):
        sim_seed = args.seed_pool[ep % len(args.seed_pool)]
        task_seed = args.task_seed_base + (ep % len(args.seed_pool))
        bridge = Path(tempfile.gettempdir()) / f'wilab_stable_ac_{os.getpid()}_{ep}'
        if bridge.exists(): shutil.rmtree(bridge)
        os.environ['WILAB_OUTPUT_ROOT'] = str(out)
        gov = SafetyGovernor(args.envelope_bytes, horizon_envelope_only=True)
        total, _, completed, trace = run_episode(
            ROOT, bridge, ep, args.density, policy, 900.0, False, False, 100.0, 20.0,
            sim_seed, args.controller_epochs, None, 3, False, None, args.safety_payload_bytes,
            args.simulation_time, 'residual', None, False, 'off', '', True, task_seed, gov,
            args.envelope_bytes)
        lp, vals, adv, rets = terms(trace); batch.append((lp, vals, adv, rets))
        row = {'episode': ep, 'sim_seed': sim_seed, 'task_seed': task_seed,
               'return': float(total), 'completed_value': float(completed),
               'decisions': len(trace), 'nonzero_reward_steps': sum(float(x['reward']) != 0 for x in trace),
               'deviation_fraction': sum(bool(x.get('residual_changed_anchor', False)) for x in trace) / len(trace) if trace else 0.0,
               'governor_interventions': sum(bool(x.get('governor_intervened', False)) for x in trace),
               'update_batch': None, 'actor_loss': None, 'critic_loss': None, 'gradient_norm': None}
        rows.append(row)
        if len(batch) >= args.batch_size:
            all_lp = torch.cat([x[0] for x in batch]); all_v = torch.cat([x[1] for x in batch])
            all_a = torch.cat([x[2] for x in batch]); all_r = torch.cat([x[3] for x in batch])
            norm_a = (all_a - all_a.mean()) / (all_a.std(unbiased=False) + 1e-6)
            actor = -(all_lp * norm_a.detach()).mean()
            critic = 0.5 * ((all_v - all_r.detach()) ** 2).mean()
            loss = actor + critic
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
            for r in rows[-len(batch):]:
                r.update({'actor_loss': float(actor.detach()), 'critic_loss': float(critic.detach()),
                          'gradient_norm': float(grad), 'update_batch': len(batch)})
            batch = []
        (out / 'training_metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        if (ep + 1) % 10 == 0:
            save_checkpoint(out / f'checkpoint_episode_{ep+1:04d}.pt', policy, optimizer, rows, cfg, ep)
        print(json.dumps(row), flush=True)
    if batch:
        print('warning: incomplete final batch not updated', flush=True)
    save_checkpoint(args.checkpoint_out, policy, optimizer, rows, cfg, args.episodes - 1)
    print(json.dumps({'status': 'complete', 'episodes': args.episodes,
                      'output_root': str(out), 'checkpoint': args.checkpoint_out}), flush=True)


if __name__ == '__main__':
    main()
