"""Run isolated all-baseline WiLab campaigns for a specified checkpoint family."""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, os, shutil, tempfile, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch

from train_wilab_closed_loop import ROOT, run_episode
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor

SEEDS = [909, 1001, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 2020]
METHODS = ["minfinish", "task_rl", "edf", "random", "shield_greedy"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def run_one(task):
    args, seed, method = task
    ck = Path(args['checkpoint']).resolve()
    payload = torch.load(ck, map_location='cpu')
    torch.manual_seed(seed)
    policy = ResidualTaskPolicy()
    if method == 'task_rl':
        task_ck = Path(args['task_rl_checkpoint']).resolve()
        policy.load_state_dict(torch.load(task_ck, map_location='cpu')['policy'], strict=False)
    policy.eval()
    mode = method
    forced = None
    bridge = Path(tempfile.gettempdir()) / f"wilab_all_baselines_{args['label']}_{seed}_{method}"
    if bridge.exists(): shutil.rmtree(bridge)
    run_out = Path(args['output_root']) / method / f'seed{seed}'
    os.environ['WILAB_OUTPUT_ROOT'] = str(run_out)
    try:
        total, _, completed, trace = run_episode(ROOT, bridge, 0, args['density'], policy, 900.0, False, False, 100.0, 20.0, seed, args['epochs'], forced, 3, False, None, args['safety_payload_bytes'], args['simulation_time'], mode, None, False, 'off', '', True, seed, SafetyGovernor(args['envelope_bytes'], horizon_envelope_only=True), args['envelope_bytes'])
        audit = json.loads((run_out/'episode_0000'/'bridge_audit.json').read_text(encoding='utf-8'))
        last = audit[-1] if audit else {}
        return {'seed': seed, 'method': method, 'completed_value': float(completed), 'return': float(total),
                'final_prr': last.get('safety_prr_cumulative'), 'final_p95_ms': last.get('safety_p95_delay_cumulative_ms'),
                'safe': bool((last.get('safety_prr_cumulative') or 0) >= .95 and (last.get('safety_p95_delay_cumulative_ms') or 999) <= 50),
                'envelope_bytes': args['envelope_bytes'], 'density': args['density'], 'epochs': args['epochs'], 'status': 'complete'}
    except Exception as exc:
        return {'seed': seed, 'method': method, 'status': 'failed', 'error': repr(exc), 'envelope_bytes': args['envelope_bytes'], 'density': args['density'], 'epochs': args['epochs']}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--density', type=int, default=35)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--simulation-time', type=float, default=12.0)
    ap.add_argument('--envelope-bytes', type=int, default=175000)
    ap.add_argument('--safety-payload-bytes', type=int, default=300)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--task-rl-checkpoint', default='checkpoints/task_pair_policy.pt')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seeds', nargs='*', type=int, default=SEEDS)
    ap.add_argument('--methods', nargs='*', default=None)
    args = ap.parse_args()
    ck = Path(args.checkpoint).resolve(); payload = torch.load(ck, map_location='cpu')
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    selected_methods = METHODS if args.methods is None else [m for m in args.methods if m in METHODS]
    manifest = {'schema_version': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'label': args.label, 'checkpoint_path': str(ck), 'checkpoint_sha256': sha256(ck),
                'seeds': args.seeds, 'methods': selected_methods, 'evaluation': vars(args),
                'environment': 'WiLabV2Xsim closed-loop', 'safety_service': {'payload_bytes': args.safety_payload_bytes, 'generation_ms': 100, 'allocation_ms': 20}}
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows = []
    existing = out/'raw_rows.json'
    if existing.exists():
        rows = json.loads(existing.read_text(encoding='utf-8'))
    done = {(r.get('seed'), r.get('method')) for r in rows if r.get('status') == 'complete'}
    task_args = vars(args).copy(); task_args['methods'] = selected_methods
    tasks = [(task_args, seed, method) for seed in args.seeds for method in selected_methods if (seed, method) not in done]
    if args.dry_run:
        print(json.dumps({'status':'dry_run','methods':selected_methods,'seeds':args.seeds,'planned_runs':len(tasks),'workers':args.workers,'output_root':str(out)}, indent=2))
        return
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())
            with (out/'raw_rows.json').open('w', encoding='utf-8') as f: json.dump(rows, f, indent=2)
    diffs=[]
    for seed in args.seeds:
        by={r['method']:r for r in rows if r['seed']==seed and r.get('status')=='complete'}
        if 'residual' not in by: continue
        for method,r in by.items():
            if method != 'residual': diffs.append({'seed':seed,'method':method,'delta_value_residual_minus_baseline':r['completed_value']-by['residual']['completed_value'],'residual_safe':by['residual']['safe'],'baseline_safe':r['safe']})
    with (out/'paired_differences.json').open('w', encoding='utf-8') as f: json.dump(diffs, f, indent=2)
    with (out/'raw_rows.csv').open('w', newline='', encoding='utf-8') as f:
        keys=sorted({k for r in rows for k in r}); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(json.dumps({'status':'complete','rows':len(rows),'output_root':str(out)}, indent=2))

if __name__ == '__main__': main()
