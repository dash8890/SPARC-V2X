"""Repeated WiLab validation without assuming deterministic counterfactual replay.

Each scenario seed keeps the same task workload (task_seed) while radio
realizations use independent simulator seeds.  Results are summarized by
policy and scenario; no run is treated as a causal twin of another run.
"""
from __future__ import annotations
import argparse, json, os, shutil, tempfile
from collections import Counter
from pathlib import Path
import torch
from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--replicates', type=int, default=5)
    ap.add_argument('--start-replicate', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--budget-bytes', type=int, default=175000)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--radio-event-log', action='store_true')
    args = ap.parse_args()
    if args.radio_event_log:
        os.environ['WILAB_RADIO_EVENT_LOGGING'] = '1'
    policy = ResidualTaskPolicy()
    policy.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['policy'], strict=False)
    policy.eval()
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario_seed in args.seeds:
        for rep in range(args.start_replicate, args.replicates):
            radio_seed = scenario_seed + 100000 * (rep + 1)
            for mode in ('residual', 'anchor'):
                run_dir = out / mode / f'seed{scenario_seed}' / f'rep{rep:02d}'
                bridge = Path(tempfile.gettempdir()) / f'wilab_repval_{mode}_{scenario_seed}_{rep}'
                if bridge.exists(): shutil.rmtree(bridge)
                os.environ['WILAB_OUTPUT_ROOT'] = str(run_dir)
                total, _, completed, trace = run_episode(
                    ROOT, bridge, 0, 50, policy, 600.0, False, False,
                    100.0, 20.0, radio_seed, args.epochs, None, 3, False,
                    args.budget_bytes, 300, 12.0, mode, None, True, 'off', '', True,
                    scenario_seed, governor=SafetyGovernor(args.budget_bytes))
                audit = json.loads((run_dir / 'episode_0000' / 'bridge_audit.json').read_text())
                z = audit[-1]
                helper_bytes = Counter(); source_bytes = Counter()
                for e in audit:
                    for h, b in (e.get('helper_payload_bytes') or {}).items(): helper_bytes[h] += int(b)
                    for a in (e.get('selected_assignments') or []): source_bytes[str(a.get('source_id'))] += int(a.get('payload_bytes', 0))
                rows.append({
                    'scenario_seed': scenario_seed, 'replicate': rep,
                    'radio_seed': radio_seed, 'method': 'ResidualAC' if mode == 'residual' else 'ValueAware',
                    'completed_value': float(completed), 'return': float(total),
                    'final_prr': z.get('safety_prr_cumulative'),
                    'final_p95_ms': z.get('safety_p95_delay_cumulative_ms'),
                    'safe': bool((z.get('safety_prr_cumulative') or 0) >= .95 and (z.get('safety_p95_delay_cumulative_ms') or 999) <= 50),
                    'mean_cbr': sum(float(e.get('cbr') or 0) for e in audit) / max(1, len(audit)),
                    'max_pending_packets': max(int(e.get('wilab_pending_packets') or 0) for e in audit),
                    'helper_payload_bytes': dict(helper_bytes),
                    'source_payload_bytes': dict(source_bytes),
                })
                print(rows[-1], flush=True)
    summary = {}
    for method in ('ResidualAC', 'ValueAware'):
        rr = [r for r in rows if r['method'] == method]
        summary[method] = {
            'n': len(rr), 'mean_value': sum(r['completed_value'] for r in rr)/len(rr),
            'safe_fraction': sum(r['safe'] for r in rr)/len(rr),
            'mean_prr': sum(r['final_prr'] for r in rr)/len(rr),
            'mean_p95_ms': sum(r['final_p95_ms'] for r in rr)/len(rr),
        }
    report = {'rows': rows, 'summary': summary, 'configuration': vars(args)}
    (out/'replicate_validation_results.json').write_text(json.dumps(report, indent=2))
    (out/'replicate_validation_report.md').write_text('# Repeated WiLab validation\n\n```json\n' + json.dumps(report, indent=2) + '\n```\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
