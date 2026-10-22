"""Run WiLab state-conditioned aggregate-byte calibration episodes."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[101,202,303,404,505,606,707,808])
    ap.add_argument('--densities', nargs='+', type=int, default=[50,52,54,56,58])
    ap.add_argument('--byte-levels', nargs='+', type=int, default=[0,250000,500000,750000,1500000])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--output-root', default='results/wilab_byte_budget_calibration')
    ap.add_argument('--safety-payload-bytes', type=int, default=1000)
    ap.add_argument('--simulation-time', type=float, default=20.0)
    args = ap.parse_args()
    base = ROOT / args.output_root
    base.mkdir(parents=True, exist_ok=True)
    runner = ROOT / 'experiments/wilab_closed_loop/train_wilab_closed_loop.py'
    jobs = [(seed, density, level) for seed in args.seeds for density in args.densities for level in args.byte_levels]

    def one(job):
        seed, density, level = job
        tag = f'seed{seed}_N{density}_B{level}'
        out = base / tag
        bridge = base / (tag + '_bridge')
        out.mkdir(parents=True, exist_ok=True); bridge.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['WILAB_OUTPUT_ROOT'] = str(out)
        env['TEMP'] = str(bridge); env['TMP'] = str(bridge)
        result = out / 'run.json'
        cmd = [PY, str(runner), '--episodes','1','--density',str(density), '--seed',str(seed),
               '--generation-ms','100','--allocation-ms','20','--controller-epochs',str(args.epochs),
               '--forced-byte-budget',str(level),'--safety-payload-bytes',str(args.safety_payload_bytes),
               '--simulation-time',str(args.simulation_time),
               '--output-json',str(result)]
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        (out / 'runner.log').write_text(p.stdout, encoding='utf-8')
        return {'seed':seed,'density':density,'byte_budget':level,'returncode':p.returncode,
                'output':str(out)}

    rows=[]
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        futures=[ex.submit(one,j) for j in jobs]
        for f in as_completed(futures):
            row=f.result(); rows.append(row); print(row, flush=True)
            (base/'progress.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (base/'manifest.json').write_text(json.dumps({'args':vars(args),'runs':rows}, indent=2), encoding='utf-8')
    print(f'COMPLETE {len(rows)}', flush=True)

if __name__ == '__main__':
    main()
