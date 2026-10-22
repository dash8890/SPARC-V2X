"""WiLab action-conditioned computation-load calibration at the frozen point."""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; PY=sys.executable

def _float_or_nan(value):
    """Parse WiLab tabular fields; WiLab uses '-' for unavailable metrics."""
    try:
        text = str(value).strip()
        return float(text) if text and text not in {'-', '--', 'nan', 'NaN'} else float('nan')
    except (TypeError, ValueError):
        return float('nan')

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=[101,202]); ap.add_argument('--densities',nargs='+',type=int,default=[60,70,80,90]); ap.add_argument('--budgets',nargs='+',type=int,default=[0,1,2,3,4]); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--output-root',default='results/wilab_boundary_calibration')
 a=ap.parse_args(); base=ROOT/a.output_root; base.mkdir(parents=True,exist_ok=True); runner=ROOT/'experiments/wilab_closed_loop/train_wilab_closed_loop.py'
 jobs=[(s,n,b) for s in a.seeds for n in a.densities for b in a.budgets]
 def one(job):
  seed,n,b=job; tag=f'seed{seed}_N{n}_G100_A20_budget{b}'; out=base/tag; bridge=base/(tag+'_bridge'); out.mkdir(parents=True,exist_ok=True); bridge.mkdir(parents=True,exist_ok=True)
  env=os.environ.copy(); env['WILAB_OUTPUT_ROOT']=str(out); env['TEMP']=str(bridge); env['TMP']=str(bridge)
  result=base/(tag+'.json'); cmd=[PY,str(runner),'--episodes','1','--density',str(n),'--seed',str(seed),'--generation-ms','100','--allocation-ms','20','--controller-epochs',str(a.epochs),'--forced-budget',str(b),'--output-json',str(result)]
  print('START',tag,flush=True); p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); (out/'runner.log').write_text(p.stdout,encoding='utf-8')
  row={'seed':seed,'density':n,'budget':b,'returncode':p.returncode}; main=next(out.glob('episode_0000/MainOut.xls'),None)
  if main:
   lines=main.read_text(errors='ignore').splitlines(); h=lines[-2].split('\t'); d=lines[-1].split('\t'); m=dict(zip(h,d)); row['prr']=_float_or_nan(m.get('Packet reception ratio C-V2X','nan')); row['median_cbr']=_float_or_nan(m.get('Median CBR','nan'))
  audit=out/'episode_0000/bridge_audit.json'
  if audit.exists():
   x=json.loads(audit.read_text()); c=[_float_or_nan(z['cbr']) for z in x]; c=[v for v in c if v == v]; row.update(cbr_mean=sum(c)/len(c) if c else float('nan'),cbr_min=min(c) if c else float('nan'),cbr_max=max(c) if c else float('nan'),selected_mean=sum(z['num_selected'] for z in x)/len(x),safe_sets=sorted({tuple(z['safe_action_set']) for z in x}))
  print('DONE',tag,row,flush=True); return row
 rows=[]
 with ThreadPoolExecutor(max_workers=min(4,len(jobs))) as ex:
  for f in as_completed([ex.submit(one,j) for j in jobs]): rows.append(f.result()); (base/'progress.json').write_text(json.dumps(rows,indent=2))
 fields=sorted({k for r in rows for k in r});
 with (base/'wilab_computation_calibration.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 print('COMPLETE',len(rows),flush=True)
if __name__=='__main__': main()
