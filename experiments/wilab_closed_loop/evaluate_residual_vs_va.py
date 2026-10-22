"""Held-out ordinary-rollout comparison: residual actor-critic vs Value-Aware."""
from __future__ import annotations
import argparse,json,os,shutil,tempfile,hashlib,datetime
from pathlib import Path
import torch
from train_wilab_closed_loop import run_episode, ROOT
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safety_governor import SafetyGovernor

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[909,1001]); ap.add_argument('--density',type=int,default=35); ap.add_argument('--epochs',type=int,default=1000); ap.add_argument('--simulation-time',type=float,default=12.0); ap.add_argument('--envelope-bytes',type=int,default=175000); ap.add_argument('--safety-payload-bytes',type=int,default=300); ap.add_argument('--horizon-envelope',action='store_true'); ap.add_argument('--output-root',default=None); ap.add_argument('--allow-overwrite',action='store_true'); args=ap.parse_args()
    checkpoint=Path(args.checkpoint).resolve()
    payload=checkpoint.read_bytes()
    checkpoint_sha256=hashlib.sha256(payload).hexdigest()
    loaded=torch.load(checkpoint,map_location='cpu')
    if args.output_root is None:
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out=Path('results') / f'wilab_eval_{checkpoint.stem}_{stamp}'
    else:
        out=Path(args.output_root)
    manifest_path=out/'run_manifest.json'
    if manifest_path.exists() and not args.allow_overwrite:
        old=json.loads(manifest_path.read_text())
        if old.get('checkpoint_sha256') != checkpoint_sha256:
            raise RuntimeError(f'Output directory already belongs to a different checkpoint: {out}. Use a new --output-root.')
    out.mkdir(parents=True,exist_ok=True)
    manifest={
        'schema_version': 1,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_metadata': {k:v for k,v in loaded.items() if k != 'policy' and isinstance(v,(str,int,float,bool,type(None),list,dict))},
        'evaluation': vars(args),
        'policy_class': 'ResidualTaskPolicy',
        'anchor': 'ValueAware',
        'environment': 'WiLabV2Xsim closed-loop',
    }
    manifest_path.write_text(json.dumps(manifest,indent=2,default=str))
    p=ResidualTaskPolicy(); p.load_state_dict(loaded['policy'],strict=False); p.eval(); rows=[]
    for seed in args.seeds:
      for mode in ('residual','anchor'):
        bridge=Path(tempfile.gettempdir())/f'wilab_eval_{mode}_{seed}';
        if bridge.exists(): shutil.rmtree(bridge)
        bridge.mkdir(parents=True); os.environ['WILAB_OUTPUT_ROOT']=str(out/mode/f'seed{seed}')
        total,_,completed,tr=run_episode(ROOT,bridge,0,args.density,p,600.0,False,False,100.0,20.0,seed,args.epochs,None,3,False,args.envelope_bytes,args.safety_payload_bytes,args.simulation_time,mode,None,True,'off','',True,seed,governor=SafetyGovernor(args.envelope_bytes, horizon_envelope_only=args.horizon_envelope))
        audit=json.loads((out/mode/f'seed{seed}'/'episode_0000'/'bridge_audit.json').read_text()); z=audit[-1]
        rows.append({'seed':seed,'method':'ResidualAC' if mode=='residual' else 'ValueAware','completed_value':float(completed),'return':float(total),'final_prr':z.get('safety_prr_cumulative'),'final_p95_ms':z.get('safety_p95_delay_cumulative_ms'),'safe':bool((z.get('safety_prr_cumulative') or 0)>=.95 and (z.get('safety_p95_delay_cumulative_ms') or 999)<=50),'envelope_bytes':args.envelope_bytes,'density':args.density,'epochs':args.epochs})
    diffs=[]
    for seed in args.seeds:
      r=next(x for x in rows if x['seed']==seed and x['method']=='ResidualAC'); a=next(x for x in rows if x['seed']==seed and x['method']=='ValueAware'); diffs.append({'seed':seed,'delta_value':r['completed_value']-a['completed_value'],'residual_safe':r['safe'],'anchor_safe':a['safe']})
    report={'rows':rows,'paired_differences':diffs,'mean_delta_value':sum(x['delta_value'] for x in diffs)/len(diffs),'residual_wins':sum(x['delta_value']>0 for x in diffs),'safety_pass':all(x['residual_safe'] and x['anchor_safe'] for x in diffs),'provenance':manifest}
    (out/'validation_results.json').write_text(json.dumps(report,indent=2)); (out/'validation_report.md').write_text('# Residual actor-critic versus Value-Aware validation\n\n'+json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
