import csv
from pathlib import Path
rows=list(csv.DictReader(open('results/wilab_byte_budget_calibration/byte_budget_summary.csv')))
rows=[r for r in rows if r['phase']=='calibration' and int(r['density'])==50]
levels=sorted(set(int(r['byte_budget']) for r in rows)); out=[]
def q(v,p):
 v=sorted(v); k=(len(v)-1)*p; lo=int(k); hi=min(len(v)-1,lo+1); return float(v[lo])+(float(v[hi])-float(v[lo]))*(k-lo)
for b in levels:
 rr=[float(r['wilab_mainout_prr']) for r in rows if int(r['byte_budget'])==b]; dd=[float(r['p95_delay_mean_ms']) for r in rows if int(r['byte_budget'])==b]
 out.append({'budget':b,'prr_mean':sum(rr)/len(rr),'prr_lower_05':q(rr,.05),'p95_max':max(dd),'n':len(rr),'qos_pass_fraction':sum(r['wilab_mainout_prr'] and float(r['wilab_mainout_prr'])>=.95 and float(r['p95_delay_mean_ms'])<=50 for r in rows if int(r['byte_budget'])==b)/len(rr)})
safe=[x for x in out if x['prr_lower_05']>=.95 and x['p95_max']<=50]; selected=max((x['budget'] for x in safe),default=0)
path=Path('results/wilab_byte_budget_calibration/WILAB_RUNLEVEL_SHIELD_REPORT.md')
lines=['# WiLab run-level byte-budget shield','', 'This shield uses authoritative run-level WiLab C-V2X PRR from `MainOut.xls` and the retained run-level P95 delay output. The ad hoc 1-s violation gate is not used. Calibration uses eight seeds at N=50.','', '|Budget (B)|Mean PRR|5th-percentile PRR|QoS pass fraction|Max P95 delay (ms)|Conservative safe|','|---:|---:|---:|---:|---:|:---:|']
for x in out: lines.append(f"|{x['budget']}|{x['prr_mean']:.4f}|{x['prr_lower_05']:.4f}|{x['qos_pass_fraction']:.3f}|{x['p95_max']:.1f}|{'yes' if x in safe else 'no'}|")
lines += ['',f'Largest conservative calibration-safe budget: **{selected} bytes**.','', 'Safety criteria: run-level PRR >= 0.95 and P95 delay <= 50 ms. Exact cumulative delay-histogram output is now instrumented for future reruns.','', 'The 1-s windowed violation results remain an internal diagnostic and are not used as the paper gate.']
path.write_text('\n'.join(lines)); print('selected',selected)
