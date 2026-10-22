"""Fit and evaluate a conservative monotone WiLab byte-budget shield."""
from __future__ import annotations
import csv,json,math
from pathlib import Path

CAL=[Path('results/wilab_byte_budget_calibration_300B_long1000'),Path('results/wilab_byte_budget_calibration_300B_long1000_rerun')]
HOLD=[Path('results/wilab_byte_budget_300B_long1000_heldout')]

def collect(roots):
    out=[]
    for root in roots:
      for d in root.glob('seed*_N50_B*'):
        if d.name.endswith('_bridge'): continue
        p=d/'episode_0000'/'bridge_audit.json'
        if not p.exists(): continue
        seed=int(d.name.split('_')[0][4:]); budget=int(d.name.split('_B')[1]); a=json.loads(p.read_text())
        for i in range(len(a)-1):
          x,y=a[i],a[i+1]
          out.append({'seed':seed,'budget':budget,'epoch':i,'cbr':x.get('cbr'),'prr_hist':x.get('safety_prr_cumulative',x.get('prr')),'p95_hist_ms':x.get('safety_p95_delay_cumulative_ms',x.get('p95_delay_ms')),'pending_packets':x.get('wilab_pending_packets'),'outcome_prr':y.get('safety_prr_cumulative',y.get('prr')),'outcome_p95_ms':y.get('safety_p95_delay_cumulative_ms',y.get('p95_delay_ms'))})
    return out
def q(v,p):
    v=sorted(v); k=(len(v)-1)*p; lo=int(math.floor(k)); hi=int(math.ceil(k)); return v[lo]+(v[hi]-v[lo])*(k-lo)
def main():
    cal=collect(CAL); hold=collect(HOLD)
    levels=sorted({int(r['budget']) for r in cal})
    table=[]; lo=1.; hi=0.
    for b in levels:
      z=[r for r in cal if int(r['budget'])==b]; pr=[float(r['outcome_prr']) for r in z]; de=[float(r['outcome_p95_ms']) for r in z]
      lo=min(lo,q(pr,.05)); hi=max(hi,q(de,.95))
      table.append({'budget':b,'n':len(z),'prr_median':q(pr,.5),'prr_lower_05':q(pr,.05),'monotone_prr_lower':lo,'p95_median_ms':q(de,.5),'p95_upper_95_ms':q(de,.95),'monotone_p95_upper_ms':hi,'safe':lo>=.95 and hi<=50})
    safe=[t['budget'] for t in table if t['safe']]; selected=max(safe) if safe else 0
    z=[r for r in hold if int(r['budget'])==selected]; violations=sum(not(float(r['outcome_prr'])>=.95 and float(r['outcome_p95_ms'])<=50) for r in z)
    out=Path('results/wilab_300B_conservative_shield'); out.mkdir(exist_ok=True)
    for name,data in [('calibration_rows.csv',cal),('heldout_rows.csv',hold),('shield_table.csv',table)]:
      with (out/name).open('w',newline='') as f:
        if data:
          w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    lines=['# WiLab-native conservative 300-B shield','',f'Calibration rows: {len(cal)} from eight seeds. Held-out rows: {len(hold)} from untouched seeds 909 and 1001 when available.','', '|Budget (B)|Rows|PRR median|PRR lower 5%|Monotone lower|P95 median (ms)|P95 upper 95% (ms)|Monotone upper|Safe|','|---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
    for t in table: lines.append(f"|{t['budget']}|{t['n']}|{t['prr_median']:.4f}|{t['prr_lower_05']:.4f}|{t['monotone_prr_lower']:.4f}|{t['p95_median_ms']:.1f}|{t['p95_upper_95_ms']:.1f}|{t['monotone_p95_upper_ms']:.1f}|{'yes' if t['safe'] else 'no'}|")
    lines += ['',f'Largest calibration-safe budget: **{selected} bytes**.',f'Held-out selected-budget rows: {len(z)}; violations: {violations}; empirical held-out violation rate: {(violations/len(z) if z else float("nan")):.4f}.','', 'The shield is a monotone empirical lookup over admitted bytes with a 5th-percentile PRR lower bound and 95th-percentile delay upper bound. It is fitted only from calibration trajectories.']
    (out/'WILAB_300B_CONSERVATIVE_SHIELD_REPORT.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if __name__=='__main__': main()
