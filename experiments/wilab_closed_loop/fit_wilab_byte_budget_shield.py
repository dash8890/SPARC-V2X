import csv,json,re,math
from pathlib import Path
from collections import defaultdict

def rows(root, phase):
    out=[]
    for d in Path(root).glob('seed*_N*_B*'):
        m=re.match(r'seed(\d+)_N(\d+)_B(\d+)$',d.name)
        fs=list(d.glob('episode_*/bridge_audit.json'))
        if not m or not fs: continue
        a=json.loads(fs[0].read_text())
        for i in range(len(a)-1):
            x=a[i]; y=a[i+1]
            out.append({'phase':phase,'seed':int(m.group(1)),'density':int(m.group(2)),'budget':int(m.group(3)),'epoch':i,
                'cbr':float(x.get('cbr',0)),'prr_hist':float(x.get('prr',0)),'p95_hist_ms':float(x.get('p95_delay_ms',0)),
                'pending_packets':float(x.get('wilab_pending_packets',0)),'outcome_prr':float(y.get('prr',0)),
                'outcome_p95_ms':float(y.get('p95_delay_ms',0))})
    return out
cal=rows('results/wilab_epoch_metric_calibration','calibration'); hold=rows('results/wilab_epoch_metric_heldout','heldout')
def q(v,p):
    v=sorted(v)
    if not v:return float('nan')
    k=(len(v)-1)*p; lo=int(math.floor(k)); hi=int(math.ceil(k))
    return v[lo]+(v[hi]-v[lo])*(k-lo)
by=defaultdict(list)
for r in cal: by[r['budget']].append(r)
levels=sorted(by)
table=[]
for b in levels:
    rr=[r['outcome_prr'] for r in by[b]]; dd=[r['outcome_p95_ms'] for r in by[b]]
    table.append({'budget':b,'prr_median':q(rr,.5),'prr_lower_05':q(rr,.05),'p95_median_ms':q(dd,.5),'p95_upper_95_ms':q(dd,.95),'n':len(rr)})
# monotone conservative envelope
minp=1.0; maxd=0.0
for t in table:
    minp=min(minp,t['prr_lower_05']); t['prr_lower_monotone']=minp
    maxd=max(maxd,t['p95_upper_95_ms']); t['p95_upper_monotone_ms']=maxd
def safe(b):
    t=next(x for x in table if x['budget']==b)
    return t['prr_lower_monotone']>=.95 and t['p95_upper_monotone_ms']<=50
safe_levels=[b for b in levels if safe(b)]
predmax=max(safe_levels) if safe_levels else 0
viol=0; eligible=0; conservative=0
for r in hold:
    if r['budget']!=predmax: continue
    eligible+=1
    actualsafe=r['outcome_prr']>=.95 and r['outcome_p95_ms']<=50
    if not actualsafe: viol+=1
for r in hold:
    if r['outcome_prr']>=.95 and r['outcome_p95_ms']<=50 and r['budget']>predmax: conservative+=1
out=Path('results/wilab_epoch_metric_calibration'); out.mkdir(exist_ok=True)
with (out/'wilab_byte_budget_shield_table.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=table[0].keys()); w.writeheader();w.writerows(table)
with (out/'wilab_per_epoch_calibration_rows.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cal[0].keys());w.writeheader();w.writerows(cal)
with (out/'wilab_per_epoch_heldout_rows.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=hold[0].keys());w.writeheader();w.writerows(hold)
lines=['# WiLab-native byte-budget shield fit','',f'Calibration rows: {len(cal)} from 8 seeds; held-out rows: {len(hold)} from 2 seeds.','', 'The target is the corrected cumulative WiLab PRR carried by the controller-facing radio metric, with the next controller epoch used as the post-action outcome. The estimator is an empirical monotone lookup over admitted computation bytes. Lower PRR bounds use the 5th calibration percentile; delay upper bounds use the 95th calibration percentile.','', '## Fitted table','', '|Budget (B)|PRR median|PRR lower 5%|Monotone lower PRR|P95 median (ms)|P95 upper 95% (ms)|Monotone upper (ms)|Safe|','|---:|---:|---:|---:|---:|---:|---:|:---:|']
for t in table: lines.append(f"|{t['budget']}|{t['prr_median']:.4f}|{t['prr_lower_05']:.4f}|{t['prr_lower_monotone']:.4f}|{t['p95_median_ms']:.2f}|{t['p95_upper_95_ms']:.2f}|{t['p95_upper_monotone_ms']:.2f}|{'yes' if safe(t['budget']) else 'no'}|")
lines += ['',f'Largest calibration-safe budget: **{predmax} bytes**.', '', '## Held-out check','',f'Rows at selected budget: {eligible}; safety violations among those rows: {viol}; empirical held-out violation rate: {(viol/eligible if eligible else float("nan")):.4f}.',f'Conservative exclusions of otherwise-safe higher-budget rows: {conservative}.','', '## Caveat','', 'This first shield fit is intentionally conservative and byte-budget conditioned. The per-epoch dataset now contains the causal radio-state fields (CBR, historical PRR/P95, pending computation backlog) needed for a richer state-conditioned model; the present lookup isolates the measured budget boundary without fitting a high-capacity predictor.']
(out/'WILAB_BYTE_BUDGET_SHIELD_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('safe_levels',safe_levels,'selected',predmax,'heldout_rows',len(hold),'violations',viol)
