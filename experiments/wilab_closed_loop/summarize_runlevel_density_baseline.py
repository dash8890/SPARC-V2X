import csv,re,json
from pathlib import Path
rows=[]
for d in Path('results/wilab_runlevel_density_baseline').glob('seed*_N*_B0'):
 m=re.match(r'seed(\d+)_N(\d+)_B0$',d.name); f=next(d.glob('episode_*/bridge_audit.json'),None)
 if not m or not f: continue
 a=json.loads(f.read_text()); last=a[-1]
 prr=float(last.get('safety_prr_cumulative') or last.get('prr') or 0); p95=float(last.get('safety_p95_delay_cumulative_ms') or last.get('p95_delay_ms') or 0)
 rows.append({'seed':int(m.group(1)),'density':int(m.group(2)),'prr':prr,'p95_delay_ms':p95,'qos_safe':int(prr>=.95 and p95<=50)})
out=Path('results/wilab_runlevel_density_baseline')
with (out/'runlevel_density_baseline.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
lines=['# WiLab zero-load run-level density feasibility','', 'Frozen configuration: 100-ms safety generation, 20-ms allocation, payload-faithful radio path, 300 controller epochs, corrected cumulative WiLab PRR, and cumulative delay-histogram P95 instrumentation.','', '|Density|Seeds|Safe seeds|Safe fraction|Mean PRR|Minimum PRR|Maximum exact P95 (ms)|','|---:|---:|---:|---:|---:|---:|---:|']
for n in sorted(set(r['density'] for r in rows)):
 x=[r for r in rows if r['density']==n]; lines.append(f"|{n}|{len(x)}|{sum(r['qos_safe'] for r in x)}|{sum(r['qos_safe'] for r in x)/len(x):.3f}|{sum(r['prr'] for r in x)/len(x):.4f}|{min(r['prr'] for r in x):.4f}|{max(r['p95_delay_ms'] for r in x):.1f}|")
lines += ['', 'Selection rule: the authoritative density must have all eight calibration seeds satisfying PRR >= 0.95 and exact P95 delay <= 50 ms.']
(out/'WILAB_RUNLEVEL_DENSITY_BASELINE_REPORT.md').write_text('\n'.join(lines)); print('rows',len(rows))
