import csv,json,re
from pathlib import Path
def collect(root):
 out=[]
 for d in Path(root).glob('seed*_N*_B0'):
  m=re.match(r'seed(\d+)_N(\d+)_B0$',d.name); af=next(d.glob('episode_*/bridge_audit.json'),None)
  if not m or not af: continue
  a=json.loads(af.read_text()); W=100
  for wi,start in enumerate(range(0,len(a)-W+1,W)):
   x=a[start]; y=a[start+W-1]; ds=float(y.get('safety_rx_cumulative') or 0)-float(x.get('safety_rx_cumulative') or 0); de=float(y.get('safety_error_cumulative') or 0)-float(x.get('safety_error_cumulative') or 0); den=ds+de
   if den<=0: continue
   seg=a[start:start+W]; prr=ds/den; p95=max(float(z.get('p95_delay_ms') or 0) for z in seg)
   out.append({'seed':int(m.group(1)),'density':int(m.group(2)),'window':wi,'prr_window':prr,'p95_peak_proxy_ms':p95,'viol':int(prr<.95 or p95>50)})
 return out
rows=[]
for root in ['results/wilab_density_window_calibration','results/wilab_density_window_calibration_retry','results/wilab_density_window_calibration_retry2']:
 rows += collect(root)
d={}
for r in rows: d[(r['seed'],r['density'],r['window'])]=r
rows=list(d.values()); out=Path('results/wilab_density_window_calibration'); out.mkdir(exist_ok=True)
with (out/'density_window_safety_raw.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(sorted(rows,key=lambda r:(r['density'],r['seed'],r['window'])))
g={}
for r in rows:g.setdefault(r['density'],[]).append(r)
lines=['# WiLab zero-computation density feasibility calibration','', 'Frozen configuration: 100-ms safety generation, 20-ms allocation, payload-faithful computation path, 300 post-warm-up controller epochs, nonoverlapping 1-s windows. The delay field is a conservative peak-P95 proxy; PRR is reconstructed from cumulative WiLab success/error counters.','', '|Density|Seeds/windows|P_violation|Mean PRR|5th pct PRR|Max P95 proxy (ms)|','|---:|---:|---:|---:|---:|---:|']
for n,rs in sorted(g.items()):
 vals=sorted(r['prr_window'] for r in rs); q5=vals[max(0,int(.05*(len(vals)-1)))]
 lines.append(f"|{n}|{len(rs)}|{sum(r['viol'] for r in rs)/len(rs):.4f}|{sum(r['prr_window'] for r in rs)/len(rs):.4f}|{q5:.4f}|{max(r['p95_peak_proxy_ms'] for r in rs):.1f}|")
lines += ['', 'Selection rule: choose the highest density whose calibration-window violation rate is at most 0.05. Positive-byte probing is performed only after this zero-byte density gate.','', 'Exact windowed P95/P99 requires cumulative delay-histogram counters, which are not yet retained by the bridge.']
(out/'WILAB_DENSITY_WINDOW_CALIBRATION_REPORT.md').write_text('\n'.join(lines))
print('rows',len(rows))
