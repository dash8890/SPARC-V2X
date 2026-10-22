import csv,json,re
from pathlib import Path

def collect(root,phase):
 out=[]
 for d in Path(root).glob('seed*_N*_B*'):
  m=re.match(r'seed(\d+)_N(\d+)_B(\d+)$',d.name); af=next(d.glob('episode_*/bridge_audit.json'),None)
  if not m or not af: continue
  a=json.loads(af.read_text()); W=100
  for wi,start in enumerate(range(0,len(a)-W+1,W)):
   x=a[start]; y=a[start+W-1]
   ds=float(y.get('safety_rx_cumulative') or 0)-float(x.get('safety_rx_cumulative') or 0)
   de=float(y.get('safety_error_cumulative') or 0)-float(x.get('safety_error_cumulative') or 0)
   den=ds+de
   if den<=0: continue
   prr=ds/den; seg=a[start:start+W]
   p95=max(float(z.get('p95_delay_ms') or 0) for z in seg)
   out.append({'phase':phase,'seed':int(m.group(1)),'density':int(m.group(2)),'byte_budget':int(m.group(3)),'window':wi,'start_epoch':start,'end_epoch':start+W-1,'rx':ds,'errors':de,'prr_window':prr,'p95_peak_proxy_ms':p95,'cbr_mean':sum(float(z.get('cbr') or 0) for z in seg)/W,'bytes_admitted':sum(float(z.get('computation_bytes_offered') or 0) for z in seg),'pending_final':float(y.get('wilab_pending_packets') or 0),'viol':int(prr<.95 or p95>50)})
 return out

cal=collect('results/wilab_windowed_calibration','calibration'); hold=collect('results/wilab_windowed_heldout','heldout'); allr=cal+hold
out=Path('results/wilab_windowed_calibration'); out.mkdir(exist_ok=True)
with (out/'windowed_safety_raw.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=allr[0].keys());w.writeheader();w.writerows(allr)
def grp(rs):
 d={}
 for r in rs:d.setdefault((r['phase'],r['byte_budget']),[]).append(r)
 return d
lines=['# WiLab 1-second windowed safety report','', 'Windows are nonoverlapping 100-epoch (1 s) windows after the 1 s warm-up. PRR is reconstructed by differencing cumulative WiLab success/error counters. The available bridge delay field is cumulative; therefore `p95_peak_proxy_ms` is the maximum per-epoch P95 in each window, a conservative proxy rather than an exact window-histogram percentile.','', '|Phase|Budget (B)|Windows|P_violation|Mean PRR|5th pct PRR|Max P95 proxy (ms)|','|---|---:|---:|---:|---:|---:|---:|']
for (ph,b),rs in sorted(grp(allr).items()):
 vals=sorted(r['prr_window'] for r in rs); q5=vals[max(0,int(.05*(len(vals)-1)))]
 lines.append(f"|{ph}|{b}|{len(rs)}|{sum(r['viol'] for r in rs)/len(rs):.4f}|{sum(r['prr_window'] for r in rs)/len(rs):.4f}|{q5:.4f}|{max(r['p95_peak_proxy_ms'] for r in rs):.1f}|")
lines += ['', 'The exact 1-s P95/P99 delay requires cumulative delay-histogram counters at each epoch. Those counters are not present in the retained runs, so no exact windowed delay claim is made.']
(out/'WILAB_WINDOWED_SAFETY_REPORT.md').write_text('\n'.join(lines))
print('windows',len(allr))
