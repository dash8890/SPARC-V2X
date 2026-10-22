import json, math, statistics, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiments.eval_cap_sparc as ev
DENSITIES=(10,25,50,80,100,125,150,175,200); CAL=tuple(range(10000,10005)); EPS=.05; Z=1.645
def upper(p,n):
 den=1+Z*Z/n; ctr=(p+Z*Z/(2*n))/den; rad=Z*math.sqrt(max(0,p*(1-p)/n+Z*Z/(4*n*n)))/den; return min(1.,ctr+rad)
robust={}; details={}
for d in DENSITIES:
 cand=[]
 for cap in range(6):
  ev.CAPS[d]=cap; rows=[ev.run(d,s,'residual') for s in CAL]; n=20*len(rows); v=sum(round(x['P_viol']*20) for x in rows); p=v/n; u=upper(p,n); util=statistics.mean(x['R_value'] for x in rows); details.setdefault(str(d),[]).append({'cap':cap,'utility':util,'violations':v,'windows':n,'upper95':u})
  if u<=EPS:cand.append((util,cap,u))
 robust[d]=max(cand,key=lambda x:x[0])[1] if cand else 0
Path('cap_sparc_robust_calibration.json').write_text(json.dumps({'epsilon':EPS,'caps':robust,'details':details},indent=2)); print(json.dumps(robust,indent=2))
