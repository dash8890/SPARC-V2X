import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'final_figures_v2'; OUT.mkdir(parents=True,exist_ok=True)
D=[10,25,50,80,100,125,150,175,200]
mapm={'residual_rl':'SPARC-V2X','value':'Value-Aware','minfinish':'Minimum-Finish','task_rl':'Task-RL'}
styles={'SPARC-V2X':('#1f77b4','o'),'Value-Aware':('#d62728','s'),'Minimum-Finish':('#2ca02c','^'),'Task-RL':('#9467bd','D')}
rows=json.loads((ROOT/'results'/'task_aware_full_kpis_10seed.json').read_text())['summary']; lookup={(r['method'],r['density']):r for r in rows}
missing=[]
for m in mapm:
 for d in D:
  r=lookup.get((m,d))
  if r is None: missing.append((mapm[m],d,'row'))
  else:
   for k in ['R_value','decision_ms_mean','decision_ms_p95']:
    if k not in r or r[k] is None: missing.append((mapm[m],d,k))
if missing: raise RuntimeError('Missing required KPI values: '+repr(missing))
vals={label:{'R_value':[lookup[(m,d)]['R_value'] for d in D],'mean':[lookup[(m,d)]['decision_ms_mean'] for d in D],'p95':[lookup[(m,d)]['decision_ms_p95'] for d in D]} for m,label in mapm.items()}
with (OUT/'fig9_data.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['density','method','R_value','mean_latency_ms','p95_latency_ms'])
 for label in vals:
  for i,d in enumerate(D): w.writerow([d,label,vals[label]['R_value'][i],vals[label]['mean'][i],vals[label]['p95'][i]])
FONT=55; TICK=49; LEG=40; LW=5; MS=16; FIG=(20,18)
def base(ax,xlab,ylab):
 ax.set_xlabel(xlab,fontsize=FONT); ax.set_ylabel(ylab,fontsize=FONT); ax.tick_params(axis='both',labelsize=TICK,width=2.5,length=10); ax.grid(True,alpha=.28,linewidth=1.5)
 for s in ax.spines.values(): s.set_linewidth(2.5)
def save(fig,stem):
 fig.tight_layout(); fig.savefig(OUT/(stem+'.png'),dpi=300,bbox_inches='tight'); fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight'); plt.close(fig)
fig,ax=plt.subplots(figsize=FIG)
for label in ['SPARC-V2X','Value-Aware','Minimum-Finish','Task-RL']:
 c,m=styles[label]; ax.plot(D,vals[label]['mean'],color=c,marker=m,markersize=MS,lw=LW,ls='-',label=f'{label} mean'); ax.plot(D,vals[label]['p95'],color=c,marker=m,markersize=MS,lw=LW,ls='--',label=f'{label} P95')
ax.axhline(10,color='black',ls=':',lw=3,label='10-ms decision interval'); base(ax,'Vehicle density','Decision latency (ms)'); ax.legend(fontsize=LEG,frameon=False,ncol=2); save(fig,'fig9a_runtime_comparison')
fig,ax=plt.subplots(figsize=FIG)
for label in ['SPARC-V2X','Value-Aware','Minimum-Finish','Task-RL']:
 c,m=styles[label]; ax.plot(vals[label]['mean'],vals[label]['R_value'],color=c,marker=m,markersize=MS,lw=LW,ls='-',label=label)
base(ax,'Mean decision latency (ms)','Value-weighted completion ratio, $R_{value}$'); ax.legend(fontsize=LEG,frameon=False); save(fig,'fig9b_utility_runtime_tradeoff')
