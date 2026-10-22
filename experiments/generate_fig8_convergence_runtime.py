import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'final_figures_v2'; OUT.mkdir(parents=True,exist_ok=True)
FONT=55; TICK=49; LEG=44; LW=5; MS=16; FIG=(20,18)
COLORS={'return':'#1f77b4','ma':'#d62728','adv':'#2ca02c','runtime':'#1f77b4','p95':'#d62728'}
log=json.loads((ROOT/'results'/'residual_task_cf_training.json').read_text())
ep=np.array([x['episode'] for x in log]); ret=np.array([x['return'] for x in log]); adv=np.array([x['cf_mean'] for x in log])
def base(ax,xlab,ylab):
 ax.set_xlabel(xlab,fontsize=FONT); ax.set_ylabel(ylab,fontsize=FONT); ax.tick_params(axis='both',labelsize=TICK,width=2.5,length=10); ax.grid(True,alpha=.28,linewidth=1.5)
 for s in ax.spines.values(): s.set_linewidth(2.5)
def save(fig,stem):
 fig.tight_layout(); fig.savefig(OUT/(stem+'.png'),dpi=300,bbox_inches='tight'); fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight'); plt.close(fig)
# 8a raw return
fig,ax=plt.subplots(figsize=FIG); ax.plot(ep,ret,color=COLORS['return'],lw=LW); base(ax,'Training episode','Episode completed-value return'); save(fig,'fig8a_convergence_reward')
# 8b exact trailing 5-episode moving average
ma=np.convolve(ret,np.ones(5)/5,mode='valid'); fig,ax=plt.subplots(figsize=FIG); ax.plot(ep[4:],ma,color=COLORS['ma'],lw=LW); base(ax,'Training episode','Five-episode moving-average completed-value return'); save(fig,'fig8b_training_return_ma')
# 8c raw cf plus trailing five-update mean, zero line
adv_ma=np.convolve(adv,np.ones(5)/5,mode='valid'); fig,ax=plt.subplots(figsize=FIG); ax.plot(ep,adv,color=COLORS['adv'],lw=LW,alpha=.45,label='Raw counterfactual advantage'); ax.plot(ep[4:],adv_ma,color=COLORS['return'],lw=LW,label='Five-episode moving average'); ax.axhline(0,color='black',linestyle='--',lw=3,label='Zero'); base(ax,'Training episode','Sampled minus anchor counterfactual advantage'); ax.legend(fontsize=LEG,frameon=False); save(fig,'fig8c_cf_advantage')
# 8d runtime
runtime=json.loads((ROOT/'results'/'final_runtime.json').read_text())['by_density']; dens=np.array([x['density'] for x in runtime]); mean=np.array([x['mean_ms'] for x in runtime]); p95=np.array([x['p95_ms'] for x in runtime]); fig,ax=plt.subplots(figsize=FIG); ax.plot(dens,mean,color=COLORS['runtime'],marker='o',markersize=MS,lw=LW,label='Mean'); ax.plot(dens,p95,color=COLORS['p95'],marker='s',markersize=MS,lw=LW,label='P95'); ax.set_ylim(0, max(10.0, float(p95.max())) * 1.08); ax.axhline(10,color='black',linestyle='--',lw=3,label='10 ms control interval'); base(ax,'Vehicle density','Decision latency (ms)'); ax.legend(fontsize=LEG,frameon=False); save(fig,'fig8d_runtime')
with (OUT/'fig8_training_runtime_data.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['episode','return','cf_advantage']); w.writerows(zip(ep,ret,adv))

