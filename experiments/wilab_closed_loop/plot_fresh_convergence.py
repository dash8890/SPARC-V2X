from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
root=Path('results/wilab_residual_actor_critic_fresh_convergence_100ep')
rows=json.loads((root/'training_metrics.json').read_text())
ep=np.array([r['episode']+1 for r in rows]); y=np.array([r['return'] for r in rows],float); w=10
ma=np.convolve(y,np.ones(w)/w,mode='valid'); mx=ep[w-1:]
plt.rcParams.update({'font.size':50,'axes.linewidth':3,'lines.linewidth':4})
fig,ax=plt.subplots(figsize=(20,18))
ax.plot(ep,y,color='#4472C4',marker='o',markersize=7,linewidth=3,alpha=.45,label='Raw episodic reward')
ax.plot(mx,ma,color='#C00000',marker='o',markersize=9,linewidth=6,label='10-episode moving average')
ax.set_xlabel('Training episode',fontsize=50); ax.set_ylabel('Completed-value episode return',fontsize=50)
ax.tick_params(axis='both',labelsize=42,width=3,length=10); ax.set_xlim(1,100); ax.set_xticks(np.arange(0,101,10)); ax.grid(True,alpha=.25,linewidth=2); ax.legend(fontsize=50,loc='upper right',frameon=False); fig.tight_layout()
out=root/'fig_fresh_convergence_100ep'; fig.savefig(out.with_suffix('.png'),dpi=200,bbox_inches='tight'); fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight'); print(out.with_suffix('.png')); print(out.with_suffix('.pdf'))
