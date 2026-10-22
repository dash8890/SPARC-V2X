import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
cap=json.loads(Path('cap_sparc_training_v2.json').read_text()); raw=json.loads(Path('results/residual_task_cf_training.json').read_text())
def series(rows):
 e=np.array([r['episode'] for r in rows]); y=np.array([r['return'] for r in rows]); w=5; ma=np.array([np.mean(y[max(0,i-w+1):i+1]) for i in range(len(y))]); return e,y,ma
ec,yc,mc=series(cap); er,yr,mr=series(raw)
plt.figure(figsize=(20,18))
plt.plot(ec,yc,color='#1f77b4',alpha=.25,linewidth=8,marker='o',markersize=6,label='Cap training reward')
plt.plot(ec,mc,color='#1f77b4',linewidth=10,label='Cap 5-episode mean')
plt.plot(er,yr,color='#d62728',alpha=.25,linewidth=8,marker='s',markersize=6,label='Raw residual-RL reward')
plt.plot(er,mr,color='#d62728',linewidth=10,label='Raw residual-RL 5-episode mean')
plt.xlabel('Training episode',fontsize=60); plt.ylabel('Completed-value reward',fontsize=60)
plt.xticks(fontsize=50); plt.yticks(fontsize=50); plt.grid(True,linewidth=3,alpha=.3); plt.legend(fontsize=40); plt.tight_layout()
out=Path('results/final_figures_v2'); plt.savefig(out/'cap_vs_raw_reward_convergence.png',dpi=220); plt.savefig(out/'cap_vs_raw_reward_convergence.pdf'); plt.close()
print(out/'cap_vs_raw_reward_convergence.png'); print(out/'cap_vs_raw_reward_convergence.pdf')
