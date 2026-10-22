import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
src=Path('results/residual_task_cf_training.json')
rows=json.loads(src.read_text())
ep=np.array([r['episode'] for r in rows]); reward=np.array([r['return'] for r in rows]); window=5
smooth=np.array([np.mean(reward[max(0,i-window+1):i+1]) for i in range(len(reward))])
out=Path('results/final_figures_v2'); out.mkdir(parents=True,exist_ok=True)
plt.figure(figsize=(20,18))
plt.plot(ep,reward,color='#1f77b4',linewidth=12,alpha=.35,marker='o',markersize=8,label='Episode reward')
plt.plot(ep,smooth,color='#d62728',linewidth=12,label='5-episode moving mean')
plt.xlabel('Training episode',fontsize=60); plt.ylabel('Completed-value reward',fontsize=60)
plt.xticks(fontsize=50); plt.yticks(fontsize=50); plt.grid(True,linewidth=3,alpha=.3); plt.legend(fontsize=45); plt.tight_layout()
plt.savefig(out/'fig8a_convergence_reward.png',dpi=220); plt.savefig(out/'fig8a_convergence_reward.pdf'); plt.close()
print('source',src); print(out/'fig8a_convergence_reward.png'); print(out/'fig8a_convergence_reward.pdf')

