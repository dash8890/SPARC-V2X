from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

root = Path('results/wilab_residual_actor_critic_stable_200ep')
rows = json.loads((root / 'training_metrics.json').read_text())
episodes = np.array([r['episode'] + 1 for r in rows])
rewards = np.array([r['return'] for r in rows], dtype=float)
window = 10
moving = np.convolve(rewards, np.ones(window) / window, mode='valid')
moving_x = episodes[window - 1:]

plt.rcParams.update({'font.size': 50, 'axes.linewidth': 3.0, 'lines.linewidth': 4.0})
fig, ax = plt.subplots(figsize=(20, 18))
ax.plot(episodes, rewards, color='#4472C4', marker='o', markersize=7,
        linewidth=3.0, alpha=0.45, label='Raw episodic reward')
ax.plot(moving_x, moving, color='#C00000', marker='o', markersize=9,
        linewidth=6.0, label='10-episode moving average')
ax.set_xlabel('Training episode', fontsize=50)
ax.set_ylabel('Completed-value episode return', fontsize=50)
ax.tick_params(axis='both', labelsize=42, width=3, length=10)
ax.set_xlim(1, 200)
ax.set_xticks(np.arange(0, 201, 25))
ax.grid(True, alpha=0.25, linewidth=2)
ax.legend(fontsize=50, loc='upper right', frameon=False)
fig.tight_layout()
out = root / 'fig_stable_training_rewards_200ep'
fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close(fig)
print(out.with_suffix('.png'))
print(out.with_suffix('.pdf'))
