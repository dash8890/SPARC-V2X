"""Fig.3a/b from extended SPARC-N (144 eps). Does NOT overwrite 96-ep fig03a/b files."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS, LW = 50, 8
OUT = Path("sparc_performance_evaluation/figures/fig03_training")
ROOT144 = Path("results/wilab_direct_ppo_144ep")
EVAL = ROOT144 / "fixed_evaluation.json"
BEST = ROOT144 / "best_checkpoint.json"
MET = ROOT144 / "training_metrics.json"
OUT.mkdir(parents=True, exist_ok=True)


def mean_band(xs):
    xs = list(xs)
    n = len(xs)
    m = mean(xs)
    if n < 2:
        return m, m, m, n, 0.0
    se = stdev(xs) / math.sqrt(n)
    return m, m - se, m + se, n, se


# --- 3(a) ---
evals = json.loads(EVAL.read_text(encoding="utf-8-sig"))
by_u = defaultdict(list)
for e in evals:
    if e.get("method") in ("residual", "direct"):
        by_u[int(e["update"])].append(float(e["completed_value"]))
updates = sorted(by_u)
means, los, his, ns, series = [], [], [], [], []
for u in updates:
    m, lo, hi, n, se = mean_band(by_u[u])
    means.append(m)
    los.append(lo)
    his.append(hi)
    ns.append(n)
    series.append(
        {"update": u, "mean_V": m, "sem_lo": lo, "sem_hi": hi, "sem": se, "n": n, "values": by_u[u]}
    )
selected_u = int(json.loads(BEST.read_text(encoding="utf-8-sig"))["update"])
(OUT / "fig03a_validation_V_absolute_144ep.json").write_text(
    json.dumps(
        {
            "source": str(EVAL).replace("\\", "/"),
            "episodes": 144,
            "y": "mean held-out validation V_done (SPARC-N extended)",
            "band": "mean ± 1·SEM",
            "selected_checkpoint_update": selected_u,
            "series": series,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

fig, ax = plt.subplots(figsize=(20, 18))
ax.fill_between(updates, los, his, color="#1f4e79", alpha=0.22, linewidth=0, label="±SEM")
ax.plot(updates, means, color="#1f4e79", linewidth=LW, marker="o", markersize=16, label="Mean")
ax.axvline(selected_u, color="0.55", linewidth=2, linestyle=":")
ax.set_title("Validation (144 episodes)", fontsize=FS)
ax.set_xlabel("PPO Update", fontsize=FS)
ax.set_ylabel(r"Held-out Validation $V_{\mathrm{done}}$", fontsize=FS)
ax.set_xticks(updates)
ax.set_ylim(25, 65)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
png = OUT / "fig03a_validation_V_144ep.png"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig03a_validation_V_144ep.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", png)

# --- 3(b) ---
rows = json.loads(MET.read_text(encoding="utf-8"))
ep = np.array([r["episode"] for r in rows], dtype=float)
ret = np.array([r["return"] for r in rows], dtype=float)
w = 32
kernel = np.ones(w) / w
smooth = np.convolve(ret, kernel, mode="same")
half = w // 2
for i in range(half):
    smooth[i] = ret[: i + half + 1].mean()
    smooth[-(i + 1)] = ret[-(i + half + 1) :].mean()

fig, ax = plt.subplots(figsize=(20, 18))
ax.plot(ep, smooth, color="#c45c26", linewidth=LW, label="Return")
ax.set_xlabel("Episode", fontsize=FS)
ax.set_ylabel("Return", fontsize=FS)
ax.set_title("Episode return (144 episodes)", fontsize=FS)
xmax = float(ep.max())
ax.set_xlim(0, xmax)
ticks = list(np.arange(0, xmax, 16))
if not ticks or ticks[-1] != xmax:
    ticks.append(xmax)
ax.set_xticks(ticks)
ax.tick_params(labelsize=FS)
ax.legend(fontsize=FS, frameon=False, loc="best")
ax.grid(True, alpha=0.3)
fig.tight_layout()
png = OUT / "fig03b_episode_return_144ep.png"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig03b_episode_return_144ep.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", png)
print("n_metrics", len(rows), "n_eval_updates", len(updates), "selected_u", selected_u)
