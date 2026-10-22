# SPARC-V2X expanded figure package

## Purpose

This document records the reproducible figure package for the final SPARC-V2X results. It separates the paper evidence into three layers: Mode-2 safety, computation/resource pressure, and task/helper allocation by the final counterfactual residual RL controller.

Figures 1 and 2 are excluded from this generated package. They remain conceptual figures for the system architecture and residual-credit mechanism.

## Reproducibility

- Notebook source: `experiments/SPARC_V2X_FINAL_FIGURES_V2.ipynb`\n- Dedicated convergence/runtime notebook: `experiments/FIG8_CONVERGENCE_RUNTIME.ipynb`
- Executed notebook: `experiments/SPARC_V2X_FINAL_FIGURES_V2_EXECUTED.ipynb`
- Output directory: `results/final_figures_v2/`
- Tabular source: `results/final_figures_v2/figure_data.csv`
- Every plotting cell produces exactly one plot and writes both PNG and PDF.
- The setup cell exposes `FIGSIZE`, `FONTSIZE`, `LINEWIDTH`, markers, colors, and hatch settings. The current defaults are 20 by 18 inches, font size 50, and line width 6.

## Generated figures

| Figure | Files | Contents |
|---|---|---|
| Fig. 3a | `fig3a_performance.png`, `fig3a_performance.pdf` | Useful-value performance by density for SPARC, Value-Aware, Minimum-Finish, Task-pair RL, EDF, Random, and Shield-Greedy. |
| Fig. 3b | `fig3b_gain.png`, `fig3b_gain.pdf` | SPARC and Value-Aware improvement relative to the Minimum-Finish reference. |
| Fig. 4a | `fig4a_prr.png`, `fig4a_prr.pdf` | Safety PRR by density for SPARC and Value-Aware. |
| Fig. 4b | `fig4b_p95.png`, `fig4b_p95.pdf` | P95 safety delay by density. |
| Fig. 4c | `fig4c_p99.png`, `fig4c_p99.pdf` | P99 safety delay by density. |
| Fig. 4d | `fig4d_cbr.png`, `fig4d_cbr.pdf` | Mean CBR by density. |
| Fig. 7 | `fig7_anchor_deviation.png`, `fig7_anchor_deviation.pdf` | Fraction of decisions that deviate from the anchor policy by density. |
| Fig. 8a | `fig8a_convergence_reward.png`, `fig8a_convergence_reward.pdf` | Raw episode completed-value return during final counterfactual RL training. |
| Fig. 8b | `fig8b_training_return_ma.png`, `fig8b_training_return_ma.pdf` | Five-episode moving-average completed-value return. |
| Fig. 8c | `fig8c_cf_advantage.png`, `fig8c_cf_advantage.pdf` | Raw and five-update moving-average sampled-minus-anchor counterfactual advantage with zero reference. |\n| Fig. 8d | `fig8d_runtime.png`, `fig8d_runtime.pdf` | Mean and P95 decision latency by density with the 10 ms reference. |

## Data provenance

Static figure data are loaded from `figure_data.csv`. Training convergence data are loaded from `results/residual_task_cf_training.json`, using the persisted `episode`, `return`, and `cf_mean` fields. The reported final method is the counterfactual residual RL checkpoint (`checkpoints/residual_task_policy_cf.pt`); v2 is retained only as an internal development phase and is not presented as a final baseline.

## Interpretation limits

The figures support the following claims: useful-value performance, safety behavior, CBR response, anchor deviation, training trajectory, and runtime tradeoff across the evaluated densities. They do not support claims about an optimizer loss that was never logged, candidate-set size beyond the frozen traces, or historical load/budget sweeps unless those conditions are rerun with the final checkpoint. The plots should therefore be described as paired held-out evaluation evidence, with the safety shield and computation scheduler interpreted as separate components.

## Re-running

Open the executed notebook or run all cells in `SPARC_V2X_FINAL_FIGURES_V2.ipynb`. The notebook uses a non-interactive Matplotlib backend and writes each figure to the output directory. Because each plot is isolated in its own cell, individual figures can be regenerated without changing the remaining package.

