# Table 3: Cost and Deployment Complexity

| Method | Trainable Params | Checkpoint (KB) | Latency p50 (ms) | Latency p99 (ms) | < 10 ms Budget? | Training WiLab (ep) | Training Time | Tuning Budget |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| B1 | 0 | 0.0 KB | 0.003 | 0.004 | **YES** (Pass) | 0 | 0.0 min (analytic) | 12 val episodes (4 caps x 3 cells) |
| B2 | 0 | 0.0 KB | 0.006 | 0.012 | **YES** (Pass) | 0 | 0.0 min (analytic) | 9 val episodes (3 grids x 3 cells) |
| B3 | 0 | 0.0 KB | 0.069 | 0.142 | **YES** (Pass) | 0 | 0.0 min (analytic) | 9 val episodes (3 grids x 3 cells) |
| B4 | 6 | 0.5 KB | 0.009 | 0.013 | **YES** (Pass) | 0 | 0.0 min (analytic) | 9 val episodes (3 margins x 3 cells) |
| B5 | 2,307 | 13.6 KB | 0.113 | 0.148 | **YES** (Pass) | 96 | 64.5 min | Fixed budget (N_tune=96) |
| B5' | 13,830 | 60.1 KB | 0.602 | 0.915 | **YES** (Pass) | 232 | 161.7 min | 0 episodes (zero per-cell tuning) |
| B5'+shield | 13,830 | 60.1 KB | 0.604 | 0.891 | **YES** (Pass) | 232 | 161.7 min | 0 episodes (zero per-cell tuning) |
| PB-CRL-core | 13,830 | 60.1 KB | 0.728 | 1.312 | **YES** (Pass) | 232 | 161.7 min | 0 episodes (zero per-cell tuning) |

### Key Findings:
- **Real-time 10 ms Constraint:** All 8 methods comfortably satisfy the 5G Mode 2 10 ms decision deadline ($p_{99} \le 1.31\text{ ms} \ll 10.0\text{ ms}$).
- **Lightweight Edge Deployment:** The proposed PB-CRL-core / B5' architecture requires only 13,830 parameters (60.1 KB checkpoint), rendering it deployable on resource-constrained C-V2X On-Board Units (OBUs).
- **Zero Per-Cell Tuning:** While baselines B1–B4 require 9–12 validation episodes per scenario cell to grid-tune byte caps and safety thresholds, PB-CRL-core is trained once across randomized scenarios and requires zero per-cell calibration at inference time.
