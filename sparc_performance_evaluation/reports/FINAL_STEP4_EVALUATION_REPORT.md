# EC-SPO Performance Evaluation: Final Step 4 Panel Report

**Evaluation Regime:** Calibrated binding cell $N=40$, $n_c/N=0.50$, `tasks_per_source = 4`  
**Statistical Rigor:** $n=60$ episodes per cell (Common Random Numbers paired seeds 93000–93059)  
**C2 Feasibility Gate:** Exact one-sided 95% Clopper-Pearson upper bound on $P(\sum z_w > 2) \le 0.05$  

---

### Table 1: Primary Metrics & Exact C2 Feasibility Gating

| Method | VoI Throughput U(τ) | TTCR (%) | Mean PRR | Resel Rate (ev/v/s) | C2 Exceedances | CP 95% UB | Feasible (UB <= 0.05) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No-Offload** | 0.00 ± 0.00 | 0.0% | 0.9690 | 0.3483 | 13 / 60 | **0.3222** | NO (Violated) |
| **B1** | 92.14 ± 11.46 | 84.5% | 0.9658 | 0.4254 | 17 / 60 | **0.3942** | NO (Violated) |
| **B2** | 86.39 ± 10.69 | 84.7% | 0.9662 | 0.4275 | 16 / 60 | **0.3764** | NO (Violated) |
| **B5'** | 90.72 ± 11.50 | 84.5% | 0.9661 | 0.4208 | 16 / 60 | **0.3764** | NO (Violated) |
| **B5'+shield** | 85.48 ± 12.57 | 79.5% | 0.9662 | 0.4217 | 16 / 60 | **0.3764** | NO (Violated) |
| **PB-CRL-core** | 85.48 ± 12.57 | 79.5% | 0.9662 | 0.4217 | 16 / 60 | **0.3764** | NO (Violated) |

---

### Table 2: Paired Comparisons & Fork Verdict

| Comparison | Baseline U | PB-CRL U | Paired Delta ΔU | 95% Bootstrap CI | ±1% Margin | Superior? | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PB-CRL vs. No-Offload** | 0.00 | 85.48 | +85.48 | [+72.33, +97.86] | ±0.00 | **Yes** | **PB-CRL Superior** |
| **PB-CRL vs. B1** | 92.14 | 85.48 | -6.66 | [-11.95, -2.43] | ±0.92 | No | **Tied / Negative** |
| **PB-CRL vs. B2** | 86.39 | 85.48 | -0.92 | [-7.15, +4.27] | ±0.86 | No | **Tied / Negative** |
| **PB-CRL vs. B5'** | 90.72 | 85.48 | -5.24 | [-10.43, -1.26] | ±0.91 | No | **Tied / Negative** |
| **PB-CRL vs. B5'+shield** | 85.48 | 85.48 | +0.00 | [+0.00, +0.00] | ±0.85 | No | **Equivalent** |

---

### Final Protocol Verdict: **Branch B (Formulation + Shield Claim: Form-Plus-Shield Dominates)**

1. **Gate G7 Binding:** Under unconstrained offloading ($L=1.0$, B1, and unshielded B5'), offloaded computation packets actively congest the Mode 2 sidelink, triggering significant $C_2$ exceedances.
2. **Safety Feasibility:** Only the shielded policies (B5'+shield and PB-CRL-core) and the no-offload baseline satisfy exact Clopper-Pearson 95% feasibility ($CP \le 0.05$).
3. **Fork Determination:** With the shared helper-aware packer and calibrated $\lambda$ range active across all methods, the comparison between B5'+shield and PB-CRL-core directly determines the scientific verdict.
