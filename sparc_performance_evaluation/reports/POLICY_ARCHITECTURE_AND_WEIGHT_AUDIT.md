# PB-CRL-core vs. B5′ vs. B5 Architecture & Checkpoint Audit

## 1. Available Checkpoints in Repository

### Checkpoint: `b5_prime_best_preserved` (`b5_prime_best_preserved_43p86.pt`)
- **File size:** 61,540 bytes
- **Top-level keys:** `['policy', 'lam_L', 'round', 'test_mean_V']`
- **Parameter count:** 13,830 parameters
- **Learned dual bound (lam_L):** `0.0`
- **Validation Score:** `43.68728935646712`
- **Weight Tensors:**
  - `log_std_lam`: shape [], mean=-1.0040, std=nan
  - `log_std_beta`: shape [], mean=-0.9960, std=nan
  - `net.0.weight`: shape [64, 16], mean=0.0009, std=0.1437
  - `net.0.bias`: shape [64], mean=0.0021, std=0.1542
  - `net.2.weight`: shape [64, 64], mean=-0.0014, std=0.0726
  - `net.2.bias`: shape [64], mean=0.0092, std=0.0751
  - `mu_lam.weight`: shape [1, 64], mean=0.0000, std=0.0039
  - `mu_lam.bias`: shape [1], mean=-4.0040, std=nan
  - `mu_beta.weight`: shape [1, 64], mean=0.0001, std=0.0039
  - `mu_beta.bias`: shape [1], mean=4.0040, std=nan
  - `value_head.0.weight`: shape [64, 64], mean=0.0013, std=0.0722
  - `value_head.0.bias`: shape [64], mean=-0.0084, std=0.0676
  - `value_head.2.weight`: shape [1, 64], mean=-0.0138, std=0.0748
  - `value_head.2.bias`: shape [1], mean=-0.0193, std=nan
  - `cost_head.0.weight`: shape [64, 64], mean=-0.0001, std=0.0716
  - `cost_head.0.bias`: shape [64], mean=-0.0057, std=0.0768
  - `cost_head.2.weight`: shape [1, 64], mean=0.0001, std=0.0711
  - `cost_head.2.bias`: shape [1], mean=-0.1086, std=nan

### Checkpoint: `b5_prime_best` (`b5_prime_best.pt`)
- **File size:** 61,540 bytes
- **Top-level keys:** `['policy', 'lam_L', 'round', 'test_mean_V']`
- **Parameter count:** 13,830 parameters
- **Learned dual bound (lam_L):** `0.04`
- **Validation Score:** `43.202770295706785`
- **Weight Tensors:**
  - `log_std_lam`: shape [], mean=-1.0504, std=nan
  - `log_std_beta`: shape [], mean=-0.9288, std=nan
  - `net.0.weight`: shape [64, 16], mean=-0.0028, std=0.1509
  - `net.0.bias`: shape [64], mean=0.0009, std=0.2193
  - `net.2.weight`: shape [64, 64], mean=0.0012, std=0.1068
  - `net.2.bias`: shape [64], mean=0.0030, std=0.1164
  - `mu_lam.weight`: shape [1, 64], mean=0.0016, std=0.0104
  - `mu_lam.bias`: shape [1], mean=0.9982, std=nan
  - `mu_beta.weight`: shape [1, 64], mean=0.0017, std=0.0113
  - `mu_beta.bias`: shape [1], mean=-0.5915, std=nan
  - `value_head.0.weight`: shape [64, 64], mean=0.0010, std=0.1134
  - `value_head.0.bias`: shape [64], mean=0.0139, std=0.0889
  - `value_head.2.weight`: shape [1, 64], mean=0.0753, std=0.5696
  - `value_head.2.bias`: shape [1], mean=0.3187, std=nan
  - `cost_head.0.weight`: shape [64, 64], mean=-0.0017, std=0.0781
  - `cost_head.0.bias`: shape [64], mean=-0.0064, std=0.0856
  - `cost_head.2.weight`: shape [1, 64], mean=0.0099, std=0.0477
  - `cost_head.2.bias`: shape [1], mean=-0.0501, std=nan

### Checkpoint: `b5_legacy_smoke` (`b5_smoke_checkpoint.pt`)
- **File size:** 13,938 bytes
- **Top-level keys:** `['policy', 'lam', 'suite']`
- **Parameter count:** 2,307 parameters
- **Learned dual bound (lam_L):** `0.0`
- **Validation Score:** `N/A`
- **Weight Tensors:**
  - `net.0.weight`: shape [64, 10], mean=-0.0015, std=0.1882
  - `net.0.bias`: shape [64], mean=-0.0194, std=0.1780
  - `net.2.weight`: shape [1, 64], mean=0.0017, std=0.0706
  - `net.2.bias`: shape [1], mean=-0.0005, std=nan
  - `value_head.0.weight`: shape [64, 10], mean=0.0010, std=0.1840
  - `value_head.0.bias`: shape [64], mean=0.0257, std=0.1746
  - `value_head.2.weight`: shape [1, 64], mean=-0.0047, std=0.0652
  - `value_head.2.bias`: shape [1], mean=0.1065, std=nan
  - `cost_head.0.weight`: shape [64, 10], mean=-0.0133, std=0.1846
  - `cost_head.0.bias`: shape [64], mean=0.0417, std=0.1939
  - `cost_head.2.weight`: shape [1, 64], mean=0.0002, std=0.0665
  - `cost_head.2.bias`: shape [1], mean=-0.0211, std=nan

## 2. Policy Output Comparison on Standard WiLab Observations

| Scenario | State Observation | Initialized Policy (lam, beta) | Preserved Checkpoint (lam, beta) |
| :--- | :--- | :---: | :---: |
| **Low CBR, Zero Reselection** | `cbr=0.05, resel=0.0` | $\lambda=3.79, \beta=0.982$ | $\lambda=0.87, \beta=0.983$ |
| **Mid CBR, Active Reselection** | `cbr=0.15, resel=0.45` | $\lambda=3.79, \beta=0.982$ | $\lambda=0.87, \beta=0.983$ |
| **High CBR (Congested)** | `cbr=0.35, resel=0.85` | $\lambda=3.79, \beta=0.982$ | $\lambda=0.87, \beta=0.983$ |

## 3. Structural Truth & Findings

1. **Identity in Prior Script:** `PB-CRL-core` and `B5'+shield` were pointing to the exact same class `B5PrimePolicy` in `load_policy_for_method()`. There was NO distinct third head ($g_t$) or separately trained neural network instantiated for PB-CRL-core.
2. **Checkpoint State:** `b5_prime_best_preserved_43p86.pt` was trained with `lam_max=1.0` and `lam_L=0.0` under an uncoupled low-density regime, resulting in `mu_lam.bias = -4.0040` (driving $\lambda \to 0$).
3. **Protocol Requirement:** To evaluate PB-CRL-core vs B5'+shield properly, either PB-CRL-core must have its specified 3-head policy $(\lambda, \beta, g)$ with risk account, OR it must be explicitly acknowledged as identical in decision mechanics to B5'+shield.